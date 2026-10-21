"""Evaluates a trained policy against the authors' NMOPSO on a benchmark leg set.
  python scripts/evaluate.py --ckpt results/ckpt/v0.pt --legs real [--nmopso results/nmopso_real]
RL front of a leg: deterministic paths for K preferences plus S stochastic samples each, the feasible
non-dominated ones. NMOPSO front: the final repository of each seed (scored by MATLAB MyCost.m).
Per leg the objectives are normalised by the ideal / nadir of the union of all fronts of that leg
(RL and every NMOPSO seed); hypervolume uses the reference point 1.1 in every objective.
Writes results/eval_<ckpt>_<legs>.json and the RL solutions for MATLAB re-scoring (matlab/eval_mycost.m)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.env import PathEnv  # noqa: E402
from rlpatrol.metrics import hypervolume, nondominated, short_safe  # noqa: E402
from rlpatrol.nmopso_cost import N_NODES  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

REF = 1.1


def pref_set(K, seed=0):
    """corners, edge midpoints, centroid, then Dirichlet(1) samples: K preferences on the 4-simplex."""
    E = np.eye(4)
    P = [*E, *[(E[i] + E[j]) / 2 for i in range(4) for j in range(i + 1, 4)], np.full(4, 0.25)]
    rng = np.random.default_rng(seed)
    while len(P) < K:
        P.append(rng.dirichlet(np.ones(4)))
    return torch.tensor(np.array(P[:K]))


def rl_solve(actor, world, legs_m, K, S, seed=0):
    """all legs at once: returns per leg (F [m, 4], enc [m, 3, n]) of all rollouts, and the wall time."""
    P = pref_set(K)
    torch.manual_seed(seed)
    nL = len(legs_m)
    rep = 1 + S
    Lr = np.repeat(legs_m, K * rep, 0)
    pr = P.repeat(rep, 1).repeat(nL, 1)
    det = torch.tensor(([True] * K + [False] * K * S) * nL)
    t0 = time.perf_counter()
    env = PathEnv(world, world.to_units(Lr[:, :3]), world.to_units(Lr[:, 3:]), pr)
    enc = []
    for _ in range(N_NODES):
        with torch.no_grad():
            d = actor.dist(env.obs())
            a = torch.where(det[:, None], d.mean, d.sample()).clamp(-1, 1)
        enc.append(torch.stack(env.act(a), -1))                    # r, psi, phi
        env.step(a)
    F = env.legs.cost(env.nodes).numpy()
    wall = time.perf_counter() - t0
    enc = torch.stack(enc, -1).numpy()                              # [B, 3, n]
    m = K * rep
    return [(F[i * m:(i + 1) * m], enc[i * m:(i + 1) * m]) for i in range(nL)], wall


def load_nmopso(d, nL):
    out = {}
    for f in sorted(Path(d).glob("leg*_seed*.mat")):
        k, s = int(f.stem[3:6]) - 1, int(f.stem.split("seed")[1])
        if k < nL:
            M = loadmat(f)
            out.setdefault(k, {})[s] = (np.atleast_2d(M["front"]).T, float(M["time"].squeeze()))
    return out


def path_metrics(world, leg, F_enc):
    """length (m) of the decoded path start -> nodes -> goal."""
    from rlpatrol.nmopso_cost import Legs
    lg = Legs(world, world.to_units(leg[None, :3]), world.to_units(leg[None, 3:]))
    e = torch.tensor(F_enc)[None]
    nodes = lg.decode(e[:, 0], e[:, 1], e[:, 2])
    P = torch.cat([lg.start[:, None], nodes, lg.goal[:, None]], 1)[0].numpy() / world.scale
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--legs", default="real")
    ap.add_argument("--nmopso", default=None)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    world = load_world()
    legs_m = loadmat(DATA / f"legs_{a.legs}.mat")["legs"]
    ck = torch.load(a.ckpt, weights_only=False)
    actor = Actor(ck["cfg"]["hidden"]); actor.load_state_dict(ck["actor"]); actor.eval()
    sols, wall = rl_solve(actor, world, legs_m, a.K, a.S)
    # single-leg latency (one leg, all K * (1 + S) rollouts)
    t0 = time.perf_counter(); rl_solve(actor, world, legs_m[:1], a.K, a.S); lat = time.perf_counter() - t0
    nm = load_nmopso(a.nmopso, len(legs_m)) if a.nmopso else {}
    rows = []
    for k, (F, enc) in enumerate(sols):
        feas = np.isfinite(F).all(1)
        rf = nondominated(F[feas]) if feas.any() else np.zeros((0, 4))
        row = dict(leg=k, rl_feasible_frac=float(feas.mean()), rl_front=len(rf))
        if feas.any():
            j = np.where(feas)[0][short_safe(F[feas])]
            row["rl_pick_F"] = F[j].tolist(); row["rl_pick_len_m"] = path_metrics(world, legs_m[k], enc[j])
        fronts = {"rl": rf}
        for s, (fr, t) in nm.get(k, {}).items():
            fr = fr[np.isfinite(fr).all(1)]
            fronts[f"nm{s}"] = fr
            row[f"nm{s}_time_s"] = t
        allF = np.vstack([f for f in fronts.values() if len(f)]) if any(len(f) for f in fronts.values()) else None
        if allF is not None:
            lo, hi = allF.min(0), allF.max(0)
            span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
            for name, fr in fronts.items():
                row[f"hv_{name}"] = hypervolume((fr - lo) / span, np.full(4, REF)) if len(fr) else 0.0
        rows.append(row)
    res = dict(ckpt=a.ckpt, legs=a.legs, K=a.K, S=a.S, rl_wall_all_s=wall, rl_latency_one_leg_s=lat, rows=rows)
    seeds = sorted({s for v in nm.values() for s in v})
    summ = {"legs": len(rows), "rl_legs_with_feasible": int(sum(r["rl_feasible_frac"] > 0 for r in rows)),
            "rl_feasible_frac_mean": float(np.mean([r["rl_feasible_frac"] for r in rows])),
            "rl_latency_one_leg_s": lat}
    both = [r for r in rows if all(f"hv_nm{s}" in r for s in seeds)] if seeds else []
    if both:
        hr = np.array([r["hv_rl"] for r in both]); hn = np.array([[r[f"hv_nm{s}"] for s in seeds] for r in both])
        summ.update({"legs_compared": len(both), "hv_rl_mean": float(hr.mean()), "hv_nmopso_mean": float(hn.mean()),
                     "rl_beats_nmopso_seed_frac": float((hr[:, None] > hn).mean()),
                     "nmopso_time_per_leg_s_median": float(np.median([r[f"nm{s}_time_s"] for r in both for s in seeds]))})
    res["summary"] = summ
    out = ROOT / "results" / f"eval_{Path(a.ckpt).stem}_{a.legs}.json"
    out.write_text(json.dumps(res, indent=1))
    enc = np.concatenate([e for _, e in sols]); legi = np.repeat(np.arange(len(sols)) + 1, len(sols[0][1]))
    savemat(ROOT / "results" / f"rl_solutions_{Path(a.ckpt).stem}_{a.legs}.mat",
            {"legs": legs_m, "leg": legi[:, None].astype(float), "r": enc[:, 0], "psi": enc[:, 1], "phi": enc[:, 2]})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
