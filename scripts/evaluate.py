"""Evaluates a trained policy against the authors' NMOPSO on a benchmark leg set.
  python scripts/evaluate.py --ckpt results/ckpt/v3.pt --legs real [--nmopso results/nmopso_real] [--refine]
Fronts per leg (all feasible, non-dominated, scored with the MyCost port; re-scored by MATLAB via
the saved solutions):
  rl         policy rollouts (deterministic + S samples per condition) plus the straight path
  rl_ls      rl plus a short CEM refinement of its paths (--refine; rlpatrol/refine.py)
  straight   the straight, level path alone (the zero action; 1 evaluation)
  random     uniform random NMOPSO particles, same count as rl
  nm<s>      NMOPSO's final repository, seed s; nm<s>+straight adds the straight path
Conditions: budget-conditioned checkpoints sweep U_SWEEP threat budgets x BUDGET_PREFS; others use
K preferences (corners, edge midpoints, centroid, Dirichlet samples).
Metrics: hypervolume with fixed normalisation (ideal 0, nadir F_SCALE, reference 1.1); IGD+ against
the union of every front of the leg (normalised by F_SCALE); additive epsilon to attain NMOPSO's
deployed (short-safe) path; RL vs NMOPSO+straight wins / ties / losses at a 1e-3 HV tolerance."""
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
from rlpatrol.env import F_SCALE, PathEnv  # noqa: E402
from rlpatrol.metrics import add_eps, hypervolume, igd_plus, nondominated, short_safe  # noqa: E402
from rlpatrol.nmopso_cost import ANGLE, N_NODES, Legs  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.refine import WEIGHTS, cem  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

REF = 1.1
TIE = 1e-3
U_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
BUDGET_PREFS = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1 / 3, 0, 1 / 3, 1 / 3]]
SCALE = F_SCALE.numpy()


def pref_set(K, seed=0):
    """corners, edge midpoints, centroid, then Dirichlet(1) samples: K preferences on the 4-simplex."""
    E = np.eye(4)
    P = [*E, *[(E[i] + E[j]) / 2 for i in range(4) for j in range(i + 1, 4)], np.full(4, 0.25)]
    rng = np.random.default_rng(seed)
    while len(P) < K:
        P.append(rng.dirichlet(np.ones(4)))
    return np.array(P[:K])


def conditions(budget, K):
    """(prefs [C, 4], budgets [C] or None)."""
    if budget:
        P = np.array([p for _ in U_SWEEP for p in BUDGET_PREFS], float)
        return P, np.repeat(U_SWEEP, len(BUDGET_PREFS)).astype(float)
    return pref_set(K), None


def rl_solve(actor, world, legs_m, prefs, budgets, S, seed=0):
    """all legs at once: per leg (F [m, 4], enc [m, 3, n]) of all rollouts (actor None: zero action),
    and the wall time."""
    torch.manual_seed(seed)
    C, nL, rep = len(prefs), len(legs_m), 1 + S
    Lr = np.repeat(legs_m, C * rep, 0)
    pr = torch.tensor(np.tile(np.tile(prefs, (rep, 1)), (nL, 1)))
    bu = None if budgets is None else torch.tensor(np.tile(np.tile(budgets, rep), nL))
    det = torch.tensor(([True] * C + [False] * C * S) * nL)
    t0 = time.perf_counter()
    env = PathEnv(world, world.to_units(Lr[:, :3]), world.to_units(Lr[:, 3:]), pr, budget=bu)
    enc = []
    for _ in range(N_NODES):
        with torch.no_grad():
            o = env.obs()
            a = torch.where(det[:, None], actor.act(o, True)[0], actor.act(o)[0]) if actor else torch.zeros(len(Lr), 3)
        enc.append(torch.stack(env.act(a), -1))                    # r, psi, phi
        env.step(a)
    F = env.legs.cost(env.nodes).numpy()
    wall = time.perf_counter() - t0
    enc = torch.stack(enc, -1).numpy()                              # [B, 3, n]
    m = C * rep
    return [(F[i * m:(i + 1) * m], enc[i * m:(i + 1) * m]) for i in range(nL)], wall


def random_search(world, legs_m, m, seed=0):
    """m uniform random NMOPSO particles per leg (CreateRandomSolution.m bounds): the same-budget baseline."""
    g = torch.Generator().manual_seed(seed)
    Lr = np.repeat(legs_m, m, 0)
    lg = Legs(world, world.to_units(Lr[:, :3]), world.to_units(Lr[:, 3:]))
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(len(Lr), N_NODES, generator=g, dtype=torch.float64)
    r = u(lg.rmin[:, None], lg.rmax[:, None]); psi = u(-ANGLE, ANGLE); phi = u(-ANGLE, ANGLE)
    F = lg.cost(lg.decode(r, psi, phi)).numpy()
    return [F[i * m:(i + 1) * m] for i in range(len(legs_m))]


def load_nmopso(d, nL):
    out = {}
    for f in sorted(Path(d).glob("leg*_seed*.mat")):
        k, s = int(f.stem[3:6]) - 1, int(f.stem.split("seed")[1])
        if k < nL:
            M = loadmat(f)
            out.setdefault(k, {})[s] = (np.atleast_2d(M["front"]).T, float(M["time"].squeeze()))
    return out


def path_len_m(world, leg, e):
    lg = Legs(world, world.to_units(leg[None, :3]), world.to_units(leg[None, 3:]))
    e = torch.tensor(e)[None]
    nodes = lg.decode(e[:, 0], e[:, 1], e[:, 2])
    P = torch.cat([lg.start[:, None], nodes, lg.goal[:, None]], 1)[0].numpy() / world.scale
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def feasible_front(F):
    F = np.asarray(F, float)
    F = F[np.isfinite(F).all(1)] if len(F) else F
    return nondominated(F) if len(F) else np.zeros((0, 4))


def hv(F):
    return hypervolume(F / SCALE, np.full(4, REF)) if len(F) else 0.0


def refine(world, legs_m, sols):
    """RL+LS: CEM from the RL front's lowest-threat, short-safe and lowest (F1, F3, F4) paths."""
    starts = []
    for F, enc in sols:
        ok = np.where(np.isfinite(F).all(1))[0]
        if len(ok) == 0:
            ok = np.arange(len(F))
            Fo = np.nan_to_num(F, posinf=1e3)
        else:
            Fo = F[ok]
        picks = [ok[np.argmin(Fo[:, 1])], ok[short_safe(Fo)], ok[np.argmin((Fo[:, [0, 2, 3]] / SCALE[[0, 2, 3]]).sum(1))]]
        starts.append(enc[picks])
    return cem(world, legs_m, np.stack(starts), F_SCALE.to(torch.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--legs", default="real")
    ap.add_argument("--nmopso", default=None)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--S", type=int, default=3)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    world = load_world()
    legs_m = loadmat(DATA / f"legs_{a.legs}.mat")["legs"]
    ck = torch.load(a.ckpt, weights_only=False)
    budget = ck["cfg"].get("budget", False)
    actor = Actor(ck["cfg"]["hidden"], ctx=5 if budget else 4); actor.load_state_dict(ck["actor"]); actor.eval()
    prefs, budgets = conditions(budget, a.K)
    sols, wall = rl_solve(actor, world, legs_m, prefs, budgets, a.S)
    t0 = time.perf_counter(); rl_solve(actor, world, legs_m[:1], prefs, budgets, a.S); lat = time.perf_counter() - t0
    straight, _ = rl_solve(None, world, legs_m, prefs[:1], None if budgets is None else budgets[:1], 0)
    sols = [(np.vstack([F, straight[k][0]]), np.concatenate([e, straight[k][1]])) for k, (F, e) in enumerate(sols)]
    n_rl = len(sols[0][0])
    rand = random_search(world, legs_m, n_rl)
    ls = None
    if a.refine:
        t0 = time.perf_counter(); lsF, lsE = refine(world, legs_m, sols); ls_wall = time.perf_counter() - t0
        ls = [(np.vstack([sols[k][0], lsF[k]]), np.concatenate([sols[k][1], lsE[k]])) for k in range(len(sols))]
    nm = load_nmopso(a.nmopso, len(legs_m)) if a.nmopso else {}
    seeds = sorted({s for v in nm.values() for s in v})
    rows = []
    for k in range(len(legs_m)):
        F, enc = sols[k]
        feas = np.isfinite(F).all(1)
        fr = {"rl": feasible_front(F), "straight": feasible_front(straight[k][0]), "random": feasible_front(rand[k])}
        if ls:
            fr["rl_ls"] = feasible_front(ls[k][0])
        picks = {}
        for s, (f, t) in nm.get(k, {}).items():
            fr[f"nm{s}"] = feasible_front(f)
            fr[f"nm{s}+straight"] = feasible_front(np.vstack([f, straight[k][0]]))
            if len(fr[f"nm{s}"]):
                picks[s] = fr[f"nm{s}"][short_safe(fr[f"nm{s}"])]
        row = dict(leg=k, rl_feasible_frac=float(feas.mean()), rl_front=len(fr["rl"]),
                   **{f"nm{s}_time_s": t for s, (_, t) in nm.get(k, {}).items()})
        if feas.any():
            j = np.where(feas)[0][short_safe(F[feas])]
            row["rl_pick_F"] = F[j].tolist(); row["rl_pick_len_m"] = path_len_m(world, legs_m[k], enc[j])
        ref = feasible_front(np.vstack([f for f in fr.values() if len(f)])) / SCALE if any(len(f) for f in fr.values()) else None
        for name, f in fr.items():
            row[f"hv_{name}"] = hv(f)
            if ref is not None:
                row[f"igd_{name}"] = igd_plus(f / SCALE, ref)
            for s, p in picks.items():                                # attain NMOPSO seed s's deployed path
                row[f"eps_{name}_to_nm{s}pick"] = add_eps(f / SCALE, p / SCALE)
        rows.append(row)

    def mean(key, R):
        v = [r[key] for r in R if key in r and np.isfinite(r[key])]
        return float(np.mean(v)) if v else None

    methods = ["rl", "straight", "random"] + (["rl_ls"] if ls else [])
    summ = {"legs": len(rows), "conditions": len(prefs), "S": a.S, "rl_evals_per_leg": n_rl,
            "rl_ls_evals_per_leg": n_rl + (len(WEIGHTS) * 64 * 3 if ls else 0),
            "rl_legs_with_feasible": int(sum(r["rl_feasible_frac"] > 0 for r in rows)),
            "rl_latency_one_leg_s": lat, "rl_wall_all_legs_s": wall, **({"rl_ls_wall_all_legs_s": ls_wall} if ls else {}),
            **{f"hv_{m}": mean(f"hv_{m}", rows) for m in methods}}
    both = [r for r in rows if all(f"hv_nm{s}" in r for s in seeds)] if seeds else []
    if both:
        cmp = {"legs_compared": len(both), "nm_seeds": seeds,
               "nmopso_time_per_leg_s_median": float(np.median([r[f"nm{s}_time_s"] for r in both for s in seeds]))}
        for m in methods + ["nm", "nm+straight"]:
            keys = [f"hv_nm{s}" for s in seeds] if m == "nm" else [f"hv_nm{s}+straight" for s in seeds] if m == "nm+straight" else [f"hv_{m}"]
            cmp[f"hv_{m}"] = float(np.mean([[r[k] for k in keys] for r in both]))
            ik = [k.replace("hv_", "igd_") for k in keys]
            cmp[f"igd_{m}"] = float(np.mean([[r[k] for k in ik] for r in both]))
        for m in methods:
            for base in ("nm", "nm+straight"):
                d = np.array([[r[f"hv_{m}"] - r[f"hv_nm{s}{'+straight' if base != 'nm' else ''}"] for s in seeds] for r in both])
                cmp[f"{m}_vs_{base}_win_tie_loss"] = [float((d > TIE).mean()), float((abs(d) <= TIE).mean()), float((d < -TIE).mean())]
            e = np.array([r[f"eps_{m}_to_nm{s}pick"] for r in both for s in seeds if f"eps_{m}_to_nm{s}pick" in r])
            cmp[f"{m}_attains_nmopso_pick_frac"] = float((e <= 1e-9).mean())
            cmp[f"{m}_eps_to_nmopso_pick_median"] = float(np.median(e))
        summ["compared"] = cmp
    res = dict(ckpt=a.ckpt, legs=a.legs, summary=summ, rows=rows)
    tag = f"{Path(a.ckpt).stem}_{a.legs}" + ("_ls" if ls else "")
    (ROOT / "results" / f"eval_{tag}.json").write_text(json.dumps(res, indent=1))
    src = ls if ls else sols
    enc = np.concatenate([e for _, e in src]); legi = np.concatenate([np.full(len(e), k + 1) for k, (_, e) in enumerate(src)])
    savemat(ROOT / "results" / f"rl_solutions_{tag}.mat",
            {"legs": legs_m, "leg": legi[:, None].astype(float), "r": enc[:, 0], "psi": enc[:, 1], "phi": enc[:, 2]})
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
