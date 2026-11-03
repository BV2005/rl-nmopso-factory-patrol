"""Generalization benchmark of vG (and the vS / vB baselines) on worlds it was not trained on.
  python scripts/eval_worlds.py --ckpt results/ckpt/vG2_seed1.pt [--baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt]
      [--sets heldout,ood,factory2_real,factory2_random,factory2_blocked] [--worlds 50] [--legs 16]
      [--candidates 3] [--eval_noise] [--tag NAME]
Sets: heldout / ood = data/worlds/{heldout,ood}.pkl (scripts/make_worlds.py; generated on the fly if
missing), legs sampled per world; factory2_* = the existing benchmark legs (data/legs_*.mat).

Fairness: every method is evaluated in the same conditions - ideal sensors and no tracking error by
default (the baselines' evaluation), sensor noise for vG and vS with --eval_noise. (Round 1 evaluated vG
with its training domain randomisation, i.e. P4 with 0.1 m tracking error.)

Two protocols:
  quick (default)    every leg 3 deterministic rollouts, balanced preference, budgets u in U; success =
                     feasible for at least one u; vG also with an unexpected box on every leg ("+box");
  --candidates S     the protocol of scripts/evaluate.py: per leg the 48 conditions of the online
                     planner (rlpatrol.planner.conditions) x (1 deterministic + S sampled) rollouts plus
                     the straight path; HV of the feasible front with evaluate.hv (comparable to the
                     hv_rl of results/eval_<ckpt>_<legs>.json on the factory2 sets)."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from rlpatrol.cost3d import cost3d  # noqa: E402
from rlpatrol.env_g import EVAL, ZBAND, Domain, build_batch, factory2_level  # noqa: E402
from rlpatrol.nmopso_cost import N_NODES  # noqa: E402
from rlpatrol.ppo_g import ConfigG, batch_from_levels, load_checkpoint, make_actor_g, rollout_g  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402
from rlpatrol.worldgen import OOD, TRAIN, generate  # noqa: E402

U = [0.3, 0.6, 1.0]
BAL = [1 / 3, 0.0, 1 / 3, 1 / 3]


def gen_set(name, n):
    p = ROOT / "data" / "worlds" / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)[:n]
    seed0, par = (10_000_000, TRAIN) if name == "heldout" else (20_000_000, OOD)
    return [generate(seed0 + i, par) for i in range(n)]


def sub_batch(b, idx):
    return {k: v[idx] for k, v in b.items()}


# --- quick protocol ---------------------------------------------------------------------------------------
def eval_vg(actor, cfg, batch, domain):
    B = len(batch["legs"])
    idx = torch.arange(B).repeat(len(U))
    pref = torch.tensor([BAL] * (B * len(U)), dtype=torch.float64)
    u = torch.tensor(np.repeat(U, B), dtype=torch.float64) if cfg["budget"] else None
    _, env = rollout_g(actor, sub_batch(batch, idx), pref, u, cfg, deterministic=True, priv=False, domain=domain)
    feas = env.feasible.reshape(len(U), B)
    F = env.F[env.feasible]
    return dict(legs=B, success=feas.any(0).float().mean().item(), feasible=feas.float().mean().item(),
                crash=env.crashed.float().mean().item(), backtracks=env.backs.float().mean().item(),
                climbs=env.n_climb.float().mean().item(), F=F.mean(0).tolist() if len(F) else [float("nan")] * 4)


def eval_baseline(ck, worlds_legs, noise):
    """vS / vB with their own environment (rlpatrol.ppo.rollout) on each world's 2D model."""
    from rlpatrol.ppo import env_kwargs, make_actor, rollout
    cfg = ck["cfg"]
    actor = make_actor(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    feas_all, F_all, succ = [], [], []
    for world, legs in worlds_legs:
        n = len(legs)
        L = np.tile(legs, (len(U), 1))
        pref = torch.tensor([BAL] * (n * len(U)), dtype=torch.float64)
        u = torch.tensor(np.repeat(U, n), dtype=torch.float64) if cfg.get("budget", False) else None
        _, _, _, _, info, _ = rollout(actor, world, L, pref, True, budget=u, **env_kwargs(cfg, noise=noise, critic=False))
        f = info["feasible"].reshape(len(U), n)
        feas_all.append(f.float().mean().item()); succ.append(f.any(0).float().mean().item())
        if info["feasible"].any():
            F_all.append(info["F"][info["feasible"]])
    F = torch.cat(F_all) if F_all else torch.zeros(0, 4)
    return dict(success=float(np.mean(succ)), feasible=float(np.mean(feas_all)),
                F=F.mean(0).tolist() if len(F) else [float("nan")] * 4)


# --- candidate protocol (scripts/evaluate.py) -----------------------------------------------------------------------
def straight_F(batch):
    legs = batch["legs"]
    s = torch.linspace(0, 1, N_NODES + 2, dtype=legs.dtype)[None, :, None]
    P = legs[:, None, :3] + s * (legs[:, None, 3:] - legs[:, None, :3])
    return cost3d(P, torch.full((len(legs),), N_NODES + 2), batch["cyl"], batch["cmask"], ZBAND).numpy()


def vg_candidates(actor, cfg, batch, S, domain, chunk=8):
    """per leg: 48 conditions x (1 + S) rollouts + the straight path -> (HV list, success list)."""
    from evaluate import feasible_front, hv
    from rlpatrol.planner import conditions
    prefs, budgets = conditions(cfg["budget"], 50)
    C = len(prefs)
    B = len(batch["legs"])
    st = straight_F(batch)
    hvs, succ = [], []
    for c0 in range(0, B, chunk):
        legs = torch.arange(c0, min(B, c0 + chunk))
        n = len(legs)
        Fs = []
        for det in [True] + [False] * S:
            idx = legs.repeat_interleave(C)
            pr = torch.tensor(np.tile(prefs, (n, 1)), dtype=torch.float64)
            bu = torch.tensor(np.tile(budgets, n), dtype=torch.float64) if budgets is not None else None
            _, env = rollout_g(actor, sub_batch(batch, idx), pr, bu, cfg, deterministic=det, priv=False, domain=domain)
            Fs.append(env.F.reshape(n, C, 4).numpy())
        F = np.concatenate(Fs, 1)
        for j, k in enumerate(legs.tolist()):
            ok = np.isfinite(F[j]).all(1)
            succ.append(bool(ok.any()))
            hvs.append(hv(feasible_front(np.vstack([F[j], st[k][None]]))))
    return hvs, succ


def baseline_candidates(ck, worlds_legs, S, noise):
    from evaluate import feasible_front, hv, rl_solve, sensor_kw
    from rlpatrol.planner import conditions
    from rlpatrol.ppo import make_actor
    cfg = ck["cfg"]
    actor = make_actor(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    prefs, budgets = conditions(cfg.get("budget", False), 50)
    hvs, succ = [], []
    for world, legs in worlds_legs:
        sols, _ = rl_solve(actor, world, legs, prefs, budgets, S, budget_obs=cfg.get("budget_obs", False),
                           route_obs=cfg.get("route_obs", False), **sensor_kw(cfg, noise))
        straight, _ = rl_solve(None, world, legs, prefs[:1], None if budgets is None else budgets[:1], 0)
        for (F, _), (Fst, _) in zip(sols, straight):
            succ.append(bool(np.isfinite(F).all(1).any()))
            hvs.append(hv(feasible_front(np.vstack([F, Fst]))))
    return hvs, succ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baselines", default="")
    ap.add_argument("--sets", default="heldout,ood,factory2_real,factory2_random,factory2_blocked")
    ap.add_argument("--worlds", type=int, default=50)
    ap.add_argument("--legs", type=int, default=16)
    ap.add_argument("--candidates", type=int, default=0, help="S > 0: the candidate-set protocol with S sampled rollouts")
    ap.add_argument("--eval_noise", action="store_true", help="sensor noise for vG and vS (default: ideal sensors)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    domain = Domain(noise=True) if a.eval_noise else EVAL
    ck = load_checkpoint(a.ckpt)
    cfg = ck["cfg"]
    actor = make_actor_g(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    base = [(Path(p).stem, load_checkpoint(p)) for p in a.baselines.split(",") if p]
    res = {"ckpt": a.ckpt, "it": ck.get("it"), "cfg": cfg, "U": U, "eval_noise": a.eval_noise,
           "candidates": a.candidates, "sets": {}}
    for s in a.sets.split(","):
        out = {}
        if s.startswith("factory2_"):
            w = load_world()
            legs = loadmat(ROOT / "data" / f"legs_{s.split('_', 1)[1]}.mat")["legs"]
            hall, boxes, cyl = factory2_level(w)
            batch = build_batch([(hall, boxes, cyl, l, None) for l in legs])
            worlds_legs = [(w, legs)]
            box_batch = None
        else:
            W = gen_set(s, a.worlds)
            levels = [{"world": i, "legs_seed": 4242 + i} for i in range(len(W))]
            known = {k: v for k, v in cfg.items() if k in ConfigG.__dataclass_fields__}
            batch = batch_from_levels(levels, W, a.legs, ConfigG(**{**known, "p_extra": 0.0}), "cpu")
            box_batch = batch_from_levels(levels, W, a.legs, ConfigG(**{**known, "p_extra": 1.0}), "cpu")
            L = batch["legs"].numpy().reshape(len(W), a.legs, 6)
            worlds_legs = [(W[i].to_world(), L[i]) for i in range(len(W))]
        if a.candidates:
            h, sc = vg_candidates(actor, cfg, batch, a.candidates, domain)
            out["vG"] = dict(hv=float(np.mean(h)), success=float(np.mean(sc)), legs=len(h))
            for name, b in base:
                h, sc = baseline_candidates(b, worlds_legs, a.candidates, a.eval_noise)
                out[name] = dict(hv=float(np.mean(h)), success=float(np.mean(sc)), legs=len(h))
        else:
            out["vG"] = eval_vg(actor, cfg, batch, domain)
            if box_batch is not None:
                out["vG+box"] = eval_vg(actor, cfg, box_batch, domain)
            for name, b in base:
                out[name] = eval_baseline(b, worlds_legs, a.eval_noise)
        res["sets"][s] = out
        print(f"== {s}")
        for k, v in out.items():
            if "hv" in v:
                print(f"  {k:14s} HV {v['hv']:.4f}  success {v['success']*100:5.1f} %  ({v['legs']} legs)", flush=True)
            else:
                print(f"  {k:14s} success {v['success']*100:5.1f} %  feasible {v['feasible']*100:5.1f} %"
                      + (f"  crash {v['crash']*100:4.1f} %  back {v['backtracks']:.2f}  climb {v['climbs']:.2f}" if "crash" in v else "")
                      + f"  F {np.round(v['F'], 4).tolist()}", flush=True)
    tag = a.tag or (Path(a.ckpt).stem + ("_cand" if a.candidates else ""))
    p = ROOT / "results" / f"eval_g_{tag}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
