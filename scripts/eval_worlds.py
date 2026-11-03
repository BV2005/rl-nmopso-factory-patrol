"""Generalization benchmark of vG (and the vS / vB baselines) on worlds it was not trained on.
  python scripts/eval_worlds.py --ckpt results/ckpt/vG_p4_seed1.pt [--baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt]
      [--sets heldout,ood,factory2_real,factory2_random,factory2_blocked] [--worlds 50] [--legs 16] [--tag NAME]
Sets: heldout / ood = data/worlds/{heldout,ood}.pkl (scripts/make_worlds.py; generated on the fly if
missing), legs sampled per world; factory2_* = the existing benchmark legs (data/legs_*.mat).
Every leg is run deterministically with the balanced preference and threat budgets u in U (3 rollouts).
vG is also run with an unexpected box on every leg long enough (the "+box" rows); the baselines see the
worlds as their 2D NMOPSO model (no unexpected boxes: they cannot sense heights / do not know them).
Reported per set: success = feasible path for at least one u, feasible share over all rollouts, crash
share (vG), mean F of the feasible paths, backtracks and climbs per leg (vG)."""
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
from rlpatrol.env_g import build_batch, factory2_level  # noqa: E402
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


def conditions(B, budget):
    pref = torch.tensor([BAL] * (B * len(U)), dtype=torch.float64)
    u = torch.tensor(np.repeat([U], B, 0).T.ravel(), dtype=torch.float64) if budget else None
    return pref, u


def repeat_batch(b, k):
    return {key: v.repeat(k, *([1] * (v.dim() - 1))) for key, v in b.items()}


def eval_vg(actor, cfg, batch):
    B = len(batch["legs"])
    pref, u = conditions(B, cfg["budget"])
    _, env = rollout_g(actor, repeat_batch(batch, len(U)), pref, u, cfg, deterministic=True, priv=False)
    feas = env.feasible.reshape(len(U), B)
    F = env.F[env.feasible]
    return dict(legs=B, success=feas.any(0).float().mean().item(), feasible=feas.float().mean().item(),
                crash=env.crashed.float().mean().item(), backtracks=env.backs.float().mean().item(),
                climbs=env.n_climb.float().mean().item(), F=F.mean(0).tolist() if len(F) else [float("nan")] * 4)


def eval_baseline(ck, worlds_legs):
    """vS / vB with their own environment (rlpatrol.ppo.rollout) on each world's 2D model."""
    from rlpatrol.ppo import env_kwargs, make_actor, rollout
    cfg = ck["cfg"]
    actor = make_actor(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    feas_all, F_all, succ = [], [], []
    for world, legs in worlds_legs:
        n = len(legs)
        L = np.tile(legs, (len(U), 1))
        pref, u = conditions(n, cfg.get("budget", False))
        _, _, _, _, info, _ = rollout(actor, world, L, pref, True, budget=u, **env_kwargs(cfg, noise=False, critic=False))
        f = info["feasible"].reshape(len(U), n)
        feas_all.append(f.float().mean().item()); succ.append(f.any(0).float().mean().item())
        if info["feasible"].any():
            F_all.append(info["F"][info["feasible"]])
    F = torch.cat(F_all) if F_all else torch.zeros(0, 4)
    return dict(success=float(np.mean(succ)), feasible=float(np.mean(feas_all)),
                F=F.mean(0).tolist() if len(F) else [float("nan")] * 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--baselines", default="")
    ap.add_argument("--sets", default="heldout,ood,factory2_real,factory2_random,factory2_blocked")
    ap.add_argument("--worlds", type=int, default=50)
    ap.add_argument("--legs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = load_checkpoint(a.ckpt)
    cfg = ck["cfg"]
    actor = make_actor_g(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    base = [(Path(p).stem, load_checkpoint(p)) for p in a.baselines.split(",") if p]
    res = {"ckpt": a.ckpt, "it": ck.get("it"), "cfg": cfg, "U": U, "sets": {}}
    for s in a.sets.split(","):
        out = {}
        if s.startswith("factory2_"):
            w = load_world()
            legs = loadmat(ROOT / "data" / f"legs_{s.split('_', 1)[1]}.mat")["legs"]
            hall, boxes, cyl = factory2_level(w)
            batch = build_batch([(hall, boxes, cyl, l, None) for l in legs])
            out["vG"] = eval_vg(actor, cfg, batch)
            worlds_legs = [(w, legs)]
        else:
            W = gen_set(s, a.worlds)
            levels = [{"world": i, "legs_seed": 4242 + i} for i in range(len(W))]
            c0 = ConfigG(**{**{k: v for k, v in cfg.items() if k in ConfigG.__dataclass_fields__}, "p_extra": 0.0})
            batch = batch_from_levels(levels, W, a.legs, c0, "cpu")
            out["vG"] = eval_vg(actor, cfg, batch)
            c1 = ConfigG(**{**c0.__dict__, "p_extra": 1.0})
            out["vG+box"] = eval_vg(actor, cfg, batch_from_levels(levels, W, a.legs, c1, "cpu"))
            L = batch["legs"].numpy().reshape(len(W), a.legs, 6)
            worlds_legs = [(W[i].to_world(), L[i]) for i in range(len(W))]
        for name, b in base:
            out[name] = eval_baseline(b, worlds_legs)
        res["sets"][s] = out
        print(f"== {s}")
        for k, v in out.items():
            print(f"  {k:14s} success {v['success']*100:5.1f} %  feasible {v['feasible']*100:5.1f} %"
                  + (f"  crash {v['crash']*100:4.1f} %  back {v['backtracks']:.2f}  climb {v['climbs']:.2f}" if "crash" in v else "")
                  + f"  F {np.round(v['F'], 4).tolist()}", flush=True)
    tag = a.tag or Path(a.ckpt).stem
    p = ROOT / "results" / f"eval_g_{tag}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
