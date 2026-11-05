"""Common benchmark of vT (time-minimising, encoding-free) against vG, vS and vB: every method's path is
scored by the same judge (rlpatrol.env_t.flight_metrics): flight time with the speed model, collisions with
the box model (obstacles + 0.5 m), threat, altitude, turn bound - and every method's planning time is measured.
  python scripts/eval_t.py --ckpt results/ckpt/vT_seed1.pt [--vg results/ckpt/vG3_box_seed1.pt]
      [--baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt]
      [--sets heldout,ood,factory2_real,factory2_random,factory2_blocked] [--worlds 50] [--legs 16] [--tag NAME]
Every method plans each leg once, deterministically, with its balanced preference (vT: time / threat / alt
1/3 each; vG / vS / vB: F1 / F3 / F4 1/3 each with threat budget u = 0.6). Success = the path reaches the
goal with no collision in the box model. vT and vG are also run with an unexpected box on every leg
(+box; the baselines cannot sense heights / their map does not have it). Planning time = wall time of
planning one leg alone (mean over --latency legs), on this machine's CPU."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.env_g import EVAL, build_batch, factory2_level  # noqa: E402
from rlpatrol.env_t import TURN_MAX, KAPPA, flight_metrics  # noqa: E402
from rlpatrol.nmopso_cost import N_NODES  # noqa: E402
from rlpatrol.ppo_g import ConfigG, batch_from_levels, load_checkpoint, make_actor_g, rollout_g  # noqa: E402
from rlpatrol.ppo_t import make_actor_t, rollout_t  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402
from rlpatrol.worldgen import OOD, TRAIN, generate  # noqa: E402


def gen_set(name, n):
    p = ROOT / "data" / "worlds" / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)[:n]
    seed0, par = (10_000_000, TRAIN) if name == "heldout" else (20_000_000, OOD)
    return [generate(seed0 + i, par) for i in range(n)]


def obstacles(batch):
    m = batch["kmask"].clone()
    m[:, -4:] = False
    return m


def walls(batch):
    return batch["kmask"] & ~obstacles(batch)


def judge(P, n, batch, reached):
    """the common metrics of the flown paths P [B, N, 3] (n points; backtracks included) that `reached` the goal;
    a path through an obstacle's or a wall's margin collides."""
    m = flight_metrics(P, n, batch["boxes"], obstacles(batch), walls=walls(batch))
    ok = reached & ~m["collides"]
    tr = m["time_s"] / m["straight_s"]
    f = lambda x: float(x[ok].mean()) if ok.any() else float("nan")  # noqa: E731
    return dict(success=float(ok.float().mean()), time_ratio=f(tr), time_s=f(m["time_s"]), length_m=f(m["length_m"]),
                threat=f(m["F"][:, 1]), alt=f(m["F"][:, 2]), max_turn_deg=float(np.degrees(m["max_turn"][ok].max())) if ok.any() else float("nan"),
                turn_rate_violations=float(((m["turn_per_m"] > KAPPA * 1.001) & ok).float().mean()),
                collides=float(m["collides"].float().mean()))


def timed(fn, reps):
    fn()                                                         # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def one(batch, i):
    return {k: v[i:i + 1] for k, v in batch.items()}


# --- the methods ---------------------------------------------------------------------------------------------
def run_vt(actor, cfg, batch):
    B = len(batch["legs"])
    pref = torch.full((B, 3), 1 / 3, dtype=torch.float64)
    _, env = rollout_t(actor, batch, pref, cfg, deterministic=True, priv=False, domain=EVAL)
    P, n = env.executed()
    r = judge(P, n, batch, env.success)
    r.update(crash=float(env.crashed.float().mean()), timeout=float((~env.success & ~env.crashed).float().mean()),
             backtracks=float(env.backs.float().mean()), climbs=float(env.n_climb.float().mean()),
             decisions=float(env.k.float().mean()))
    r["plan_ms"] = timed(lambda: rollout_t(actor, one(batch, 0), pref[:1], cfg, deterministic=True, priv=False, domain=EVAL), 5)
    return r


def run_vg(actor, cfg, batch):
    B = len(batch["legs"])
    pref = torch.tensor([[1 / 3, 0, 1 / 3, 1 / 3]] * B, dtype=torch.float64)
    u = torch.full((B,), 0.6, dtype=torch.float64) if cfg.get("budget", True) else None
    _, env = rollout_g(actor, batch, pref, u, cfg, deterministic=True, priv=False, domain=EVAL)
    P, n = env.executed()
    r = judge(P, n, batch, ~env.crashed)
    r.update(crash=float(env.crashed.float().mean()), backtracks=float(env.backs.float().mean()))
    r["plan_ms"] = timed(lambda: rollout_g(actor, one(batch, 0), pref[:1], None if u is None else u[:1], cfg,
                                           deterministic=True, priv=False, domain=EVAL), 5)
    return r


def run_baseline(ck, worlds_legs, batch):
    """vS / vB plan in their own (NMOPSO) environment on each world's 2D model; their 10-node paths are
    then judged like every other method's."""
    from rlpatrol.ppo import env_kwargs, make_actor, rollout
    cfg = ck["cfg"]
    actor = make_actor(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    Ps = []
    lat = None
    for world, legs in worlds_legs:
        n = len(legs)
        pref = torch.tensor([[1 / 3, 0, 1 / 3, 1 / 3]] * n, dtype=torch.float64)
        u = torch.full((n,), 0.6, dtype=torch.float64) if cfg.get("budget", False) else None
        kw = env_kwargs(cfg, noise=False, critic=False)
        _, _, _, _, _, env = rollout(actor, world, legs, pref, True, budget=u, **kw)
        nodes = world.to_metres(env.nodes.cpu().numpy())
        Ps.append(np.concatenate([legs[:, None, :3], nodes, legs[:, None, 3:]], 1))
        if lat is None:
            lat = timed(lambda: rollout(actor, world, legs[:1], pref[:1], True, budget=None if u is None else u[:1], **kw), 5)
    P = torch.tensor(np.concatenate(Ps), dtype=torch.float64)
    r = judge(P, torch.full((len(P),), N_NODES + 2), batch, torch.ones(len(P), dtype=torch.bool))
    r["plan_ms"] = lat
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vg", default="")
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
    actor = make_actor_t(cfg); actor.load_state_dict(ck["actor"]); actor.eval()
    vgs = []
    for p in [p for p in a.vg.split(",") if p]:
        c = load_checkpoint(p)
        ag = make_actor_g(c["cfg"]); ag.load_state_dict(c["actor"]); ag.eval()
        vgs.append((Path(p).stem, ag, c["cfg"]))
    base = [(Path(p).stem, load_checkpoint(p)) for p in a.baselines.split(",") if p]
    res = {"ckpt": a.ckpt, "it": ck.get("it"), "cfg": cfg, "sets": {}}
    for s in a.sets.split(","):
        out = {}
        if s.startswith("factory2_"):
            w = load_world()
            legs = loadmat(ROOT / "data" / f"legs_{s.split('_', 1)[1]}.mat")["legs"]
            hall, boxes, cyl = factory2_level(w)
            batch = build_batch([(hall, boxes, cyl, l, None) for l in legs])
            box_batch, worlds_legs = None, [(w, legs)]
        else:
            W = gen_set(s, a.worlds)
            levels = [{"world": i, "legs_seed": 4242 + i} for i in range(len(W))]
            c0 = ConfigG(p_extra=0.0, lmin=cfg.get("lmin", 1.5), lmax=cfg.get("lmax", 12.0))
            batch = batch_from_levels(levels, W, a.legs, c0, "cpu")
            box_batch = batch_from_levels(levels, W, a.legs, ConfigG(**{**c0.__dict__, "p_extra": 1.0}), "cpu")
            L = batch["legs"].numpy().reshape(len(W), a.legs, 6)
            worlds_legs = [(W[i].to_world(), L[i]) for i in range(len(W))]
        out["vT"] = run_vt(actor, cfg, batch)
        if box_batch is not None:
            out["vT+box"] = run_vt(actor, cfg, box_batch)
        for name, ag, c in vgs:
            out[name] = run_vg(ag, c, batch)
            if box_batch is not None:
                out[name + "+box"] = run_vg(ag, c, box_batch)
        for name, b in base:
            out[name] = run_baseline(b, worlds_legs, batch)
        res["sets"][s] = out
        print(f"== {s}")
        for k, v in out.items():
            print(f"  {k:18s} success {v['success']*100:5.1f} %  time/straight {v['time_ratio']:.3f}  time {v['time_s']:.1f} s  "
                  f"len {v['length_m']:.1f} m  threat {v['threat']:.3f}  alt {v['alt']:.3f}  max turn {v['max_turn_deg']:.0f} deg  "
                  f"turn-rate viol {v['turn_rate_violations']*100:.1f} %  plan {v['plan_ms']:.1f} ms/leg"
                  + (f"  crash {v['crash']*100:.1f} %" if "crash" in v else ""), flush=True)
    tag = a.tag or Path(a.ckpt).stem
    p = ROOT / "results" / f"eval_t_{tag}.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
