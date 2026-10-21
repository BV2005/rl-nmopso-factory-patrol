"""Path quality and robustness of the path each method would fly (the short-safe pick of its front,
as plan_factory_patrol.m selects), on the 73 real legs x 3 seeds, in physical units:
  quality     length (m), length / straight distance, min clearance to the real obstacle boxes (m),
              max and total turning (deg), max altitude deviation from 1.8 m, flight time at 1 m/s
  tracking    probability of a collision when the waypoints are flown with Gaussian xy error
              sigma = 0.1 / 0.2 / 0.3 m (PX4 SITL tracking error in the patrol demos: max 0.36 m)
  map error   share of paths that collide if the obstacles are delta = 0.1 / 0.2 / 0.3 m larger than modelled
  failure     share of (leg, seed) without a feasible path; on the 200 random legs: RL failures, and
              legs where NMOPSO cannot start (no feasible particle among 1000 random ones: its
              initialisation loop resamples until one is feasible)
Collision = horizontal distance from the flown polyline to a box < DRONE_R (0.3 m, the drone half-size
of factory_layout.m's 0.4 m inflation minus its 0.1 m margin).
  python scripts/robustness.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import conditions, rl_solve  # noqa: E402
from rlpatrol.metrics import short_safe  # noqa: E402
from rlpatrol.nmopso_cost import ANGLE, N_NODES, Legs  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

DRONE_R = 0.3
SIGMAS = (0.1, 0.2, 0.3)
DELTAS = (0.1, 0.2, 0.3)
N_NOISE = 200
SEEDS = (1, 2, 3)


def nm_front(d, k, s):
    M = loadmat(ROOT / "results" / d / f"leg{k + 1:03d}_seed{s}.mat")
    return np.atleast_2d(M["front"]).T, np.stack([np.atleast_2d(M[n]) for n in ("r", "psi", "phi")], 1)


def polyline(world, leg, enc):
    """[n + 2, 3] metres: start, decoded nodes, goal."""
    lg = Legs(world, world.to_units(leg[None, :3]), world.to_units(leg[None, 3:]))
    e = torch.as_tensor(enc)[None]
    nodes = lg.decode(e[:, 0], e[:, 1], e[:, 2])
    return world.to_metres(torch.cat([lg.start[:, None], nodes, lg.goal[:, None]], 1)[0].numpy())


def clearance(P, rects, per_seg=10):
    """min horizontal distance from polylines P [..., m, 3] to the boxes; returns [...]."""
    t = np.linspace(0, 1, per_seg, endpoint=False)
    a, b = P[..., :-1, None, :2], P[..., 1:, None, :2]
    q = (a + t[:, None] * (b - a)).reshape(*P.shape[:-2], -1, 2)
    q = np.concatenate([q, P[..., -1:, :2]], -2)
    dx = np.maximum(np.maximum(rects[:, 0] - q[..., None, 0], 0), q[..., None, 0] - rects[:, 1])
    dy = np.maximum(np.maximum(rects[:, 2] - q[..., None, 1], 0), q[..., None, 1] - rects[:, 3])
    return np.hypot(dx, dy).min((-1, -2))


def clearance_profile(P, rects, ds=0.02):
    """arc length s and clearance c at points every ds metres along the polyline."""
    seg = np.diff(P[:, :2], axis=0); L = np.linalg.norm(seg, axis=1)
    pts, s0 = [], 0.0
    S = []
    for a, e, l in zip(P[:-1, :2], seg, L):
        t = np.arange(0, l, ds) / max(l, 1e-12)
        pts.append(a + t[:, None] * e); S.append(s0 + t * l); s0 += l
    q = np.vstack(pts + [P[-1:, :2]]); s = np.concatenate(S + [[s0]])
    dx = np.maximum(np.maximum(rects[:, 0] - q[:, None, 0], 0), q[:, None, 0] - rects[:, 1])
    dy = np.maximum(np.maximum(rects[:, 2] - q[:, None, 1], 0), q[:, None, 1] - rects[:, 3])
    return s, np.hypot(dx, dy).min(1)


def quality(P, leg, rects):
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=1)
    u = seg / np.maximum(L[:, None], 1e-12)
    ang = np.degrees(np.arccos(np.clip((u[:-1] * u[1:]).sum(1), -1, 1)))
    s, c = clearance_profile(P, rects)
    inner = (s >= 0.5) & (s <= s[-1] - 0.5)
    return {"length_m": float(L.sum()), "length_ratio": float(L.sum() / np.linalg.norm(leg[3:] - leg[:3])),
            "clearance_m": float(clearance(P, rects)),
            "clearance_mean_m": float(c.mean()),
            "clearance_inner_min_m": float(c[inner].min()) if inner.any() else float(c.min()),
            "near_frac": float((c < DRONE_R + 0.3).mean()),       # share of the path within 0.3 m of the drone touching
            "max_turn_deg": float(ang.max()), "total_turn_deg": float(ang.sum()),
            "max_alt_dev_m": float(np.abs(P[1:-1, 2] - 1.8).max()), "time_s_at_1mps": float(L.sum())}


def main():
    torch.set_num_threads(3)
    w = load_world()
    rects = w.rects[:, :4]
    legs = loadmat(DATA / "legs_real.mat")["legs"]
    W = loadmat(DATA / "warm_v2_real.mat")
    rng = np.random.default_rng(0)
    methods = ["rl", "nm500", "nm500+straight", "cold50+straight", "hybrid"]
    Q = {m: [] for m in methods}; fail = {m: 0 for m in methods}; paths = {m: [] for m in methods}
    for k in range(len(legs)):
        c = int(W["cnt"][k, 0])
        Er = np.stack([W[n][k, :c] for n in ("r", "psi", "phi")], 1)
        lg = Legs(w, w.to_units(np.repeat(legs[k:k + 1, :3], c, 0)), w.to_units(np.repeat(legs[k:k + 1, 3:], c, 0)))
        Fr = lg.cost(lg.decode(*(torch.as_tensor(Er[:, i]) for i in range(3)))).numpy()
        for s in SEEDS:
            Fn, En = nm_front("nmopso_real", k, s)
            Fc, Ec = nm_front("nmopso_cold50", k, s)
            Fw, Ew = nm_front("nmopso_warm50", k, s)
            cand = {"rl": (Fr, Er), "nm500": (Fn, En), "nm500+straight": (np.vstack([Fn, Fr[:1]]), np.concatenate([En, Er[:1]])),
                    "cold50+straight": (np.vstack([Fc, Fr[:1]]), np.concatenate([Ec, Er[:1]])),
                    "hybrid": (np.vstack([Fr, Fw]), np.concatenate([Er, Ew]))}
            for m, (F, E) in cand.items():
                ok = np.where(np.isfinite(F).all(1))[0]
                if len(ok) == 0:
                    fail[m] += 1; continue
                j = ok[short_safe(F[ok])]
                P = polyline(w, legs[k], E[j])
                Q[m].append(quality(P, legs[k], rects)); paths[m].append(P)
    out = {"legs": len(legs), "seeds": list(SEEDS), "drone_radius_m": DRONE_R, "methods": {}}
    for m in methods:
        q = {key: np.array([r[key] for r in Q[m]]) for key in Q[m][0]}
        P = np.stack(paths[m])                                            # [N, n + 2, 3]
        row = {"planning_failure_frac": fail[m] / (len(legs) * len(SEEDS))}
        for key, v in q.items():
            row[key] = {"mean": float(v.mean()), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
                        "p95": float(np.percentile(v, 95))}
        row["collides_nominal_frac"] = float((q["clearance_m"] < DRONE_R).mean())
        row["map_error_collision_frac"] = {str(d): float((q["clearance_m"] < DRONE_R + d).mean()) for d in DELTAS}
        row["tracking_collision_prob"] = {}
        for sg in SIGMAS:
            col = []
            for i in range(0, len(P), 16):
                B = P[i:i + 16]
                noise = rng.normal(0, sg, (len(B), N_NOISE, B.shape[1], 2))
                noise[:, :, [0, -1]] = 0                                   # start and goal are exact
                Pn = np.repeat(B[:, None], N_NOISE, 1).copy()
                Pn[..., :2] += noise
                col.append((clearance(Pn, rects) < DRONE_R).mean(1))
            col = np.concatenate(col)
            row["tracking_collision_prob"][str(sg)] = {"mean": float(col.mean()), "p95_over_paths": float(np.percentile(col, 95)),
                                                      "paths_over_5pct": float((col > 0.05).mean())}
        out["methods"][m] = row

    # planning failure on the 200 random legs: RL, and NMOPSO's initialisation
    R = loadmat(DATA / "legs_random.mat")["legs"]
    ck = torch.load(ROOT / "results" / "ckpt" / "v2.pt", weights_only=False)
    actor = Actor(ck["cfg"]["hidden"]); actor.load_state_dict(ck["actor"]); actor.eval()
    prefs, budgets = conditions(False, 50)
    sols, _ = rl_solve(actor, w, R, prefs, budgets, 3)
    straight, _ = rl_solve(None, w, R, prefs[:1], None, 0)
    rl_ok = np.array([np.isfinite(np.vstack([F, straight[k][0]])).all(1).any() for k, (F, _) in enumerate(sols)])
    g = torch.Generator().manual_seed(1)
    nm_p = []
    for k in range(len(R)):
        lg = Legs(w, w.to_units(np.repeat(R[k:k + 1, :3], 1000, 0)), w.to_units(np.repeat(R[k:k + 1, 3:], 1000, 0)))
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(1000, N_NODES, generator=g, dtype=torch.float64)
        F = lg.cost(lg.decode(u(lg.rmin[:, None], lg.rmax[:, None]), u(-ANGLE, ANGLE), u(-ANGLE, ANGLE))).numpy()
        nm_p.append(np.isfinite(F).all(1).mean())
    nm_p = np.array(nm_p)
    Lr = np.linalg.norm(R[:, 3:5] - R[:, :2], axis=1)
    bins = [(1.5, 5), (5, 8), (8, 11), (11, 15)]
    out["random_legs"] = {
        "legs": len(R),
        "rl_failure_frac": float((~rl_ok).mean()),
        "nmopso_cannot_start_frac": float((nm_p == 0).mean()),
        "nmopso_start_prob_lt_50pct_frac": float((1 - (1 - nm_p) ** 100 < 0.5).mean()),
        "rl_fails_where_nmopso_can_start": int((~rl_ok & (nm_p > 0)).sum()),
        "rl_solves_where_nmopso_cannot_start": int((rl_ok & (nm_p == 0)).sum()),
        "by_length_m": {f"{a}-{b}": {"legs": int(((Lr >= a) & (Lr < b)).sum()),
                                     "rl_failure_frac": float((~rl_ok[(Lr >= a) & (Lr < b)]).mean()),
                                     "nmopso_cannot_start_frac": float((nm_p[(Lr >= a) & (Lr < b)] == 0).mean())} for a, b in bins}}
    (ROOT / "results" / "robustness.json").write_text(json.dumps(out, indent=1))

    # compact table
    print(f"{'method':16s} {'fail':>5s} {'len m':>6s} {'ratio':>6s} {'clrmean':>7s} {'inmin p5':>8s} {'near%':>6s} {'maxturn':>7s} "
          f"{'coll0':>5s} {'map+.2':>6s} {'map+.3':>6s} {'trk.1':>6s} {'trk.2':>6s} {'trk.3':>6s}")
    for m, r in out["methods"].items():
        t = r["tracking_collision_prob"]; me = r["map_error_collision_frac"]
        print(f"{m:16s} {r['planning_failure_frac']:5.2f} {r['length_m']['mean']:6.2f} {r['length_ratio']['mean']:6.3f} "
              f"{r['clearance_mean_m']['mean']:7.2f} {r['clearance_inner_min_m']['p5']:8.2f} {100 * r['near_frac']['mean']:6.1f} "
              f"{r['max_turn_deg']['mean']:7.1f} {r['collides_nominal_frac']:5.2f} {me['0.2']:6.2f} {me['0.3']:6.2f} "
              f"{t['0.1']['mean']:6.3f} {t['0.2']['mean']:6.3f} {t['0.3']['mean']:6.3f}")
    print(json.dumps(out["random_legs"], indent=1))


if __name__ == "__main__":
    main()
