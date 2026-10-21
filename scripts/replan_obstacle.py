"""Robustness to an unexpected obstacle: a 0.6 m object (person, cart) appears at the midpoint of a
real leg (legs >= 3 m whose midpoint is free and >= 1.2 m from both ends). The planner model sees it
as a cylinder of radius 0.3 + 0.4 m inflation = 0.7 m (CreateModel.m extra_threats).
  prep     - writes data/legs_obstacle.mat (legs, extra, idx), replans with RL in the new world
             (timed per leg) and writes warm seeds data/warm_v2_obstacle.mat
  (MATLAB) - run_nmopso_variant(legs_obstacle, results/obst_cold50, 1:3, 4, 50, '', 300)
             run_nmopso_variant(legs_obstacle, results/obst_warm50, 1:3, 4, 50, warm_v2_obstacle, 300)
  analyse  - does each method's originally deployed path hit the new obstacle; replanning success,
             time, hypervolume and deployed-path quality per method
  python scripts/replan_obstacle.py prep|analyse"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import SCALE, TIE, conditions, feasible_front, hv, rl_solve  # noqa: E402
from make_warm import pick  # noqa: E402
from robustness import DRONE_R, nm_front, polyline, quality  # noqa: E402
from rlpatrol.metrics import short_safe  # noqa: E402
from rlpatrol.nmopso_cost import Legs  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.world import DATA, load_world, with_extra  # noqa: E402

OBST_HALF = 0.3            # physical half-width of the unexpected obstacle (m)
R_MODEL = OBST_HALF + 0.4  # as factory_layout.m inflates obstacles
SEEDS = (1, 2, 3)


def load_actor():
    ck = torch.load(ROOT / "results" / "ckpt" / "v2.pt", weights_only=False)
    a = Actor(ck["cfg"]["hidden"]); a.load_state_dict(ck["actor"]); a.eval()
    return a


def costs(world, leg, E):
    lg = Legs(world, world.to_units(np.repeat(leg[None, :3], len(E), 0)), world.to_units(np.repeat(leg[None, 3:], len(E), 0)))
    return lg.cost(lg.decode(*(torch.as_tensor(E[:, i]) for i in range(3)))).numpy()


def prep():
    torch.set_num_threads(3)
    w = load_world()
    legs = loadmat(DATA / "legs_real.mat")["legs"]
    sel, extra = [], []
    for k, L in enumerate(legs):
        a, b = L[:2], L[3:5]
        mid = (a + b) / 2
        if np.linalg.norm(b - a) >= 3.0 and w.free(*mid) and min(np.linalg.norm(mid - a), np.linalg.norm(mid - b)) >= R_MODEL + 0.5:
            sel.append(k); extra.append([*mid, R_MODEL])
    sel, extra = np.array(sel), np.array(extra)
    actor = load_actor()
    prefs, budgets = conditions(False, 50)
    M = 50
    n = 10
    R_, PS, PH, cnt, times = (np.zeros((len(sel), M, n)), np.zeros((len(sel), M, n)), np.zeros((len(sel), M, n)),
                              np.zeros(len(sel)), np.zeros(len(sel)))
    for i, (k, ex) in enumerate(zip(sel, extra)):
        wk = with_extra(w, ex)
        t0 = time.perf_counter()
        sols, _ = rl_solve(actor, wk, legs[k:k + 1], prefs, budgets, 3)
        straight, _ = rl_solve(None, wk, legs[k:k + 1], prefs[:1], None, 0)
        F = np.vstack([sols[0][0], straight[0][0]]); E = np.concatenate([sols[0][1], straight[0][1]])
        idx = pick(F, M - 1)
        times[i] = time.perf_counter() - t0
        e = np.concatenate([straight[0][1][:1], E[idx]])
        R_[i, :len(e)], PS[i, :len(e)], PH[i, :len(e)] = e[:, 0], e[:, 1], e[:, 2]
        cnt[i] = len(e)
    savemat(DATA / "legs_obstacle.mat", {"legs": legs[sel], "extra": extra, "idx": sel[:, None].astype(float)})
    savemat(DATA / "warm_v2_obstacle.mat", {"r": R_, "psi": PS, "phi": PH, "cnt": cnt[:, None], "wall_s": times.sum(), "times": times})
    print(f"{len(sel)} obstacle legs of {len(legs)}; RL replanning {np.median(times):.2f} s median per leg (incl. straight check)")


def analyse():
    torch.set_num_threads(3)
    w = load_world()
    O = loadmat(DATA / "legs_obstacle.mat"); legs, extra, idx = O["legs"], O["extra"], O["idx"].ravel().astype(int)
    Wo = loadmat(DATA / "warm_v2_obstacle.mat")
    W0 = loadmat(DATA / "warm_v2_real.mat")
    rects = np.vstack([w.rects[:, :4], [[e[0] - OBST_HALF, e[0] + OBST_HALF, e[1] - OBST_HALF, e[1] + OBST_HALF] for e in extra]])
    res = {m: {"ok": [], "hv": [], "time": [], "q": []} for m in ("rl", "cold50+straight", "hybrid")}
    hit_before = {m: [] for m in ("rl", "nm500+straight")}
    timeouts = {"cold50+straight": 0, "hybrid": 0}                     # NMOPSO runs cancelled (could not start)
    for i, (L, ex, k) in enumerate(zip(legs, extra, idx)):
        wk = with_extra(w, ex)
        box = rects[[*range(len(w.rects)), len(w.rects) + i]]
        # originally deployed paths (old world) vs the new obstacle
        c0 = int(W0["cnt"][k, 0]); E0 = np.stack([W0[n][k, :c0] for n in ("r", "psi", "phi")], 1); F0 = costs(w, L, E0)
        ok = np.where(np.isfinite(F0).all(1))[0]
        if len(ok):
            P = polyline(w, L, E0[ok[short_safe(F0[ok])]]); hit_before["rl"].append(quality(P, L, box)["clearance_m"] < DRONE_R)
        for s in SEEDS:
            Fn, En = nm_front("nmopso_real", k, s)
            F = np.vstack([Fn, F0[:1]]); E = np.concatenate([En, E0[:1]]); ok = np.where(np.isfinite(F).all(1))[0]
            P = polyline(w, L, E[ok[short_safe(F[ok])]]); hit_before["nm500+straight"].append(quality(P, L, box)["clearance_m"] < DRONE_R)
        # replanning in the new world
        c = int(Wo["cnt"][i, 0]); Er = np.stack([Wo[n][i, :c] for n in ("r", "psi", "phi")], 1); Fr = costs(wk, L, Er)
        st = Fr[:1]
        for s in SEEDS:
            cand = {"rl": (Fr, Er, float(Wo["times"][0, i]))}
            for m, d in (("cold50+straight", "obst_cold50"), ("hybrid", "obst_warm50")):
                f = ROOT / "results" / d / f"leg{i + 1:03d}_seed{s}.mat"
                if f.exists():
                    Fm, Em = nm_front(d, i, s)
                    tm = float(loadmat(f)["time"].squeeze())
                else:                                                  # timed out: no NMOPSO front
                    to = f.with_name(f.stem + "_timeout.mat")
                    Fm, Em, tm = np.zeros((0, 4)), np.zeros((0, 3, 10)), float(loadmat(to)["timedOut"].squeeze())
                    timeouts[m] += 1
                if m == "hybrid":
                    cand[m] = (np.vstack([Fr, Fm]), np.concatenate([Er, Em]), tm + float(Wo["times"][0, i]))
                else:
                    cand[m] = (np.vstack([Fm, st]), np.concatenate([Em, Er[:1]]), tm)
            for m, (F, E, t) in cand.items():
                ok = np.where(np.isfinite(F).all(1))[0]
                res[m]["ok"].append(len(ok) > 0); res[m]["time"].append(t); res[m]["hv"].append(hv(feasible_front(F)))
                if len(ok):
                    res[m]["q"].append(quality(polyline(wk, L, E[ok[short_safe(F[ok])]]), L, box))
    out = {"legs": len(legs), "seeds": list(SEEDS), "obstacle_half_width_m": OBST_HALF, "model_radius_m": R_MODEL,
           "original_path_hits_new_obstacle_frac": {m: float(np.mean(v)) for m, v in hit_before.items()},
           "nmopso_timeout_frac": {m: v / (len(legs) * len(SEEDS)) for m, v in timeouts.items()}, "replan": {}}
    for m, r in res.items():
        q = {key: np.array([x[key] for x in r["q"]]) for key in r["q"][0]} if r["q"] else {}
        out["replan"][m] = {"success_frac": float(np.mean(r["ok"])), "time_median_s": float(np.median(r["time"])),
                            "hv_mean": float(np.mean(r["hv"])),
                            "length_ratio_mean": float(q["length_ratio"].mean()), "clearance_p5_m": float(np.percentile(q["clearance_m"], 5)),
                            "clearance_median_m": float(np.median(q["clearance_m"])), "collides_frac": float((q["clearance_m"] < DRONE_R).mean()),
                            "max_turn_deg_mean": float(q["max_turn_deg"].mean())}
    for m in ("rl", "hybrid"):
        d = np.array(res[m]["hv"]) - np.array(res["cold50+straight"]["hv"])
        out["replan"][m]["vs_cold50+straight_win_tie_loss"] = [float((d > TIE).mean()), float((abs(d) <= TIE).mean()), float((d < -TIE).mean())]
    (ROOT / "results" / "replan_obstacle.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prep": prep, "analyse": analyse}[sys.argv[1]]()
