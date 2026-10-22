"""The legs a policy finds no feasible path for: are they long, blocked, or impossible? (motivates round B)
  python scripts/hard_legs.py results/eval_v5_seed1_random.json [--legs random]
Reads an evaluate.py output (its rows give the feasible front size per leg) and, per leg of the set:
  length_m              straight distance
  straight_clearance_m  closest approach of the straight line to a drone-inflated cylinder (< 0: blocked)
  free_path_m           shortest collision-free horizontal route (8-connected grid of 1 unit = 0.1 m, cylinders
                        inflated by the drone size, map bounds); computed for the failed legs only
  free_over_straight    free_path_m / length_m. The NMOPSO encoding spans at most 3 x the straight distance
                        (10 steps of <= 0.3 x it), so a ratio > 3 cannot be solved by any method using it.
Writes results/hard_legs_<eval stem>.json (summary, failure rate by length, per failed leg) and prints them."""
import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.nmopso_cost import DRONE  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

ENCODING_SPAN = 3.0             # NMOPSO.m VarMax.r = 3 * dist / n: the n nodes span <= 3 x the straight distance


def straight_clearance(world, th, leg):
    a, b = world.to_units(leg[:3])[:2], world.to_units(leg[3:])[:2]
    e = b - a
    t = np.clip(((th[:, :2] - a) @ e) / (e @ e), 0, 1)
    d = np.linalg.norm(a + t[:, None] * e - th[:, :2], axis=1) - th[:, 3] - DRONE
    return float(d.min() / world.scale)


def free_path(occ, lo, a, g, scale):
    """8-connected Dijkstra from a to g (NMOPSO units) on the occupancy grid; metres, inf if cut off."""
    s = (int(round(a[1] - lo[1])), int(round(a[0] - lo[0])))
    t = (int(round(g[1] - lo[1])), int(round(g[0] - lo[0])))
    if occ[s] or occ[t]:
        return float("nan")
    D, h = {s: 0.0}, [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if u == t:
            return d / scale
        if d > D[u]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                v = (u[0] + dy, u[1] + dx)
                if (dx or dy) and 0 <= v[0] < occ.shape[0] and 0 <= v[1] < occ.shape[1] and not occ[v]:
                    nd = d + (1.4142135623730951 if dx and dy else 1.0)
                    if nd < D.get(v, 1e18):
                        D[v] = nd; heapq.heappush(h, (nd, v))
    return float("inf")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_json", help="an evaluate.py output, e.g. results/eval_v5_seed1_random.json")
    ap.add_argument("--legs", default=None, help="leg set (default: the one recorded in eval_json)")
    a = ap.parse_args()
    ev = json.loads(Path(a.eval_json).read_text())
    world = load_world()
    th, b = world.threats.numpy(), world.bounds
    legs = loadmat(DATA / f"legs_{a.legs or ev['legs']}.mat")["legs"]
    feas = np.array([r["rl_front"] > 0 for r in ev["rows"]])
    X, Y = np.meshgrid(np.arange(b[0], b[1] + 1), np.arange(b[2], b[3] + 1))
    occ = np.zeros(X.shape, bool)
    for x, y, _, R in th:
        occ |= (X - x) ** 2 + (Y - y) ** 2 < (R + DRONE) ** 2

    L = np.linalg.norm(legs[:, 3:6] - legs[:, :3], axis=1)
    C = np.array([straight_clearance(world, th, leg) for leg in legs])
    summ = {}
    for name, m in (("feasible", feas), ("failed", ~feas)):
        summ[name] = {"legs": int(m.sum()), "length_median_m": float(np.median(L[m])),
                      "length_range_m": [float(L[m].min()), float(L[m].max())],
                      "straight_blocked_frac": float(np.mean(C[m] < 0)),
                      "straight_clearance_median_m": float(np.median(C[m]))}
        print(f"{name:8} {m.sum():3} legs  length median {np.median(L[m]):5.1f} m ({L[m].min():.1f}-{L[m].max():.1f})"
              f"  straight line blocked {np.mean(C[m] < 0) * 100:3.0f}%  median straight clearance {np.median(C[m]):+.2f} m")
    by_len = []
    for lo_, hi_ in ((0, 4), (4, 8), (8, 12), (12, 16)):
        m = (L >= lo_) & (L < hi_)
        by_len.append({"length_m": [lo_, hi_], "legs": int(m.sum()), "fail_frac": float(np.mean(~feas[m])) if m.any() else None})
        print(f"  legs {lo_:2}-{hi_:2} m: {m.sum():3} legs, fail {np.mean(~feas[m]) * 100 if m.any() else 0:3.0f}%")
    failed = []
    for i in np.where(~feas)[0]:
        fp = free_path(occ, b[[0, 2]], world.to_units(legs[i, :3]), world.to_units(legs[i, 3:]), world.scale)
        failed.append({"leg": int(i), "length_m": float(L[i]), "straight_clearance_m": float(C[i]),
                       "free_path_m": fp, "free_over_straight": fp / float(L[i]),
                       "solvable_in_encoding": bool(fp / L[i] <= ENCODING_SPAN)})
        print(f"  failed leg {i:3}: {L[i]:5.1f} m, straight clearance {C[i]:+.2f} m, shortest free path {fp:5.1f} m"
              f" ({fp / L[i]:.2f} x){'' if fp / L[i] <= ENCODING_SPAN else '  > 3 x: impossible in the encoding'}")
    n_ok = sum(f["solvable_in_encoding"] for f in failed)
    print(f"{n_ok} of {len(failed)} failed legs are solvable within the encoding's {ENCODING_SPAN:.0f} x span")
    out = ROOT / "results" / f"hard_legs_{Path(a.eval_json).stem.removeprefix('eval_')}.json"
    out.write_text(json.dumps({"eval": a.eval_json, "summary": summ, "fail_by_length": by_len,
                               "failed_solvable_in_encoding": n_ok, "failed": failed}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
