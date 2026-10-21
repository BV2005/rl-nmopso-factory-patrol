"""Independent collision check of a Gazebo flight (the demo's run_logs/<run>/drone_states.csv, 10 Hz,
factory-frame positions): per drone, the smallest distance from its centre to every real obstacle box
of the factory2 world (counted only while the drone is below the box top + DRONE_R), and between
drones (3-D, and horizontally at the same level |dz| < 0.6 m); only airborne samples (z > 0.3 m).
  collision  distance to a box < DRONE_R, or between drones < 2 DRONE_R
  near miss  distance to a box < 2 DRONE_R, or between drones < 1.0 m
  python scripts/flight_check.py <run dir (Windows or \\\\wsl.localhost path)> [--out results/flight_<name>.json]"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DRONE_R = 0.3


def obstacle_boxes():
    """[K, 5] boxes [xmin xmax ymin ymax top] metres: the obstacles, the low ones the patrol flies over,
    and the four walls (data/factory2_model.json; numpy only, so it runs with the simulator's python)."""
    J = json.loads((ROOT / "data" / "factory2_model.json").read_text())
    h = J["hall"]
    walls = np.array([[h[0] - 0.3, h[0], h[2], h[3], 12], [h[1], h[1] + 0.3, h[2], h[3], 12],
                      [h[0], h[1], h[2] - 0.3, h[2], 12], [h[0], h[1], h[3], h[3] + 0.3, 12]])
    return np.vstack([np.atleast_2d(np.array(J["rects"], float)), np.atleast_2d(np.array(J["overflown"], float)), walls])


def load_states(run):
    rows = list(csv.DictReader(open(Path(run) / "drone_states.csv")))
    out = {}
    for r in rows:
        try:
            d = int(r["drone"]); t = float(r["t"]); x, y, z = float(r["x"]), float(r["y"]), float(r["z"])
        except (KeyError, ValueError):
            continue
        out.setdefault(d, []).append((t, x, y, z, r.get("phase", "")))
    return {d: np.array([v[:4] for v in sorted(vs)]) for d, vs in out.items()}, \
           {d: [v[4] for v in sorted(vs)] for d, vs in out.items()}


def box_clearance(P, boxes):
    """[N] distance from points P [N, 3] to the nearest box [xmin xmax ymin ymax top] the point is not above."""
    dx = np.maximum(np.maximum(boxes[:, 0] - P[:, None, 0], 0), P[:, None, 0] - boxes[:, 1])
    dy = np.maximum(np.maximum(boxes[:, 2] - P[:, None, 1], 0), P[:, None, 1] - boxes[:, 3])
    dz = np.maximum(P[:, None, 2] - boxes[:, 4], 0)                  # above the top: vertical gap
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2).min(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    boxes = obstacle_boxes()
    S, phases = load_states(a.run)
    rep = {"run": str(a.run), "drones": {}, "pairs": {}}
    for d, X in S.items():
        air = X[:, 3] > 0.3
        c = box_clearance(X[air, 1:4], boxes)
        k = int(np.argmin(c))
        rep["drones"][d] = {"samples_airborne": int(air.sum()), "min_box_clearance_m": float(c[k]),
                            "at": [round(float(v), 2) for v in X[air][k, 1:4]],
                            "phase": [p for p, f in zip(phases[d], air) if f][k],
                            "collision": bool(c[k] < DRONE_R), "near_miss": bool(c[k] < 2 * DRONE_R)}
    ids = sorted(S)
    for i in ids:
        for j in ids:
            if j <= i:
                continue
            A, B = S[i], S[j]
            t = np.intersect1d(np.round(A[:, 0], 1), np.round(B[:, 0], 1))
            Ai = A[np.isin(np.round(A[:, 0], 1), t)]; Bj = B[np.isin(np.round(B[:, 0], 1), t)]
            _, ia = np.unique(np.round(Ai[:, 0], 1), return_index=True); _, ib = np.unique(np.round(Bj[:, 0], 1), return_index=True)
            Ai, Bj = Ai[ia], Bj[ib]
            air = (Ai[:, 3] > 0.3) & (Bj[:, 3] > 0.3)
            d3 = np.linalg.norm(Ai[air, 1:4] - Bj[air, 1:4], axis=1)
            same = np.abs(Ai[air, 3] - Bj[air, 3]) < 0.6
            dh = np.linalg.norm(Ai[air, 1:3] - Bj[air, 1:3], axis=1)
            k = int(np.argmin(d3)) if len(d3) else None
            rep["pairs"][f"{i}-{j}"] = {"samples_both_airborne": int(air.sum()),
                                        "min_sep_3d_m": float(d3.min()) if len(d3) else None,
                                        "at_t": float(Ai[air][k, 0]) if k is not None else None,
                                        "min_sep_same_level_m": float(dh[same].min()) if same.any() else None,
                                        "collision": bool(len(d3) and d3.min() < 2 * DRONE_R),
                                        "near_miss": bool(len(d3) and d3.min() < 1.0)}
    rep["any_obstacle_collision"] = any(v["collision"] for v in rep["drones"].values())
    rep["any_drone_collision"] = any(v["collision"] for v in rep["pairs"].values())
    s = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
