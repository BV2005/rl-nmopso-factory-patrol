"""Pad-safe transits for the factory2_auto_full patrol: the same pads, zones and patrol loops, but each
drone's transit (pad -> loop start, loop end -> pad) is routed around the other drones' pads.
In factory2_auto_full the launch-area keep-out of the earlier layouts is gone, so transits ran over
neighbouring pads, where a drone hovers before departure and before landing (Gazebo flight
rl_20261021_152127: drones 2 and 3 collided over drone 3's pad).
Routing as plan_factory_patrol.m route(): shortest 8-connected path on a 0.2 m grid of free cells
(free_space.m: cylMargin outside the obstacle cylinders, inside the wall margin), here also outside a
KEEP-radius disc around every other pad; shortened by line of sight; split into legs <= MAX_LEG.
Writes data/padsafe_preview.json (preview format: transit_in / patrol_points / transit_out per drone,
plus keep-out cylinders per drone) and data/legs_padsafe.mat (the new transit legs, with their
keep-out cylinders as extra [K x 3m] metres for NMOPSO's extra_threats).
  python scripts/padsafe_routes.py"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from make_legs import PREVIEW  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

KEEP = 2.6          # m: keep-out radius around other drones' pads (> follower yield radius 2.5 m)
R_THREAT = 2.5      # m: the same keep-out for the leg planner (MyCost adds the 0.1 m drone size)
GRID = 0.2
MAX_LEG = 6.5       # factory_layout.m maxLeg


def route(w, a, b, pads, keep=KEEP):
    h, m = w.hall, 1.5                                                     # factory_layout wallMargin
    xs = np.arange(h[0] + m, h[1] - m + 1e-9, GRID); ys = np.arange(h[2] + m, h[3] - m + 1e-9, GRID)
    GX, GY = np.meshgrid(xs, ys)
    free = w.free(GX, GY)
    for p in pads:
        free &= np.hypot(GX - p[0], GY - p[1]) > keep
    idx = np.flatnonzero(free); ny, nx = free.shape
    node = -np.ones(free.size, int); node[idx] = np.arange(len(idx))
    r, c = np.unravel_index(idx, free.shape)
    S, T, W = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = r + dr, c + dc
        ok = (r2 >= 0) & (r2 < ny) & (c2 >= 0) & (c2 < nx)
        k2 = np.full(len(r), -1); k2[ok] = node[np.ravel_multi_index((r2[ok], c2[ok]), free.shape)]
        ok = k2 >= 0
        S.append(np.flatnonzero(ok)); T.append(k2[ok]); W.append(np.full(ok.sum(), GRID * np.hypot(dr, dc)))
    S, T, W = np.concatenate(S), np.concatenate(T), np.concatenate(W)
    G = coo_matrix((W, (S, T)), shape=(len(idx), len(idx))).tocsr()
    near = lambda q: int(np.argmin((GX.ravel()[idx] - q[0]) ** 2 + (GY.ravel()[idx] - q[1]) ** 2))
    s, t = near(a), near(b)
    dist, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
    if not np.isfinite(dist[t]):
        raise RuntimeError(f"no route {a} -> {b}")
    path = [t]
    while path[-1] != s:
        path.append(pred[path[-1]])
    Q = np.vstack([a[:2], np.c_[GX.ravel()[idx][path[::-1]], GY.ravel()[idx][path[::-1]]], b[:2]])

    def clear(p, q):                                                       # line of sight over free cells
        tt = np.linspace(0, 1, 60)
        P = p + tt[:, None] * (q - p)
        ci = np.clip(np.round((P[:, 0] - xs[0]) / GRID).astype(int), 0, nx - 1)
        ri = np.clip(np.round((P[:, 1] - ys[0]) / GRID).astype(int), 0, ny - 1)
        return free[ri, ci].all()

    K, i = [0], 0
    while i < len(Q) - 1:
        j = len(Q) - 1
        while j > i + 1 and not clear(Q[i], Q[j]):
            j -= 1
        K.append(j); i = j
    Q = Q[K]
    V = [Q[0]]
    for p, q in zip(Q[:-1], Q[1:]):
        n = int(np.ceil(np.linalg.norm(q - p) / MAX_LEG))
        V += [p + (q - p) * k / n for k in range(1, n + 1)]
    V = np.array(V)
    keep_pts = [V[0]] + [v for v in V[1:-1] if np.linalg.norm(v - V[0]) >= 0.5 and np.linalg.norm(v - V[-1]) >= 0.5] + [V[-1]]
    return np.c_[np.array(keep_pts), np.full(len(keep_pts), w.z_patrol)]


def min_dist_to(P, pts):
    """smallest horizontal distance from polyline P to the points."""
    best = np.inf
    for a, b in zip(P[:-1, :2], P[1:, :2]):
        e = b - a
        for p in pts:
            t = np.clip(np.dot(p - a, e) / max(np.dot(e, e), 1e-12), 0, 1)
            best = min(best, np.linalg.norm(a + t * e - p))
    return best


def main():
    w = load_world()
    J = json.loads(PREVIEW.read_text())
    homes = {d["id"]: np.array(d["home"][:2], float) for d in J["drones"]}
    legs, extra = [], []
    for d in J["drones"]:
        others = [homes[i] for i in homes if i != d["id"]]
        loop = np.asarray(d["patrol_points"], float)
        home = np.r_[homes[d["id"]], w.z_patrol]
        d["transit_in"] = route(w, home, loop[0], others).tolist()
        d["transit_out"] = route(w, loop[-1], home, others).tolist()
        d["keepout"] = [[*p, R_THREAT] for p in others]
        for part in ("transit_in", "transit_out"):
            P = np.asarray(d[part])
            for a, b in zip(P[:-1], P[1:]):
                legs.append(np.r_[a, b]); extra.append(np.ravel(d["keepout"]))
            print(f"drone {d['id']} {part}: {len(P) - 1} legs, {np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1).sum():.1f} m, "
                  f"min distance to other pads {min_dist_to(P, others):.2f} m")
        print(f"drone {d['id']} loop: min distance to other pads {min_dist_to(loop, others):.2f} m")
    J["layout"] = "factory2_auto_full + pad-safe transits"
    (DATA / "padsafe_preview.json").write_text(json.dumps(J, indent=1))
    savemat(DATA / "legs_padsafe.mat", {"legs": np.array(legs), "extra": np.array(extra)})
    print(f"{len(legs)} new transit legs -> data/legs_padsafe.mat")


if __name__ == "__main__":
    main()
