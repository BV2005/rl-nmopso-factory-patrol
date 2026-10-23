"""Route guidance for the leg-planning MDP (round B): shortest collision-free horizontal routes between
the cells of a coarse grid, precomputed once per world, and the per-step route features read from it.

Grid: CELL-unit cells (0.5 m) over the map bounds. A cell is free if its centre clears every
drone-inflated cylinder; 8-connected neighbours are linked if both are free and the straight segment
between their centres clears every inflated cylinder. All-pairs Dijkstra gives dist [N, N] (units) and
nxt [N, N]: nxt[g, c] is the next cell from c on a shortest route to g (undirected graph:
the predecessor of c on the route from g). Blocked cells snap to the nearest free cell.

Features at a point q for goal g (route_features):
  waypoint   the farthest of the next LOOKAHEAD route cells visible from q (the segment q -> cell clears
             every inflated cylinder); the goal itself when it is visible from q. Observed as the cos and
             sin of its bearing relative to the goal bearing (cos 1, sin 0 whenever the goal is visible).
  route_len  |q - cell(q)| + dist[cell(g), cell(q)] + |cell(g) - g| (units; the straight distance when
             the goal is visible). Observed relative to the reach left and to the straight distance."""
import hashlib
import math

import numpy as np
import torch

from .nmopso_cost import DRONE
from .world import DATA

CELL = 5.0                     # units (0.5 m)
LOOKAHEAD = 8                  # route cells considered for the waypoint (4 m)
_ROUTE_TABLES = {}


def seg_clear(th, a, b):
    """segments a -> b ([..., 2] each, units) clear every drone-inflated cylinder of th [M, 4]: [...] bool."""
    e = b - a
    c = th[:, :2]
    ee = (e * e).sum(-1, keepdim=True)
    t = (((c - a[..., None, :]) * e[..., None, :]).sum(-1) / ee.clamp_min(1e-12)).clamp(0, 1)
    d = (a[..., None, :] + t[..., None] * e[..., None, :] - c).norm(dim=-1)
    return (d >= th[:, 3] + DRONE).all(-1)


def _build(world):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    th = world.threats.detach().cpu()
    b = np.asarray(world.bounds, float)
    nx, ny = int(math.ceil((b[1] - b[0]) / CELL)) + 1, int(math.ceil((b[3] - b[2]) / CELL)) + 1
    X, Y = np.meshgrid(b[0] + CELL * np.arange(nx), b[2] + CELL * np.arange(ny))      # [ny, nx]
    ctr = torch.tensor(np.stack([X.ravel(), Y.ravel()], -1), dtype=th.dtype)            # cell k = iy * nx + ix
    d = (ctr[:, None] - th[None, :, :2]).norm(dim=-1)
    free = (d >= th[:, 3] + DRONE).all(-1).numpy()
    rows, cols, wts = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        jy, jx = iy + dy, ix + dx
        ok = (jy < ny) & (jx >= 0) & (jx < nx)
        a, c = (iy * nx + ix)[ok], (jy * nx + jx)[ok]
        ok2 = free[a] & free[c]
        a, c = a[ok2], c[ok2]
        clear = seg_clear(th, ctr[a], ctr[c]).numpy()
        rows.append(a[clear]); cols.append(c[clear]); wts.append(np.full(clear.sum(), CELL * math.hypot(dx, dy)))
    N = nx * ny
    G = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    dist, pred = dijkstra(G, directed=False, return_predecessors=True)
    nxt = np.where(pred < 0, np.arange(N)[None, :], pred)                                # at the goal: stay
    fi = np.where(free)[0]
    dd = ((ctr.numpy()[:, None] - ctr.numpy()[fi][None]) ** 2).sum(-1)
    snap = np.where(free, np.arange(N), fi[dd.argmin(1)])
    return {"origin": torch.tensor(b[[0, 2]]), "nx": nx, "ny": ny, "centres": ctr,
            "dist": torch.tensor(dist, dtype=torch.float32), "nxt": torch.tensor(nxt.astype(np.int64)),
            "snap": torch.tensor(snap.astype(np.int64)), "free": torch.tensor(free)}


def route_table(world):
    """the world's route table (cached in memory per device and on disk in data/route_<sig>.pt)."""
    dev = world.threats.device
    key = (id(world), str(dev))
    if key in _ROUTE_TABLES:
        return _ROUTE_TABLES[key]
    th = world.threats.detach().cpu()
    sig = hashlib.sha1(th.numpy().tobytes() + np.asarray(world.bounds).tobytes()
                       + f"{CELL} {DRONE}".encode()).hexdigest()[:12]
    cache = DATA / f"route_{sig}.pt"
    if cache.exists():
        tab = torch.load(cache, map_location="cpu")
    else:
        tab = _build(world)
        torch.save(tab, cache)
    tab = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in tab.items()}
    tab["centres"] = tab["centres"].to(world.threats.dtype)
    _ROUTE_TABLES[key] = tab
    return tab


def cell_of(tab, p):
    """nearest free cell of points p [..., 2] (units)."""
    i = ((p - tab["origin"].to(p)) / CELL).round().long()
    ix, iy = i[..., 0].clamp(0, tab["nx"] - 1), i[..., 1].clamp(0, tab["ny"] - 1)
    return tab["snap"][iy * tab["nx"] + ix]


def route_features(world, q, goal):
    """q, goal [B, 2] (units) -> (waypoint [B, 2], route length [B]) as in the module doc."""
    tab = route_table(world)
    th = world.threats
    cq, cg = cell_of(tab, q), cell_of(tab, goal)
    ctr = tab["centres"]
    L = (q - ctr[cq]).norm(dim=-1) + tab["dist"][cg, cq].to(q.dtype) + (ctr[cg] - goal).norm(dim=-1)
    hops, c = [], cq
    for _ in range(LOOKAHEAD):                                              # the next route cells
        c = tab["nxt"][cg, c]
        hops.append(c)
    H = torch.stack(hops, 1)                                                # [B, K]
    P = ctr[H]                                                              # [B, K, 2]
    vis = seg_clear(th, q[:, None].expand_as(P), P)                         # [B, K]
    k = torch.where(vis.any(1), (vis * torch.arange(1, LOOKAHEAD + 1, device=q.device)).argmax(1),
                    torch.zeros_like(cq))
    wp = P[torch.arange(len(q), device=q.device), k]
    goal_vis = seg_clear(th, q, goal)
    straight = (goal - q).norm(dim=-1)
    wp = torch.where(goal_vis[:, None], goal, wp)
    L = torch.where(goal_vis, straight, torch.maximum(L, straight))
    return wp, L
