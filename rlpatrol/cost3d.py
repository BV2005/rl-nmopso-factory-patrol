"""MyCost-3D: NMOPSO's MyCost.m objectives with obstacle heights, batched over per-episode worlds.

MyCost.m treats every obstacle as an infinitely tall cylinder. Here every cylinder also has a top (the
height of the box it covers, rlpatrol/worldgen.cylinders): a path segment whose lowest point is more
than DRONE above a top is not a collision, and its threat contribution decays over DANGER above the
top. With tops = +inf this is exactly MyCost (rlpatrol/nmopso_cost.Legs.cost; checked to 1e-12 by
scripts/test_generalizable.py), so 2D paths keep NMOPSO's objectives.

Input paths are in metres; the cost is computed in NMOPSO units (x 10), as MyCost.m; it is translation
invariant, so the per-world origin does not matter. Paths have a variable number of nodes: P [B, N, 3]
holds start, k nodes and the goal packed at the front, n_pts [B] = k + 2 valid points.
Objectives (as MyCost.m): J1 path length 1 - straight/length (inf if a segment is shorter than rmin),
J2 threat (inf on collision), J3 altitude deviation from the patrol band, J4 smoothness (mean turn angle / pi)."""
import math

import torch

from .nmopso_cost import DANGER, DRONE, N_NODES

UNITS = 10.0
TOP_INF = 1e9
DANGER_M = DANGER / UNITS        # 1 m, MyCost.m danger_dist
DRONE_M = DRONE / UNITS          # 0.1 m, MyCost.m drone_size


def cost3d(P, n_pts, cyl, cmask, zband=(1.6, 2.0), details=False):
    """P [B, N, 3] metres (packed start, nodes, goal), n_pts [B]; cyl [B, M, 4] metres [x y R top];
    cmask [B, M] valid cylinders -> F [B, 4] (inf where infeasible); with details also the soft
    violation (>= 0, 0 iff J1 and J2 finite) and J2 without the collision infinity."""
    U = P * UNITS
    B, N, _ = U.shape
    dev, dt = U.device, U.dtype
    ar = torch.arange(N, device=dev)
    goal = U[torch.arange(B, device=dev), n_pts - 1]
    dist = (goal - U[:, 0]).norm(dim=-1)                               # PP
    rmin = 3 * dist / N_NODES / 9                                      # NMOPSO VarMin.r
    seg = U[:, 1:] - U[:, :-1]                                         # [B, N-1, 3]
    sval = ar[None, :-1] < (n_pts[:, None] - 1)                        # valid segments
    sl = seg.norm(dim=-1)
    short = (sl <= rmin[:, None]) & sval
    traj = (sl * sval).sum(-1)
    J1 = (1 - dist / traj.clamp_min(1e-12)).abs()
    J1 = torch.where(short.any(-1), torch.full_like(J1, math.inf), J1)

    c = cyl * UNITS
    a, e = U[:, :-1, None, :2], seg[:, :, None, :2]                    # [B, S, 1, 2]
    cc = c[:, None, :, :2]                                             # [B, 1, M, 2]
    ee = (e * e).sum(-1)
    t = torch.where(ee > 0, ((cc - a) * e).sum(-1) / ee.clamp_min(1e-300), torch.zeros_like(ee)).clamp(0, 1)
    d = (a + t[..., None] * e - cc).norm(dim=-1)                       # DistP2S [B, S, M]
    R, top = c[:, None, :, 2], c[:, None, :, 3]
    zlow = torch.minimum(U[:, :-1, 2], U[:, 1:, 2])[..., None]         # lowest point of the segment
    above = zlow - top - DRONE                                         # > 0: the segment passes over it
    tc_h = (1 - (d - DRONE - R) / DANGER).clamp(0, 1)
    tc_v = (1 - above / DANGER).clamp(0, 1)
    valid = sval[..., None] & cmask[:, None, :]
    tc = tc_h * tc_v * valid
    hit = (d < R + DRONE) & (above < 0) & valid
    nseg, ncyl = sval.sum(-1).clamp_min(1), cmask.sum(-1).clamp_min(1)
    J2soft = tc.sum((-1, -2)) / (nseg * ncyl)
    J2 = torch.where(hit.any(-1).any(-1), torch.full_like(J2soft, math.inf), J2soft)

    nval = (ar[None] >= 1) & (ar[None] < n_pts[:, None] - 1)           # the nodes (not start / goal)
    z = U[..., 2]
    zmin, zmax = zband[0] * UNITS, zband[1] * UNITS
    dev_z = (z - (zmax + zmin) / 2).abs() / ((zmax - zmin) / 2)
    J3 = (dev_z * nval).sum(-1) / nval.sum(-1).clamp_min(1)
    J3 = torch.where(((z < 0) & nval).any(-1), torch.full_like(J3, math.inf), J3)

    s1, s2 = seg[:, :-1], seg[:, 1:]
    pval = ar[None, :-2] < (n_pts[:, None] - 2)                        # consecutive valid segment pairs
    ang = torch.atan2(torch.linalg.cross(s1, s2).norm(dim=-1), (s1 * s2).sum(-1)) / math.pi
    J4 = (ang * pval).sum(-1) / pval.sum(-1).clamp_min(1)
    F = torch.stack([J1, J2, J3, J4], -1)
    if not details:
        return F
    pen = (((R + DRONE - d).clamp_min(0) * (above < 0) * valid) / DANGER).sum((-1, -2))
    shortv = (((rmin[:, None] - sl).clamp_min(0) / rmin[:, None].clamp_min(1e-12)) * sval).sum(-1) + short.float().sum(-1) * 0.1
    return F, pen + shortv, J2soft


def segment_hits(a, b, cyl, cmask):
    """[B] bool: the segments a -> b (metres, [B, 3]) collide with a cylinder (3D rule of cost3d)."""
    return segment_check(a, b, cyl, cmask)[0]


def segment_check(a, b, cyl, cmask):
    """(hit [B] bool, margin [B] metres): the collision of segment_hits and the smallest horizontal
    clearance (distance - R - DRONE) to a cylinder the segment does not pass over (+inf if none)."""
    P = torch.stack([a, b], 1)
    U = P * UNITS
    c = cyl * UNITS
    s, e = U[:, 0, None, :2], (U[:, 1] - U[:, 0])[:, None, :2]
    ee = (e * e).sum(-1)
    t = torch.where(ee > 0, ((c[..., :2] - s) * e).sum(-1) / ee.clamp_min(1e-300), torch.zeros_like(ee)).clamp(0, 1)
    d = (s + t[..., None] * e - c[..., :2]).norm(dim=-1)
    zlow = torch.minimum(U[:, 0, 2], U[:, 1, 2])[:, None]
    below = (zlow - c[..., 3] - DRONE < 0) & cmask
    hit = ((d < c[..., 2] + DRONE) & below).any(-1)
    gap = torch.where(below, (d - c[..., 2] - DRONE) / UNITS, torch.full_like(d, math.inf))
    return hit, gap.min(-1).values


# --- the box collision model (round 3) ---------------------------------------------------------------------------
# NMOPSO's cylinder cover makes the collision zone 0.5-1.0 m wider than the boxes, by an amount that depends
# on the obstacle and on where along it - invisible to the sensors. The box model's zone is every box grown
# by one fixed margin (e.g. 0.5 m): the scan can report the distance to it exactly (scan3d margin), in
# training and in flight (inflated_bins). J1, J3, J4 are MyCost's; J2 is MyCost's threat with the
# segment-to-box distance (minus the margin) in place of the segment-to-cylinder distance (minus R + DRONE).
def _pbox(px, py, b):
    return torch.hypot(torch.clamp(torch.maximum(b[..., 0] - px, px - b[..., 1]), min=0),
                       torch.clamp(torch.maximum(b[..., 2] - py, py - b[..., 3]), min=0))


def seg_box_dist(a, b, boxes):
    """horizontal distance (m) between segments a -> b ([..., S, 2]) and boxes ([..., K, >= 4]):
    [..., S, K]; 0 where the segment crosses the box."""
    a, b = a[..., :, None, :], b[..., :, None, :]
    bx = boxes[..., None, :, :]
    d = b - a
    eps = 1e-12
    dx = torch.where(d[..., 0].abs() < eps, torch.full_like(d[..., 0], eps), d[..., 0])
    dy = torch.where(d[..., 1].abs() < eps, torch.full_like(d[..., 1], eps), d[..., 1])
    tx1, tx2 = (bx[..., 0] - a[..., 0]) / dx, (bx[..., 1] - a[..., 0]) / dx
    ty1, ty2 = (bx[..., 2] - a[..., 1]) / dy, (bx[..., 3] - a[..., 1]) / dy
    tin = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
    tout = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))
    cross = (tout >= tin) & (tout >= 0) & (tin <= 1)
    dist = torch.minimum(_pbox(a[..., 0], a[..., 1], bx), _pbox(b[..., 0], b[..., 1], bx))
    ee = (d * d).sum(-1).clamp_min(1e-300)
    for cx, cy in ((0, 2), (0, 3), (1, 2), (1, 3)):                 # box corners to the segment
        c = torch.stack([bx[..., cx], bx[..., cy]], -1)
        t = (((c - a) * d).sum(-1) / ee).clamp(0, 1)
        dist = torch.minimum(dist, (a + t[..., None] * d - c).norm(dim=-1))
    return torch.where(cross, torch.zeros_like(dist), dist)


def cost3d_box(P, n_pts, boxes, kmask, margin=0.5, zband=(1.6, 2.0), details=False):
    """MyCost-3D with the box collision model: boxes [B, K, 5] metres (obstacles only, no walls), kmask [B, K]."""
    B = len(P)
    none = torch.zeros(B, 1, 4, dtype=P.dtype, device=P.device)
    F0, short, _ = cost3d(P, n_pts, none, torch.zeros(B, 1, dtype=torch.bool, device=P.device), zband, details=True)
    N = P.shape[1]
    ar = torch.arange(N, device=P.device)
    sval = ar[None, :-1] < (n_pts[:, None] - 1)
    d = seg_box_dist(P[:, :-1, :2], P[:, 1:, :2], boxes)                # [B, S, K]
    zlow = torch.minimum(P[:, :-1, 2], P[:, 1:, 2])[..., None]
    above = zlow - boxes[:, None, :, 4] - DRONE_M
    valid = sval[..., None] & kmask[:, None, :]
    tc = (1 - (d - margin) / DANGER_M).clamp(0, 1) * (1 - above / DANGER_M).clamp(0, 1) * valid
    hit = (d < margin) & (above < 0) & valid
    J2soft = tc.sum((-1, -2)) / (sval.sum(-1).clamp_min(1) * kmask.sum(-1).clamp_min(1))
    J2 = torch.where(hit.any(-1).any(-1), torch.full_like(J2soft, math.inf), J2soft)
    F = torch.stack([F0[:, 0], J2, F0[:, 2], F0[:, 3]], -1)
    if not details:
        return F
    pen = (((margin - d).clamp_min(0) * (above < 0) * valid) / DANGER_M).sum((-1, -2))
    return F, pen + short, J2soft


def segment_check_box(a, b, boxes, kmask, margin=0.5):
    """(hit [B], clearance [B] m) of segments a -> b ([B, 3]) in the box model (clearance = distance -
    margin to the nearest box it does not pass over)."""
    d = seg_box_dist(a[:, None, :2], b[:, None, :2], boxes)[:, 0]      # [B, K]
    zlow = torch.minimum(a[:, 2], b[:, 2])[:, None]
    below = (zlow - boxes[..., 4] - DRONE_M < 0) & kmask
    hit = ((d < margin) & below).any(-1)
    gap = torch.where(below, d - margin, torch.full_like(d, math.inf))
    return hit, gap.min(-1).values
