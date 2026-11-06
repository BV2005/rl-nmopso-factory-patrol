"""The x500_patrol scan for per-episode worlds, with an obstacle-height channel.

Same sensor model, constants and bin layout as rlpatrol/sensors.py (2D lidar plane + forward depth
camera, SCAN_BINS bins over 270 deg, ranges normalised by SCAN_MAX), but every episode has its own
obstacle boxes (boxes [B, K, 5], kmask [B, K]) instead of one shared world, and a second channel:

  heights [B, SCAN_BINS]   the top of the nearest obstacle in each bin, / H_MAX (0: no return).
                           Known only where the depth camera produced the nearest return and the
                           obstacle's top is inside its vertical field of view; otherwise 1.0
                           ("tall or unknown": e.g. a lidar-only return, a rack whose top is above
                           the view) - the climb action treats unknown as not overflyable.

With one world broadcast to every episode the range channel equals sensors.sim_scan exactly
(scripts/test_generalizable.py)."""
import math

import torch

from . import sensors as S

H_MAX = 4.0


def walls(hall):
    """[B, 4, 5] the hall walls of hall [B, 4] as boxes (as sensors.scene_boxes)."""
    x0, x1, y0, y1 = hall.unbind(-1)
    T, H = S.WALL_T, torch.full_like(x0, S.WALL_TOP)
    return torch.stack([torch.stack([x0 - T, x0, y0 - T, y1 + T, H], -1), torch.stack([x1, x1 + T, y0 - T, y1 + T, H], -1),
                        torch.stack([x0 - T, x1 + T, y0 - T, y0, H], -1), torch.stack([x0 - T, x1 + T, y1, y1 + T, H], -1)], 1)


def _ray_boxes(o, d, boxes):
    """o [B, 2], d [B, N, 2], boxes [B, K, 5] -> t_in, t_out [B, N, K]."""
    big = torch.finfo(d.dtype).max / 4
    eps = 1e-12
    dx = torch.where(d[..., 0].abs() < eps, torch.full_like(d[..., 0], eps), d[..., 0])[..., None]
    dy = torch.where(d[..., 1].abs() < eps, torch.full_like(d[..., 1], eps), d[..., 1])[..., None]
    ox, oy = o[:, None, None, 0], o[:, None, None, 1]
    bx0, bx1 = boxes[:, None, :, 0], boxes[:, None, :, 1]
    by0, by1 = boxes[:, None, :, 2], boxes[:, None, :, 3]
    tx1, tx2 = (bx0 - ox) / dx, (bx1 - ox) / dx
    ty1, ty2 = (by0 - oy) / dy, (by1 - oy) / dy
    t_in = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2)).clamp(-big, big)
    t_out = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2)).clamp(-big, big)
    return t_in, t_out


def sim_scan_b(boxes, kmask, pos, yaw, spec=None, generator=None, margin=0.0):
    """-> (ranges [B, SCAN_BINS] normalised as sensors.to_obs, heights [B, SCAN_BINS] / H_MAX), float32.
    margin > 0: the ranges are to the obstacles grown by margin (the box collision model's zone), i.e. what
    inflated_bins computes from the real point cloud in flight."""
    spec = spec or S.SensorSpec()
    if margin:
        boxes = boxes.clone()
        boxes[..., 0] -= margin; boxes[..., 2] -= margin
        boxes[..., 1] += margin; boxes[..., 3] += margin
    dt, dev = pos.dtype, pos.device
    B = len(pos)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dt, device=dev)

    if spec.noise:
        pos = pos + torch.stack([randn(B) * spec.pos_sigma, randn(B) * spec.pos_sigma, randn(B) * spec.z_sigma], -1)
        yaw = yaw + randn(B) * spec.yaw_sigma
    J = max(1, int(spec.beams_per_bin))
    off = (torch.linspace(-0.5, 0.5, J, dtype=dt, device=dev) if J > 1 else torch.zeros(1, dtype=dt, device=dev)) * S.BIN
    rel = (S.bin_centres(dt, dev)[:, None] + off[None]).reshape(-1)
    ang = yaw[:, None] + rel[None]
    d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)
    top = boxes[:, None, :, 4]                                          # [B, 1, K]
    inf = torch.tensor(math.inf, dtype=dt, device=dev)
    km = kmask[:, None, :]

    t_in, t_out = _ray_boxes(pos[:, :2], d, boxes)
    t0 = t_in.clamp_min(0)
    hit = (t_out >= t0) & (top > pos[:, None, None, 2] + S.LIDAR_Z) & km
    tl = torch.where(hit, t0, inf)
    r_l = tl.min(-1).values
    r_l = torch.where((r_l >= S.LIDAR_RANGE[0]) & (r_l <= S.LIDAR_RANGE[1]), r_l, inf)

    cam = pos[:, :2] + S.DEPTH_X * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    t_in, t_out = _ray_boxes(cam, d, boxes)
    z_cam = pos[:, None, None, 2] + S.DEPTH_Z
    tanv = math.tan(S.DEPTH_VFOV / 2)
    t_top = (z_cam - top) / tanv
    t_hit = torch.maximum(t_in.clamp_min(0), t_top)
    hit = (t_out >= t_hit) & (top >= S.OBST_MIN_H) & km
    td = torch.where(hit, t_hit, inf)
    r_d, k_d = td.min(-1)
    top_d = torch.gather(boxes[..., 4], 1, k_d.reshape(B, -1)).reshape(r_d.shape)
    r_d = torch.where((r_d >= S.DEPTH_RANGE[0]) & (r_d <= S.DEPTH_RANGE[1]), r_d, inf)
    infov = (rel.abs() <= S.DEPTH_HFOV / 2)[None]
    r_d = torch.where(infov, r_d, inf)
    # the top is measurable if it is inside the camera's view at that range
    top_seen = top_d <= pos[:, None, 2] + S.DEPTH_Z + r_d * tanv

    if spec.noise:
        r_l = r_l + randn(*r_l.shape) * spec.lidar_sigma
        r_d = r_d + randn(*r_d.shape) * spec.depth_k * torch.where(torch.isfinite(r_d), r_d, torch.zeros_like(r_d)) ** 2
    r = torch.minimum(r_l, r_d).clamp_min(0)
    h = torch.where(torch.isfinite(r), torch.where((r_d <= r_l) & top_seen, (top_d / H_MAX).clamp(0, 1),
                                                    torch.ones_like(r)), torch.zeros_like(r))
    h = torch.where(r < S.SCAN_MAX, h, torch.zeros_like(h))           # beyond the range cap: nothing
    r = r.reshape(B, S.SCAN_BINS, J).min(-1).values
    h = h.reshape(B, S.SCAN_BINS, J).max(-1).values
    if spec.noise and spec.dropout > 0:
        drop = torch.rand(B, S.SCAN_BINS, generator=generator, dtype=dt, device=dev) < spec.dropout
        r = torch.where(drop, torch.full_like(r, math.inf), r)
        h = torch.where(drop, torch.zeros_like(h), h)
    return (r / S.SCAN_MAX).clamp(max=1.0).nan_to_num(1.0).float(), h.float()


def _beam_angles(beams_per_bin):
    import numpy as np
    c = S.bin_centres().numpy()
    off = (np.linspace(-0.5, 0.5, beams_per_bin) if beams_per_bin > 1 else np.zeros(1)) * S.BIN
    return (c[:, None] + off[None]).reshape(-1)                           # [SCAN_BINS * beams_per_bin]


def _inflated_beams(points, margin, th):
    """points [N, 2] (sensor frame) -> per beam th [M]: metres to the zone of radius margin around the points
    (inf without one) and the index of the point that sets it (-1 without one)."""
    import numpy as np
    p = np.asarray(points, float).reshape(-1, 2)
    if not len(p):
        return np.full(len(th), np.inf), np.full(len(th), -1)
    r = np.hypot(p[:, 0], p[:, 1])
    ang = np.arctan2(p[:, 1], p[:, 0])
    dlt = ang[None] - th[:, None]
    lat, along = r[None] * np.sin(dlt), r[None] * np.cos(dlt)
    # the beam meets the disc if it passes within margin and the disc is ahead of the sensor (or contains it):
    # a disc behind the sensor (along < 0, sensor outside it) is not on the beam
    ok = (np.abs(lat) < margin) & ((along >= 0) | (r[None] < margin))
    t = np.where(ok, (along - np.sqrt(np.clip(margin ** 2 - lat ** 2, 0, None))).clip(min=0), np.inf)
    k = t.argmin(1)
    tb = t[np.arange(len(th)), k]
    return tb, np.where(np.isfinite(tb), k, -1)


def _square_beams(points, margin, th, yaw):
    """as _inflated_beams, but the zone around each point is the axis-aligned square of half-size margin in world
    axes (the drone heads yaw): for the faces of axis-aligned boxes exactly the box grown by margin of training
    (sim_scan_b), and never smaller than the disc of radius margin."""
    import numpy as np
    p = np.asarray(points, float).reshape(-1, 2)
    if not len(p):
        return np.full(len(th), np.inf), np.full(len(th), -1)
    c, s = math.cos(yaw), math.sin(yaw)
    wx, wy = c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]
    a = th + yaw
    dx, dy = np.cos(a), np.sin(a)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)[:, None]
    dy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)[:, None]
    tx1, tx2 = (wx[None] - margin) / dx, (wx[None] + margin) / dx
    ty1, ty2 = (wy[None] - margin) / dy, (wy[None] + margin) / dy
    t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
    t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    t0 = np.maximum(t_in, 0)
    t = np.where(t_out >= t0, t0, np.inf)
    k = t.argmin(1)
    tb = t[np.arange(len(th)), k]
    return tb, np.where(np.isfinite(tb), k, -1)


def inflated_bins(points, margin, beams_per_bin=40):
    """flight side of sim_scan_b(margin=...): sensor points [N, 2] (metres, drone frame: x forward, y left,
    origin at the sensor) -> [SCAN_BINS] metres to the zone of radius `margin` around the points along each
    bin's beams (the nearest per bin; inf without one). For a dense cloud on an obstacle's surface this is
    the range to the obstacle grown by margin, as the simulated scan in training."""
    tb, _ = _inflated_beams(points, margin, _beam_angles(beams_per_bin))
    return tb.reshape(S.SCAN_BINS, -1).min(1)


def _depth_columns(points, cell=math.radians(0.25), depth=0.3):
    """levelled depth points [N, 3] (camera frame, z up relative to the optical centre) -> per azimuth cell of
    `cell` rad: the nearest point's horizontal (x, y) and the top of that column (the highest point within
    `depth` m behind the nearest one). A dense cloud reduced to the points that set ranges and obstacle tops."""
    import numpy as np
    p = np.asarray(points, float).reshape(-1, 3)
    if not len(p):
        return np.zeros((0, 2)), np.zeros(0)
    rng = np.hypot(p[:, 0], p[:, 1])
    c = np.floor((np.arctan2(p[:, 1], p[:, 0]) + math.pi) / cell).astype(np.int64)
    order = np.lexsort((rng, c))
    c_s = c[order]
    first = np.r_[True, c_s[1:] != c_s[:-1]]
    cells, start = c_s[first], order[first]                               # the nearest point of each cell
    near = np.full(c.max() + 1, np.inf)
    near[cells] = rng[start]
    inside = rng <= near[c] + depth
    top = np.full(c.max() + 1, -np.inf)
    np.maximum.at(top, c[inside], p[inside, 2])
    return p[start, :2], top[cells]


def flight_scan(lidar_xy=None, depth_points=None, drone_z=1.8, margin=0.0, yaw=0.0, beams_per_bin=40, beams=False):
    """the scan of sim_scan_b(margin=...) from real sensor data, without noise or dropout (flight):
      lidar_xy      [N, 2] lidar returns in the drone frame (x forward, y left; origin base_link = the lidar),
                    e.g. from a sensor_msgs/LaserScan (laserscan_points), or None
      depth_points  [M, 3] depth-camera points in the camera frame (x forward, y left, z up; origin the optical
                    centre), already levelled with the drone's roll and pitch (sensors.level), or None
      drone_z       base_link height above the floor (m)
      yaw           the drone's heading (rad, world ENU): the zone around a return is a square in world axes
                    (_square_beams), so a box's returns give exactly the grown box of training
    -> (ranges [SCAN_BINS] normalised by SCAN_MAX, heights [SCAN_BINS] / H_MAX), float32, as in training:
    ranges to the zone `margin` around every return (lidar from base_link, depth from the camera), heights the
    top of the nearest obstacle where the depth camera gave the nearest return and that top is inside its
    vertical view (1.0: tall or unknown, e.g. a lidar-only return; 0: nothing within SCAN_MAX).
    beams=True also returns a full-resolution scan for the safety check of a step (vt_pilot.VTPilot.clear), with
    round zones (distance < margin to a return: the collision model of training, which the square zones of the
    observation over-approximate at corners): dict(r [M] metres to the zone per beam from its sensor, inf: none; th [M] beam angles
    relative to the heading; x, y [M] where the beam enters the zone, in the drone frame (x forward, y left, origin
    base_link: depth beams start at the camera, DEPTH_X ahead); top [M] the obstacle top in metres where known, inf:
    tall or unknown), M = SCAN_BINS * beams_per_bin."""
    import numpy as np
    th = _beam_angles(beams_per_bin)
    lid = np.zeros((0, 2)) if lidar_xy is None else lidar_xy
    xy, top = np.zeros((0, 2)), np.zeros(0)
    if depth_points is not None and len(depth_points):
        p = np.asarray(depth_points, float).reshape(-1, 3)
        p = p[np.isfinite(p).all(1)]
        h = drone_z + S.DEPTH_Z + p[:, 2]
        rng = np.hypot(p[:, 0], p[:, 1])
        ang = np.arctan2(p[:, 1], p[:, 0])
        ok = (h >= S.OBST_MIN_H) & (rng >= S.DEPTH_RANGE[0]) & (rng <= S.DEPTH_RANGE[1]) & (np.abs(ang) <= S.DEPTH_HFOV / 2)
        xy, top = _depth_columns(np.c_[p[ok, :2], h[ok]])
    tanv = math.tan(S.DEPTH_VFOV / 2)

    def zone(beam_fn):
        """per beam: range to the zone (lidar from base_link, depth from the camera), which sensor, the known top."""
        r_l, _ = beam_fn(lid)
        r_d, k = beam_fn(xy)
        r_d = np.where(np.abs(th) <= S.DEPTH_HFOV / 2, r_d, np.inf)       # beams outside the camera's view
        top_d = np.where(k >= 0, top[np.maximum(k, 0)] if len(top) else np.inf, np.inf)
        seen = top_d <= drone_z + S.DEPTH_Z + r_d * tanv
        return np.minimum(r_l, r_d), r_d <= r_l, seen, top_d

    # the policy's observation: square zones, as training's grown boxes
    r, by_depth, seen, top_d = zone(lambda pts: _square_beams(pts, margin, th, yaw))
    hgt = np.where(np.isfinite(r), np.where(by_depth & seen, np.clip(top_d / H_MAX, 0, 1), 1.0), 0.0)
    hgt = np.where(r < S.SCAN_MAX, hgt, 0.0)
    fine = None
    if beams:
        # the safety check: discs, i.e. distance < margin to a return - training's collision model (segment_check_box),
        # which the square zones of the observation over-approximate at obstacle corners
        rc, dep, seen_c, top_c = zone(lambda pts: _inflated_beams(pts, margin, th))
        ok = np.isfinite(rc)
        rf = np.where(ok, rc, 0.0)
        fine = dict(r=rc, th=th, top=np.where(ok & dep & seen_c, top_c, np.inf),
                    x=np.where(ok, np.where(dep, S.DEPTH_X, 0.0) + rf * np.cos(th), np.inf), y=np.where(ok, rf * np.sin(th), np.inf))
    r = r.reshape(S.SCAN_BINS, -1).min(1)
    hgt = hgt.reshape(S.SCAN_BINS, -1).max(1)
    out = np.clip(np.nan_to_num(r / S.SCAN_MAX, nan=1.0, posinf=1.0), 0, 1).astype(np.float32), hgt.astype(np.float32)
    return (*out, fine) if beams else out
