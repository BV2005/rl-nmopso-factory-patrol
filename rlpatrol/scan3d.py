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


def sim_scan_b(boxes, kmask, pos, yaw, spec=None, generator=None):
    """-> (ranges [B, SCAN_BINS] normalised as sensors.to_obs, heights [B, SCAN_BINS] / H_MAX), float32."""
    spec = spec or S.SensorSpec()
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
