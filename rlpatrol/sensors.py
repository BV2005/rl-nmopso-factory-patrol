"""Perception of the x500_patrol drone (gazebo/models/x500_patrol), shared by training and flight.

The sensor-input policy (PathEnv(sensor_obs=True)) does not see the oracle geometry of the default
observation (exact rays to the NMOPSO cylinders, whole-map shortest routes). It sees a fused
"virtual scan": SCAN_BINS range bins over the lidar's 270 deg field of view, centred on the drone's
heading, each the nearest obstacle return in that sector from
  * the 2D lidar (lidar_2d_v2: 1080 beams over +-135 deg, 0.1-30 m, one horizontal plane LIDAR_Z
    above base_link): it sees only obstacles taller than its scan plane;
  * the forward depth camera (OAK-D Lite stereo depth: 73 deg x 58 deg, 0.2-19.1 m): every depth
    point at least OBST_MIN_H above the floor, i.e. also the machines below the lidar plane, but only
    inside its field of view and only once the obstacle top enters the vertical field of view.
Ranges are capped at SCAN_MAX and normalised to [0, 1] (1: nothing within SCAN_MAX). No return
behind the drone (the 90 deg blind sector) - the policy has to cope with that as the real drone does.

The same bin layout is produced in two ways, which must agree:
  sim_scan()            batched ray casting against the true obstacle boxes (training, offline
                        evaluation), with optional sensor noise (domain randomisation): range noise,
                        depth noise growing with range^2 (stereo), bin dropout, pose and yaw error;
  bins_from_laserscan() / bins_from_points() / to_obs()
                        the real sensor messages in flight (sensor_msgs/LaserScan from the lidar,
                        the depth camera's point cloud), see gazebo/nmopso_demo/sensor_follower.py.

Frames: world metres, x east, y north, z up (Gazebo ENU); angles counter-clockwise from the drone's
heading (a LaserScan's angle convention: angle_min = -135 deg is on the right)."""
import math
from dataclasses import dataclass

import numpy as np
import torch

# --- x500_patrol geometry (keep equal to gazebo/models/x500_patrol/model.sdf) -------------------
LIDAR_Z = 0.315                     # lidar scan plane above base_link (include z 0.26 + sensor 0.055)
LIDAR_FOV = 2 * 2.356195            # 270 deg, centred forward
LIDAR_RANGE = (0.1, 30.0)
DEPTH_X, DEPTH_Z = 0.132, 0.261     # depth camera optical centre ahead of / above base_link
                                    # (camera_link at 0.12 0 0.242 + sensor at 0.01233 0 0.01878)
DEPTH_HFOV = 1.274                  # OAK-D Lite stereo depth (PX4's OakD-Lite model)
DEPTH_W, DEPTH_H = 320, 240         # x500_patrol depth image (PX4's model: 640 x 480; same FOV)
DEPTH_VFOV = 2 * math.atan(math.tan(DEPTH_HFOV / 2) * DEPTH_H / DEPTH_W)
DEPTH_RANGE = (0.2, 19.1)

# --- observation ------------------------------------------------------------------------------
SCAN_BINS = 27                      # 10 deg bins over the lidar field of view
BIN = LIDAR_FOV / SCAN_BINS
SCAN_MAX = 6.0                      # m: ranges are capped here and normalised by it
OBST_MIN_H = 1.1                    # m: depth points lower than this are not obstacles at patrol height
                                    # (factory2: loading dock 1.0 m, conveyor 0.8 m are overflown; the
                                    # 1.2 m workbench counts, as it does for NMOPSO)
WALL_T, WALL_TOP = 0.3, 10.0        # hall walls as boxes just outside world.hall


def bin_centres(dtype=torch.float64, device=None):
    """[SCAN_BINS] bin centre angles relative to the heading (rad, counter-clockwise)."""
    k = torch.arange(SCAN_BINS, dtype=dtype, device=device)
    return -LIDAR_FOV / 2 + (k + 0.5) * BIN


@dataclass
class SensorSpec:
    """sensor model used by sim_scan(); noise=False gives ideal sensors (exact geometry)."""
    noise: bool = True
    beams_per_bin: int = 3          # simulated rays per bin: both edges + centre (the real lidar has 40 per bin)
    lidar_sigma: float = 0.02       # m, lidar range noise
    depth_k: float = 0.0067         # stereo depth noise sigma = depth_k * range^2 (about 2 % at 3 m)
    dropout: float = 0.02           # probability that a bin returns nothing (dark, shiny, glass)
    pos_sigma: float = 0.05         # m, horizontal pose error of the state estimate
    z_sigma: float = 0.05           # m, height error
    yaw_sigma: float = 0.02         # rad, heading error


_BOXES = {}


def scene_boxes(world):
    """[K, 5] tensor of the physical obstacle boxes [xmin xmax ymin ymax top] (metres, uninflated:
    world.rects) plus the four hall walls, on the world's device (cached per world)."""
    dev, dt = world.threats.device, world.threats.dtype
    rects = np.asarray(world.rects if world.rects is not None else np.zeros((0, 5)), float).reshape(-1, 5)
    hall = np.asarray(world.hall, float)
    key = (rects.tobytes(), hall.tobytes(), str(dev), str(dt))       # by content: worlds come and go
    if key not in _BOXES:
        x0, x1, y0, y1 = (float(v) for v in hall)
        walls = [[x0 - WALL_T, x0, y0 - WALL_T, y1 + WALL_T, WALL_TOP],
                 [x1, x1 + WALL_T, y0 - WALL_T, y1 + WALL_T, WALL_TOP],
                 [x0 - WALL_T, x1 + WALL_T, y0 - WALL_T, y0, WALL_TOP],
                 [x0 - WALL_T, x1 + WALL_T, y1, y1 + WALL_T, WALL_TOP]]
        if len(_BOXES) > 256:
            _BOXES.clear()
        _BOXES[key] = torch.tensor(np.vstack([rects, walls]), dtype=dt, device=dev)
    return _BOXES[key]


def units_to_m(world, q):
    """NMOPSO units [..., 3] (tensor) -> world metres (World.to_metres for tensors)."""
    s = world.scale
    return torch.stack([(q[..., 0] - 1) / s + float(world.hall[0]),
                        (q[..., 1] - 1) / s + float(world.hall[2]),
                        q[..., 2] / s], -1)


def ray_boxes(o, d, boxes):
    """2D slab test. o [B, 2] origins, d [B, N, 2] unit directions, boxes [K, 5] ->
    (t_in, t_out) [B, N, K]: the ray o + t d is inside box k for t in [t_in, t_out]; a miss has
    t_out < max(t_in, 0). An origin inside a box has t_in < 0 <= t_out."""
    big = torch.finfo(d.dtype).max / 4
    eps = 1e-12
    dx = torch.where(d[..., 0].abs() < eps, torch.full_like(d[..., 0], eps), d[..., 0])[..., None]
    dy = torch.where(d[..., 1].abs() < eps, torch.full_like(d[..., 1], eps), d[..., 1])[..., None]
    ox, oy = o[:, None, None, 0], o[:, None, None, 1]
    tx1, tx2 = (boxes[:, 0] - ox) / dx, (boxes[:, 1] - ox) / dx
    ty1, ty2 = (boxes[:, 2] - oy) / dy, (boxes[:, 3] - oy) / dy
    t_in = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2)).clamp(-big, big)
    t_out = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2)).clamp(-big, big)
    return t_in, t_out


def sim_scan(boxes, pos, yaw, spec=None, generator=None, metres=False):
    """the fused virtual scan of drones at pos [B, 3] (world metres, z = base_link height) heading yaw
    [B] (rad, ENU) among boxes [K, 5] -> [B, SCAN_BINS] float32, normalised (to_obs convention), or
    the raw ranges in metres (inf: no return) with metres=True."""
    spec = spec or SensorSpec()
    dt, dev = pos.dtype, pos.device
    B = len(pos)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=dt, device=dev)

    if spec.noise:                                   # the scan is taken where the drone believes it is not
        pos = pos + torch.stack([randn(B) * spec.pos_sigma, randn(B) * spec.pos_sigma, randn(B) * spec.z_sigma], -1)
        yaw = yaw + randn(B) * spec.yaw_sigma
    J = max(1, int(spec.beams_per_bin))
    # beams on the bin edges (and evenly between): a real bin is the minimum over all its 40 lidar
    # beams, and the nearest point of a flat face inside a sector is at a sector edge or at the face
    # normal, so edge beams keep the simulated bins from reading farther than the real ones
    off = (torch.linspace(-0.5, 0.5, J, dtype=dt, device=dev) if J > 1 else torch.zeros(1, dtype=dt, device=dev)) * BIN
    rel = (bin_centres(dt, dev)[:, None] + off[None]).reshape(-1)                 # [N] = [K * J]
    ang = yaw[:, None] + rel[None]                                                 # [B, N]
    d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                          # [B, N, 2]
    top = boxes[:, 4]

    # lidar: obstacles taller than its scan plane
    t_in, t_out = ray_boxes(pos[:, :2], d, boxes)
    t0 = t_in.clamp_min(0)
    hit = (t_out >= t0) & (top > pos[:, None, None, 2] + LIDAR_Z)
    r_l = torch.where(hit, t0, torch.full_like(t0, math.inf)).min(-1).values       # [B, N]
    r_l = torch.where((r_l >= LIDAR_RANGE[0]) & (r_l <= LIDAR_RANGE[1]), r_l, torch.full_like(r_l, math.inf))

    # depth camera: the nearest point of the obstacle column inside the vertical field of view and
    # at least OBST_MIN_H above the floor, along each azimuth inside the horizontal field of view
    cam = pos[:, :2] + DEPTH_X * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    t_in, t_out = ray_boxes(cam, d, boxes)
    z_cam = pos[:, None, None, 2] + DEPTH_Z
    t_top = (z_cam - top) / math.tan(DEPTH_VFOV / 2)          # from here on the top is in view
    t_hit = torch.maximum(t_in.clamp_min(0), t_top)
    hit = (t_out >= t_hit) & (top >= OBST_MIN_H)
    r_d = torch.where(hit, t_hit, torch.full_like(t_hit, math.inf)).min(-1).values
    r_d = torch.where((r_d >= DEPTH_RANGE[0]) & (r_d <= DEPTH_RANGE[1]), r_d, torch.full_like(r_d, math.inf))
    r_d = torch.where((rel.abs() <= DEPTH_HFOV / 2)[None], r_d, torch.full_like(r_d, math.inf))

    if spec.noise:
        r_l = r_l + randn(*r_l.shape) * spec.lidar_sigma
        r_d = r_d + randn(*r_d.shape) * spec.depth_k * torch.where(torch.isfinite(r_d), r_d, torch.zeros_like(r_d)) ** 2
    r = torch.minimum(r_l, r_d).clamp_min(0).reshape(B, SCAN_BINS, J).min(-1).values
    if spec.noise and spec.dropout > 0:
        drop = torch.rand(B, SCAN_BINS, generator=generator, dtype=dt, device=dev) < spec.dropout
        r = torch.where(drop, torch.full_like(r, math.inf), r)
    if metres:
        return r
    return (r / SCAN_MAX).clamp(max=1.0).nan_to_num(1.0).float()


# --- real sensor messages -> the same bins (numpy, used in flight) ------------------------------------
def _bin_min(ang, r):
    out = np.full(SCAN_BINS, np.inf)
    k = np.floor((ang + LIDAR_FOV / 2) / BIN).astype(int)
    ok = (k >= 0) & (k < SCAN_BINS) & np.isfinite(r)
    np.minimum.at(out, k[ok], r[ok])
    return out


def bins_from_laserscan(ranges, angle_min, angle_increment, range_min=LIDAR_RANGE[0], range_max=LIDAR_RANGE[1]):
    """sensor_msgs/LaserScan fields -> [SCAN_BINS] metres (inf: no return in the bin)."""
    r = np.asarray(ranges, float)
    ang = angle_min + angle_increment * np.arange(len(r))
    r = np.where((r >= range_min) & (r <= range_max), r, np.inf)
    return _bin_min(ang, r)


def bins_from_points(points, drone_z, roll=0.0, pitch=0.0):
    """depth-camera points [N, 3] in the camera frame (x forward, y left, z up, origin the optical
    centre) -> [SCAN_BINS] metres. The points are levelled with the drone's roll and pitch, kept if
    they are at least OBST_MIN_H above the floor (drone_z: base_link height) and inside the depth
    range and horizontal field of view; each bin is the nearest horizontal range."""
    p = np.asarray(points, float).reshape(-1, 3)
    p = p[np.isfinite(p).all(1)]
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    p = p @ (Ry @ Rx).T
    h = drone_z + DEPTH_Z + p[:, 2]
    rng = np.hypot(p[:, 0], p[:, 1])
    ang = np.arctan2(p[:, 1], p[:, 0])
    ok = (h >= OBST_MIN_H) & (rng >= DEPTH_RANGE[0]) & (rng <= DEPTH_RANGE[1]) & (np.abs(ang) <= DEPTH_HFOV / 2)
    return _bin_min(ang[ok], rng[ok])


def to_obs(*bins_m):
    """fuse per-sensor bins (metres) by their minimum and normalise: [SCAN_BINS] float32 in [0, 1]."""
    r = np.minimum.reduce([np.asarray(b, float) for b in bins_m]) if len(bins_m) > 1 else np.asarray(bins_m[0], float)
    return np.clip(np.nan_to_num(r / SCAN_MAX, nan=1.0, posinf=1.0), 0.0, 1.0).astype(np.float32)


def segment_clear(scan_obs, heading, direction, length, radius):
    """sensor-side safety check of the next straight segment: True unless a return in a bin whose
    sector comes within `radius` of the segment (heading and direction in rad, ENU; length, radius
    in metres) is nearer than the segment end. Bins with no return within SCAN_MAX count as clear;
    the blind sector behind the drone is not checked (the drone has just come from there)."""
    r = np.asarray(scan_obs, float) * SCAN_MAX
    rel = (direction - heading + math.pi) % (2 * math.pi) - math.pi
    c = bin_centres().numpy()
    for k in range(SCAN_BINS):
        if r[k] >= SCAN_MAX - 1e-9:
            continue
        lo, hi = c[k] - BIN / 2, c[k] + BIN / 2
        # the closest angular gap between the bin sector and the segment direction
        gap = 0.0 if lo <= rel <= hi else min(abs(rel - lo), abs(rel - hi))
        if gap >= math.pi / 2:                     # beside / behind the start: only if it is within radius
            if r[k] < radius:
                return False
            continue
        # a return at range r[k] in that sector is within `radius` of the segment's line if
        # r sin(gap) < radius, and in front of the segment end if r cos(gap) < length + radius
        if r[k] * math.sin(gap) < radius and r[k] * math.cos(gap) < length + radius:
            return False
    return True
