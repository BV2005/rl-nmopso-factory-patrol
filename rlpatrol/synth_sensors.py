"""Synthetic raw sensor data of the x500_patrol drone, for testing the flight pipeline without Gazebo.

Ray-casts what Gazebo's sensors of gazebo/models/x500_patrol return among obstacle boxes [K, 5] (xmin xmax
ymin ymax top, metres, standing on the floor z = 0; include the hall walls):
  lidar_scan()   the 2D lidar (lidar_2d_v2): 1080 beams over +-135 deg in the plane LIDAR_Z above base_link,
                 ranges 0.1-30 m, as sensor_msgs/LaserScan fields
  depth_cloud()  the depth camera (320 x 240, horizontal FOV DEPTH_HFOV, 0.2-19.1 m along the optical axis) as
                 the points of its point cloud in the camera frame (x forward, y left, z up; origin the optical
                 centre), floor included
The drone is level (roll = pitch = 0). numpy only."""
import math

import numpy as np

from . import sensors as S

LIDAR_BEAMS = 1080


def _slab(o, d, lo, hi):
    """rays o [3] + t d [N, 3] against boxes lo, hi [K, 3] -> t_in, t_out [N, K] (miss: t_out < max(t_in, 0))."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.where(np.abs(d) < 1e-12, 1e-12, d)
    t1 = (lo[None] - o[None, None]) * inv[:, None]
    t2 = (hi[None] - o[None, None]) * inv[:, None]
    return np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)


def lidar_scan(boxes, pos, yaw):
    """-> (ranges [LIDAR_BEAMS] with inf for no return, angle_min, angle_increment) of the drone at pos [3]
    (world, base_link) heading yaw (rad, ENU)."""
    b = np.asarray(boxes, float).reshape(-1, 5)
    z = pos[2] + S.LIDAR_Z
    b = b[b[:, 4] > z]                                                   # only obstacles reaching the scan plane
    a_min = -S.LIDAR_FOV / 2
    inc = S.LIDAR_FOV / (LIDAR_BEAMS - 1)
    ang = yaw + a_min + inc * np.arange(LIDAR_BEAMS)
    d = np.c_[np.cos(ang), np.sin(ang), np.zeros(LIDAR_BEAMS)]
    o = np.array([pos[0], pos[1], z])
    if len(b):
        lo = np.c_[b[:, 0], b[:, 2], np.full(len(b), -1.0)]
        hi = np.c_[b[:, 1], b[:, 3], b[:, 4]]
        t_in, t_out = _slab(o, d, lo, hi)
        t0 = np.maximum(t_in, 0)
        t = np.where(t_out >= t0, t0, np.inf).min(1)
    else:
        t = np.full(LIDAR_BEAMS, np.inf)
    t = np.where((t >= S.LIDAR_RANGE[0]) & (t <= S.LIDAR_RANGE[1]), t, np.inf)
    return t, a_min, inc


def depth_cloud(boxes, pos, yaw, width=S.DEPTH_W, height=S.DEPTH_H):
    """-> [N, 3] the depth camera's points (camera frame) of the drone at pos [3] heading yaw: every pixel whose
    ray hits a box or the floor within the depth range."""
    b = np.asarray(boxes, float).reshape(-1, 5)
    f = (width / 2) / math.tan(S.DEPTH_HFOV / 2)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dc = np.c_[np.ones(u.size), -(u.ravel() - (width - 1) / 2) / f, -(v.ravel() - (height - 1) / 2) / f]   # x = 1
    c, s = math.cos(yaw), math.sin(yaw)
    dw = np.c_[c * dc[:, 0] - s * dc[:, 1], s * dc[:, 0] + c * dc[:, 1], dc[:, 2]]
    o = np.array([pos[0] + S.DEPTH_X * c, pos[1] + S.DEPTH_X * s, pos[2] + S.DEPTH_Z])
    t = np.full(len(dw), np.inf)
    if len(b):
        lo = np.c_[b[:, 0], b[:, 2], np.zeros(len(b))]
        hi = np.c_[b[:, 1], b[:, 3], b[:, 4]]
        t_in, t_out = _slab(o, dw, lo, hi)
        t0 = np.maximum(t_in, 0)
        t = np.where(t_out >= t0, t0, np.inf).min(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dw[:, 2] < -1e-9, -o[2] / dw[:, 2], np.inf)
    t = np.minimum(t, t_floor)
    ok = (t >= S.DEPTH_RANGE[0]) & (t <= S.DEPTH_RANGE[1])               # x component of dc is 1: t = depth
    return dc[ok] * t[ok, None]
