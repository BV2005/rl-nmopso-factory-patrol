"""Local reactive planner of the global-local architecture: VFH+ on the fused x500_patrol scan.

The global planner (LegPlanner, vB on the prior map + obstacle memory) gives the path; this module only
decides, every control tick, how to get past what the sensors see and the map does not:
  clearance()     nearest return in the front sectors (brake / avoid gating)
  path_blocked()  is the straight segment to a point (the global path's lookahead) blocked in the scan
  vfh_heading()   VFH+ (Ulrich & Borenstein 1998) on the SCAN_BINS polar histogram: bins nearer than
                  d_act are blocked, each blocked bin is enlarged by asin(radius / range) so the drone's
                  width fits through what remains, and the free direction nearest the target, the current
                  heading and the previous choice wins. None if every direction is blocked.
Angles are world ENU bearings (rad); scans are sensors.to_obs / sim_scan observations (normalised bins)."""
import math
from dataclasses import dataclass

import numpy as np

from . import sensors as S


@dataclass
class VFHParams:
    d_act: float = 2.5          # m: returns nearer than this block their direction
    d_brake: float = 0.6        # m: clearance below this -> brake
    radius: float = 0.45        # m: drone half-width + margin (bin enlargement, segment checks)
    mu_target: float = 5.0      # cost weights: towards the target ...
    mu_head: float = 2.0        # ... little turning ...
    mu_prev: float = 2.0        # ... and consistent with the previous choice (no dithering)
    step: float = 0.5           # m: length of one avoidance step
    front: float = math.radians(35)   # half-width of the front sector used by clearance() (brake gating)


def wrap(a):
    return (np.asarray(a, float) + math.pi) % (2 * math.pi) - math.pi


def ranges(scan_obs):
    return np.asarray(scan_obs, float) * S.SCAN_MAX


def clearance(scan_obs, p=VFHParams()):
    """nearest return (m) within +-front of the heading; SCAN_MAX if none."""
    c = S.bin_centres().numpy()
    r = ranges(scan_obs)[np.abs(c) <= p.front + S.BIN / 2]
    return float(r.min()) if len(r) else S.SCAN_MAX


def path_blocked(scan_obs, heading, pos, point, p=VFHParams()):
    """True if the straight segment pos -> point is blocked in the scan (sensors.segment_clear)."""
    v = np.asarray(point[:2], float) - np.asarray(pos[:2], float)
    return not S.segment_clear(scan_obs, heading, math.atan2(v[1], v[0]), float(np.linalg.norm(v)), p.radius)


def blocked_bins(scan_obs, p=VFHParams()):
    """[SCAN_BINS] bool: VFH+ binary histogram with the obstacle enlargement."""
    r = ranges(scan_obs)
    c = S.bin_centres().numpy()
    blocked = np.zeros(S.SCAN_BINS, bool)
    for k in np.where(r < p.d_act)[0]:
        g = math.asin(min(1.0, p.radius / max(r[k], 1e-6)))
        blocked |= np.abs(c - c[k]) <= g + S.BIN / 2
    return blocked


def vfh_heading(scan_obs, heading, target_bearing, prev=None, p=VFHParams()):
    """the world bearing to steer (rad) or None if no direction in the field of view is free."""
    c = S.bin_centres().numpy()
    free = ~blocked_bins(scan_obs, p)
    if not free.any():
        return None
    th = heading + c[free]                                      # candidate world bearings
    cost = p.mu_target * np.abs(wrap(th - target_bearing)) + p.mu_head * np.abs(wrap(th - heading))
    if prev is not None:
        cost = cost + p.mu_prev * np.abs(wrap(th - prev))
    return float(wrap(th[int(np.argmin(cost))]))
