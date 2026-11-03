"""Obstacle memory: the light "sensed layer" of the global-local architecture.

The prior map knows the factory as it was mapped. When the local planner cannot get past something,
the obstacle it saw is stored here as a box (with a time to live), and the global planner replans in
memory.world(now) = the prior world plus the remembered boxes (world.with_extra: NMOPSO cylinders for
the path cost, physical boxes for the simulated sensors and the A* route grid). Without it a replan on
the prior map would return the same blocked path."""
import math

import numpy as np

from . import sensors as S
from .world import with_extra


class ObstacleMemory:
    def __init__(self, world, ttl=300.0, explained=0.3, cluster=0.5, min_size=0.3, max_range=4.0):
        self.prior = world
        self.ttl, self.explained, self.cluster, self.min_size = ttl, explained, cluster, min_size
        self.max_range = max_range                          # farther returns are too coarse (10 deg bins)
        self.items = []                                    # (box [5], t_seen)
        self._prior_boxes = S.scene_boxes(world).cpu().numpy()

    def points_from_scan(self, scan_obs, pos, heading, sector=None):
        """world xy of the scan's returns (bin centre direction x range), optionally only bins whose
        relative angle is inside sector = (lo, hi) rad; returns the prior map explains are dropped."""
        r = np.asarray(scan_obs, float) * S.SCAN_MAX
        c = S.bin_centres().numpy()
        ok = r < min(S.SCAN_MAX - 1e-6, self.max_range)
        if sector is not None:
            ok &= (c >= sector[0]) & (c <= sector[1])
        a = heading + c[ok]
        P = np.c_[pos[0] + r[ok] * np.cos(a), pos[1] + r[ok] * np.sin(a)]
        # a return is placed at its bin's centre angle: up to r tan(BIN / 2) off sideways
        tol = self.explained + r[ok] * math.tan(S.BIN / 2)
        keep = [not self._near_prior(p, t) for p, t in zip(P, tol)]
        return P[np.asarray(keep, bool)] if len(P) else P

    def _near_prior(self, p, d=None):
        d = self.explained if d is None else d
        b = self._prior_boxes
        return bool(((p[0] >= b[:, 0] - d) & (p[0] <= b[:, 1] + d) & (p[1] >= b[:, 2] - d) & (p[1] <= b[:, 3] + d)).any())

    def boxes_from_points(self, P, top=2.5):
        """single-linkage clusters (cluster m) of the points -> axis-aligned boxes, at least min_size wide."""
        P = np.asarray(P, float).reshape(-1, 2)
        if not len(P):
            return []
        lab = -np.ones(len(P), int)
        for i in range(len(P)):
            if lab[i] >= 0:
                continue
            lab[i] = i
            stack = [i]
            while stack:
                j = stack.pop()
                near = np.where((lab < 0) & (np.hypot(*(P - P[j]).T) <= self.cluster))[0]
                lab[near] = i
                stack.extend(near.tolist())
        out = []
        for l in np.unique(lab):
            Q = P[lab == l]
            x0, y0 = Q.min(0)
            x1, y1 = Q.max(0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hx, hy = max((x1 - x0) / 2, self.min_size / 2), max((y1 - y0) / 2, self.min_size / 2)
            out.append(np.array([cx - hx, cx + hx, cy - hy, cy + hy, top]))
        return out

    def boxes_from_scan(self, scan_obs, pos, heading, sector=None, depth=0.6):
        """boxes of the unexplained returns, each extended `depth` m away from the drone (the scan sees
        only the near face; the obstacle continues behind it)."""
        out = []
        for b in self.boxes_from_points(self.points_from_scan(scan_obs, pos, heading, sector)):
            dx, dy = (b[0] + b[1]) / 2 - pos[0], (b[2] + b[3]) / 2 - pos[1]
            n = math.hypot(dx, dy) or 1.0
            ex, ey = depth * dx / n, depth * dy / n
            out.append(np.array([min(b[0], b[0] + ex), max(b[1], b[1] + ex), min(b[2], b[2] + ey), max(b[3], b[3] + ey), b[4]]))
        return out

    def add(self, box, now):
        self.items.append((np.asarray(box, float), float(now)))

    def active(self, now):
        """[k, 5] boxes seen within ttl."""
        self.items = [(b, t) for b, t in self.items if now - t <= self.ttl]
        return np.array([b for b, _ in self.items]).reshape(-1, 5)

    def world(self, now):
        """the prior world plus the active boxes (each also as its circumscribed NMOPSO cylinder)."""
        B = self.active(now)
        if not len(B):
            return self.prior
        cyl = np.c_[(B[:, 0] + B[:, 1]) / 2, (B[:, 2] + B[:, 3]) / 2,
                    np.hypot(B[:, 1] - B[:, 0], B[:, 3] - B[:, 2]) / 2]
        return with_extra(self.prior, cyl, boxes=B)

    def near(self, path_xy, now, dist=1.5, age=60.0):
        """True if the path passes within dist of a box seen less than age s ago (adaptive budget)."""
        P = np.asarray(path_xy, float).reshape(-1, 2)
        for b, t in self.items:
            if now - t > age:
                continue
            dx = np.maximum(np.maximum(b[0] - P[:, 0], 0), P[:, 0] - b[1])
            dy = np.maximum(np.maximum(b[2] - P[:, 1], 0), P[:, 1] - b[3])
            if (np.hypot(dx, dy) < dist).any():
                return True
        return False
