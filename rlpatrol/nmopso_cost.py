"""Batched PyTorch port of the authors' NMOPSO path model, used only to make RL training fast:
SphericalToCart2.m + TransfomationMatrix.m (decoding) and MyCost.m (objectives F1-F4).
It is checked against the MATLAB originals by scripts/verify_cost.py; reported results are always
re-scored with the unmodified MATLAB MyCost.m (matlab/eval_mycost.m).

Units are NMOPSO's (1 unit = 0.1 m). n = 10 path nodes between start and goal, as CreateModel.m."""
import math

import torch

N_NODES = 10
ANGLE = math.pi / 4          # NMOPSO.m AngleRange: psi, phi in [-pi/4, pi/4]
DRONE = 1.0                  # MyCost.m drone_size
DANGER = 10.0 * DRONE        # MyCost.m danger_dist


def tmat(r, phi, psi):
    """TransfomationMatrix.m: Rot_z(phi) * Rot_y(-psi) * Trans_x(r), batched [..., 4, 4]."""
    cz, sz = torch.cos(phi), torch.sin(phi)
    cy, sy = torch.cos(-psi), torch.sin(-psi)
    z, o = torch.zeros_like(r), torch.ones_like(r)
    rz = torch.stack([torch.stack([cz, -sz, z, z], -1), torch.stack([sz, cz, z, z], -1),
                      torch.stack([z, z, o, z], -1), torch.stack([z, z, z, o], -1)], -2)
    ry = torch.stack([torch.stack([cy, z, sy, z], -1), torch.stack([z, o, z, z], -1),
                      torch.stack([-sy, z, cy, z], -1), torch.stack([z, z, z, o], -1)], -2)
    tx = torch.stack([torch.stack([o, z, z, r], -1), torch.stack([z, o, z, z], -1),
                      torch.stack([z, z, o, z], -1), torch.stack([z, z, z, o], -1)], -2)
    return rz @ ry @ tx


class Legs:
    """A batch of legs (start, goal in NMOPSO units, [B, 3]) in one world."""

    def __init__(self, world, start, goal):
        self.w = world
        dev, dt = world.threats.device, world.threats.dtype
        self.start = torch.as_tensor(start, dtype=dt, device=dev)
        self.goal = torch.as_tensor(goal, dtype=dt, device=dev)
        d = self.goal - self.start
        self.dist = d.norm(dim=-1)                                       # PP in MyCost.m
        self.rmax = 3 * self.dist / N_NODES                              # NMOPSO.m VarMax.r
        self.rmin = self.rmax / 9                                        # VarMin.r
        phis = torch.atan2(d[:, 1], d[:, 0])
        psis = torch.atan2(d[:, 2], d[:, :2].norm(dim=-1))
        T0 = torch.eye(4, dtype=dt, device=dev).repeat(len(d), 1, 1)
        T0[:, :3, 3] = self.start
        self.T0 = T0 @ tmat(torch.zeros_like(phis), phis, psis)          # startPosition
        b = torch.as_tensor(world.bounds, dtype=dt, device=dev)
        self.lo, self.hi = b[[0, 2, 4]], b[[1, 3, 5]]

    def __len__(self):
        return len(self.dist)

    def clamp(self, p):
        return torch.minimum(torch.maximum(p, self.lo), self.hi)

    def decode(self, r, psi, phi):
        """SphericalToCart2.m: [B, n] each -> nodes [B, n, 3] (clamped to the map bounds)."""
        T = torch.eye(4, dtype=r.dtype, device=r.device).repeat(len(r), 1, 1)
        out = []
        for i in range(r.shape[1]):
            T = T @ tmat(r[:, i], phi[:, i], psi[:, i])
            out.append(self.clamp((self.T0 @ T)[:, :3, 3]))
        return torch.stack(out, 1)

    def cost(self, nodes, details=False):
        """MyCost.m: nodes [B, n, 3] -> F [B, 4] = [J1 path length, J2 threat, J3 altitude, J4 smoothness]
        (inf where infeasible). details=True also returns a soft constraint violation per path
        (>= 0, 0 iff J1 and J2 are finite), for shaping only."""
        P = torch.cat([self.start[:, None], nodes, self.goal[:, None]], 1)   # [B, N, 3], N = n + 2
        seg = P[:, 1:] - P[:, :-1]                                            # [B, N-1, 3]
        sl = seg.norm(dim=-1)
        short = sl <= self.rmin[:, None]                                      # Traj = 0 -> J1 = inf
        traj = sl.sum(-1)
        J1 = (1 - self.dist / traj).abs()
        J1 = torch.where(short.any(-1), torch.full_like(J1, math.inf), J1)

        th = self.w.threats                                                   # [M, 4]
        a, e = P[:, :-1, None, :2], seg[:, :, None, :2]                      # [B, S, 1, 2]
        c = th[None, None, :, :2]                                             # [1, 1, M, 2]
        ee = (e * e).sum(-1)
        t = torch.where(ee > 0, ((c - a) * e).sum(-1) / ee.clamp_min(1e-300), torch.zeros_like(ee)).clamp(0, 1)
        dist = (a + t[..., None] * e - c).norm(dim=-1)                        # DistP2S, [B, S, M]
        R = th[:, 3]
        tc = (1 - (dist - DRONE - R) / DANGER).clamp(0, 1)                    # 0 beyond the danger zone
        hit = dist < R + DRONE
        J2 = tc.mean((-1, -2))
        J2 = torch.where(hit.any(-1).any(-1), torch.full_like(J2, math.inf), J2)

        z = nodes[..., 2]
        zmin, zmax = self.lo[2], self.hi[2]
        J3 = ((z - (zmax + zmin) / 2).abs() / ((zmax - zmin) / 2)).mean(-1)
        J3 = torch.where((z < 0).any(-1), torch.full_like(J3, math.inf), J3)

        s1, s2 = seg[:, :-1], seg[:, 1:]                                      # feasible paths: no zero segments
        ang = torch.atan2(torch.linalg.cross(s1, s2).norm(dim=-1), (s1 * s2).sum(-1)) / math.pi
        J4 = ang.mean(-1)
        F = torch.stack([J1, J2, J3, J4], -1)
        if not details:
            return F
        pen = ((R + DRONE - dist).clamp_min(0) / DANGER).sum((-1, -2))
        shortv = ((self.rmin[:, None] - sl).clamp_min(0) / self.rmin[:, None]).sum(-1) + short.float().sum(-1) * 0.1
        return F, pen + shortv
