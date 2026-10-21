"""Leg-planning MDP over NMOPSO's own path encoding, batched in PyTorch.

An episode builds one path of n = 10 nodes. Step i chooses node i's (r, psi, phi) in NMOPSO.m's
bounds (r in [VarMin.r, VarMax.r], psi and phi in [-pi/4, pi/4]), decoded exactly as
SphericalToCart2.m does (chained transforms from the start pose, clamped to the map). The episode
ends after node n; the path is closed to the goal and scored with MyCost.m (ported).

Reward: a preference w on the 4-simplex is drawn per episode and observed by the policy.
  terminal  -(w . F / F_SCALE)                     if the path is feasible (all objectives finite)
            -(INFEASIBLE + violation)              otherwise (soft violation for a gradient)
  every step  potential-based shaping Phi(s') - Phi(s), Phi = -SHAPE * |q - goal| / PP and
            Phi = 0 at the end, so the optimal policy is unchanged (Ng et al. 1999).
"""
import hashlib
import math

import numpy as np
import torch

from .world import DATA
from .nmopso_cost import ANGLE, DRONE, N_NODES, Legs, tmat

N_RAYS = 16
RAY_MAX = 40.0                         # units (4 m)
F_SCALE = torch.tensor([0.5, 0.02, 1.0, 0.2])   # p95 of NMOPSO's fronts on the factory2_auto plan (scripts/front_stats.py)
S_MAX = 5.0                                      # scaled cost of a feasible path is capped here
INFEASIBLE = S_MAX + 1.0                         # so any feasible path beats any infeasible one
SHAPE = 0.5
OBS_DIM = 3 + 3 + 1 + 1 + 1 + 1 + N_RAYS + 4


RAY_DIRS = 64
_RAY_TABLES = {}


def ray_table(world):
    """[ny, nx, RAY_DIRS] float32: from every unit cell centre (x, y = 0 .. mapsize), the distance along
    each global direction to the first drone-inflated cylinder or the map bounds, / RAY_MAX, capped at 1."""
    key = id(world)
    if key in _RAY_TABLES:
        return _RAY_TABLES[key]
    sig = hashlib.sha1(world.threats.numpy().tobytes() + np.asarray(world.bounds).tobytes()
                       + f"{RAY_DIRS} {RAY_MAX} {DRONE}".encode()).hexdigest()[:12]
    cache = DATA / f"raytable_{sig}.pt"
    if cache.exists():
        _RAY_TABLES[key] = torch.load(cache)
        return _RAY_TABLES[key]
    nx, ny = int(world.bounds[1]) + 2, int(world.bounds[3]) + 2
    th = world.threats
    ang = torch.arange(RAY_DIRS, dtype=th.dtype) * (2 * math.pi / RAY_DIRS)
    d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                     # [K, 2]
    lo = torch.as_tensor(world.bounds[[0, 2]], dtype=th.dtype)
    hi = torch.as_tensor(world.bounds[[1, 3]], dtype=th.dtype)
    R = th[:, 3] + DRONE
    gy, gx = torch.meshgrid(torch.arange(ny, dtype=th.dtype), torch.arange(nx, dtype=th.dtype), indexing="ij")
    P = torch.stack([gx.ravel(), gy.ravel()], -1)
    out = []
    for q in P.split(4096):
        o = q[:, None, None] - th[None, None, :, :2]                          # [C, 1, M, 2]
        bq = (o * d[None, :, None]).sum(-1)                                   # [C, K, M]
        cq = (o * o).sum(-1) - R ** 2
        disc = bq ** 2 - cq
        t = -bq - torch.sqrt(disc.clamp_min(0))
        t = torch.where((disc >= 0) & (t >= 0), t, torch.full_like(t, math.inf))
        t = torch.where(cq < 0, torch.zeros_like(t), t)                       # inside a cylinder: 0
        hit = t.min(-1).values
        tb = torch.where(d > 0, (hi - q[:, None]) / d.clamp_min(1e-9),
                         torch.where(d < 0, (lo - q[:, None]) / d.clamp_max(-1e-9), torch.full_like(q[:, None], math.inf)))
        hit = torch.minimum(hit, tb.min(-1).values.clamp_min(0))
        out.append((hit / RAY_MAX).clamp(max=1.0).float())
    tab = torch.cat(out).reshape(ny, nx, RAY_DIRS)
    torch.save(tab, cache)
    _RAY_TABLES[key] = tab
    return tab


def sample_pref(B, gen=None, dtype=torch.float64):
    """uniform on the simplex (Dirichlet(1,1,1,1))."""
    e = -torch.log(torch.rand(B, 4, generator=gen, dtype=dtype).clamp_min(1e-12))
    return e / e.sum(-1, keepdim=True)


class PathEnv:
    def __init__(self, world, start, goal, pref):
        self.legs = Legs(world, start, goal)
        self.w = world
        self.pref = pref.to(self.legs.dist.dtype)
        B, dt = len(self.legs), self.legs.dist.dtype
        self.T = torch.eye(4, dtype=dt).repeat(B, 1, 1)                      # chain since startPosition
        self.q = self.legs.start.clone()                                    # last path point (clamped)
        self.nodes = torch.zeros(B, N_NODES, 3, dtype=dt)
        self.i = 0
        ang = torch.arange(N_RAYS, dtype=dt) * (2 * math.pi / N_RAYS)
        self.ray_dir = torch.stack([torch.cos(ang), torch.sin(ang)], -1)   # local frame

    def pose(self):
        return self.legs.T0 @ self.T

    def potential(self):
        return -SHAPE * (self.q - self.legs.goal).norm(dim=-1) / self.legs.dist

    def rays(self, q, yaw):
        """distance from q along N_RAYS horizontal directions (relative to yaw) to the first cylinder
        surface inflated by the drone size, or to the map bounds, / RAY_MAX (capped at 1); looked up
        in the precomputed table of ray_table() (1-unit cells, RAY_DIRS global directions)."""
        tab = ray_table(self.w)
        ix = q[:, 0].round().long().clamp(0, tab.shape[1] - 1)
        iy = q[:, 1].round().long().clamp(0, tab.shape[0] - 1)
        k = torch.arange(N_RAYS) * (RAY_DIRS // N_RAYS)
        dk = (yaw / (2 * math.pi / RAY_DIRS)).round().long()
        return tab[iy[:, None], ix[:, None], (k[None] + dk[:, None]) % RAY_DIRS]

    def obs(self):
        P = self.pose()
        R, p = P[:, :3, :3], P[:, :3, 3]
        D = self.legs.dist[:, None]
        g = (R.transpose(1, 2) @ (self.legs.goal - self.q)[..., None])[..., 0] / D
        off = (R.transpose(1, 2) @ (p - self.q)[..., None])[..., 0] / D
        zlo, zhi = self.legs.lo[2], self.legs.hi[2]
        zb = (self.q[:, 2:3] - (zlo + zhi) / 2) / ((zhi - zlo) / 2)
        left = torch.full_like(D, (N_NODES - self.i) / N_NODES)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        o = torch.cat([g, off.clamp(-2, 2), zb, left, D / 100, self.legs.rmax[:, None] / RAY_MAX,
                       self.rays(self.q, yaw), self.pref], -1)
        return o.float()

    def act(self, a):
        """a [B, 3] in [-1, 1] -> (r, psi, phi) in NMOPSO's bounds."""
        a = a.to(self.T.dtype).clamp(-1, 1)
        r = self.legs.rmin + (a[:, 0] + 1) / 2 * (self.legs.rmax - self.legs.rmin)
        return r, a[:, 1] * ANGLE, a[:, 2] * ANGLE

    def step(self, a):
        phi0 = self.potential()
        r, psi, phi = self.act(a)
        self.T = self.T @ tmat(r, phi, psi)
        self.q = self.legs.clamp(self.pose()[:, :3, 3])
        self.nodes[:, self.i] = self.q
        self.i += 1
        done = self.i == N_NODES
        if not done:
            return (self.potential() - phi0).float(), False, None
        F, viol = self.legs.cost(self.nodes, details=True)
        feas = torch.isfinite(F).all(-1)
        s = (self.pref * (F / F_SCALE.to(F.dtype))).sum(-1).clamp(max=S_MAX)
        rew = torch.where(feas, -s, -(INFEASIBLE + viol)) - phi0               # Phi(terminal) = 0
        return rew.float(), True, {"F": F, "feasible": feas}
