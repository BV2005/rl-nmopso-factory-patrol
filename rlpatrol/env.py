"""Leg-planning MDP over NMOPSO's own path encoding, batched in PyTorch.

An episode builds one path of n = 10 nodes. Step i outputs node i's (r, psi, phi) in NMOPSO.m's
bounds (r in [VarMin.r, VarMax.r], psi and phi in [-pi/4, pi/4]), decoded exactly as
SphericalToCart2.m does (chained transforms from the start pose, clamped to the map). The episode
ends after node n; the path is closed to the goal and scored with MyCost.m (ported).

Action (residual, a in (-1, 1)^3): the zero action is "straight at the goal, level at the band
centre". a_r scales the even-spacing step length (r_nom = |goal - p| / segments left) by
exp(SIG_R a_r); a_yaw turns the world bearing from the goal bearing by a_yaw * YAW_MAX; a_z sets the
height the step aims for, band centre + a_z * half band. The world direction is converted exactly to
the chain's local (phi, psi) (Rz(phi) Ry(-psi) e_x = d_local) and clipped to NMOPSO's bounds, so every
path is an ordinary NMOPSO particle.

Reward: a preference w on the 4-simplex is drawn per episode and observed by the policy.
  terminal  -(w . F / F_SCALE)                     if the path is feasible (all objectives finite)
            -(INFEASIBLE + min(violation, 3))      otherwise (soft violation for a gradient)
  optional  potential-based shaping (shape > 0): Phi = -shape * |q - goal| / PP, Phi = 0 at the end.
  optional  straight prior (act_pen > 0): -act_pen * |a|^2 every step, so the policy leaves the
            straight level path only where it pays; a training regulariser, never part of the score.
  optional  dense threat (dense): J2 is a mean over segments, so each step is paid -w2 * (its new
            segment's share of J2) / F_SCALE[1], plus the growth of the budget overshoot; the terminal
            reward is the usual one minus what was already paid, so every path's return is unchanged.
            At most S_MAX is paid early, so that remainder is never positive (a capped cost or a
            collision would otherwise refund it and credit the later steps).
Observation option (budget_obs, with budget conditioning): BUDGET_OBS entries before u, in
F_SCALE[1] units: the budget left (u * straight F2 - threat of the segments so far) and the threat
of flying straight from the current point to the goal over the remaining segments.
"""
import hashlib
import math

import numpy as np
import torch

from .world import DATA
from .nmopso_cost import ANGLE, DANGER, DRONE, N_NODES, Legs, tmat

N_RAYS = 16
RAY_MAX = 40.0                         # units (4 m)
F_SCALE = torch.tensor([0.5, 0.02, 1.0, 0.2])   # p95 of NMOPSO's fronts on the factory2_auto plan (scripts/front_stats.py)
S_MAX = 5.0                                      # scaled cost of a feasible path is capped here
INFEASIBLE = S_MAX + 1.0                         # so any feasible path beats any infeasible one
SIG_R = 1.0                                      # a_r = +-1: step length x e^+-1
YAW_MAX = math.pi / 2                            # a_yaw = +-1: +-90 deg from the goal bearing
OBS_DIM = 3 + 1 + 1 + 3 + 1 + 1 + 1 + 2 + N_RAYS + 4      # + 1 (threat budget u) when budget conditioning is on
BUDGET_PEN = 5.0                                 # per F_SCALE[1] of threat above the budget
BUDGET_OBS = 2                                   # extra observation entries with budget_obs
N_SEGS = N_NODES + 1                             # start -> node 1 -> ... -> node n -> goal


RAY_DIRS = 64
_RAY_TABLES = {}


def ray_table(world):
    """[ny, nx, RAY_DIRS] float32: from every unit cell centre (x, y = 0 .. mapsize), the distance along
    each global direction to the first drone-inflated cylinder or the map bounds, / RAY_MAX, capped at 1."""
    dev = world.threats.device
    key = (id(world), str(dev))
    if key in _RAY_TABLES:
        return _RAY_TABLES[key]
    th = world.threats.detach().cpu()
    sig = hashlib.sha1(th.numpy().tobytes() + np.asarray(world.bounds).tobytes()
                       + f"{RAY_DIRS} {RAY_MAX} {DRONE}".encode()).hexdigest()[:12]
    cache = DATA / f"raytable_{sig}.pt"
    if cache.exists():
        _RAY_TABLES[key] = torch.load(cache, map_location="cpu").to(dev)
        return _RAY_TABLES[key]
    nx, ny = int(world.bounds[1]) + 2, int(world.bounds[3]) + 2
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
    _RAY_TABLES[key] = tab.to(dev)
    return _RAY_TABLES[key]


def direct_rays(world, q, yaw):
    """exact ray casting from q [B, 3] along N_RAYS directions relative to yaw [B] (the geometry of
    ray_table, without its grid and direction rounding), / RAY_MAX capped at 1."""
    th = world.threats
    ang = yaw[:, None] + torch.arange(N_RAYS, dtype=th.dtype, device=th.device) * (2 * math.pi / N_RAYS)
    d = torch.stack([torch.cos(ang), torch.sin(ang)], -1)                     # [B, K, 2]
    o = q[:, None, None, :2] - th[None, None, :, :2]                          # [B, 1, M, 2]
    R = th[:, 3] + DRONE
    bq = (o * d[:, :, None]).sum(-1)
    cq = (o * o).sum(-1) - R ** 2
    disc = bq ** 2 - cq
    t = -bq - torch.sqrt(disc.clamp_min(0))
    t = torch.where((disc >= 0) & (t >= 0), t, torch.full_like(t, math.inf))
    t = torch.where(cq < 0, torch.zeros_like(t), t)
    hit = t.min(-1).values
    lo = torch.as_tensor(world.bounds[[0, 2]], dtype=th.dtype, device=th.device)
    hi = torch.as_tensor(world.bounds[[1, 3]], dtype=th.dtype, device=th.device)
    qq = q[:, None, :2]
    tb = torch.where(d > 0, (hi - qq) / d.clamp_min(1e-9), torch.where(d < 0, (lo - qq) / d.clamp_max(-1e-9), torch.full_like(d, math.inf)))
    hit = torch.minimum(hit, tb.min(-1).values.clamp_min(0))
    return (hit / RAY_MAX).clamp(max=1.0).float()


def sample_pref(B, gen=None, dtype=torch.float64):
    """uniform on the simplex (Dirichlet(1,1,1,1))."""
    e = -torch.log(torch.rand(B, 4, generator=gen, dtype=dtype).clamp_min(1e-12))
    return e / e.sum(-1, keepdim=True)




def seg_threat(world, P):
    """each segment's share of MyCost.m's soft J2: points P [B, K, 3] -> [B, K-1], summing to J2 when
    P is the whole path (start, n nodes, goal); same formula as Legs.cost (horizontal distance)."""
    th = world.threats
    a, e = P[:, :-1, None, :2], (P[:, 1:] - P[:, :-1])[:, :, None, :2]
    c = th[None, None, :, :2]
    ee = (e * e).sum(-1)
    t = torch.where(ee > 0, ((c - a) * e).sum(-1) / ee.clamp_min(1e-300), torch.zeros_like(ee)).clamp(0, 1)
    dist = (a + t[..., None] * e - c).norm(dim=-1)
    tc = (1 - (dist - DRONE - th[:, 3]) / DANGER).clamp(0, 1)
    return tc.sum(-1) / (N_SEGS * len(th))


class PathEnv:
    def __init__(self, world, start, goal, pref, shape=0.0, scalar="linear", act_pen=0.0, budget=None, budget_pen=BUDGET_PEN,
                 dense=False, budget_obs=False):
        """budget: None, or u [B] - threat-budget conditioning: the path should keep F2 <= u * F2 of
        the straight (zero-action) path (its F2 without the collision infinity); u is observed.
        dense, budget_obs: see the module doc."""
        if budget_obs and budget is None:
            raise ValueError("budget_obs needs budget conditioning")
        if dense and scalar != "linear":
            raise ValueError("dense threat needs the linear scalarisation (a max does not split by segment)")
        self.legs = Legs(world, start, goal)
        self.w = world
        self.pref = pref.to(self.legs.dist)                               # dtype and device of the world
        self.shape, self.scalar, self.act_pen, self.budget_pen = shape, scalar, act_pen, budget_pen
        B, dt, dev = len(self.legs), self.legs.dist.dtype, self.legs.dist.device
        self.T = torch.eye(4, dtype=dt, device=dev).repeat(B, 1, 1)          # chain since startPosition
        self.q = self.legs.start.clone()                                    # last path point (clamped)
        self.nodes = torch.zeros(B, N_NODES, 3, dtype=dt, device=dev)
        self.i = 0
        self.dense, self.budget_obs = dense, budget_obs
        self.prev = self.legs.start.clone()                                 # start of the next segment
        self.c = torch.zeros(B, dtype=dt, device=dev)                       # J2 share of the segments so far
        self.paid = torch.zeros(B, dtype=dt, device=dev)                    # dense rewards paid so far
        self.zmid = (self.legs.lo[2] + self.legs.hi[2]) / 2
        self.zhalf = (self.legs.hi[2] - self.legs.lo[2]) / 2
        self.u = None
        if budget is not None:
            ref = PathEnv(world, start, goal, pref)
            for _ in range(N_NODES):
                ref.step(torch.zeros(B, 3, device=dev))
            self.f2_straight = ref.legs.cost(ref.nodes, details=True)[2]
            self.u = torch.as_tensor(budget, dtype=dt, device=dev)
            self.eps = self.u * self.f2_straight

    def pose(self):
        return self.legs.T0 @ self.T

    def potential(self):
        return -self.shape * (self.q - self.legs.goal).norm(dim=-1) / self.legs.dist

    def rays(self, q, yaw):
        """distance from q along N_RAYS horizontal directions (relative to yaw) to the first cylinder
        surface inflated by the drone size, or to the map bounds, / RAY_MAX (capped at 1); looked up
        in the precomputed table of ray_table() (1-unit cells, RAY_DIRS global directions)."""
        if self.w.direct_rays:
            return direct_rays(self.w, q, yaw)
        tab = ray_table(self.w)
        ix = q[:, 0].round().long().clamp(0, tab.shape[1] - 1)
        iy = q[:, 1].round().long().clamp(0, tab.shape[0] - 1)
        k = torch.arange(N_RAYS, device=q.device) * (RAY_DIRS // N_RAYS)
        dk = (yaw / (2 * math.pi / RAY_DIRS)).round().long()
        return tab[iy[:, None], ix[:, None], (k[None] + dk[:, None]) % RAY_DIRS]

    def segs_left(self):
        return N_NODES - self.i + 1                                         # this node ... goal

    def obs(self):
        P = self.pose()
        R, p = P[:, :3, :3], P[:, :3, 3]
        D = self.legs.dist[:, None]
        gv = self.legs.goal - p
        g = (R.transpose(1, 2) @ gv[..., None])[..., 0] / D                 # goal, chain frame
        reach = gv.norm(dim=-1, keepdim=True) / (self.segs_left() * self.legs.rmax[:, None])
        zc = ((p[:, 2:3] - self.zmid) / self.zhalf).clamp(-5, 5)            # unclamped chain height
        off = ((self.q - p) / 10).clamp(-2, 2)                              # clamping offset (m)
        left = torch.full_like(D, (N_NODES - self.i) / N_NODES)
        bear = torch.atan2(self.legs.goal[:, 1] - self.q[:, 1], self.legs.goal[:, 0] - self.q[:, 0])
        head = torch.atan2(R[:, 1, 0], R[:, 0, 0]) - bear                   # chain heading vs goal bearing
        o = torch.cat([g, reach, zc, off, left, D / 100, self.legs.rmax[:, None] / RAY_MAX,
                       torch.cos(head)[:, None], torch.sin(head)[:, None],
                       self.rays(self.q, bear)] + self.budget_state() + ([self.u[:, None]] if self.u is not None else [])
                      + [self.pref], -1)
        return o.float()

    def budget_state(self):
        """[] or [budget left, straight threat to go] ([B, 1] each, F_SCALE[1] units, clamped to +-5)."""
        if not self.budget_obs:
            return []
        n = self.segs_left()
        t = torch.linspace(0, 1, n + 1, dtype=self.q.dtype, device=self.q.device)[None, :, None]
        togo = seg_threat(self.w, self.q[:, None] + t * (self.legs.goal - self.q)[:, None]).sum(-1)
        f2 = F_SCALE[1].item()
        return [((self.eps - self.c) / f2).clamp(-5, 5)[:, None], (togo / f2).clamp(-5, 5)[:, None]]

    def act(self, a):
        """residual a [B, 3] in [-1, 1] -> (r, psi, phi) in NMOPSO's bounds (see module doc)."""
        a = a.to(self.T).clamp(-1, 1)
        P = self.pose()
        R, p = P[:, :3, :3], P[:, :3, 3]
        gv = self.legs.goal - p
        r_nom = gv.norm(dim=-1) / self.segs_left()
        r = torch.minimum(torch.maximum(r_nom * torch.exp(SIG_R * a[:, 0]), self.legs.rmin * 1.001), self.legs.rmax)
        yaw = torch.atan2(gv[:, 1], gv[:, 0]) + a[:, 1] * YAW_MAX
        dz = self.zmid + a[:, 2] * self.zhalf - p[:, 2]
        el = torch.asin((dz / r).clamp(-1, 1))
        d = torch.stack([torch.cos(el) * torch.cos(yaw), torch.cos(el) * torch.sin(yaw), torch.sin(el)], -1)
        dl = (R.transpose(1, 2) @ d[..., None])[..., 0]
        phi = torch.atan2(dl[:, 1], dl[:, 0]).clamp(-ANGLE, ANGLE)
        psi = torch.atan2(dl[:, 2], dl[:, :2].norm(dim=-1)).clamp(-ANGLE, ANGLE)
        return r, psi, phi

    def scalarise(self, F):
        Fs = F / F_SCALE.to(F)
        if self.scalar == "tcheby":                                         # augmented Tchebycheff
            return (self.pref * Fs).max(-1).values + 0.05 * (self.pref * Fs).sum(-1)
        return (self.pref * Fs).sum(-1)

    def step(self, a):
        phi0 = self.potential()
        pen = (self.act_pen * a.to(self.T).pow(2).sum(-1)).float()   # straight prior (training only)
        r, psi, phi = self.act(a)
        self.T = self.T @ tmat(r, phi, psi)
        self.q = self.legs.clamp(self.pose()[:, :3, 3])
        self.nodes[:, self.i] = self.q
        self.i += 1
        c0, self.c = self.c, self.c + seg_threat(self.w, torch.stack([self.prev, self.q], 1))[:, 0]
        self.prev = self.q
        done = self.i == N_NODES
        if not done:
            d = torch.zeros_like(self.c)
            if self.dense:                                                  # this segment's threat, as it is flown
                d = -self.pref[:, 1] * (self.c - c0) / F_SCALE[1].item()
                if self.u is not None:
                    d = d - self.budget_pen * ((self.c - self.eps).clamp_min(0) - (c0 - self.eps).clamp_min(0)) / F_SCALE[1].item()
                d = (self.paid + d).clamp_min(-S_MAX) - self.paid               # paid >= -S_MAX: the terminal
                self.paid = self.paid + d                                       # remainder is never a refund
            return (self.potential() - phi0 + d).float() - pen, False, None
        F, viol, _ = self.legs.cost(self.nodes, details=True)
        feas = torch.isfinite(F).all(-1)
        Ff = F.nan_to_num(posinf=1e6)
        s = self.scalarise(Ff)
        if self.u is not None:                                              # F2 is a budget, not a weight
            s = s + self.budget_pen * (Ff[:, 1] - self.eps).clamp_min(0) / F_SCALE[1].item()
        s = s.clamp(max=S_MAX)
        rew = torch.where(feas, -s, -(INFEASIBLE + viol.clamp(max=3.0))) - phi0   # Phi(terminal) = 0
        rew = rew - self.paid                                               # dense: the return is unchanged
        return rew.float() - pen, True, {"F": F, "feasible": feas}
