"""Leg-planning MDP of the generalizable policy vG (docs/generalizable_rl.md), batched over per-episode worlds.

Every episode has its own factory (a generated world, or factory2 for the benchmark) and possibly an
unexpected box on its leg. An episode is one leg, at most T_DEC decisions; the path is a stack of at
most N_NODES nodes (NMOPSO's encoding size):
  advance    next node: step r = r_nom * exp(a_r) in [rmin, rmax], heading = goal bearing + a_yaw * 90 deg,
             clipped to +-45 deg from the current heading (NMOPSO's turn bound), height band centre +
             a_z * half band, climb angle <= 45 deg
  climb      the same step, height above the band up to z_max_local (legal only if the height scan
             shows a known, low enough obstacle ahead)
  backtrack  return to the previous node (legal from node 1 on, at most MAX_BACK times); the direction
             just tried from there stays tabu (observed; choosing a tabu direction again is penalised)
The episode ends when N_NODES nodes are placed, when T_DEC decisions are used (the path is closed to
the goal as is), or at a collision (a step into an obstacle, 3D rule of cost3d). The terminal reward is
vS's: -cap(scalarised MyCost-3D + budget overshoot), or -(INFEASIBLE + violation); per decision
-C_BACK * backtracked length / leg length, -C_CLIMB * height above the band, -C_TABU, -act_pen |a|^2.

Observation (egocentric, nothing absolute): CORE (goal in the heading frame, reach, height in the band,
headroom to z_max_local, nodes / decisions / backtracks left, heading vs goal bearing, leg length),
the fused range scan, the obstacle-height scan (rlpatrol/scan3d.py), the tabu mask of the current node,
the legal modes, then the conditioning (u with budget conditioning, the preference). The privileged
critic also sees ideal (noise-free) ranges and heights, the true goal visibility and the threat to go."""
import math
from dataclasses import dataclass

import numpy as np
import torch

from . import sensors as S
from .cost3d import TOP_INF, cost3d, segment_hits
from .env import BUDGET_PEN, F_SCALE, INFEASIBLE, S_MAX, SOFT_SCALE, sample_pref
from .nmopso_cost import N_NODES
from .scan3d import sim_scan_b, walls
from .worldgen import cylinders

T_DEC = 20
MAX_BACK = 5
ZBAND = (1.6, 2.0)
SIG_R, YAW_MAX, TURN = 1.0, math.pi / 2, math.pi / 4
C_BACK, C_CLIMB, C_TABU = 0.5, 0.2, 0.1
NB = S.SCAN_BINS
CORE = 12
OBS_BASE = CORE + 3 * NB + 3          # + ctx (4 or 5)
PRIV = 2 * NB + 2
ADVANCE, CLIMB, BACK = 0, 1, 2


def obs_dim(budget):
    return OBS_BASE + (5 if budget else 4)


@dataclass
class Domain:
    """per-episode domain randomisation ranges (a phase's dr level picks them)."""
    noise: bool = True
    zmax_local: tuple = (2.6, 2.6)
    exec_sigma: float = 0.0           # m, node execution (tracking) error
    mount_sigma: float = 0.0          # m, sensor mounting height error


DR = {0: Domain(), 1: Domain(zmax_local=(2.2, 3.0), exec_sigma=0.1, mount_sigma=0.03)}


# --- batches of levels ----------------------------------------------------------------------------------
def factory2_level(world):
    """factory2 (rlpatrol.world.World) as a level source: its boxes for the sensors; its NMOPSO cylinders
    as infinitely tall (2D MyCost, as the benchmark)."""
    c = np.c_[world.cyl_m, np.full(len(world.cyl_m), TOP_INF)]
    return np.asarray(world.hall, float), np.asarray(world.rects, float), c


def build_batch(items, device="cpu", dtype=torch.float64):
    """items: per episode (hall [4], boxes [k, 5], cyl [m, 4], leg [6], extra box [5] or None) ->
    padded tensors (hall walls added to the boxes; the extra box also gets its cylinder cover)."""
    B = len(items)
    bx, cy = [], []
    for hall, boxes, cyl, leg, extra in items:
        if extra is not None:
            boxes = np.vstack([boxes, extra[None]])
            cyl = np.vstack([cyl, cylinders(extra[None])])
        bx.append(boxes)
        cy.append(cyl)
    K, M = max(len(b) for b in bx), max(len(c) for c in cy)
    boxes = np.zeros((B, K, 5)); kmask = np.zeros((B, K), bool)
    cyl = np.zeros((B, M, 4)); cmask = np.zeros((B, M), bool)
    for i, (b, c) in enumerate(zip(bx, cy)):
        boxes[i, :len(b)], kmask[i, :len(b)] = b, True
        cyl[i, :len(c)], cmask[i, :len(c)] = c, True
    hall = torch.tensor(np.array([it[0] for it in items]), dtype=dtype, device=device)
    t = lambda a, d=dtype: torch.tensor(a, dtype=d, device=device)  # noqa: E731
    boxes_t = torch.cat([t(boxes), walls(hall)], 1)
    kmask_t = torch.cat([t(kmask, torch.bool), torch.ones(B, 4, dtype=torch.bool, device=device)], 1)
    return dict(hall=hall, boxes=boxes_t, kmask=kmask_t, cyl=t(cyl), cmask=t(cmask, torch.bool),
                legs=t(np.array([it[3] for it in items])))


# --- the environment -------------------------------------------------------------------------------------
class GenEnv:
    def __init__(self, batch, pref, budget=None, modes="all", domain=Domain(), budget_pen=BUDGET_PEN, act_pen=0.02,
                 wall_margin=1.0, generator=None):
        self.b = batch
        legs = batch["legs"]
        dev, dt = legs.device, legs.dtype
        B = len(legs)
        self.B, self.dev, self.dt = B, dev, dt
        self.start, self.goal = legs[:, :3].clone(), legs[:, 3:].clone()
        self.D = (self.goal - self.start)[:, :2].norm(dim=-1).clamp_min(1e-6)
        self.rmax = 3 * (self.goal - self.start).norm(dim=-1) / N_NODES
        self.rmin = self.rmax / 9
        self.pref = pref.to(device=dev, dtype=dt)
        self.u = None if budget is None else torch.as_tensor(budget, dtype=dt, device=dev)
        self.modes, self.dom, self.budget_pen, self.act_pen = modes, domain, budget_pen, act_pen
        self.gen = generator
        h = batch["hall"]
        self.lo = torch.stack([h[:, 0] + wall_margin, h[:, 2] + wall_margin], -1)
        self.hi = torch.stack([h[:, 1] - wall_margin, h[:, 3] - wall_margin], -1)
        self.zmax_local = self._uniform(*domain.zmax_local)
        self.mount = torch.randn(B, dtype=dt, device=dev, generator=generator) * domain.mount_sigma
        self.spec = S.SensorSpec(noise=domain.noise)
        self.nodes = torch.zeros(B, N_NODES + 1, 3, dtype=dt, device=dev)     # [0] = start
        self.nodes[:, 0] = self.start
        self.head = torch.zeros(B, N_NODES + 1, dtype=dt, device=dev)
        self.head[:, 0] = self._bearing(self.start, self.goal)
        self.k = torch.zeros(B, dtype=torch.long, device=dev)                  # nodes placed
        self.t = 0
        self.backs = torch.zeros(B, dtype=torch.long, device=dev)
        self.tabu = torch.zeros(B, N_NODES + 1, NB, dtype=torch.bool, device=dev)
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        self.crashed = torch.zeros(B, dtype=torch.bool, device=dev)
        self.flown_back = torch.zeros(B, dtype=dt, device=dev)
        self.n_climb = torch.zeros(B, dtype=torch.long, device=dev)
        self.F = torch.full((B, 4), math.inf, dtype=dt, device=dev)
        self.feasible = torch.zeros(B, dtype=torch.bool, device=dev)
        if self.u is not None:                                                 # the straight path's threat
            s = torch.linspace(0, 1, N_NODES + 2, dtype=dt, device=dev)[None, :, None]
            Pst = self.start[:, None] + s * (self.goal - self.start)[:, None]
            F = cost3d(Pst, torch.full((B,), N_NODES + 2, device=dev), batch["cyl"], batch["cmask"], ZBAND, details=True)
            self.eps = self.u * F[2]

    def _uniform(self, lo, hi):
        return lo + (hi - lo) * torch.rand(self.B, dtype=self.dt, device=self.dev, generator=self.gen)

    @staticmethod
    def _bearing(a, b):
        return torch.atan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])

    @property
    def cur(self):
        return self.nodes[torch.arange(self.B, device=self.dev), self.k]

    @property
    def heading(self):
        return self.head[torch.arange(self.B, device=self.dev), self.k]

    # --- observation ---------------------------------------------------------------------------------
    def scans(self, noise=True):
        p = self.cur.clone()
        p[:, 2] = p[:, 2] + self.mount
        spec = self.spec if noise else S.SensorSpec(noise=False)
        return sim_scan_b(self.b["boxes"], self.b["kmask"], p, self.heading, spec, self.gen)

    def legal(self, heights=None, ranges=None):
        """[B, 3] bool: advance always; climb if allowed and a known obstacle ahead is low enough;
        backtrack from node 1 on, at most MAX_BACK times."""
        L = torch.zeros(self.B, 3, dtype=torch.bool, device=self.dev)
        L[:, ADVANCE] = True
        if self.modes != "all":
            return L
        L[:, BACK] = (self.k >= 1) & (self.backs < MAX_BACK)
        front = (S.bin_centres(self.dt, self.dev).abs() <= math.radians(30))[None]
        near = (ranges * S.SCAN_MAX < 3.0) & front
        from .scan3d import H_MAX
        top = torch.where(near, heights.to(self.dt) * H_MAX, torch.zeros_like(heights, dtype=self.dt))
        known = ~(near & (heights >= 0.999))
        L[:, CLIMB] = near.any(-1) & known.all(-1) & (top.max(-1).values + 0.3 <= self.zmax_local)
        return L

    def obs_pair(self, priv=True):
        """(actor obs [B, OBS], critic obs [B, OBS + PRIV] or None, legal [B, 3])."""
        p, hd = self.cur, self.heading
        g = self.goal - p
        c, s = torch.cos(hd), torch.sin(hd)
        gx, gy = c * g[:, 0] + s * g[:, 1], -s * g[:, 0] + c * g[:, 1]
        left = (N_NODES - self.k).to(self.dt)
        bear = self._bearing(p, self.goal)
        mid, half = (ZBAND[0] + ZBAND[1]) / 2, (ZBAND[1] - ZBAND[0]) / 2
        core = torch.stack([gx / self.D, gy / self.D, g[:, 2], g[:, :2].norm(dim=-1) / (left.clamp_min(1) * self.rmax),
                            ((p[:, 2] - mid) / half).clamp(-5, 5), self.zmax_local - p[:, 2], left / N_NODES,
                            torch.full_like(left, (T_DEC - self.t) / T_DEC), self.backs.to(self.dt) / MAX_BACK,
                            torch.cos(hd - bear), torch.sin(hd - bear), self.D / 20], -1)
        r, h = self.scans()
        tabu = self.tabu[torch.arange(self.B, device=self.dev), self.k].to(self.dt)
        L = self.legal(h, r)
        ctx = ([self.u[:, None]] if self.u is not None else []) + [self.pref]
        o = torch.cat([core, r.to(self.dt), h.to(self.dt), tabu, L.to(self.dt)] + ctx, -1).float()
        if not priv:
            return o, None, L
        ri, hi = self.scans(noise=False)
        vis = (~segment_hits(p, self.goal, self.b["cyl"], self.b["cmask"])).to(self.dt)
        Pg = torch.stack([p, self.goal], 1)
        tg = cost3d(Pg, torch.full((self.B,), 2, device=self.dev), self.b["cyl"], self.b["cmask"], ZBAND, details=True)[2]
        oc = torch.cat([core, r.to(self.dt), h.to(self.dt), tabu, L.to(self.dt), ri.to(self.dt), hi.to(self.dt),
                        vis[:, None], (tg / F_SCALE[1].item()).clamp(-5, 5)[:, None]] + ctx, -1).float()
        return o, oc, L

    # --- dynamics ------------------------------------------------------------------------------------------
    def _proposal(self, mode, a):
        p, hd = self.cur, self.heading
        g = self.goal - p
        left = (N_NODES - self.k).clamp_min(1).to(self.dt)
        r_nom = g.norm(dim=-1) / left
        r = torch.minimum(torch.maximum(r_nom * torch.exp(SIG_R * a[:, 0]), self.rmin * 1.001), self.rmax)
        yaw = self._bearing(p, self.goal) + a[:, 1] * YAW_MAX
        d = torch.atan2(torch.sin(yaw - hd), torch.cos(yaw - hd)).clamp(-TURN, TURN)
        yaw = hd + d
        mid, half = (ZBAND[0] + ZBAND[1]) / 2, (ZBAND[1] - ZBAND[0]) / 2
        zt = torch.where(mode == CLIMB, ZBAND[1] + (a[:, 2] + 1) / 2 * (self.zmax_local - ZBAND[1]), mid + a[:, 2] * half)
        dz = (zt - p[:, 2]).clamp(-r * 0.999, r * 0.999)                        # climb angle <= 45 deg
        hor = torch.sqrt((r ** 2 - dz ** 2).clamp_min(0))
        q = torch.stack([p[:, 0] + hor * torch.cos(yaw), p[:, 1] + hor * torch.sin(yaw), p[:, 2] + dz], -1)
        q[:, :2] = torch.minimum(torch.maximum(q[:, :2], self.lo), self.hi)
        return q, d

    def step(self, mode, a):
        """mode [B] long, a [B, 3] in [-1, 1] -> reward [B] float32, done (all episodes finished)."""
        a = a.to(self.dt).clamp(-1, 1)
        mode = torch.where(self.alive, mode, torch.full_like(mode, ADVANCE))
        rew = torch.zeros(self.B, dtype=self.dt, device=self.dev)
        idx = torch.arange(self.B, device=self.dev)
        move = self.alive & (mode != BACK)
        back = self.alive & (mode == BACK) & (self.k >= 1)
        # backtrack: return to the previous node; flown twice
        if back.any():
            seg = (self.nodes[idx, self.k] - self.nodes[idx, (self.k - 1).clamp_min(0)]).norm(dim=-1)
            rew = rew - torch.where(back, C_BACK * seg / self.D, torch.zeros_like(seg))
            self.flown_back = self.flown_back + torch.where(back, seg, torch.zeros_like(seg))
            self.k = torch.where(back, self.k - 1, self.k)
            self.backs = self.backs + back.long()
        # advance / climb
        if move.any():
            q, d = self._proposal(mode, a)
            if self.dom.exec_sigma > 0:
                e = torch.randn(self.B, 2, dtype=self.dt, device=self.dev, generator=self.gen) * self.dom.exec_sigma
                q = q.clone(); q[:, :2] = q[:, :2] + e
            kb = ((d + S.LIDAR_FOV / 2) / S.BIN).floor().long().clamp(0, NB - 1)
            was = self.tabu[idx, self.k, kb]
            self.tabu[idx[move], self.k[move], kb[move]] = True
            rew = rew - torch.where(move & was, torch.full_like(rew, C_TABU), torch.zeros_like(rew))
            rew = rew - torch.where(move, self.act_pen * a.pow(2).sum(-1), torch.zeros_like(rew))
            climbz = (q[:, 2] - ZBAND[1]).clamp_min(0)
            rew = rew - torch.where(move, C_CLIMB * climbz, torch.zeros_like(rew))
            self.n_climb = self.n_climb + (move & (mode == CLIMB)).long()
            hit = segment_hits(self.cur, q, self.b["cyl"], self.b["cmask"]) & move
            ok = move & ~hit
            nk = (self.k + 1).clamp(max=N_NODES)
            self.nodes[idx[ok], nk[ok]] = q[ok]
            self.head[idx[ok], nk[ok]] = self._bearing(self.cur, q)[ok]
            self.tabu[idx[ok], nk[ok]] = False
            self.k = torch.where(ok, nk, self.k)
            crash = hit
            if crash.any():
                rew = rew - torch.where(crash, torch.full_like(rew, INFEASIBLE + 3.0), torch.zeros_like(rew))
                self.crashed |= crash
                self.alive &= ~crash
        self.t += 1
        fin = self.alive & ((self.k >= N_NODES) | (self.t >= T_DEC))
        if fin.any():
            rew = rew + torch.where(fin, self._terminal(), torch.zeros_like(rew))
            self.alive &= ~fin
        return rew.float(), not bool(self.alive.any())

    def path(self):
        """packed [B, N_NODES + 2, 3] start, nodes, goal and n_pts [B]."""
        P = torch.zeros(self.B, N_NODES + 2, 3, dtype=self.dt, device=self.dev)
        P[:, :N_NODES + 1] = self.nodes
        idx = torch.arange(self.B, device=self.dev)
        P[idx, self.k + 1] = self.goal
        return P, self.k + 2

    def _terminal(self):
        P, n = self.path()
        F, viol, J2s = cost3d(P, n, self.b["cyl"], self.b["cmask"], ZBAND, details=True)
        feas = torch.isfinite(F).all(-1)
        Ff = F.nan_to_num(posinf=1e6)
        s = (self.pref * Ff / F_SCALE.to(Ff)).sum(-1)
        if self.u is not None:
            s = s + self.budget_pen * (Ff[:, 1] - self.eps).clamp_min(0) / F_SCALE[1].item()
        s = S_MAX * torch.tanh(s / SOFT_SCALE)
        upd = self.alive & ((self.k >= N_NODES) | (self.t >= T_DEC))
        self.F = torch.where(upd[:, None], F, self.F)
        self.feasible = torch.where(upd, feas, self.feasible)
        return torch.where(feas, -s, -(INFEASIBLE + viol.clamp(max=3.0)))


def sample_conditions(B, budget, u_lo=0.15, u_hi=1.2, gen=None):
    """preference (w2 = 0 with budget conditioning) and threat budget u, as ppo.sample_budget_pref."""
    w = sample_pref(B, gen)
    if not budget:
        return w, None
    w[:, 1] = 0
    w = w / w.sum(-1, keepdim=True)
    return w, u_lo + (u_hi - u_lo) * torch.rand(B, generator=gen, dtype=torch.float64)
