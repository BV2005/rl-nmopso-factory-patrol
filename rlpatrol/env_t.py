"""vT: a step-wise, encoding-free leg planner that minimises flight time (docs/time_optimal_rl.md).

No NMOPSO path encoding: the policy places waypoints one step at a time until it reaches the goal (or
T_MAX decisions). Each step:
  advance    step length r in [R_MIN, R_MAX] m (fixed metres, not relative to the leg), heading change
             d_psi within the turn bound |d_psi| <= min(TURN_MAX, KAPPA * r) (45 deg per step and at most
             45 deg per 0.5 m: a minimum turn radius, so the path is flyable at speed), height in the band
  climb      the same, height above the band up to z_max_local (legal only over a known low obstacle)
  backtrack  return to the previous waypoint (the direction tried is tabu there)
Final approach: when the goal is within GOAL_SNAP, inside the turn bound and the straight segment is
clear, the step goes to the goal and the episode succeeds.

Collision model: obstacle boxes grown by BOX_MARGIN (cost3d.cost3d_box / segment_check_box); the scan
reports ranges to that zone (scan3d margin), so the policy sees what it is judged on.

Objectives (all paths of all methods are scored by flight_metrics with the same model):
  F_time    flight time / straight-line time at cruise - 1 (0 = straight, no turns, no climbs)
  F_threat  mean per-segment threat: 1 - clearance / D_DANGER beyond the margin, clipped to [0, 1]
  F_alt     mean deviation from the patrol band centre (/ half band) + climb energy (sum of rises / leg)
Flight time model per segment: max(horizontal length / V_CRUISE, |dz| / V_Z) + T_TURN * |turn| / 45 deg;
a backtrack flies the segment back plus T_REVERSE. Planning time is not an objective (one network pass
per decision, ~constant): it is measured by scripts/eval_t.py.

Reward: terminal -S_MAX tanh(w . F / SCALE / (4 S_MAX)) on success, -(FAIL + remaining distance / leg) on
a collision or a timeout; per step a potential shaping on the distance to the goal (return-neutral), a
proximity penalty c_prox (1 - clearance / d_safe), -C_TABU for a tabu direction.
Observation (egocentric): goal in the heading frame, distance and height to the goal, height in the band,
headroom, decisions / backtracks left, heading vs goal bearing, the last turn and step, the range scan,
the height scan, the tabu mask, the legal modes, then the preference w (3)."""
import math

import torch

from . import sensors as S
from .cost3d import DANGER_M, DRONE_M, seg_box_dist, segment_check_box
from .env_g import ZBAND, Domain
from .scan3d import H_MAX, sim_scan_b

T_MAX = 60
R_MIN, R_MAX = 0.4, 1.5
TURN_MAX = math.pi / 4
KAPPA = TURN_MAX / 0.5                 # rad per metre: at most 45 deg per 0.5 m
GOAL_SNAP = 1.5
BOX_MARGIN = 0.5
V_CRUISE, V_Z = 1.0, 0.5               # m/s
T_TURN, T_REVERSE = 0.5, 1.5           # s per 45 deg of turn; s to reverse for a backtrack
MAX_BACK = 3
C_TABU = 0.1
FAIL = 6.0
S_MAX = 5.0
SHAPE = 1.0
SCALE_T = torch.tensor([0.5, 0.1, 0.3], dtype=torch.float64)       # typical F_time, F_threat, F_alt
NB = S.SCAN_BINS
CORE = 14
OBS_BASE = CORE + 3 * NB + 3
PRIV = 2 * NB + 2
CTX = 3
ADVANCE, CLIMB, BACK = 0, 1, 2


def obs_dim():
    return OBS_BASE + CTX


def _wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def segment_time(a, b, turn):
    """flight time (s) of segments a -> b ([..., 3]) entered with a heading change `turn` (rad)."""
    d = b - a
    th = torch.maximum(d[..., :2].norm(dim=-1) / V_CRUISE, d[..., 2].abs() / V_Z)
    return th + T_TURN * turn.abs() / TURN_MAX


def flight_metrics(P, n, boxes, kmask, margin=BOX_MARGIN, zband=ZBAND):
    """the common judge of every method's path: P [B, N, 3] metres (packed start .. goal), n [B] points ->
    dict of [B] tensors: time_s, length_m, F (time, threat, alt) [B, 3], collides (box model), max_turn
    (rad, largest heading change at a waypoint), turn_per_m (largest |turn| / following step length)."""
    B, N, _ = P.shape
    dev, dt = P.device, P.dtype
    ar = torch.arange(N, device=dev)
    sval = ar[None, :-1] < (n[:, None] - 1)
    seg = P[:, 1:] - P[:, :-1]
    hd = torch.atan2(seg[..., 1], seg[..., 0])
    hl = seg[..., :2].norm(dim=-1)
    turn = torch.zeros(B, N - 1, dtype=dt, device=dev)
    turn[:, 1:] = _wrap(hd[:, 1:] - hd[:, :-1])
    turn = torch.where((hl > 1e-9) & torch.roll(hl > 1e-9, 1, 1), turn, torch.zeros_like(turn)) * sval
    t = (segment_time(P[:, :-1], P[:, 1:], turn) * sval).sum(-1)
    L = (seg.norm(dim=-1) * sval).sum(-1)
    goal = P[torch.arange(B, device=dev), n - 1]
    D = (goal - P[:, 0])[:, :2].norm(dim=-1).clamp_min(1e-6)
    t_st = D / V_CRUISE
    d = seg_box_dist(P[:, :-1, :2], P[:, 1:, :2], boxes)                # [B, S, K]
    zlow = torch.minimum(P[:, :-1, 2], P[:, 1:, 2])[..., None]
    below = (zlow - boxes[:, None, :, 4] - DRONE_M < 0) & kmask[:, None, :] & sval[..., None]
    clear = torch.where(below, d - margin, torch.full_like(d, math.inf)).min(-1).values
    collides = ((clear < 0) & sval).any(-1)
    thr = ((1 - clear / DANGER_M).clamp(0, 1) * sval).sum(-1) / sval.sum(-1).clamp_min(1)
    nval = (ar[None] >= 1) & (ar[None] < n[:, None] - 1)
    mid, half = (zband[0] + zband[1]) / 2, (zband[1] - zband[0]) / 2
    zdev = (((P[..., 2] - mid).abs() / half) * nval).sum(-1) / nval.sum(-1).clamp_min(1)
    rise = (seg[..., 2].clamp_min(0) * sval).sum(-1) / D
    F = torch.stack([t / t_st - 1, thr, zdev + rise], -1)
    tpm = (turn[:, 1:].abs() / hl[:, 1:].clamp_min(1e-6) * sval[:, 1:]).max(-1).values if N > 2 else torch.zeros(B, dtype=dt, device=dev)
    return dict(time_s=t, length_m=L, F=F, collides=collides, max_turn=turn.abs().max(-1).values, turn_per_m=tpm,
                straight_s=t_st)


class TEnv:
    def __init__(self, batch, pref, domain=Domain(), act_pen=0.02, c_prox=0.5, d_safe=0.5, box_margin=BOX_MARGIN,
                 c_back_time=1.0, generator=None):
        self.b = batch
        legs = batch["legs"]
        dev, dt = legs.device, legs.dtype
        B = len(legs)
        self.B, self.dev, self.dt, self.gen = B, dev, dt, generator
        self.start, self.goal = legs[:, :3].clone(), legs[:, 3:].clone()
        self.D = (self.goal - self.start)[:, :2].norm(dim=-1).clamp_min(1e-6)
        self.t_st = self.D / V_CRUISE
        self.pref = pref.to(device=dev, dtype=dt)
        self.dom, self.act_pen, self.c_prox, self.d_safe, self.margin = domain, act_pen, c_prox, d_safe, box_margin
        self.c_back_time = c_back_time
        self.obst = batch["kmask"].clone()
        self.obst[:, -4:] = False                                              # walls are not obstacles
        h = batch["hall"]
        self.lo = torch.stack([h[:, 0] + 1.0, h[:, 2] + 1.0], -1)
        self.hi = torch.stack([h[:, 1] - 1.0, h[:, 3] - 1.0], -1)
        self.zmax_local = domain.zmax_local[0] + (domain.zmax_local[1] - domain.zmax_local[0]) * torch.rand(
            B, dtype=dt, device=dev, generator=generator)
        self.mount = torch.randn(B, dtype=dt, device=dev, generator=generator) * domain.mount_sigma
        self.spec = S.SensorSpec(noise=domain.noise)
        self.pts = torch.zeros(B, T_MAX + 2, 3, dtype=dt, device=dev)         # waypoint stack, [0] = start
        self.pts[:, 0] = self.start
        self.hds = torch.zeros(B, T_MAX + 2, dtype=dt, device=dev)
        self.hds[:, 0] = torch.atan2(self.goal[:, 1] - self.start[:, 1], self.goal[:, 0] - self.start[:, 0])
        self.k = torch.zeros(B, dtype=torch.long, device=dev)
        self.tabu = torch.zeros(B, T_MAX + 2, NB, dtype=torch.bool, device=dev)
        self.t = 0
        self.backs = torch.zeros(B, dtype=torch.long, device=dev)
        self.n_climb = torch.zeros(B, dtype=torch.long, device=dev)
        self.last_turn = torch.zeros(B, dtype=dt, device=dev)
        self.last_r = torch.zeros(B, dtype=dt, device=dev)
        self.time = torch.zeros(B, dtype=dt, device=dev)                       # flight time so far (incl. backtracks)
        self.alive = torch.ones(B, dtype=torch.bool, device=dev)
        self.success = torch.zeros(B, dtype=torch.bool, device=dev)
        self.crashed = torch.zeros(B, dtype=torch.bool, device=dev)
        self.F = torch.full((B, 3), math.inf, dtype=dt, device=dev)

    # --- state -------------------------------------------------------------------------------------------
    @property
    def cur(self):
        return self.pts[torch.arange(self.B, device=self.dev), self.k]

    @property
    def heading(self):
        return self.hds[torch.arange(self.B, device=self.dev), self.k]

    def _bearing(self, a, b):
        return torch.atan2(b[:, 1] - a[:, 1], b[:, 0] - a[:, 0])

    def scans(self, noise=True):
        p = self.cur.clone()
        p[:, 2] = p[:, 2] + self.mount
        spec = self.spec if noise else S.SensorSpec(noise=False)
        return sim_scan_b(self.b["boxes"], self.b["kmask"], p, self.heading, spec, self.gen, margin=self.margin)

    def legal(self, r, h):
        L = torch.zeros(self.B, 3, dtype=torch.bool, device=self.dev)
        L[:, ADVANCE] = True
        L[:, BACK] = (self.k >= 1) & (self.backs < MAX_BACK)
        front = (S.bin_centres(self.dt, self.dev).abs() <= math.radians(30))[None]
        near = (r.to(self.dt) * S.SCAN_MAX < 3.0) & front
        top = torch.where(near, h.to(self.dt) * H_MAX, torch.zeros_like(h, dtype=self.dt))
        known = ~(near & (h >= 0.999))
        L[:, CLIMB] = near.any(-1) & known.all(-1) & (top.max(-1).values + 0.3 <= self.zmax_local)
        return L

    def obs_pair(self, priv=True):
        p, hd = self.cur, self.heading
        g = self.goal - p
        c, s = torch.cos(hd), torch.sin(hd)
        gx, gy = c * g[:, 0] + s * g[:, 1], -s * g[:, 0] + c * g[:, 1]
        dist = g[:, :2].norm(dim=-1)
        bear = self._bearing(p, self.goal)
        mid, half = (ZBAND[0] + ZBAND[1]) / 2, (ZBAND[1] - ZBAND[0]) / 2
        core = torch.stack([(gx / 10).clamp(-3, 3), (gy / 10).clamp(-3, 3), (dist / 10).clamp(0, 3), g[:, 2],
                            ((p[:, 2] - mid) / half).clamp(-5, 5), self.zmax_local - p[:, 2],
                            torch.full_like(dist, (T_MAX - self.t) / T_MAX), self.backs.to(self.dt) / MAX_BACK,
                            torch.cos(hd - bear), torch.sin(hd - bear), self.last_turn / TURN_MAX, self.last_r / R_MAX,
                            (self.time / self.t_st / 3).clamp(0, 3), self.D / 20], -1)
        r, h = self.scans()
        tabu = self.tabu[torch.arange(self.B, device=self.dev), self.k].to(self.dt)
        L = self.legal(r, h)
        o = torch.cat([core, r.to(self.dt), h.to(self.dt), tabu, L.to(self.dt), self.pref], -1).float()
        if not priv:
            return o, None, L
        ri, hi = self.scans(noise=False)
        vis = (~segment_check_box(p, self.goal, self.b["boxes"], self.obst, self.margin)[0]).to(self.dt)
        oc = torch.cat([core, r.to(self.dt), h.to(self.dt), tabu, L.to(self.dt), ri.to(self.dt), hi.to(self.dt),
                        vis[:, None], (dist / 10)[:, None], self.pref], -1).float()
        return o, oc, L

    # --- dynamics --------------------------------------------------------------------------------------------
    def _potential(self):
        return -SHAPE * (self.goal - self.cur)[:, :2].norm(dim=-1) / self.D

    def _proposal(self, mode, a):
        mode = mode.to(self.dev)                                                # callers may pass CPU tensors
        a = a.to(device=self.dev, dtype=self.dt).clamp(-1, 1)
        p, hd = self.cur, self.heading
        r = R_MIN + (a[:, 0] + 1) / 2 * (R_MAX - R_MIN)
        lim = torch.clamp(KAPPA * r, max=TURN_MAX)
        turn = a[:, 1] * lim                                                    # within the turn bound
        yaw = hd + turn
        mid, half = (ZBAND[0] + ZBAND[1]) / 2, (ZBAND[1] - ZBAND[0]) / 2
        zt = torch.where(mode == CLIMB, ZBAND[1] + (a[:, 2] + 1) / 2 * (self.zmax_local - ZBAND[1]), mid + a[:, 2] * half)
        dz = (zt - p[:, 2]).clamp(-r * 0.999, r * 0.999)
        hor = torch.sqrt((r ** 2 - dz ** 2).clamp_min(0))
        q = torch.stack([p[:, 0] + hor * torch.cos(yaw), p[:, 1] + hor * torch.sin(yaw), p[:, 2] + dz], -1)
        q = torch.cat([torch.minimum(torch.maximum(q[:, :2], self.lo), self.hi), q[:, 2:]], -1)
        # final approach: the goal is near, inside the turn bound and the straight segment is clear
        g = self.goal
        dg = (g - p)[:, :2].norm(dim=-1)
        tg = _wrap(self._bearing(p, g) - hd)
        clear = ~segment_check_box(p, g, self.b["boxes"], self.obst, self.margin)[0]
        snap = (dg <= GOAL_SNAP) & (tg.abs() <= torch.clamp(KAPPA * dg.clamp_min(R_MIN), max=TURN_MAX)) & clear & (mode != BACK)
        q = torch.where(snap[:, None], g, q)
        turn = torch.where(snap, tg, turn)
        r = torch.where(snap, (g - p).norm(dim=-1), r)
        return q, turn, r, snap

    def step(self, mode, a):
        a = a.to(device=self.dev, dtype=self.dt).clamp(-1, 1)
        mode = mode.to(self.dev)
        mode = torch.where(self.alive, mode, torch.full_like(mode, ADVANCE))
        idx = torch.arange(self.B, device=self.dev)
        phi0 = self._potential()
        rew = torch.zeros(self.B, dtype=self.dt, device=self.dev)
        back = self.alive & (mode == BACK) & (self.k >= 1)
        move = self.alive & ~back
        if back.any():
            prev = self.pts[idx, (self.k - 1).clamp_min(0)]
            seg = (self.cur - prev).norm(dim=-1)
            self.time = self.time + torch.where(back, self.c_back_time * (seg / V_CRUISE + T_REVERSE), torch.zeros_like(seg))
            self.k = torch.where(back, self.k - 1, self.k)
            self.backs = self.backs + back.long()
        reached = torch.zeros(self.B, dtype=torch.bool, device=self.dev)
        if move.any():
            q, turn, r, snap = self._proposal(mode, a)
            if self.dom.exec_sigma > 0:
                e = torch.randn(self.B, 2, dtype=self.dt, device=self.dev, generator=self.gen) * self.dom.exec_sigma
                q = torch.where(snap[:, None], q, torch.cat([q[:, :2] + e, q[:, 2:]], -1))
            kb = ((turn + S.LIDAR_FOV / 2) / S.BIN).floor().long().clamp(0, NB - 1)
            was = self.tabu[idx, self.k, kb]
            self.tabu[idx[move], self.k[move], kb[move]] = True
            rew = rew - torch.where(move & was & ~snap, torch.full_like(rew, C_TABU), torch.zeros_like(rew))
            rew = rew - torch.where(move & ~snap, self.act_pen * a.pow(2).sum(-1), torch.zeros_like(rew))
            hit, clear = segment_check_box(self.cur, q, self.b["boxes"], self.obst, self.margin)
            hit = hit & move
            rew = rew - torch.where(move & ~hit, self.c_prox * (1 - clear / self.d_safe).clamp(0, 1), torch.zeros_like(rew))
            ok = move & ~hit
            # the flown turn (the step may have been clamped to the hall), timed from the second segment on
            # (the first one leaves the start in whatever direction the drone already faces), as flight_metrics
            flown = torch.where((q - self.cur)[:, :2].norm(dim=-1) > 1e-9, _wrap(self._bearing(self.cur, q) - self.heading),
                                torch.zeros_like(turn))
            tturn = torch.where(self.k >= 1, flown, torch.zeros_like(flown))
            self.time = self.time + torch.where(ok, segment_time(self.cur, q, tturn), torch.zeros_like(self.time))
            self.n_climb = self.n_climb + (ok & (mode == CLIMB)).long()
            nk = (self.k + 1).clamp(max=T_MAX + 1)
            self.pts[idx[ok], nk[ok]] = q[ok]
            self.hds[idx[ok], nk[ok]] = torch.where(r > 1e-9, self._bearing(self.cur, q), self.heading)[ok]
            self.tabu[idx[ok], nk[ok]] = False
            self.last_turn = torch.where(ok, flown, self.last_turn)
            self.last_r = torch.where(ok, r, self.last_r)
            self.k = torch.where(ok, nk, self.k)
            reached = ok & snap
            if hit.any():
                rem = (self.goal - self.cur)[:, :2].norm(dim=-1) / self.D
                rew = rew - torch.where(hit, FAIL + rem.clamp(max=3), torch.zeros_like(rew))
                self.crashed |= hit
                self.alive &= ~hit
        self.t += 1
        rew = rew + torch.where(self.alive & ~reached, self._potential() - phi0, -phi0 * self.alive)   # Phi(end) = 0
        if reached.any():
            rew = rew + torch.where(reached, self._terminal(reached), torch.zeros_like(rew))
            self.success |= reached
            self.alive &= ~reached
        timeout = self.alive & (self.t >= T_MAX)
        if timeout.any():
            rem = (self.goal - self.cur)[:, :2].norm(dim=-1) / self.D
            rew = rew - torch.where(timeout, FAIL + rem.clamp(max=3), torch.zeros_like(rew))
            self.alive &= ~timeout
        return rew.float(), not bool(self.alive.any())

    def path(self):
        """packed [B, T_MAX + 2, 3] waypoints (start .. current; the goal when reached) and n [B]."""
        return self.pts.clone(), self.k + 1

    def _terminal(self, mask):
        P, n = self.path()
        m = flight_metrics(P, n, self.b["boxes"], self.obst, self.margin)
        # the flown time includes backtracks; the path metrics only the final path
        F = m["F"].clone()
        F[:, 0] = self.time / self.t_st - 1
        self.F = torch.where(mask[:, None], F, self.F)
        s = (self.pref * F / SCALE_T.to(F)).sum(-1)
        return -S_MAX * torch.tanh(s / (4 * S_MAX))


def sample_pref3(B, gen=None):
    e = -torch.log(torch.rand(B, 3, generator=gen, dtype=torch.float64).clamp_min(1e-12))
    return e / e.sum(-1, keepdim=True)
