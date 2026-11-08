"""Flight-side decision core of vT (rlpatrol/env_t.py): turns live sensor scans into the next waypoint of a leg.

No ROS here, so it runs the same in Gazebo (gazebo/nmopso_demo/vt_follower.py) and in the offline test with
synthetic sensors (scripts/test_vt_flight.py).

The policy sees exactly its training observation: a vT environment (TEnv) is advanced with the accepted
decisions, so the goal in the heading frame, decisions / backtracks left, the last turn and step, time used,
tabu directions and legal modes are computed as in training. Two things differ:
  * the scans: the environment's simulated scan is replaced by the flight scan (scan3d.flight_scan of the real
    lidar and depth data, taken where the drone is) - FlightEnv.scan_override;
  * the map: the environment knows only the hall walls (it is bookkeeping, the policy never sees a map). The
    safety of every step is checked against the flight scan (clear()), plus the walls.

decide() tries, in order, and returns the first step the scan shows clear:
  1. the policy's deterministic decision (mode and action),
  2. `samples` stochastic decisions of the policy,
  3. a sweep of the action space (step length x heading change, the policy's height action; advance, and climb
     where it is legal), nearest to the policy's own action first,
  4. a backtrack to the previous waypoint (known free: the drone came from there), where legal.
None if nothing is clear (the caller hovers, rescans, and gives up on the leg after a while).

A step is clear if no return of the scan - ranges to the zone `margin` around every obstacle - lies within `radius`
of the segment before its end, checked on the full-resolution scan (1080 beams of 0.25 deg; the policy itself sees
the 27 bins it was trained on), except returns whose obstacle top is known and at least DRONE_M below the segment
(the 3-D overfly rule of training, for climbs).

Times: the environment's flight time (time model of env_t) is reported per leg next to the flown time measured
in the simulator, which is what the Gazebo test is for."""
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

from . import sensors as S
from .cost3d import DRONE_M, segment_check_box
from .env_g import ZBAND, Domain, build_batch
from .env_t import ADVANCE, BACK, CLIMB, TEnv, flight_metrics
from .ppo_g import load_checkpoint
from .ppo_t import env_options_t, make_actor_t
from .scan3d import H_MAX

FLIGHT = Domain(noise=False)            # no simulated noise / tracking error: the real ones are flown
MODES = {ADVANCE: "advance", CLIMB: "climb", BACK: "back"}


class FlightEnv(TEnv):
    """TEnv whose scans come from outside (the drone's sensors) when scan_override = (ranges, heights) is set."""
    scan_override = None

    def scans(self, noise=True):
        if self.scan_override is not None:
            return self.scan_override
        return super().scans(noise)


class Decision(NamedTuple):
    mode: int                           # ADVANCE / CLIMB / BACK
    action: torch.Tensor                # [1, 3] in [-1, 1]
    target: np.ndarray                  # [3] world metres: the next waypoint
    source: str                         # policy | sample | sweep | backtrack
    tries: int                          # candidates checked (1 = the policy's own decision)
    snap: bool                          # the final approach to the goal


class VTPilot:
    def __init__(self, ckpt, hall, pref=(1 / 3, 1 / 3, 1 / 3), zmax_local=2.6, radius=0.1, samples=16, seed=0):
        """ckpt: a vT checkpoint (path or loaded dict); hall [4] xmin xmax ymin ymax (m)."""
        ck = load_checkpoint(ckpt) if isinstance(ckpt, (str, Path)) else ckpt
        self.cfg = ck["cfg"]
        self.it = ck.get("it")
        self.actor = make_actor_t(self.cfg)
        self.actor.load_state_dict(ck["actor"])
        self.actor.eval()
        self.hall = np.asarray(hall, float)
        self.pref = torch.tensor([list(pref)], dtype=torch.float64)
        self.opts = env_options_t(self.cfg)
        self.margin = float(self.opts["box_margin"])
        self.zmax_local, self.radius, self.samples = float(zmax_local), float(radius), int(samples)
        torch.manual_seed(seed)                                         # the stochastic candidates
        self.env = None

    # --- a leg -------------------------------------------------------------------------------------------
    def start_leg(self, start, goal):
        leg = np.r_[np.asarray(start, float), np.asarray(goal, float)]
        b = build_batch([(self.hall, np.zeros((0, 5)), np.zeros((0, 4)), leg, None)])
        self.env = FlightEnv(b, self.pref, domain=FLIGHT, **self.opts)
        self.env.zmax_local = torch.full((1,), self.zmax_local, dtype=torch.float64)
        self.goal = np.asarray(goal, float)

    @property
    def cur(self):
        return self.env.cur[0].numpy().copy()

    @property
    def heading(self):
        """the heading the policy assumes (ENU rad): the direction of the last flown segment."""
        return float(self.env.heading[0])

    @property
    def success(self):
        return bool(self.env.success[0])

    @property
    def over(self):
        """the leg's episode has ended (success, or out of decisions)."""
        return not bool(self.env.alive[0])

    def report(self):
        """the leg as the environment saw it: decisions, modes, model flight time, executed polyline."""
        e = self.env
        P, n = e.executed()
        m = flight_metrics(P, n, e.b["boxes"], e.obst, self.margin)
        return dict(success=self.success, decisions=int(e.t), backtracks=int(e.backs[0]), climbs=int(e.n_climb[0]),
                    model_time_s=float(e.time[0]), straight_time_s=float(e.t_st[0]),
                    model_time_ratio=float(e.time[0] / e.t_st[0]), max_turn_deg=math.degrees(float(m["max_turn"][0])),
                    executed=P[0, :int(n[0])].numpy().tolist())

    # --- decisions ---------------------------------------------------------------------------------------
    def clear(self, q, ranges, heights, beams=None):
        """True if the step from the current waypoint to q [3] is clear in the scan (and of the walls). With the
        full-resolution scan (beams, scan3d.flight_scan(beams=True)): no zone return within `radius` of the segment
        before its end, on any 0.25 deg beam; else the 10 deg bins (sensors.segment_clear: a return anywhere in its
        sector, so steps that skirt a zone at a shallow angle are refused). Returns of an obstacle whose top is known
        and at least DRONE_M below the segment are flown over (the 3-D rule of training)."""
        p = self.cur
        v = np.asarray(q[:2], float) - p[:2]
        L = float(np.hypot(*v))
        zlow = min(p[2], float(q[2]))
        if L > 1e-6 and beams is not None:
            x, y, top = (np.asarray(beams[k], float) for k in ("x", "y", "top"))
            seen = np.isfinite(x)                                         # beams with a zone return
            x, y = np.where(seen, x, 0.0), np.where(seen, y, 0.0)
            phi = math.atan2(v[1], v[0]) - self.heading                   # the step in the drone frame
            along = x * math.cos(phi) + y * math.sin(phi)
            lat = np.abs(-x * math.sin(phi) + y * math.cos(phi))
            use = seen & ~(top + DRONE_M <= zlow) & (along > 0)
            if bool((use & (lat < self.radius) & (along < L + self.radius)).any()):
                return False
        elif L > 1e-6:
            r = np.asarray(ranges, float).copy()
            top = np.asarray(heights, float) * H_MAX
            over = (np.asarray(heights) > 0) & (np.asarray(heights) < 0.999) & (top + DRONE_M <= zlow)
            r[over] = 1.0                                                # known low obstacle, flown over
            if not S.segment_clear(r, self.heading, math.atan2(v[1], v[0]), L, self.radius):
                return False
        e = self.env
        qt = torch.as_tensor(np.asarray(q, float), dtype=e.dt)[None]
        return not bool(segment_check_box(e.cur, qt, e.b["boxes"], e.coll, e.margin)[0][0])

    def _candidates(self, o, L):
        with torch.no_grad():
            m0, a0 = self.actor.act(o, L, True)[:2]
            yield "policy", int(m0[0]), a0
            for _ in range(self.samples):
                m, a = self.actor.act(o, L)[:2]
                yield "sample", int(m[0]), a
        modes = [ADVANCE] + ([CLIMB] if bool(L[0, CLIMB]) else [])
        grid = [(mode, torch.tensor([[ar, at, float(a0[0, 2])]], dtype=a0.dtype))
                for mode in modes for ar in (1.0, 0.0, -1.0) for at in np.linspace(-1, 1, 9)]
        for mode, a in sorted(grid, key=lambda g: (g[0] != int(m0[0]), float((g[1] - a0).norm()))):
            yield "sweep", mode, a
        if bool(L[0, BACK]):
            yield "backtrack", BACK, torch.zeros(1, 3, dtype=a0.dtype)

    def decide(self, ranges, heights, beams=None):
        """ranges, heights [SCAN_BINS] (scan3d.flight_scan, taken at the current waypoint facing self.heading; the
        policy's observation) and optionally its full-resolution beams (the safety check) -> the first clear
        Decision, or None."""
        e = self.env
        e.scan_override = (torch.as_tensor(np.asarray(ranges, np.float32))[None],
                           torch.as_tensor(np.asarray(heights, np.float32))[None])
        o, _, L = e.obs_pair(False)
        tries = 0
        for source, mode, a in self._candidates(o, L):
            if not bool(L[0, mode]):
                continue
            tries += 1
            if mode == BACK:
                q = e.pts[0, int(e.k[0]) - 1].numpy().copy()
                return Decision(mode, a, q, source, tries, False)
            q, _, _, snap = e._proposal(torch.tensor([mode]), a)
            q = q[0].numpy().copy()
            if self.clear(q, ranges, heights, beams):
                return Decision(mode, a, q, source, tries, bool(snap[0]))
        return None

    def commit(self, d):
        """advance the environment with an accepted decision -> the new current waypoint (= d.target)."""
        self.env.step(torch.tensor([d.mode]), d.action)
        if bool(self.env.crashed[0]):                                  # the walls (checked in clear()): not expected
            raise RuntimeError("the environment reports a wall collision for a step the pilot accepted")
        return self.cur
