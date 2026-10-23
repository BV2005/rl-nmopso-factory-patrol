"""Online leg planner: a trained policy proposes paths for one leg, the deployed path is picked from them.

This is the runtime face of the method evaluated in scripts/evaluate.py, packaged for a caller that
needs one path per leg (the ROS coordinator's 'rl' algorithm in factory_patrol_ws):
  candidates  K preferences x (1 deterministic + S sampled) policy rollouts, plus the straight level
              path (the zero action), all ordinary NMOPSO particles scored with the MyCost port;
  pick        the short-safe path among the feasible candidates (rlpatrol.metrics.short_safe, the rule
              plan_factory_patrol.m flies);
  fallback    if no candidate is feasible and refine is on, the CEM refinement of rlpatrol.refine
              (the RL+LS method of evaluate.py --refine), started from the least-violating candidates.
              If that also fails the plan is reported infeasible: no unchecked path is ever returned.

Frames: world metres, x = Gazebo X (east), y = Gazebo Y (north), z = height above the floor. The
planner works in the trained patrol band (world.z_patrol); the path is horizontal-geometry only,
because MyCost.m treats every obstacle as an infinitely tall cylinder.

Legs whose end points are not free (inside a cylinder or its margin, e.g. a task above a machine)
are planned between the nearest free points; the caller connects the real end points to them.

Keep-in box (keep_in = (xmin, xmax, ymin, ymax) world metres, optional): the flight controller may
clamp its targets to a smaller area than the NMOPSO model's map bounds (patrol_controller.c: |x| <= 12,
|y| <= 7, against 13.35 / 8.35 here). A path through the difference would be flown clamped, i.e. not
at all. With keep_in, a point outside the box is not free (end points, A* routes, line of sight) and
a candidate path with a node outside it is not feasible, so only paths the controller flies are picked.

Routes (plan_route). NMOPSO's encoding (10 nodes, step <= 0.3 x the straight distance, turns <= 45
deg per node) cannot represent detours much longer than the leg, e.g. between two adjacent rack aisles
around the rack. The offline pipeline never asks for such legs: planner/factory_patrol builds the
patrol from line-of-sight lanes and NMOPSO (or the policy) plans each leg. Online, plan_route plays
that role: it first asks the policy for the direct leg; if no candidate is feasible it splits the leg
at line-of-sight via points of an A* route over the same free-space map (world.free) and the policy
plans every sub-leg (a sub-leg that still fails is split again at its route midpoint)."""
import heapq
import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .env import F_SCALE, PathEnv
from .metrics import short_safe
from .nmopso_cost import N_NODES
from .ppo import Actor, load_checkpoint, net_shape, resolve_device
from .refine import WEIGHTS, cem
from .world import load_world

SCALE = F_SCALE.numpy()
U_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]           # threat budgets u
BUDGET_PREFS = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1 / 3, 0, 1 / 3, 1 / 3]]  # w over F1, F3, F4


def pref_set(K, seed=0):
    """corners, edge midpoints, centroid, then Dirichlet(1) samples: K preferences on the 4-simplex."""
    E = np.eye(4)
    P = [*E, *[(E[i] + E[j]) / 2 for i in range(4) for j in range(i + 1, 4)], np.full(4, 0.25)]
    rng = np.random.default_rng(seed)
    while len(P) < K:
        P.append(rng.dirichlet(np.ones(4)))
    return np.array(P[:K])


def conditions(budget, K):
    """the rollout conditions (prefs [C, 4], budgets [C] or None): K preferences, or for a
    threat-budget checkpoint every U_SWEEP budget x BUDGET_PREFS preference (C = 48, K unused)."""
    if budget:
        P = np.array([p for _ in U_SWEEP for p in BUDGET_PREFS], float)
        return P, np.repeat(U_SWEEP, len(BUDGET_PREFS)).astype(float)
    return pref_set(K), None


@dataclass
class LegPlan:
    feasible: bool
    path: np.ndarray                 # [m, 3] world metres: start (free), nodes, goal (free); z = band height
    F: list                          # [F1..F4] of the picked path (MyCost port); inf if infeasible
    method: str                      # rl | straight | rl_ls | none
    start_free: np.ndarray           # planned start (x, y): the requested start or the nearest free point
    goal_free: np.ndarray
    n_candidates: int = 0
    n_feasible: int = 0
    plan_time_s: float = 0.0
    reason: str = ""
    extra: dict = field(default_factory=dict)


class LegPlanner:
    def __init__(self, ckpt, n_prefs=50, n_samples=3, refine=True, device="cpu", threads=0, seed=0,
                 free_search_radius=3.0, keep_in=None):
        self.dev = resolve_device(device)
        if threads:
            torch.set_num_threads(threads)
        ck = load_checkpoint(ckpt, map_location=self.dev)
        cfg = ck["cfg"]
        ctx, extra = net_shape(cfg)
        self.actor = Actor(cfg["hidden"], ctx=ctx, extra=extra).to(self.dev)
        self.actor.load_state_dict(ck["actor"])
        self.actor.eval()
        self.ckpt_info = {"path": str(ckpt), "iteration": ck.get("it", len(ck.get("hist", [])) - 1),
                          "seed": cfg.get("seed"), "hidden": cfg["hidden"], "format": ck.get("format", 1),
                          **{k: cfg.get(k, False) for k in ("budget", "budget_obs", "route_obs")}}
        self.world = load_world(device=self.dev)
        self.budget_obs, self.route_obs = cfg.get("budget_obs", False), cfg.get("route_obs", False)
        self.prefs, self.budgets = conditions(cfg.get("budget", False), n_prefs)
        self.ckpt_info["conditions"] = len(self.prefs)
        self.S, self.refine, self.seed, self.free_r = n_samples, refine, seed, free_search_radius
        self.keep_in = None if keep_in is None else tuple(float(v) for v in keep_in)
        self.ckpt_info["keep_in"] = self.keep_in

    # --- free space --------------------------------------------------------------------------------
    def in_box(self, x, y):
        """inside the keep-in box (everywhere without one); x, y arrays or scalars, world metres."""
        if self.keep_in is None:
            return np.ones(np.shape(x), bool)
        x0, x1, y0, y1 = self.keep_in
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

    def free(self, x, y):
        """world.free inside the keep-in box."""
        return self.world.free(x, y) & self.in_box(x, y)

    def is_free(self, xy):
        return bool(self.free(np.float64(xy[0]), np.float64(xy[1])))

    def nearest_free(self, xy):
        """xy itself if free, else the nearest free point on rings of 0.05 m up to free_search_radius
        (72 directions per ring); None if there is none."""
        xy = np.asarray(xy, float)
        if self.is_free(xy):
            return xy.copy()
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        for r in np.arange(0.05, self.free_r + 1e-9, 0.05):
            c = xy + r * np.c_[np.cos(ang), np.sin(ang)]
            ok = self.free(c[:, 0], c[:, 1])
            if ok.any():
                return c[np.argmax(ok)]
        return None

    # --- planning ----------------------------------------------------------------------------------
    def _rollouts(self, leg, seed):
        """all candidates for one leg: F [m, 4] (numpy), nodes [m, n, 3] (units), encodings [m, 3, n]."""
        torch.manual_seed(seed)
        C, rep = len(self.prefs), 1 + self.S
        L = np.repeat(leg[None], C * rep + 1, 0)                    # + 1: the straight path
        pr = torch.tensor(np.vstack([np.tile(self.prefs, (rep, 1)), self.prefs[:1]]))
        bu = None if self.budgets is None else torch.tensor(np.r_[np.tile(self.budgets, rep), self.budgets[:1]])
        det = torch.tensor([True] * C + [False] * C * self.S + [True], device=self.dev)
        straight = torch.zeros(len(L), dtype=torch.bool, device=self.dev)
        straight[-1] = True
        w = self.world
        env = PathEnv(w, w.to_units(L[:, :3]), w.to_units(L[:, 3:]), pr, budget=bu, budget_obs=self.budget_obs,
                      route_obs=self.route_obs)
        enc = []
        with torch.no_grad():
            for _ in range(N_NODES):
                o = env.obs()
                a = torch.where(det[:, None], self.actor.act(o, True)[0], self.actor.act(o)[0])
                a = torch.where(straight[:, None], torch.zeros_like(a), a)
                if not torch.isfinite(a).all():
                    raise FloatingPointError("policy produced a non-finite action")
                enc.append(torch.stack(env.act(a), -1))
                env.step(a)
            F = env.legs.cost(env.nodes)
        return F.cpu().numpy(), env.nodes.cpu().numpy(), torch.stack(enc, -1).cpu().numpy()

    def plan(self, start_xy, goal_xy, seed=None):
        t0 = time.perf_counter()
        seed = self.seed if seed is None else seed
        s_free, g_free = self.nearest_free(start_xy), self.nearest_free(goal_xy)
        nothing = np.zeros((0, 3))
        if s_free is None or g_free is None:
            which = "start" if s_free is None else "goal"
            return LegPlan(False, nothing, [math.inf] * 4, "none", s_free, g_free,
                           plan_time_s=time.perf_counter() - t0,
                           reason=f"no free point within {self.free_r} m of the {which}")
        z = self.world.z_patrol
        if np.linalg.norm(g_free - s_free) < 0.3:                  # nothing to plan: one short hop
            P = np.array([[*s_free, z], [*g_free, z]])
            return LegPlan(True, P, [0.0] * 4, "trivial", s_free, g_free, 1, 1, time.perf_counter() - t0)
        leg = np.array([s_free[0], s_free[1], z, g_free[0], g_free[1], z])
        F, nodes, enc = self._rollouts(leg, seed)
        feas = np.isfinite(F).all(1) & self._nodes_in_box(nodes)
        method = "rl"
        n_cand, n_feas = len(F), int(feas.sum())
        if not feas.any() and self.refine:
            Fo = np.nan_to_num(F, posinf=1e3)
            picks = [np.argmin(Fo[:, 1]), np.argmin((Fo / SCALE).sum(1)),
                     np.argmin((Fo[:, [0, 2, 3]] / SCALE[[0, 2, 3]]).sum(1))]
            Fl, El = cem(self.world, leg[None], enc[picks][None], F_SCALE.to(torch.float64).to(self.dev), seed=seed)
            F, enc = Fl[0], El[0]
            feas = np.isfinite(F).all(1)
            n_cand += len(F)
            n_feas = int(feas.sum())
            method = "rl_ls"
            if feas.any():
                from .nmopso_cost import Legs                     # decode the refined encodings
                lg = Legs(self.world, self.world.to_units(leg[None, :3]), self.world.to_units(leg[None, 3:]))
                e = torch.as_tensor(enc, device=self.dev)
                nodes = torch.cat([lg.decode(e[i:i + 1, 0], e[i:i + 1, 1], e[i:i + 1, 2]) for i in range(len(e))]
                                  ).cpu().numpy()
                feas &= self._nodes_in_box(nodes)
                n_feas = int(feas.sum())
        if not feas.any():
            return LegPlan(False, nothing, [math.inf] * 4, "none", s_free, g_free, n_cand, 0,
                           time.perf_counter() - t0, reason="no feasible candidate (policy"
                           + (" + CEM refinement)" if self.refine else ")"))
        idx = np.where(feas)[0]
        j = idx[short_safe(F[idx])]
        if method == "rl" and j == len(F) - 1:
            method = "straight"
        w = self.world
        P = np.vstack([[*s_free, z], w.to_metres(nodes[j]), [*g_free, z]])
        return LegPlan(True, P, F[j].tolist(), method, s_free, g_free, n_cand, n_feas, time.perf_counter() - t0)

    def _nodes_in_box(self, nodes):
        """[m] bool: every node of each path [m, n, 3] (NMOPSO units) inside the keep-in box."""
        if self.keep_in is None:
            return np.ones(len(nodes), bool)
        P = self.world.to_metres(nodes)
        return self.in_box(P[..., 0], P[..., 1]).all(-1)

    # --- routes: direct leg, else line-of-sight sub-legs of an A* route -----------------------------
    def _grid(self):
        if not hasattr(self, "_free_grid"):
            h, res = self.world.hall, 0.2
            xs, ys = np.arange(h[0], h[1] + 1e-9, res), np.arange(h[2], h[3] + 1e-9, res)
            X, Y = np.meshgrid(xs, ys, indexing="ij")
            self._free_grid = (xs, ys, res, self.free(X, Y))
        return self._free_grid

    def line_of_sight(self, a, b, step=0.05):
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / step)) + 1)
        t = np.linspace(0, 1, n)[:, None]
        q = np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
        return bool(self.free(q[:, 0], q[:, 1]).all())

    def astar(self, a, b):
        """8-connected A* on a 0.2 m grid of world.free from free point a to free point b; [m, 2] metres
        (exact a and b at the ends) or None."""
        xs, ys, res, G = self._grid()
        nx, ny = G.shape
        ok = lambda c: 0 <= c[0] < nx and 0 <= c[1] < ny and G[c]

        def snap(p):
            """the free grid cell nearest to p (a free point on the margin may round into a blocked cell)."""
            fx, fy = (p[0] - xs[0]) / res, (p[1] - ys[0]) / res
            cand = [(int(math.floor(fx)) + dx, int(math.floor(fy)) + dy) for dx in range(-2, 4) for dy in range(-2, 4)]
            cand = [c for c in cand if ok(c)]
            return min(cand, key=lambda c: (c[0] - fx) ** 2 + (c[1] - fy) ** 2) if cand else None

        s, g = snap(a), snap(b)
        if s is None or g is None:
            return None
        moves = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        hfun = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        openq, came, cost = [(hfun(s), 0.0, s)], {s: None}, {s: 0.0}
        while openq:
            _, c0, c = heapq.heappop(openq)
            if c == g:
                break
            if c0 > cost[c]:
                continue
            for dx, dy, w in moves:
                n_ = (c[0] + dx, c[1] + dy)
                if not ok(n_) or (dx and dy and not (ok((c[0] + dx, c[1])) and ok((c[0], c[1] + dy)))):
                    continue
                nc = c0 + w
                if nc < cost.get(n_, math.inf):
                    cost[n_], came[n_] = nc, c
                    heapq.heappush(openq, (nc + hfun(n_), nc, n_))
        if g not in came:
            return None
        cells, c = [], g
        while c is not None:
            cells.append(c); c = came[c]
        P = np.array([[xs[i], ys[j]] for i, j in reversed(cells)], float)
        P[0], P[-1] = a, b
        return P

    def via_points(self, route):
        """greedy string pulling: the farthest route point still in line of sight, repeatedly."""
        out, i = [route[0]], 0
        while i < len(route) - 1:
            j = len(route) - 1
            while j > i + 1 and not self.line_of_sight(route[i], route[j]):
                j -= 1
            out.append(route[j]); i = j
        return np.array(out)

    def plan_route(self, start_xy, goal_xy, seed=None, max_depth=4):
        """the direct leg if the policy finds a feasible path, else the sub-legs between line-of-sight
        via points of an A* route (each planned by the policy, split again at its route midpoint if it
        fails). Returns a LegPlan whose path concatenates the sub-legs; extra['legs'] lists them."""
        t0 = time.perf_counter()
        seed = self.seed if seed is None else seed
        direct = self.plan(start_xy, goal_xy, seed)
        if direct.feasible or direct.method == "none" and "no free point" in direct.reason:
            direct.extra = {"legs": [self._leg_summary(direct)], "decomposed": False}
            return direct
        s, g = direct.start_free, direct.goal_free
        route = self.astar(s, g)
        if route is None:
            direct.reason = "no free route between the start and the goal (A* on world.free)"
            direct.plan_time_s = time.perf_counter() - t0
            return direct
        legs, n_cand, n_feas = [], direct.n_candidates, direct.n_feasible

        def solve(a_i, b_i, depth):
            nonlocal n_cand, n_feas
            r = self.plan(route[a_i], route[b_i], seed + len(legs) + 1)
            n_cand += r.n_candidates
            if r.feasible:
                n_feas += r.n_feasible
                legs.append(r)
                return True
            if depth >= max_depth or b_i - a_i < 2:
                legs.append(r)
                return False
            m = (a_i + b_i) // 2
            return solve(a_i, m, depth + 1) and solve(m, b_i, depth + 1)

        vias = self.via_points(route)
        vi = [int(np.argmin(np.linalg.norm(route - v, axis=1))) for v in vias]
        ok = all(solve(a, b, 0) for a, b in zip(vi[:-1], vi[1:]))
        dt = time.perf_counter() - t0
        info = {"legs": [self._leg_summary(r) for r in legs], "decomposed": True,
                "via_points": np.round(vias, 3).tolist()}
        if not ok:
            bad = legs[-1]
            return LegPlan(False, np.zeros((0, 3)), [math.inf] * 4, "none", s, g, n_cand, n_feas, dt,
                           reason=f"sub-leg {np.round(bad.start_free, 2).tolist()} -> "
                                  f"{np.round(bad.goal_free, 2).tolist()}: {bad.reason}", extra=info)
        P = np.vstack([legs[0].path] + [r.path[1:] for r in legs[1:]])
        F = np.sum([r.F for r in legs], 0).tolist()          # objectives summed over the sub-legs (diagnostic)
        methods = sorted({r.method for r in legs})
        return LegPlan(True, P, F, "route:" + "+".join(methods), s, g, n_cand, n_feas, dt, extra=info)

    @staticmethod
    def _leg_summary(r):
        return {"feasible": r.feasible, "method": r.method, "F": [round(float(f), 6) for f in r.F],
                "start": np.round(r.start_free, 3).tolist() if r.start_free is not None else None,
                "goal": np.round(r.goal_free, 3).tolist() if r.goal_free is not None else None,
                "candidates": r.n_candidates, "feasible_candidates": r.n_feasible,
                "time_s": round(r.plan_time_s, 4)}
