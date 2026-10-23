"""NMOPSO as an online leg planner: a vectorised port of the authors' NMOPSO.m (Duong, Bui & Phung 2025).

Same problem as the authors' code and as the RL policy: path = n = 10 nodes in navigation variables
(r, psi, phi), decoded by SphericalToCart2.m and scored by MyCost.m (both through the verified PyTorch
port in rlpatrol.nmopso_cost). The search follows NMOPSO.m step by step:

  init        nPop random particles (CreateRandomSolution.m), re-drawn until one is feasible
              (the authors' isInit loop; given up after max_init_rounds, then the leg is infeasible)
  velocity    v = w v + c1 r1 (pbest - x) + c2 r2 (leader - x), clamped to +-alpha (VarMax - VarMin);
              x += v; velocity mirrored where x left its bounds; x clamped (per variable r, psi, phi)
  leader      per particle from the repository by hypergrid roulette, P(cell) ~ exp(-beta N_cell)
              (SelectLeader.m, CreateGrid.m, FindGridIndex.m)
  mutation    with probability pm = (1 - (it-1)/(MaxIt-1))^(1/mu): x + randn * pbest * tanh(delta |rep|)
              (Mutate.m), accepted if it dominates, kept out if dominated, else with probability 0.5
  pbest       replaced if dominated by the new position, else with probability 0.5 if incomparable
  repository  non-dominated members of rep + swarm, at most nRep; extra members deleted by grid
              roulette, P(cell) ~ exp(gamma N_cell) (DeleteOneRepMember.m); w *= wdamp per iteration

Parameters are the authors' MAIN/NMOPSO.m values (nPop 100, MaxIt 500, nRep 50, w 1, wdamp 0.98,
c1 = c2 = 1.5, nGrid 5, alpha 0.1, beta = gamma = 2, mu 0.5, delta 20). Two deliberate differences:
  - Mutate.m clamps with max(VarMax, x) then min(VarMin, .), which sets every mutated variable to its
    lower bound; this port clamps to [VarMin, VarMax] as the paper describes.
  - The repository holds feasible paths only (finite F1..F4): MyCost.m returns inf for infeasible
    paths, which would make the hypergrid bounds infinite.
The swarm is updated synchronously (all particles of an iteration from the same repository), which is
what NMOPSO.m does too: the repository is only updated after the particle loop.

The deployed path is the short-safe member of the final repository (rlpatrol.metrics.short_safe, the
rule plan_factory_patrol.m flies), exactly as for the RL planner. Everything else (free space, keep-in
box, A* route decomposition of blocked legs) is inherited from LegPlanner, so NMOPSO and RL differ only
in how the candidate paths of a leg are produced.
"""
import math
import time

import numpy as np
import torch

from .metrics import short_safe
from .nmopso_cost import ANGLE, N_NODES, Legs
from .planner import LegPlan, LegPlanner
from .world import load_world


class NMOPSOPlanner(LegPlanner):
    def __init__(self, max_it=500, n_pop=100, n_rep=50, w=1.0, wdamp=0.98, c1=1.5, c2=1.5, n_grid=5,
                 grid_alpha=0.1, beta=2.0, gamma=2.0, mu=0.5, delta=20.0, max_init_rounds=20,
                 device="cpu", threads=0, seed=0, free_search_radius=3.0, keep_in=None):
        self.dev = torch.device(device if device != "auto" else "cpu")
        if threads:
            torch.set_num_threads(threads)
        self.world = load_world(device=self.dev)
        self.cfg = dict(max_it=max_it, n_pop=n_pop, n_rep=n_rep, w=w, wdamp=wdamp, c1=c1, c2=c2, n_grid=n_grid,
                        grid_alpha=grid_alpha, beta=beta, gamma=gamma, mu=mu, delta=delta,
                        max_init_rounds=max_init_rounds)
        self.refine, self.seed, self.free_r = False, seed, free_search_radius
        self.keep_in = None if keep_in is None else tuple(float(v) for v in keep_in)
        self.ckpt_info = {"planner": "nmopso", **self.cfg, "keep_in": self.keep_in}

    # --- repository helpers (numpy, minimisation) ------------------------------------------------------
    @staticmethod
    def _dominates(A, B):
        """[len(A), len(B)] bool: A_i dominates B_j (Dominates.m: <= everywhere, < somewhere, A_i finite)."""
        le = (A[:, None, :] <= B[None, :, :]).all(-1)
        lt = (A[:, None, :] < B[None, :, :]).any(-1)
        return le & lt & np.isfinite(A).all(1)[:, None]

    def _nondominated(self, F):
        return ~self._dominates(F, F).any(0)

    def _grid_index(self, F):
        """CreateGrid.m + FindGridIndex.m: one cell index per row of F (finite)."""
        c = self.cfg
        lo, hi = F.min(0), F.max(0)
        d = hi - lo
        lo, hi = lo - c["grid_alpha"] * d, hi + c["grid_alpha"] * d
        sub = np.empty(F.shape, int)
        for j in range(F.shape[1]):
            ub = np.r_[np.linspace(lo[j], hi[j], c["n_grid"] + 1), np.inf]
            sub[:, j] = np.minimum(np.searchsorted(ub, F[:, j], side="right") + 1, c["n_grid"])
        idx = sub[:, 0].copy()
        for j in range(1, F.shape[1]):
            idx = (idx - 1) * c["n_grid"] + sub[:, j]
        return idx

    def _roulette_cells(self, gi, pressure, k, rng):
        """k member indices: a cell by roulette with P ~ exp(pressure * N_cell), then a random member."""
        cells, counts = np.unique(gi, return_counts=True)
        p = np.exp(pressure * (counts - counts.min() if pressure > 0 else counts))
        p /= p.sum()
        chosen = rng.choice(len(cells), size=k, p=p)
        out = np.empty(k, int)
        for i, ci in enumerate(chosen):
            members = np.flatnonzero(gi == cells[ci])
            out[i] = members[rng.integers(len(members))]
        return out

    # --- one leg -----------------------------------------------------------------------------------
    def _search(self, leg, seed):
        """NMOPSO.m on one leg; returns (F [m, 4], nodes [m, n, 3] units) of the final repository."""
        c, rng = self.cfg, np.random.default_rng(seed)
        P, n = c["n_pop"], N_NODES
        w_ = self.world
        lg = Legs(w_, np.repeat(w_.to_units(leg[None, :3]), 2 * P, 0), np.repeat(w_.to_units(leg[None, 3:]), 2 * P, 0))
        rmax = float(lg.rmax[0])                                      # lg holds 2P copies: swarm + mutants
        lo = np.array([rmax / 9, -ANGLE, -ANGLE])[:, None]            # [3, 1]: r, psi, phi
        hi = np.array([rmax, ANGLE, ANGLE])[:, None]
        vmax = 0.5 * (hi - lo)

        def evaluate(X):                                             # X [P, 3, n] -> F [P, 4], nodes
            t = torch.as_tensor(X, dtype=w_.threats.dtype, device=self.dev)
            m = len(X)
            sub = lg if m == len(lg) else Legs(w_, lg.start[:m], lg.goal[:m])
            nodes = sub.decode(t[:, 0], t[:, 1], t[:, 2])
            return sub.cost(nodes).cpu().numpy(), nodes.cpu().numpy()

        X = None
        for _ in range(c["max_init_rounds"]):                        # the authors' isInit loop
            X = lo + rng.random((P, 3, n)) * (hi - lo)
            F, nodes = evaluate(X)
            if np.isfinite(F).all(1).any():
                break
        else:
            return np.zeros((0, 4)), np.zeros((0, n, 3))
        V = np.zeros_like(X)
        PB, PBF = X.copy(), F.copy()
        feas = np.isfinite(F).all(1)
        nd = self._nondominated(F) & feas
        RX, RF, RN = X[nd], F[nd], nodes[nd]
        w = c["w"]
        for it in range(1, c["max_it"] + 1):
            gi = self._grid_index(RF)
            L = RX[self._roulette_cells(gi, -c["beta"], P, rng)]       # a leader per particle
            V = w * V + c["c1"] * rng.random(X.shape) * (PB - X) + c["c2"] * rng.random(X.shape) * (L - X)
            V = np.clip(V, -vmax, vmax)
            X = X + V
            out = (X < lo) | (X > hi)
            V[out] = -V[out]                                         # velocity mirroring
            X = np.clip(X, lo, hi)
            # mutation (Mutate.m with the bounds clamp fixed) on a pm fraction of the swarm; the mutants
            # are scored in the same batch as the moved swarm (one MyCost call per iteration)
            pm = (1 - (it - 1) / max(c["max_it"] - 1, 1)) ** (1 / c["mu"])
            mi = np.flatnonzero(rng.random(P) < pm)
            b = math.tanh(c["delta"] * len(RX))
            XM = np.clip(X[mi] + rng.standard_normal(X[mi].shape) * PB[mi] * b, lo, hi)
            FA, NA = evaluate(np.concatenate([X, XM]))
            F, nodes, FM, NM = FA[:P], NA[:P], FA[P:], NA[P:]
            if len(mi):
                new_dom = self._dominates(FM, F[mi]).diagonal()
                old_dom = self._dominates(F[mi], FM).diagonal()
                take = new_dom | (~old_dom & (rng.random(len(mi)) < 0.5))
                t = mi[take]
                X[t], F[t], nodes[t] = XM[take], FM[take], NM[take]
            # personal best
            new_dom = self._dominates(F, PBF).diagonal()
            old_dom = self._dominates(PBF, F).diagonal()
            take = new_dom | (~old_dom & (rng.random(P) < 0.5))
            PB[take], PBF[take] = X[take], F[take]
            # repository: non-dominated feasible members of rep + swarm, pruned to nRep
            feas = np.isfinite(F).all(1)
            AX, AF, AN = np.r_[RX, X[feas]], np.r_[RF, F[feas]], np.r_[RN, nodes[feas]]
            AF_u, ui = np.unique(AF, axis=0, return_index=True)       # drop exact duplicates
            AX, AF, AN = AX[ui], AF_u, AN[ui]
            keep = self._nondominated(AF)
            RX, RF, RN = AX[keep], AF[keep], AN[keep]
            while len(RX) > c["n_rep"]:
                d = self._roulette_cells(self._grid_index(RF), c["gamma"], 1, rng)[0]
                RX, RF, RN = np.delete(RX, d, 0), np.delete(RF, d, 0), np.delete(RN, d, 0)
            w *= c["wdamp"]
        return RF, RN

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
        if np.linalg.norm(g_free - s_free) < 0.3:
            P = np.array([[*s_free, z], [*g_free, z]])
            return LegPlan(True, P, [0.0] * 4, "trivial", s_free, g_free, 1, 1, time.perf_counter() - t0)
        leg = np.array([s_free[0], s_free[1], z, g_free[0], g_free[1], z])
        F, nodes = self._search(leg, seed)
        ok = self._nodes_in_box(nodes) if len(nodes) else np.zeros(0, bool)
        n_cand = len(F)
        if not ok.any():
            return LegPlan(False, nothing, [math.inf] * 4, "none", s_free, g_free, n_cand, 0,
                           time.perf_counter() - t0, reason="no feasible NMOPSO repository member")
        idx = np.flatnonzero(ok)
        j = idx[short_safe(F[idx])]
        path = np.vstack([[*s_free, z], self.world.to_metres(nodes[j]), [*g_free, z]])
        return LegPlan(True, path, F[j].tolist(), "nmopso", s_free, g_free, n_cand, int(ok.sum()),
                       time.perf_counter() - t0)
