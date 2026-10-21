"""Warm-start seeds for the RL-warm-started NMOPSO (matlab/run_nmopso_variant.m): per leg, the straight
path plus up to M - 1 feasible, non-dominated policy paths (spread over the front by F2 order), in
NMOPSO's (r, psi, phi) encoding.
  python scripts/make_warm.py --ckpt results/ckpt/v2.pt --legs real --M 50"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import conditions, rl_solve  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402


def pick(F, M):
    """indices of up to M feasible non-dominated rows, evenly spread in F2 order."""
    ok = np.where(np.isfinite(F).all(1))[0]
    if len(ok) == 0:
        return ok
    G = F[ok]
    nd = np.array([not ((G <= g).all(1) & (G < g).any(1)).any() for g in G])
    cand = ok[nd]
    cand = cand[np.unique(F[cand], axis=0, return_index=True)[1]]
    cand = cand[np.argsort(F[cand, 1])]
    if len(cand) > M:
        cand = cand[np.round(np.linspace(0, len(cand) - 1, M)).astype(int)]
    return cand


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--legs", default="real")
    ap.add_argument("--M", type=int, default=50)
    ap.add_argument("--S", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(2)
    world = load_world()
    legs_m = loadmat(DATA / f"legs_{a.legs}.mat")["legs"]
    ck = torch.load(a.ckpt, weights_only=False)
    budget = ck["cfg"].get("budget", False)
    actor = Actor(ck["cfg"]["hidden"], ctx=5 if budget else 4); actor.load_state_dict(ck["actor"]); actor.eval()
    prefs, budgets = conditions(budget, 50)
    t0 = time.perf_counter()
    sols, _ = rl_solve(actor, world, legs_m, prefs, budgets, a.S)
    straight, _ = rl_solve(None, world, legs_m, prefs[:1], None if budgets is None else budgets[:1], 0)
    K, n = len(legs_m), sols[0][1].shape[-1]
    R = np.zeros((K, a.M, n)); PS = np.zeros((K, a.M, n)); PH = np.zeros((K, a.M, n)); cnt = np.zeros(K)
    for k in range(K):
        e = [straight[k][1][0]] + [sols[k][1][i] for i in pick(sols[k][0], a.M - 1)]
        e = np.stack(e)
        R[k, :len(e)], PS[k, :len(e)], PH[k, :len(e)] = e[:, 0], e[:, 1], e[:, 2]
        cnt[k] = len(e)
    wall = time.perf_counter() - t0
    out = DATA / f"warm_{Path(a.ckpt).stem}_{a.legs}.mat"
    savemat(out, {"r": R, "psi": PS, "phi": PH, "cnt": cnt[:, None], "wall_s": wall})
    print(f"{out}: {K} legs, seeds per leg {cnt.min():.0f}-{cnt.max():.0f} (mean {cnt.mean():.1f}), {wall:.1f} s for all legs")
