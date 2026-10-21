"""Checks the PyTorch port (rlpatrol/nmopso_cost.py) against the authors' MATLAB SphericalToCart2.m
and MyCost.m on random solutions drawn as CreateRandomSolution.m does, on real and random legs."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.nmopso_cost import ANGLE, N_NODES, Legs  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

MATLAB = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"


def main(per_leg=200, seed=0):
    w = load_world()
    legs_m = np.vstack([loadmat(DATA / "legs_real.mat")["legs"][:20], loadmat(DATA / "legs_random.mat")["legs"][:20]])
    K = len(legs_m)
    lg = Legs(w, w.to_units(legs_m[:, :3]), w.to_units(legs_m[:, 3:]))
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(K).repeat_interleave(per_leg)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(len(idx), N_NODES, generator=g, dtype=torch.float64)
    r = u(lg.rmin[idx, None], lg.rmax[idx, None])
    psi, phi = u(-ANGLE, ANGLE), u(-ANGLE, ANGLE)
    sub = Legs(w, lg.start[idx], lg.goal[idx])
    xyz = sub.decode(r, psi, phi)
    F = sub.cost(xyz).numpy()
    tmp = ROOT / "results" / "verify"; tmp.mkdir(parents=True, exist_ok=True)
    savemat(tmp / "in.mat", {"legs": legs_m, "leg": (idx + 1).numpy()[:, None].astype(float),
                             "r": r.numpy(), "psi": psi.numpy(), "phi": phi.numpy()})
    subprocess.run([MATLAB, "-batch", f"cd('{ROOT / 'matlab'}'); eval_mycost('{tmp / 'in.mat'}', '{tmp / 'out.mat'}')"], check=True)
    M = loadmat(tmp / "out.mat")
    Fm, Xm = M["cost"], M["xyz"]
    fin_m, fin_p = np.isfinite(Fm).all(1), np.isfinite(F).all(1)
    both = fin_m & fin_p
    print(f"solutions: {len(F)}, feasible MATLAB {fin_m.sum()}, Python {fin_p.sum()}, feasibility agrees on {np.mean(fin_m == fin_p) * 100:.3f} %")
    print(f"node positions: max |diff| {np.abs(Xm - xyz.numpy()).max():.3e} units")
    print(f"per-objective inf pattern agrees: {(np.isinf(Fm) == np.isinf(F)).all(0)}")
    print(f"costs of feasible: max |diff| per objective {np.abs(Fm[both] - F[both]).max(0)}")
    ok = (fin_m == fin_p).all() and np.abs(Fm[both] - F[both]).max() < 1e-9 and np.abs(Xm - xyz.numpy()).max() < 1e-9
    print("VERIFIED" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
