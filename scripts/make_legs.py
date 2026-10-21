"""Benchmark legs (world metres, rows [ax ay az bx by bz], patrol height 1.8 m):
  data/legs_real.mat   - every NMOPSO leg of the latest plan (factory2_auto_full preview: transits
                         and patrol loops of the 3 drones, consecutive patrol points)
  data/legs_random.mat - 200 held-out random legs between free points, 1.5-15 m long (seed 12345;
                         training samples its legs with other seeds)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rlpatrol.legs import random_legs  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

PREVIEW = Path(r"C:\Users\bhara\Downloads\NMOPSO_repro\factory_patrol\out_factory2_auto_full\preview_paths.json")


def real_legs():
    J = json.loads(PREVIEW.read_text())
    legs = []
    for d in J["drones"]:
        for part in ("transit_in", "patrol_points", "transit_out"):
            P = np.asarray(d[part], float)
            for a, b in zip(P[:-1], P[1:]):
                if np.linalg.norm(a[:2] - b[:2]) >= 0.05:           # vertical moves are not NMOPSO legs
                    legs.append(np.r_[a, b])
    return np.array(legs)


if __name__ == "__main__":
    w = load_world()
    R = real_legs()
    X = random_legs(w, 200, np.random.default_rng(12345))
    savemat(DATA / "legs_real.mat", {"legs": R})
    savemat(DATA / "legs_random.mat", {"legs": X})
    L = np.linalg.norm(R[:, 3:5] - R[:, :2], axis=1)
    print(f"real legs: {len(R)}, length {L.min():.2f}-{L.max():.2f} m (median {np.median(L):.2f})")
    L = np.linalg.norm(X[:, 3:5] - X[:, :2], axis=1)
    print(f"random legs: {len(X)}, length {L.min():.2f}-{L.max():.2f} m (median {np.median(L):.2f})")
