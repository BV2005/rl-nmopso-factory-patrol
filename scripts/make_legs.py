"""Benchmark legs (world metres, rows [ax ay az bx by bz], patrol height 1.8 m):
  data/legs_real.mat   - every NMOPSO leg of the latest plan (factory2_auto_full preview: transits
                         and patrol loops of the 3 drones, consecutive patrol points)
  data/legs_random.mat - 200 held-out random legs between free points, 1.5-15 m long (seed 12345;
                         training samples its legs with other seeds)
  data/legs_blocked.mat - (python scripts/make_legs.py blocked; round B) 200 held-out random legs as above
                         (seed 24680) whose straight line is blocked (the goal is not visible from the start)
                         and whose shortest collision-free route (rlpatrol/route.py) is at most 3 x the
                         straight distance, the span of the NMOPSO encoding: legs that need a detour and
                         that the encoding can solve. Chosen without any policy."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

from rlpatrol.legs import random_legs  # noqa: E402
from rlpatrol.route import route_features, seg_clear  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

PREVIEW = DATA / "factory2_auto_full_preview.json"   # planner/factory_patrol dry run, layout factory2_auto_full


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


def blocked_legs(w, n, rng, span=3.0):
    out = np.zeros((0, 6))
    while len(out) < n:
        X = random_legs(w, 4 * n, rng)
        q, g = torch.tensor(w.to_units(X[:, :3])[:, :2]), torch.tensor(w.to_units(X[:, 3:])[:, :2])
        _, L = route_features(w, q, g)
        ok = ~seg_clear(w.threats, q, g) & (L <= span * (g - q).norm(dim=-1))
        out = np.vstack([out, X[ok.numpy()]])
    return out[:n]


if __name__ == "__main__":
    w = load_world()
    if sys.argv[1:] == ["blocked"]:
        B = blocked_legs(w, 200, np.random.default_rng(24680))
        savemat(DATA / "legs_blocked.mat", {"legs": B})
        L = np.linalg.norm(B[:, 3:5] - B[:, :2], axis=1)
        print(f"blocked legs: {len(B)}, length {L.min():.2f}-{L.max():.2f} m (median {np.median(L):.2f})")
        sys.exit()
    R = real_legs()
    X = random_legs(w, 200, np.random.default_rng(12345))
    savemat(DATA / "legs_real.mat", {"legs": R})
    savemat(DATA / "legs_random.mat", {"legs": X})
    L = np.linalg.norm(R[:, 3:5] - R[:, :2], axis=1)
    print(f"real legs: {len(R)}, length {L.min():.2f}-{L.max():.2f} m (median {np.median(L):.2f})")
    L = np.linalg.norm(X[:, 3:5] - X[:, :2], axis=1)
    print(f"random legs: {len(X)}, length {L.min():.2f}-{L.max():.2f} m (median {np.median(L):.2f})")
