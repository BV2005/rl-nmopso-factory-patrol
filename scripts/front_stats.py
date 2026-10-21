"""Ranges of NMOPSO's objectives on the stored Pareto fronts of the factory2_auto plan (sets F_SCALE)."""
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

d = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\bhara\Downloads\NMOPSO_repro\factory_patrol\out_factory2_auto\legs")
fr, ch, tm = [], [], []
for f in d.glob("*.mat"):
    r = loadmat(f, squeeze_me=True, struct_as_record=False)["r"]
    front = np.atleast_2d(r.front)
    if front.size == 0:
        continue
    fr.append(front.T); ch.append(np.ravel(r.F)); tm.append(float(r.time))
fr, ch = np.vstack(fr), np.vstack(ch)
print(f"{len(ch)} legs, {len(fr)} front points, NMOPSO time per leg: median {np.median(tm):.0f} s, max {max(tm):.0f} s")
for name, X in (("front", fr), ("chosen", ch)):
    print(name, "p5 ", np.percentile(X, 5, 0).round(4), "\n      p50", np.percentile(X, 50, 0).round(4),
          "\n      p95", np.percentile(X, 95, 0).round(4))
