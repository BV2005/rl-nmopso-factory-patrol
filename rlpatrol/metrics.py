"""Pareto-front metrics (minimisation)."""
import numpy as np


def nondominated(F):
    """rows of F (finite) not dominated by another row; duplicates kept once."""
    F = np.unique(np.asarray(F, float), axis=0)
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        if keep[i]:
            dom = (F <= F[i]).all(1) & (F < F[i]).any(1)
            if dom.any():
                keep[i] = False
    return F[keep]


def hypervolume(F, ref):
    """exact hypervolume of the points F (minimisation) dominated region bounded by ref
    (HSO slicing over the first objective; fine for fronts of a few hundred points in 4-D)."""
    F = np.asarray(F, float)
    F = F[(F < ref).all(1)] if len(F) else F
    if len(F) == 0:
        return 0.0
    F = nondominated(F)
    if F.shape[1] == 1:
        return float(ref[0] - F[:, 0].min())
    F = F[np.argsort(F[:, 0])]
    v = 0.0
    for i in range(len(F)):
        hi = F[i + 1, 0] if i + 1 < len(F) else ref[0]
        if hi > F[i, 0]:
            v += (hi - F[i, 0]) * hypervolume(F[: i + 1, 1:], ref[1:])
    return v


def short_safe(F):
    """the path plan_factory_patrol.m flies under select = 'short-safe': threat F2 within 10 % of the
    front's F2 range of its minimum, then the lowest F1 (ties: F2, then F3 + F4). Returns the row index."""
    F = np.asarray(F, float)
    f2 = F[:, 1]
    cand = np.where(f2 <= f2.min() + 0.1 * (f2.max() - f2.min()) + 1e-12)[0]
    key = np.lexsort((F[cand, 2] + F[cand, 3], F[cand, 1], F[cand, 0]))
    return cand[key[0]]
