"""Random legs between free points of the world (world metres, rows [ax ay az bx by bz])."""
import numpy as np
import torch


def random_legs(world, n, rng, lmin=1.5, lmax=15.0, margin=1.6):
    """n legs, both ends free (free_space.m), horizontal length uniform in [lmin, lmax] after
    rejection, at patrol height."""
    h = world.hall
    lo, hi = [h[0] + margin, h[2] + margin], [h[1] - margin, h[3] - margin]
    out = np.zeros((0, 6))
    while len(out) < n:
        k = 4 * (n - len(out)) + 64
        a = rng.uniform(lo, hi, (k, 2))
        ang = rng.uniform(0, 2 * np.pi, k)
        L = rng.uniform(lmin, lmax, k)
        b = a + L[:, None] * np.c_[np.cos(ang), np.sin(ang)]
        ok = (b >= lo).all(1) & (b <= hi).all(1)
        ok &= world.free(a[:, 0], a[:, 1]) & world.free(b[:, 0], b[:, 1])
        z = np.full((ok.sum(), 1), world.z_patrol)
        out = np.vstack([out, np.hstack([a[ok], z, b[ok], z])])
    return out[:n]


def blocked_legs(world, n, rng, span=3.0):
    """n random legs whose straight line is blocked (the goal is not visible from the start) and whose
    shortest collision-free route (rlpatrol/route.py) is at most span x the straight distance: legs that
    need a detour the NMOPSO encoding can fly (data/legs_blocked.mat, round B; training: blocked_frac)."""
    from .route import route_features, seg_clear
    dev, dt = world.threats.device, world.threats.dtype
    out = np.zeros((0, 6))
    while len(out) < n:
        X = random_legs(world, 4 * n, rng)
        q = torch.tensor(world.to_units(X[:, :3])[:, :2], dtype=dt, device=dev)
        g = torch.tensor(world.to_units(X[:, 3:])[:, :2], dtype=dt, device=dev)
        _, L = route_features(world, q, g)
        ok = ~seg_clear(world.threats, q, g) & (L <= span * (g - q).norm(dim=-1))
        out = np.vstack([out, X[ok.cpu().numpy()]])
    return out[:n]
