"""Random legs between free points of the world (world metres, rows [ax ay az bx by bz])."""
import numpy as np


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
