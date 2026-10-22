"""Short local refinement of policy paths: batched cross-entropy method (CEM) in NMOPSO's own
(r, psi, phi) encoding and bounds, scored with the MyCost port. Used as the optional RL+LS row."""
import torch

from .nmopso_cost import ANGLE, Legs

# weightings of the starts (on F / F_SCALE): low threat, balanced, short / smooth
WEIGHTS = torch.tensor([[0.02, 0.94, 0.02, 0.02], [0.3, 0.4, 0.15, 0.15], [0.4, 0.05, 0.25, 0.3]], dtype=torch.float64)


def cem(world, legs_m, starts, F_SCALE, pop=64, iters=3, elite=8, seed=0):
    """legs_m [L, 6] metres; starts [L, J, 3, n] encodings (J = len(WEIGHTS)), start j refined for
    WEIGHTS[j]. Returns all evaluated (F [L, J*pop*iters, 4], enc [L, J*pop*iters, 3, n])."""
    L, J, _, n = starts.shape
    lg = Legs(world, world.to_units(legs_m[:, :3]), world.to_units(legs_m[:, 3:]))
    dev = lg.start.device
    g = torch.Generator(device=dev).manual_seed(seed)
    F_SCALE = F_SCALE.to(lg.start)
    rmin = lg.rmin[:, None, None].expand(L, J, n); rmax = lg.rmax[:, None, None].expand(L, J, n)
    lo = torch.stack([rmin, torch.full_like(rmin, -ANGLE), torch.full_like(rmin, -ANGLE)], 2)   # [L, J, 3, n]
    hi = torch.stack([rmax, torch.full_like(rmax, ANGLE), torch.full_like(rmax, ANGLE)], 2)
    mu = torch.as_tensor(starts, dtype=torch.float64, device=dev).clone()
    sd = 0.1 * (hi - lo)
    big = Legs(world, lg.start.repeat_interleave(J * pop, 0), lg.goal.repeat_interleave(J * pop, 0))
    w = WEIGHTS.to(dev)[None, :, None, :].expand(L, J, pop, 4)
    allF, allE = [], []
    for _ in range(iters):
        x = mu[:, :, None] + sd[:, :, None] * torch.randn(L, J, pop, 3, n, generator=g, dtype=torch.float64, device=dev)
        x = torch.minimum(torch.maximum(x, lo[:, :, None]), hi[:, :, None])
        x[..., 0, :] = torch.maximum(x[..., 0, :], rmin[:, :, None] * 1.001)
        flat = x.reshape(-1, 3, n)
        F, viol, _ = big.cost(big.decode(flat[:, 0], flat[:, 1], flat[:, 2]), details=True)
        F = F.reshape(L, J, pop, 4)
        s = (w * (F.nan_to_num(posinf=0) / F_SCALE)).sum(-1)
        s = torch.where(torch.isfinite(F).all(-1), s, 100 + viol.reshape(L, J, pop))
        idx = s.argsort(-1)[..., :elite]
        el = torch.gather(x, 2, idx[..., None, None].expand(L, J, elite, 3, n))
        mu, sd = el.mean(2), el.std(2).clamp_min(1e-4)
        allF.append(F.reshape(L, -1, 4)); allE.append(x.reshape(L, -1, 3, n))
    return torch.cat(allF, 1).cpu().numpy(), torch.cat(allE, 1).cpu().numpy()
