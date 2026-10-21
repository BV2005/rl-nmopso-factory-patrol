"""Preference-conditioned PPO for the leg-planning MDP (rlpatrol/env.py)."""
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .env import N_NODES, OBS_DIM, PathEnv, sample_pref
from .legs import random_legs


@dataclass
class Config:
    iters: int = 300
    batch: int = 4096               # episodes per iteration (x 10 steps)
    epochs: int = 10
    minibatch: int = 8192
    lr: float = 3e-4
    lr_v: float = 1e-3
    gamma: float = 1.0
    lam: float = 1.0
    clip: float = 0.2
    target_kl: float = 0.02         # stop the epochs of an iteration beyond this approximate KL
    ent: float = 0.0
    hidden: int = 256
    log_std0: float = -1.2          # initial std of the pre-tanh Gaussian
    real_frac: float = 0.7          # share of training legs with the real legs' lengths
    lmin: float = 0.6
    lmax: float = 6.5
    lmax_long: float = 15.0
    shape: float = 0.0
    scalar: str = "linear"          # linear | tcheby
    act_pen: float = 0.0            # straight prior: -act_pen * |a|^2 per step
    budget: bool = False            # threat-budget conditioning (F2 <= u * F2 of the straight path)
    u_lo: float = 0.15
    u_hi: float = 1.2
    budget_pen: float = 5.0         # reward per F_SCALE[1] of threat above the budget
    seed: int = 1
    threads: int = 0                # 0: torch default


class PrefMLP(nn.Module):
    """MLP whose every hidden layer also sees the conditioning (the last ctx observation entries:
    the preference, and the threat budget u when on)."""

    def __init__(self, h, out, ctx=4):
        super().__init__()
        self.ctx = ctx
        self.l1 = nn.Linear(OBS_DIM + ctx - 4, h)
        self.l2 = nn.Linear(h + ctx, h)
        self.l3 = nn.Linear(h + ctx, h)
        self.out = nn.Linear(h + ctx, out)

    def forward(self, o):
        w = o[..., -self.ctx:]
        x = torch.tanh(self.l1(o))
        x = torch.tanh(self.l2(torch.cat([x, w], -1)))
        x = torch.tanh(self.l3(torch.cat([x, w], -1)))
        return self.out(torch.cat([x, w], -1))


class Actor(nn.Module):
    """tanh-squashed Gaussian: the action a = tanh(u) is always inside (-1, 1)."""

    def __init__(self, h, log_std0=-1.2, ctx=4):
        super().__init__()
        self.mu = PrefMLP(h, 3, ctx)
        nn.init.zeros_(self.mu.out.weight); nn.init.zeros_(self.mu.out.bias)   # start at the zero action
        self.log_std = nn.Parameter(torch.full((3,), log_std0))

    def dist(self, o):
        return torch.distributions.Normal(self.mu(o), self.log_std.exp())

    def act(self, o, deterministic=False):
        d = self.dist(o)
        u = d.mean if deterministic else d.sample()
        return torch.tanh(u), u, self.logp(d, u)

    @staticmethod
    def logp(d, u):
        return (d.log_prob(u) - 2 * (math.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)


def training_legs(world, n, rng, cfg):
    k = int(round(cfg.real_frac * n))
    return np.vstack([random_legs(world, k, rng, cfg.lmin, cfg.lmax),
                      random_legs(world, n - k, rng, 1.5, cfg.lmax_long)])


def sample_budget_pref(B, cfg, gen=None):
    """preference over F1, F3, F4 (w2 = 0: threat is a budget) and budget u ~ U(u_lo, u_hi)."""
    w = sample_pref(B, gen)
    w[:, 1] = 0
    w = w / w.sum(-1, keepdim=True)
    u = cfg.u_lo + (cfg.u_hi - cfg.u_lo) * torch.rand(B, generator=gen, dtype=torch.float64)
    return w, u


def rollout(actor, world, legs_m, pref, deterministic=False, shape=0.0, scalar="linear", act_pen=0.0, budget=None,
            budget_pen=5.0):
    env = PathEnv(world, world.to_units(legs_m[:, :3]), world.to_units(legs_m[:, 3:]), pref, shape, scalar, act_pen, budget,
                  budget_pen)
    O, U, LP, R = [], [], [], []
    info = None
    for _ in range(N_NODES):
        o = env.obs()
        with torch.no_grad():
            a, u, lp = actor.act(o, deterministic)
        r, done, info = env.step(a)
        O.append(o); U.append(u); LP.append(lp); R.append(r)
    return torch.stack(O, 1), torch.stack(U, 1), torch.stack(LP, 1), torch.stack(R, 1), info, env


def det_probe(actor, world, legs, cfg):
    """deterministic paths on fixed legs: random preferences, and the four corner preferences."""
    n = len(legs)
    g = torch.Generator().manual_seed(5)
    if cfg.budget:
        w, u = sample_budget_pref(n, cfg, g)
        _, _, _, _, info, _ = rollout(actor, world, legs, w, True, budget=u)
    else:
        _, _, _, _, info, _ = rollout(actor, world, legs, sample_pref(n, g), True)
    feas = info["feasible"]; F = info["F"][feas]
    out = {"det_feasible": feas.float().mean().item(), "det_F": F.mean(0).tolist() if len(F) else [math.nan] * 4}
    if cfg.budget:                       # F2 / straight F2 achieved for a tight and a loose budget
        w = torch.full((n, 4), 1 / 3, dtype=torch.float64); w[:, 1] = 0
        out["budget_F2_ratio"] = []
        for uu in (0.3, 1.0):
            _, _, _, _, info, env = rollout(actor, world, legs, w, True, budget=torch.full((n,), uu, dtype=torch.float64))
            ok = info["feasible"] & (env.f2_straight > 0)
            out["budget_F2_ratio"].append((info["F"][ok, 1] / env.f2_straight[ok]).median().item() if ok.any() else math.nan)
        return out
    corner = []
    for j in range(4):
        w = torch.zeros(n, 4, dtype=torch.float64); w[:, j] = 1
        _, _, _, _, info, _ = rollout(actor, world, legs, w, True)
        Fj = info["F"][info["feasible"]]
        corner.append(Fj[:, j].mean().item() if len(Fj) else math.nan)
    out["corner_Fj"] = corner          # F_j under w = e_j (should be the lowest F_j of any preference)
    return out


def train(world, cfg: Config, log=print, ckpt=None):
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed + 777)       # test legs use seed 12345
    probe_legs = training_legs(world, 1024, np.random.default_rng(999), cfg)
    ctx = 5 if cfg.budget else 4
    actor, crit = Actor(cfg.hidden, cfg.log_std0, ctx), PrefMLP(cfg.hidden, 1, ctx)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(crit.parameters(), lr=cfg.lr_v)
    ret_scale = None
    hist = []
    for it in range(cfg.iters):
        t0 = time.time()
        legs = training_legs(world, cfg.batch, rng, cfg)
        if cfg.budget:
            pref, bud = sample_budget_pref(cfg.batch, cfg)
        else:
            pref, bud = sample_pref(cfg.batch), None
        O, U, LP, R, info, _ = rollout(actor, world, legs, pref, shape=cfg.shape, scalar=cfg.scalar,
                                       act_pen=cfg.act_pen, budget=bud, budget_pen=cfg.budget_pen)
        B, T = R.shape
        s = R.sum(1).std().item() + 1e-6                     # return normalisation (running)
        ret_scale = s if ret_scale is None else 0.95 * ret_scale + 0.05 * s
        Rn = R / ret_scale
        with torch.no_grad():
            V = crit(O.reshape(B * T, -1)).reshape(B, T)
        adv = torch.zeros_like(Rn); last = torch.zeros(B)
        for t in reversed(range(T)):
            nv = V[:, t + 1] if t + 1 < T else torch.zeros(B)
            delta = Rn[:, t] + cfg.gamma * nv - V[:, t]
            last = delta + cfg.gamma * cfg.lam * last
            adv[:, t] = last
        ret = adv + V
        o, u, lp, ad, rt = (x.reshape(B * T, *x.shape[2:]) for x in (O, U, LP, adv, ret))
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        kl, ep = 0.0, 0
        for ep in range(cfg.epochs):
            kls = []
            for idx in torch.randperm(B * T).split(cfg.minibatch):
                d = actor.dist(o[idx])
                nlp = Actor.logp(d, u[idx])
                lr_ = nlp - lp[idx]
                ratio = lr_.exp()
                pl = -torch.min(ratio * ad[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad[idx]).mean()
                loss = pl - cfg.ent * d.entropy().sum(-1).mean()
                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
                vl = (crit(o[idx]).squeeze(-1) - rt[idx]).pow(2).mean()
                opt_v.zero_grad(); vl.backward()
                nn.utils.clip_grad_norm_(crit.parameters(), 1.0); opt_v.step()
                kls.append(((ratio - 1) - lr_).mean().item())
            kl = float(np.mean(kls))
            if kl > cfg.target_kl:
                break
        feas = info["feasible"].float().mean().item()
        row = dict(it=it, ret=R.sum(1).mean().item(), feasible=feas, kl=kl, epochs=ep + 1,
                   std=actor.log_std.exp().tolist(), sec=time.time() - t0)
        if it % 10 == 0 or it == cfg.iters - 1:
            row.update(det_probe(actor, world, probe_legs, cfg))
            log(f"it {it:4d}  return {row['ret']:+.3f}  feas {feas*100:5.1f} %  det feas {row['det_feasible']*100:5.1f} %  "
                f"det F {np.round(row['det_F'], 4).tolist()}  "
                + (f"F2/straight at u=0.3,1.0 {np.round(row['budget_F2_ratio'], 3).tolist()}  " if cfg.budget else
                   f"corner F_j {np.round(row['corner_Fj'], 4).tolist()}  ") +
                f"kl {kl:.3f} ep {ep + 1}  std {np.round(row['std'], 3).tolist()}  {row['sec']:.1f} s")
        hist.append(row)
        if ckpt and (it % 25 == 0 or it == cfg.iters - 1):
            torch.save({"actor": actor.state_dict(), "critic": crit.state_dict(), "cfg": asdict(cfg), "hist": hist}, ckpt)
    return actor, crit, hist
