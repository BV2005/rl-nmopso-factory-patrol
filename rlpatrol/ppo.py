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
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    ent: float = 1e-3
    vf: float = 0.5
    hidden: int = 256
    lmin: float = 1.0
    lmax: float = 15.0
    seed: int = 1
    threads: int = 0                # 0: torch default


class Actor(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.mu = nn.Sequential(nn.Linear(OBS_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                                nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 3))
        self.log_std = nn.Parameter(torch.full((3,), -0.5))

    def dist(self, o):
        return torch.distributions.Normal(self.mu(o), self.log_std.exp())


def critic(h):
    return nn.Sequential(nn.Linear(OBS_DIM, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(),
                         nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))


def rollout(actor, world, legs_m, pref, deterministic=False):
    env = PathEnv(world, world.to_units(legs_m[:, :3]), world.to_units(legs_m[:, 3:]), pref)
    O, A, LP, R = [], [], [], []
    info = None
    for _ in range(N_NODES):
        o = env.obs()
        with torch.no_grad():
            d = actor.dist(o)
            a = d.mean if deterministic else d.sample()
        r, done, info = env.step(a)
        O.append(o); A.append(a); LP.append(d.log_prob(a).sum(-1)); R.append(r)
    return torch.stack(O, 1), torch.stack(A, 1), torch.stack(LP, 1), torch.stack(R, 1), info, env


def train(world, cfg: Config, log=print, ckpt=None):
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed + 777)       # test legs use seed 12345
    actor, crit = Actor(cfg.hidden), critic(cfg.hidden)
    opt = torch.optim.Adam(list(actor.parameters()) + list(crit.parameters()), lr=cfg.lr)
    hist = []
    for it in range(cfg.iters):
        t0 = time.time()
        legs = random_legs(world, cfg.batch, rng, cfg.lmin, cfg.lmax)
        pref = sample_pref(cfg.batch)
        O, A, LP, R, info, _ = rollout(actor, world, legs, pref)
        B, T = R.shape
        with torch.no_grad():
            V = crit(O.reshape(B * T, -1)).reshape(B, T)
        adv = torch.zeros_like(R); last = torch.zeros(B)
        for t in reversed(range(T)):
            nv = V[:, t + 1] if t + 1 < T else torch.zeros(B)
            delta = R[:, t] + cfg.gamma * nv - V[:, t]
            last = delta + cfg.gamma * cfg.lam * last
            adv[:, t] = last
        ret = adv + V
        o, a, lp, ad, rt = (x.reshape(B * T, *x.shape[2:]) for x in (O, A, LP, adv, ret))
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        for _ in range(cfg.epochs):
            for idx in torch.randperm(B * T).split(cfg.minibatch):
                d = actor.dist(o[idx])
                nlp = d.log_prob(a[idx]).sum(-1)
                ratio = (nlp - lp[idx]).exp()
                pl = -torch.min(ratio * ad[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad[idx]).mean()
                vl = (crit(o[idx]).squeeze(-1) - rt[idx]).pow(2).mean()
                loss = pl + cfg.vf * vl - cfg.ent * d.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(crit.parameters()), 0.5)
                opt.step()
        feas = info["feasible"].float().mean().item()
        F = info["F"][info["feasible"]]
        row = dict(it=it, ret=R.sum(1).mean().item(), feasible=feas,
                   F=F.mean(0).tolist() if len(F) else [math.nan] * 4,
                   std=actor.log_std.exp().tolist(), sec=time.time() - t0)
        hist.append(row)
        if it % 10 == 0 or it == cfg.iters - 1:
            log(f"it {it:4d}  return {row['ret']:+.3f}  feasible {feas*100:5.1f} %  "
                f"F {np.round(row['F'], 4).tolist()}  std {np.round(row['std'], 3).tolist()}  {row['sec']:.1f} s")
        if ckpt and (it % 50 == 0 or it == cfg.iters - 1):
            torch.save({"actor": actor.state_dict(), "critic": crit.state_dict(), "cfg": asdict(cfg), "hist": hist}, ckpt)
    return actor, crit, hist
