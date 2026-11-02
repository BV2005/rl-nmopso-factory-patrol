"""Preference-conditioned PPO for the leg-planning MDP (rlpatrol/env.py).

Training state is checkpointed in full (networks, optimisers, return scale, RNG states, iteration)
so an interrupted run resumes where it stopped (train(..., resume=...)); see scripts/train.py."""
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn

from .env import BUDGET_OBS, N_NODES, N_RAYS, OBS_DIM, ROUTE_OBS, SENSOR_OBS, PathEnv, sample_pref
from .sensors import SensorSpec
from .legs import blocked_legs, random_legs


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
    dense: bool = False             # pay the threat (w2) and the budget overshoot per segment (same return)
    budget_obs: bool = False        # observe the budget left and the straight path's threat to go (needs budget)
    route_obs: bool = False         # observe the shortest collision-free route (waypoint bearing, length left)
    soft_cap: bool = False          # cap the scaled cost by S_MAX * tanh(s / SOFT_SCALE), never flat (round C)
    sensor_obs: bool = False        # the actor sees the x500_patrol lidar + depth scan instead of oracle geometry
    priv_critic: bool = False       # asymmetric actor-critic: the critic also sees the oracle features (needs sensor_obs)
    sensor_noise: bool = True       # sensor_obs: simulate sensor noise, dropout and pose error (domain randomisation)
    recurrent: bool = False         # GRU actor: memory of the episode's earlier observations (RecurrentActor)
    gru_hidden: int = 128           # GRU state size (recurrent)
    blocked_frac: float = 0.0       # share of training legs whose straight line is blocked (legs.blocked_legs)
    seed: int = 1
    threads: int = 0                # 0: torch default
    device: str = "cpu"             # cpu | cuda | auto (cuda if available)
    ckpt_every: int = 10            # write the resumable checkpoint every N iterations (and at the end)
    snapshot_every: int = 0         # also keep <name>_it<N>.pt every N iterations (0: never)
    probe_every: int = 10           # deterministic probe (det feasibility, F, corner F_j) every N iterations
    max_bad_iters: int = 3          # abort after this many consecutive iterations with non-finite values


# Config fields that may change when a run is resumed (they do not change what is learned)
RESUMABLE = {"iters", "device", "threads", "ckpt_every", "snapshot_every", "probe_every", "max_bad_iters"}


def resolve_device(name):
    """'cpu' | 'cuda' | 'cuda:N' | 'auto' -> torch.device (cuda always with its index, so devices compare equal)."""
    if isinstance(name, torch.device):
        name = str(name)
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False "
                               f"(torch {torch.__version__}, built for CUDA {torch.version.cuda})")
        dev = torch.device("cuda", torch.cuda.current_device() if dev.index is None else dev.index)
    return dev


class PrefMLP(nn.Module):
    """MLP whose every hidden layer also sees the conditioning (the last ctx observation entries:
    the preference, and the threat budget u when on)."""

    def __init__(self, h, out, ctx=4, extra=0):
        super().__init__()
        self.ctx = ctx
        self.l1 = nn.Linear(OBS_DIM + extra + ctx - 4, h)
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

    def __init__(self, h, log_std0=-1.2, ctx=4, extra=0):
        super().__init__()
        self.mu = PrefMLP(h, 3, ctx, extra)
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

    # the episode interface of RecurrentActor; an MLP has no state
    recurrent = False

    def reset(self, B=None):
        pass

    def commit(self):
        pass

    def discard(self):
        pass

    def dist_seq(self, O):
        return self.dist(O)


class RecurrentActor(nn.Module):
    """tanh-squashed Gaussian like Actor, with a GRU memory over the episode's observations: the node
    decision at step t sees the current observation and a summary of steps 0..t-1 (the scans of
    obstacles already passed, where the openings were). The GRU state depends on the observations
    only, never on the sampled actions, so several actions can be drawn for one observation.

    Episode interface (every caller that steps an episode):
      reset(B)        new episodes (zero state)
      act(o) / dist(o)  the decision for observation o; calling it again with the same tensor o
                      (deterministic + sampled candidates) reuses the same step
      commit()        after the step was taken: the state moves on to include o
      discard()       no step taken (e.g. the drone hovers and scans again): forget o
      dist_seq(O)     [n, T, D] whole episodes from the zero state, with gradients (PPO update)"""
    recurrent = True

    def __init__(self, h, log_std0=-1.2, ctx=4, extra=0, gru=128):
        super().__init__()
        self.ctx = ctx
        self.enc = nn.Linear(OBS_DIM + extra + ctx - 4, h)
        self.gru = nn.GRUCell(h, gru)
        self.l2 = nn.Linear(h + gru + ctx, h)          # the current features skip past the GRU
        self.l3 = nn.Linear(h + ctx, h)
        self.out = nn.Linear(h + ctx, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)       # start at the zero action
        self.log_std = nn.Parameter(torch.full((3,), log_std0))
        self.h, self._pend = None, None

    def step_mu(self, o, h):
        w = o[..., -self.ctx:]
        x = torch.tanh(self.enc(o))
        h = self.gru(x, h)
        z = torch.tanh(self.l2(torch.cat([x, h, w], -1)))
        z = torch.tanh(self.l3(torch.cat([z, w], -1)))
        return self.out(torch.cat([z, w], -1)), h

    def zero_state(self, n, ref):
        return torch.zeros(n, self.gru.hidden_size, dtype=ref.dtype, device=ref.device)

    def dist_seq(self, O):
        h, mus = self.zero_state(len(O), O), []
        for t in range(O.shape[1]):
            mu, h = self.step_mu(O[:, t], h)
            mus.append(mu)
        return torch.distributions.Normal(torch.stack(mus, 1), self.log_std.exp())

    def reset(self, B=None):
        self.h, self._pend = None, None

    def dist(self, o):
        if self._pend is not None:
            if self._pend[0] is o:
                return self._pend[1]
            raise RuntimeError("RecurrentActor: a new observation before commit() or discard() of the last one")
        h = self.h if self.h is not None else self.zero_state(len(o), o)
        mu, h2 = self.step_mu(o, h)
        d = torch.distributions.Normal(mu, self.log_std.exp())
        self._pend = (o, d, h2)
        return d

    def act(self, o, deterministic=False):
        d = self.dist(o)
        u = d.mean if deterministic else d.sample()
        return torch.tanh(u), u, Actor.logp(d, u)

    def commit(self):
        if self._pend is not None:
            self.h = self._pend[2].detach()
            self._pend = None

    def discard(self):
        self._pend = None


def make_actor(cfg, log_std0=-1.2):
    """the actor of a Config or a checkpoint's cfg dict (recurrent: RecurrentActor, else Actor)."""
    get = _getter(cfg)
    ctx, extra = net_shape(cfg)
    if get("recurrent", False):
        return RecurrentActor(get("hidden", 256), log_std0, ctx, extra, get("gru_hidden", 128))
    return Actor(get("hidden", 256), log_std0, ctx, extra)


def _getter(cfg):
    return cfg.get if isinstance(cfg, dict) else lambda k, d=None: getattr(cfg, k, d)


def net_shape(cfg):
    """(ctx, extra) of a Config or a checkpoint's cfg dict: conditioning entries (preference, + u with
    budget conditioning) and extra observation entries of the actor (budget_obs, route_obs, sensor_obs)."""
    get = _getter(cfg)
    return ((5 if get("budget", False) else 4),
            (BUDGET_OBS if get("budget_obs", False) else 0) + (ROUTE_OBS if get("route_obs", False) else 0)
            + (SENSOR_OBS if get("sensor_obs", False) else 0))


def critic_extra(cfg):
    """extra observation entries of the critic: the actor's, plus with priv_critic the privileged
    oracle features of PathEnv.obs_pair (exact rays, route features, budget state with budget)."""
    get = _getter(cfg)
    _, extra = net_shape(cfg)
    if get("priv_critic", False):
        extra += N_RAYS + ROUTE_OBS + (BUDGET_OBS if get("budget", False) else 0)
    return extra


def env_kwargs(cfg, noise=None, critic=True):
    """the PathEnv observation options of a Config or a checkpoint's cfg dict. noise overrides
    sensor_noise (e.g. False: ideal sensors at evaluation); critic=False leaves out the privileged
    critic observation (inference needs the actor's only)."""
    get = _getter(cfg)
    noise = get("sensor_noise", True) if noise is None else noise
    return dict(budget_obs=get("budget_obs", False), route_obs=get("route_obs", False),
                sensor_obs=get("sensor_obs", False), priv=bool(critic and get("priv_critic", False)),
                sensor_spec=SensorSpec(noise=bool(noise)))


def check_config(cfg):
    if cfg.sensor_obs and (cfg.route_obs or cfg.budget_obs):
        raise ValueError("sensor_obs: route_obs / budget_obs are oracle features; use priv_critic to give them "
                         "to the critic only")
    if cfg.priv_critic and not cfg.sensor_obs:
        raise ValueError("priv_critic needs sensor_obs")


def training_legs(world, n, rng, cfg):
    nb = int(round(getattr(cfg, "blocked_frac", 0.0) * n))       # blocked legs (detours), off by default
    k = int(round(cfg.real_frac * (n - nb)))
    parts = [random_legs(world, k, rng, cfg.lmin, cfg.lmax), random_legs(world, n - nb - k, rng, 1.5, cfg.lmax_long)]
    if nb:
        parts.append(blocked_legs(world, nb, rng))
    return np.vstack(parts)


def sample_budget_pref(B, cfg, gen=None):
    """preference over F1, F3, F4 (w2 = 0: threat is a budget) and budget u ~ U(u_lo, u_hi)."""
    w = sample_pref(B, gen)
    w[:, 1] = 0
    w = w / w.sum(-1, keepdim=True)
    u = cfg.u_lo + (cfg.u_hi - cfg.u_lo) * torch.rand(B, generator=gen, dtype=torch.float64)
    return w, u


def rollout(actor, world, legs_m, pref, deterministic=False, shape=0.0, scalar="linear", act_pen=0.0, budget=None,
            budget_pen=5.0, dense=False, budget_obs=False, route_obs=False, soft_cap=False, sensor_obs=False,
            priv=False, sensor_spec=None):
    """one batch of episodes -> (O, U, LP, R [B, T, ...], terminal info, env); the critic's
    observations [B, T, ...] are env.OC (the same tensor as O unless priv)."""
    env = PathEnv(world, world.to_units(legs_m[:, :3]), world.to_units(legs_m[:, 3:]), pref, shape, scalar, act_pen, budget,
                  budget_pen, dense, budget_obs, route_obs, soft_cap, sensor_obs=sensor_obs, priv=priv,
                  sensor_spec=sensor_spec)
    O, OC, U, LP, R = [], [], [], [], []
    info = None
    actor.reset(len(legs_m))
    for _ in range(N_NODES):
        o, oc = env.obs_pair()
        with torch.no_grad():
            a, u, lp = actor.act(o, deterministic)
        actor.commit()
        r, done, info = env.step(a)
        O.append(o); OC.append(oc); U.append(u); LP.append(lp); R.append(r)
    O = torch.stack(O, 1)
    env.OC = torch.stack(OC, 1) if priv else O
    return O, torch.stack(U, 1), torch.stack(LP, 1), torch.stack(R, 1), info, env


def det_probe(actor, world, legs, cfg):
    """deterministic paths on fixed legs: random preferences, and the four corner preferences."""
    n = len(legs)
    g = torch.Generator().manual_seed(5)
    kw = env_kwargs(cfg)
    kw_nb = {**kw, "budget_obs": False}                 # the rollouts without budget conditioning
    if cfg.budget:
        w, u = sample_budget_pref(n, cfg, g)
        _, _, _, _, info, _ = rollout(actor, world, legs, w, True, budget=u, **kw)
    else:
        _, _, _, _, info, _ = rollout(actor, world, legs, sample_pref(n, g), True, **kw_nb)
    feas = info["feasible"]; F = info["F"][feas]
    out = {"det_feasible": feas.float().mean().item(), "det_F": F.mean(0).tolist() if len(F) else [math.nan] * 4}
    if cfg.budget:                       # F2 / straight F2 achieved for a tight and a loose budget
        w = torch.full((n, 4), 1 / 3, dtype=torch.float64); w[:, 1] = 0
        out["budget_F2_ratio"] = []
        for uu in (0.3, 1.0):
            _, _, _, _, info, env = rollout(actor, world, legs, w, True, budget=torch.full((n,), uu, dtype=torch.float64),
                                            **kw)
            ok = info["feasible"] & (env.f2_straight > 0)
            out["budget_F2_ratio"].append((info["F"][ok, 1] / env.f2_straight[ok]).median().item() if ok.any() else math.nan)
        return out
    corner = []
    for j in range(4):
        w = torch.zeros(n, 4, dtype=torch.float64); w[:, j] = 1
        _, _, _, _, info, _ = rollout(actor, world, legs, w, True, **kw_nb)
        Fj = info["F"][info["feasible"]]
        corner.append(Fj[:, j].mean().item() if len(Fj) else math.nan)
    out["corner_Fj"] = corner          # F_j under w = e_j (should be the lowest F_j of any preference)
    return out


def _rng_state(rng, dev):
    st = {"np": rng.bit_generator.state, "torch": torch.get_rng_state()}
    if dev.type == "cuda":
        st["cuda"] = torch.cuda.get_rng_state(dev)
    return st


def _set_rng_state(rng, st, dev):
    rng.bit_generator.state = st["np"]
    torch.set_rng_state(st["torch"].cpu() if hasattr(st["torch"], "cpu") else st["torch"])
    if dev.type == "cuda" and "cuda" in st:
        torch.cuda.set_rng_state(st["cuda"].cpu(), dev)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def save_checkpoint(path, state):
    """atomic: write <path>.tmp then rename, so an interruption never leaves a truncated checkpoint."""
    path = str(path)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def _finite(*xs):
    return all(bool(torch.isfinite(x).all()) for x in xs)


def train(world, cfg: Config, log=print, ckpt=None, resume=None, metrics=None, snapshot=None):
    """PPO on the leg-planning MDP.
    ckpt      path of the resumable checkpoint (rewritten every cfg.ckpt_every iterations and at the end)
    resume    a checkpoint dict from load_checkpoint() to continue from (its cfg must match cfg except
              for the RESUMABLE fields)
    metrics   path of a JSONL file; one row per iteration is appended
    snapshot  format string with {it}, e.g. 'results/ckpt/v5_it{it:04d}.pt' (cfg.snapshot_every)
    The world must already be on cfg's device (load_world(device=...))."""
    check_config(cfg)
    dev = resolve_device(cfg.device)
    if world.threats.device != dev:
        raise ValueError(f"world is on {world.threats.device}, cfg.device resolves to {dev}")
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 777)       # test legs use seed 12345
    probe_legs = training_legs(world, 1024, np.random.default_rng(999), cfg)
    ctx, extra = net_shape(cfg)
    actor = make_actor(cfg, cfg.log_std0).to(dev)
    crit = PrefMLP(cfg.hidden, 1, ctx, critic_extra(cfg)).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(crit.parameters(), lr=cfg.lr_v)
    ret_scale = None
    hist = []
    it0 = 0
    if resume is not None:
        if "opt_a" not in resume:
            raise ValueError("checkpoint has no optimiser state (written before resumable checkpoints); "
                             "it can be evaluated but not resumed")
        old = resume["cfg"]
        dflt = {f.name: f.default for f in fields(Config)}           # fields added after the checkpoint was written
        diff = {k: (old.get(k, dflt[k]), v) for k, v in asdict(cfg).items()
                if k not in RESUMABLE and old.get(k, dflt[k]) != v}
        if diff:
            raise ValueError(f"config differs from the checkpoint's in non-resumable fields: {diff}")
        actor.load_state_dict(resume["actor"]); crit.load_state_dict(resume["critic"])
        opt_a.load_state_dict(resume["opt_a"]); opt_v.load_state_dict(resume["opt_v"])
        ret_scale, hist, it0 = resume["ret_scale"], list(resume["hist"]), resume["it"] + 1
        _set_rng_state(rng, resume["rng"], dev)
        log(f"resumed at iteration {it0} of {cfg.iters} (checkpoint iteration {resume['it']}, device {dev})")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    mf = open(metrics, "a", buffering=1) if metrics else None

    def state(it):
        return {"format": 2, "actor": actor.state_dict(), "critic": crit.state_dict(),
                "opt_a": opt_a.state_dict(), "opt_v": opt_v.state_dict(), "cfg": asdict(cfg), "hist": hist,
                "it": it, "ret_scale": ret_scale, "rng": _rng_state(rng, dev), "torch": torch.__version__}

    bad = 0
    it = it0 - 1
    for it in range(it0, cfg.iters):
        t0 = time.time()
        legs = training_legs(world, cfg.batch, rng, cfg)
        if cfg.budget:
            pref, bud = sample_budget_pref(cfg.batch, cfg)
        else:
            pref, bud = sample_pref(cfg.batch), None
        O, U, LP, R, info, renv = rollout(actor, world, legs, pref, shape=cfg.shape, scalar=cfg.scalar,
                                          act_pen=cfg.act_pen, budget=bud, budget_pen=cfg.budget_pen,
                                          dense=cfg.dense, soft_cap=cfg.soft_cap, **env_kwargs(cfg))
        OC = renv.OC
        if not _finite(O, OC, U, LP, R):
            bad += 1
            log(f"it {it:4d}  non-finite observation/action/reward in the rollout: update skipped ({bad} in a row)")
            if bad >= cfg.max_bad_iters:
                raise FloatingPointError(f"{bad} consecutive iterations with non-finite rollouts; last good "
                                         f"checkpoint: {ckpt}")
            continue
        B, T = R.shape
        s = R.sum(1).std().item() + 1e-6                     # return normalisation (running)
        ret_scale = s if ret_scale is None else 0.95 * ret_scale + 0.05 * s
        Rn = R / ret_scale
        with torch.no_grad():
            V = crit(OC.reshape(B * T, -1)).reshape(B, T)
        adv = torch.zeros_like(Rn); last = torch.zeros(B, device=dev)
        for t in reversed(range(T)):
            nv = V[:, t + 1] if t + 1 < T else torch.zeros(B, device=dev)
            delta = Rn[:, t] + cfg.gamma * nv - V[:, t]
            last = delta + cfg.gamma * cfg.lam * last
            adv[:, t] = last
        ret = adv + V
        o, oc, u, lp, ad, rt = (x.reshape(B * T, *x.shape[2:]) for x in (O, OC, U, LP, adv, ret))
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        snap_a = {k: v.clone() for k, v in actor.state_dict().items()}
        snap_v = {k: v.clone() for k, v in crit.state_dict().items()}
        kl, ep, skipped = 0.0, 0, 0
        pls, vls, ents, cfs = [], [], [], []
        if actor.recurrent:                                  # whole episodes: the GRU needs the sequence
            o, u, lp, ad = (x.reshape(B, T, *x.shape[1:]) for x in (o, u, lp, ad))
            oc, rt = oc.reshape(B, T, -1), rt.reshape(B, T)
            n_items, mb = B, max(1, cfg.minibatch // T)
        else:
            n_items, mb = B * T, cfg.minibatch
        for ep in range(cfg.epochs):
            kls = []
            for idx in torch.randperm(n_items, device=dev).split(mb):
                d = actor.dist_seq(o[idx])
                nlp = Actor.logp(d, u[idx])
                lr_ = nlp - lp[idx]
                ratio = lr_.exp()
                pl = -torch.min(ratio * ad[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad[idx]).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl - cfg.ent * ent
                vl = (crit(oc[idx]).squeeze(-1) - rt[idx]).pow(2).mean()
                if not _finite(loss, vl):
                    skipped += 1
                    continue
                opt_a.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
                opt_v.zero_grad(); vl.backward()
                nn.utils.clip_grad_norm_(crit.parameters(), 1.0); opt_v.step()
                kls.append(((ratio - 1) - lr_).mean().item())
                pls.append(pl.item()); vls.append(vl.item()); ents.append(ent.item())
                cfs.append(((ratio - 1).abs() > cfg.clip).float().mean().item())
            kl = float(np.mean(kls)) if kls else float("nan")
            if not kls or kl > cfg.target_kl:
                break
        params_ok = _finite(*actor.state_dict().values(), *crit.state_dict().values())
        if skipped or not params_ok:
            if not params_ok:                                # never keep (or checkpoint) broken weights
                actor.load_state_dict(snap_a); crit.load_state_dict(snap_v)
            bad += 1
            log(f"it {it:4d}  {skipped} minibatch(es) with non-finite loss"
                + ("" if params_ok else ", non-finite weights: iteration rolled back") + f" ({bad} in a row)")
            if bad >= cfg.max_bad_iters:
                raise FloatingPointError(f"{bad} consecutive iterations with non-finite losses; last good "
                                         f"checkpoint: {ckpt}")
        else:
            bad = 0
        feas = info["feasible"].float().mean().item()
        Ff = info["F"][info["feasible"]]
        row = dict(it=it, ret=R.sum(1).mean().item(), ret_std=R.sum(1).std().item(), feasible=feas,
                   F_feasible=Ff.mean(0).tolist() if len(Ff) else [math.nan] * 4,
                   ep_len=T, env_steps=(it + 1) * B * T,
                   policy_loss=float(np.mean(pls)) if pls else math.nan,
                   value_loss=float(np.mean(vls)) if vls else math.nan,
                   entropy=float(np.mean(ents)) if ents else math.nan,
                   clip_frac=float(np.mean(cfs)) if cfs else math.nan,
                   kl=kl, epochs=ep + 1, skipped_minibatches=skipped, ret_scale=ret_scale,
                   std=actor.log_std.exp().tolist(), sec=time.time() - t0)
        if dev.type == "cuda":
            row["gpu_mem_peak_mb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 20
        if it % cfg.probe_every == 0 or it == cfg.iters - 1:
            row.update(det_probe(actor, world, probe_legs, cfg))
            log(f"it {it:4d}  return {row['ret']:+.3f}  feas {feas*100:5.1f} %  det feas {row['det_feasible']*100:5.1f} %  "
                f"det F {np.round(row['det_F'], 4).tolist()}  "
                + (f"F2/straight at u=0.3,1.0 {np.round(row['budget_F2_ratio'], 3).tolist()}  " if cfg.budget else
                   f"corner F_j {np.round(row['corner_Fj'], 4).tolist()}  ") +
                f"kl {kl:.3f} ep {ep + 1}  std {np.round(row['std'], 3).tolist()}  "
                f"pi {row['policy_loss']:+.4f} v {row['value_loss']:.4f}  {row['sec']:.1f} s"
                + (f"  gpu {row['gpu_mem_peak_mb']:.0f} MB" if "gpu_mem_peak_mb" in row else ""))
        hist.append(row)
        if mf:
            mf.write(json.dumps(row, allow_nan=True) + "\n")
        if ckpt and (it % cfg.ckpt_every == 0 or it == cfg.iters - 1):
            save_checkpoint(ckpt, state(it))
        if snapshot and cfg.snapshot_every and it > 0 and it % cfg.snapshot_every == 0:
            save_checkpoint(snapshot.format(it=it), state(it))
    if ckpt and it >= it0:
        save_checkpoint(ckpt, state(it))
    if mf:
        mf.close()
    return actor, crit, hist
