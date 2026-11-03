"""PPO for the generalizable policy vG (rlpatrol/env_g.py): hybrid actions, privileged critic, procedurally
generated worlds with Prioritized Level Replay (Jiang et al. 2021), domain randomisation.

A level is (world id in the training pool, legs seed): its episodes are legs sampled in that world (with
probability p_extra an unexpected box on the leg). Without PLR every iteration draws fresh levels
uniformly; with PLR a buffer keeps levels scored by the mean |GAE advantage| of their last rollout and
replays them by rank (temperature beta) mixed with staleness (rho), with p_new fresh levels.

Phases (configs/vG_p1..p4.yaml): P1 advance only, uniform levels, sensor noise; P2 + climb / backtrack;
P3 + PLR; P4 + full domain randomisation (z_max_local, tracking error, sensor mount)."""
import json
import math
import os
import pickle
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .env_g import ADVANCE, BACK, C_BACK, C_CLIMB, DR, MAX_BACK, OBS_BASE, PRIV, T_DEC, GenEnv, sample_conditions
from .scan3d import walls
from .worldgen import TRAIN, cylinders, generate, unexpected_box

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ConfigG:
    iters: int = 2000
    batch: int = 8192
    epochs: int = 10
    minibatch: int = 8192
    lr: float = 3e-4
    lr_v: float = 1e-3
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.02
    ent: float = 0.003
    hidden: int = 256
    log_std0: float = -1.2
    budget: bool = True
    u_lo: float = 0.15
    u_hi: float = 1.2
    budget_pen: float = 50.0
    act_pen: float = 0.02
    modes: str = "all"                 # advance | all (advance, climb, backtrack)
    plr: bool = False
    p_new: float = 0.3
    plr_capacity: int = 4000
    plr_beta: float = 0.1
    plr_rho: float = 0.3
    levels_per_iter: int = 64
    p_extra: float = 0.3               # share of episodes with an unexpected box on the leg
    dr: int = 0                        # 0: sensor noise only; 1: full domain randomisation
    pool: str = "data/worlds/train.pkl"
    pool_size: int = 2000              # generated if the pool file is missing
    lmin: float = 1.5
    lmax: float = 12.0
    seed: int = 1
    threads: int = 0
    device: str = "cpu"
    ckpt_every: int = 10
    snapshot_every: int = 0
    probe_every: int = 20
    # round 2 (defaults = round 1)
    w_len: float = 1.0                 # weights of F1 path length, F3 altitude, F4 smoothness in the terminal cost
    w_alt: float = 1.0
    w_smooth: float = 1.0
    c_back: float = C_BACK
    c_climb: float = C_CLIMB
    max_back: int = MAX_BACK
    c_prox: float = 0.0                # per-step proximity penalty (fewer crashes)
    d_safe: float = 0.5                # m
    noise_decay: bool = False          # anneal the exploration: log_std cap -> log_std_end, entropy bonus -> 0
    log_std_end: float = -2.2


RESUMABLE = {"iters", "device", "threads", "ckpt_every", "snapshot_every", "probe_every"}


# --- networks ---------------------------------------------------------------------------------------------
class CtxMLP(nn.Module):
    """3 tanh layers; every layer also sees the conditioning (the last ctx inputs), as ppo.PrefMLP."""

    def __init__(self, n_in, h, out, ctx):
        super().__init__()
        self.ctx = ctx
        self.l1, self.l2, self.l3 = nn.Linear(n_in, h), nn.Linear(h + ctx, h), nn.Linear(h + ctx, h)
        self.out = nn.Linear(h + ctx, out)

    def forward(self, o):
        w = o[..., -self.ctx:]
        x = torch.tanh(self.l1(o))
        x = torch.tanh(self.l2(torch.cat([x, w], -1)))
        x = torch.tanh(self.l3(torch.cat([x, w], -1)))
        return self.out(torch.cat([x, w], -1))


class ActorG(nn.Module):
    """mode ~ Categorical(masked logits), step u ~ Normal(mu, std), a = tanh(u); the step's log-probability
    counts for advance and climb only (a backtrack has no step)."""

    def __init__(self, n_obs, ctx, h=256, log_std0=-1.2):
        super().__init__()
        self.body = CtxMLP(n_obs, h, 6, ctx)
        nn.init.zeros_(self.body.out.weight)
        with torch.no_grad():
            self.body.out.bias.zero_()
            self.body.out.bias[ADVANCE] = 2.0                  # start mostly advancing, straight at the goal
        self.log_std = nn.Parameter(torch.full((3,), log_std0))

    def dist(self, o, legal):
        z = self.body(o)
        logits = z[..., :3].masked_fill(~legal, -1e9)
        return torch.distributions.Categorical(logits=logits), torch.distributions.Normal(z[..., 3:], self.log_std.exp())

    @staticmethod
    def logp(dm, dc, mode, u):
        lc = (dc.log_prob(u) - 2 * (math.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)
        return dm.log_prob(mode) + torch.where(mode != BACK, lc, torch.zeros_like(lc))

    def act(self, o, legal, deterministic=False):
        dm, dc = self.dist(o, legal)
        mode = dm.probs.argmax(-1) if deterministic else dm.sample()
        u = dc.mean if deterministic else dc.sample()
        return mode, torch.tanh(u), u, self.logp(dm, dc, mode, u)


def ctx_dim(cfg):
    return 5 if _get(cfg, "budget") else 4


def _get(cfg, k, d=None):
    return cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d)


def make_actor_g(cfg):
    c = ctx_dim(cfg)
    return ActorG(OBS_BASE + c, c, _get(cfg, "hidden", 256), _get(cfg, "log_std0", -1.2))


def make_critic_g(cfg):
    c = ctx_dim(cfg)
    return CtxMLP(OBS_BASE + PRIV + c, _get(cfg, "hidden", 256), 1, c)


# --- worlds and levels ----------------------------------------------------------------------------------------
def load_pool(path, size, params=TRAIN, seed0=0, log=print):
    p = ROOT / path
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    log(f"world pool {p} missing: generating {size} worlds (scripts/make_worlds.py caches them)")
    pool = [generate(seed0 + i, params) for i in range(size)]
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(pool, f)
    return pool


def level_episodes(world, n, legs_seed, p_extra, lmin, lmax):
    """the legs and unexpected boxes of a level (deterministic in legs_seed)."""
    rng = np.random.default_rng(legs_seed)
    legs = world.sample_legs(n, rng, lmin, lmax)
    extra = np.full((n, 5), np.nan)
    L = np.linalg.norm(legs[:, 3:5] - legs[:, :2], axis=1)
    put = (rng.random(n) < p_extra) & (L >= 3.0)
    for i in np.where(put)[0]:
        extra[i] = unexpected_box(legs[i], rng)
    return legs, extra


def batch_from_levels(levels, worlds, per_level, cfg, device, dtype=torch.float64):
    """episodes of the levels -> padded batch tensors (as env_g.build_batch, vectorised by level)."""
    Lw = [worlds[lv["world"]] for lv in levels]
    K = max(len(w.boxes) for w in Lw) + 1
    M = max(len(w.cyl) for w in Lw) + 6
    nL = len(levels)
    boxes = np.zeros((nL, K, 5)); kmask = np.zeros((nL, K), bool)
    cyl = np.zeros((nL, M, 4)); cmask = np.zeros((nL, M), bool)
    halls = np.zeros((nL, 4))
    legs, extras = [], []
    for i, (lv, w) in enumerate(zip(levels, Lw)):
        boxes[i, :len(w.boxes)], kmask[i, :len(w.boxes)] = w.boxes, True
        cyl[i, :len(w.cyl)], cmask[i, :len(w.cyl)] = w.cyl, True
        halls[i] = w.hall
        lg, ex = level_episodes(w, per_level, lv["legs_seed"], cfg.p_extra, cfg.lmin, cfg.lmax)
        legs.append(lg); extras.append(ex)
    ep = np.repeat(np.arange(nL), per_level)
    legs, extras = np.vstack(legs), np.vstack(extras)
    B = len(ep)
    bx, km = boxes[ep], kmask[ep]
    cy, cm = cyl[ep], cmask[ep]
    has = ~np.isnan(extras[:, 0])
    bx[has, K - 1], km[has, K - 1] = extras[has], True
    nl = np.array([len(w.cyl) for w in Lw])[ep]
    for i in np.where(has)[0]:
        c = cylinders(extras[i][None])[:6]
        cy[i, nl[i]:nl[i] + len(c)], cm[i, nl[i]:nl[i] + len(c)] = c, True
    t = lambda a, d=dtype: torch.tensor(a, dtype=d, device=device)  # noqa: E731
    hall = t(halls[ep])
    return dict(hall=hall, boxes=torch.cat([t(bx), walls(hall)], 1),
                kmask=torch.cat([t(km, torch.bool), torch.ones(B, 4, dtype=torch.bool, device=device)], 1),
                cyl=t(cy), cmask=t(cm, torch.bool), legs=t(legs), level=torch.tensor(ep, device=device))


class PLR:
    """Prioritized Level Replay buffer of (world, legs_seed) levels."""

    def __init__(self, n_worlds, capacity, p_new, beta, rho, rng):
        self.n_worlds, self.capacity, self.p_new, self.beta, self.rho = n_worlds, capacity, p_new, beta, rho
        self.rng = rng
        self.levels = []                                    # dicts: world, legs_seed, score, last

    def _new(self):
        return {"world": int(self.rng.integers(self.n_worlds)), "legs_seed": int(self.rng.integers(2 ** 62)),
                "score": 0.0, "last": -1}

    def sample(self, n, it, enabled=True):
        if not enabled or len(self.levels) < n:
            return [self._new() for _ in range(n)]
        out = []
        s = np.array([lv["score"] for lv in self.levels])
        rank = np.empty(len(s)); rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
        ps = (1 / rank) ** (1 / self.beta); ps /= ps.sum()
        st = np.array([it - lv["last"] for lv in self.levels], float); st /= max(st.sum(), 1e-12)
        P = (1 - self.rho) * ps + self.rho * st
        P /= P.sum()
        for _ in range(n):
            if self.rng.random() < self.p_new:
                out.append(self._new())
            else:
                out.append(self.levels[int(self.rng.choice(len(self.levels), p=P))])
        return out

    def update(self, levels, scores, it):
        for lv, sc in zip(levels, scores):
            lv["score"], lv["last"] = float(sc), it
            if not any(lv is x for x in self.levels):
                self.levels.append(lv)
        if len(self.levels) > self.capacity:
            self.levels.sort(key=lambda x: -x["score"])
            del self.levels[self.capacity:]

    def state(self):
        return {"levels": self.levels, "rng": self.rng.bit_generator.state}

    def load(self, st):
        self.levels = st["levels"]
        self.rng.bit_generator.state = st["rng"]


# --- rollouts -----------------------------------------------------------------------------------------------------
def env_options(cfg):
    """the GenEnv reward options of a ConfigG or a checkpoint's cfg dict (round-1 checkpoints: defaults)."""
    g = lambda k, d: _get(cfg, k, d)  # noqa: E731
    return dict(modes=g("modes", "all"), budget_pen=g("budget_pen", 50.0), act_pen=g("act_pen", 0.02),
                wq=(g("w_len", 1.0), 1.0, g("w_alt", 1.0), g("w_smooth", 1.0)), c_back=g("c_back", C_BACK),
                c_climb=g("c_climb", C_CLIMB), c_prox=g("c_prox", 0.0), d_safe=g("d_safe", 0.5),
                max_back=g("max_back", MAX_BACK))


def rollout_g(actor, batch, pref, budget, cfg, deterministic=False, priv=True, generator=None, domain=None):
    """domain: override the training domain randomisation (evaluation: env_g.EVAL)."""
    env = GenEnv(batch, pref, budget, domain=domain or DR[_get(cfg, "dr", 0)], generator=generator, **env_options(cfg))
    O, OC, MO, U, LP, R, AL, LG = [], [], [], [], [], [], [], []
    for _ in range(T_DEC):
        alive = env.alive.clone()
        o, oc, legal = env.obs_pair(priv)
        with torch.no_grad():
            mode, a, u, lp = actor.act(o, legal, deterministic)
        r, done = env.step(mode, a)
        O.append(o); OC.append(oc); MO.append(mode); U.append(u); LP.append(lp); R.append(r); AL.append(alive); LG.append(legal)
        if done:
            break
    st = lambda xs: torch.stack(xs, 1)                       # noqa: E731
    return dict(O=st(O), OC=st(OC) if priv else None, MODE=st(MO), U=st(U), LP=st(LP), R=st(R), ALIVE=st(AL),
                LEGAL=st(LG)), env


def episode_stats(env):
    fin = ~env.alive
    return dict(feasible=env.feasible.float().mean().item(), crash=env.crashed.float().mean().item(),
                backtracks=env.backs.float().mean().item(), climbs=env.n_climb.float().mean().item(),
                nodes=env.k.float().mean().item(), finished=fin.float().mean().item())


# --- training -------------------------------------------------------------------------------------------------------
def save_checkpoint(path, state):
    torch.save(state, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", str(path))


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


def _finite(*xs):
    return all(bool(torch.isfinite(x).all()) for x in xs)


def train_g(cfg: ConfigG, log=print, ckpt=None, resume=None, metrics=None, snapshot=None, worlds=None):
    from .ppo import resolve_device
    dev = resolve_device(cfg.device)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    worlds = worlds if worlds is not None else load_pool(cfg.pool, cfg.pool_size, log=log)
    rng = np.random.default_rng(cfg.seed + 777)
    plr = PLR(len(worlds), cfg.plr_capacity, cfg.p_new, cfg.plr_beta, cfg.plr_rho, np.random.default_rng(cfg.seed + 99))
    probe_rng = np.random.default_rng(999)
    probe_levels = [{"world": int(probe_rng.integers(len(worlds))), "legs_seed": int(probe_rng.integers(2 ** 62))} for _ in range(16)]
    actor, crit = make_actor_g(cfg).to(dev), make_critic_g(cfg).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(crit.parameters(), lr=cfg.lr_v)
    ret_scale, hist, it0 = None, [], 0
    if resume is not None:
        dflt = {f.name: f.default for f in fields(ConfigG)}
        diff = {k: (resume["cfg"].get(k, dflt[k]), v) for k, v in asdict(cfg).items()
                if k not in RESUMABLE and resume["cfg"].get(k, dflt[k]) != v}
        if diff:
            raise ValueError(f"config differs from the checkpoint in non-resumable fields: {diff}")
        actor.load_state_dict(resume["actor"]); crit.load_state_dict(resume["critic"])
        opt_a.load_state_dict(resume["opt_a"]); opt_v.load_state_dict(resume["opt_v"])
        ret_scale, hist, it0 = resume["ret_scale"], list(resume["hist"]), resume["it"] + 1
        plr.load(resume["plr"]); rng.bit_generator.state = resume["rng"]
        torch.set_rng_state(resume["torch_rng"])
        log(f"resumed at iteration {it0}")
    mf = open(metrics, "a", buffering=1) if metrics else None

    def state(it):
        return {"format": "vG1", "actor": actor.state_dict(), "critic": crit.state_dict(), "opt_a": opt_a.state_dict(),
                "opt_v": opt_v.state_dict(), "cfg": asdict(cfg), "hist": hist, "it": it, "ret_scale": ret_scale,
                "plr": plr.state(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                "torch": torch.__version__}

    per_level = max(1, cfg.batch // cfg.levels_per_iter)
    it = it0 - 1
    for it in range(it0, cfg.iters):
        t0 = time.time()
        levels = plr.sample(cfg.levels_per_iter, it, cfg.plr)
        batch = batch_from_levels(levels, worlds, per_level, cfg, dev)
        B = len(batch["legs"])
        pref, u = sample_conditions(B, cfg.budget, cfg.u_lo, cfg.u_hi)
        roll, env = rollout_g(actor, batch, pref, u, cfg)
        O, OC, MO, U, LP, R, AL, LG = (roll[k] for k in ("O", "OC", "MODE", "U", "LP", "R", "ALIVE", "LEGAL"))
        if not _finite(O, OC, U, LP, R):
            log(f"it {it:4d}  non-finite rollout: skipped")
            continue
        T = R.shape[1]
        m = AL.float()
        s = (R * m).sum(1).std().item() + 1e-6
        ret_scale = s if ret_scale is None else 0.95 * ret_scale + 0.05 * s
        Rn = R / ret_scale
        with torch.no_grad():
            V = crit(OC.reshape(B * T, -1)).reshape(B, T) * m
        adv = torch.zeros_like(Rn)
        last = torch.zeros(B, device=dev)
        for k in reversed(range(T)):
            nv = V[:, k + 1] * m[:, k + 1] if k + 1 < T else torch.zeros(B, device=dev)
            delta = (Rn[:, k] + cfg.gamma * nv - V[:, k]) * m[:, k]
            last = delta + cfg.gamma * cfg.lam * last * (m[:, k + 1] if k + 1 < T else 0)
            adv[:, k] = last
        ret = adv + V
        # PLR scores: mean |advantage| per level
        lvl = batch["level"]
        absadv = (adv.abs() * m).sum(1) / m.sum(1).clamp_min(1)
        scores = torch.zeros(len(levels), device=dev).index_add_(0, lvl, absadv) / per_level
        plr.update(levels, scores.cpu().numpy(), it)
        keep = AL.reshape(-1)
        o, oc, mo, uu, lp, lg = (x.reshape(B * T, *x.shape[2:])[keep] for x in (O, OC, MO, U, LP, LG))
        ad, rt = adv.reshape(-1)[keep], ret.reshape(-1)[keep]
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        n = len(ad)
        frac = it / max(cfg.iters - 1, 1)
        ent_coef = cfg.ent * (1 - frac) if cfg.noise_decay else cfg.ent
        kls, pls, vls, ents = [], [], [], []
        ep = 0
        for ep in range(cfg.epochs):
            ek = []
            for idx in torch.randperm(n, device=dev).split(cfg.minibatch):
                dm, dc = actor.dist(o[idx], lg[idx])
                nlp = ActorG.logp(dm, dc, mo[idx], uu[idx])
                lr_ = nlp - lp[idx]
                ratio = lr_.exp()
                pl = -torch.min(ratio * ad[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * ad[idx]).mean()
                ent = dm.entropy().mean() + dc.entropy().sum(-1).mean() * 0.1
                vl = (crit(oc[idx]).squeeze(-1) - rt[idx]).pow(2).mean()
                if not _finite(pl, vl):
                    continue
                opt_a.zero_grad(); (pl - ent_coef * ent).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
                opt_v.zero_grad(); vl.backward()
                nn.utils.clip_grad_norm_(crit.parameters(), 1.0); opt_v.step()
                ek.append(((ratio - 1) - lr_).mean().item())
                pls.append(pl.item()); vls.append(vl.item()); ents.append(ent.item())
            kl = float(np.mean(ek)) if ek else math.nan
            kls.append(kl)
            if not ek or kl > cfg.target_kl:
                break
        if cfg.noise_decay:                                 # cap the exploration noise, linearly in training time
            with torch.no_grad():
                actor.log_std.clamp_(max=cfg.log_std0 + frac * (cfg.log_std_end - cfg.log_std0))
        st = episode_stats(env)
        modes = torch.bincount(mo, minlength=3).float() / max(len(mo), 1)
        row = dict(it=it, ret=(R * m).sum(1).mean().item(), **st, mode_frac=modes.tolist(), kl=kls[-1] if kls else math.nan,
                   epochs=ep + 1, policy_loss=float(np.mean(pls)) if pls else math.nan,
                   value_loss=float(np.mean(vls)) if vls else math.nan, entropy=float(np.mean(ents)) if ents else math.nan,
                   std=actor.log_std.exp().tolist(), plr_levels=len(plr.levels), steps=T, sec=time.time() - t0)
        if cfg.probe_every and (it % cfg.probe_every == 0 or it == cfg.iters - 1):
            pb = batch_from_levels([dict(lv, score=0.0, last=-1) for lv in probe_levels], worlds, 16, cfg, dev)
            g = torch.Generator().manual_seed(5)
            pw, pu = sample_conditions(len(pb["legs"]), cfg.budget, cfg.u_lo, cfg.u_hi, g)
            _, penv = rollout_g(actor, pb, pw, pu, cfg, deterministic=True, priv=False)
            row.update({f"det_{k}": v for k, v in episode_stats(penv).items()})
            log(f"it {it:4d}  ret {row['ret']:+.3f}  feas {st['feasible']*100:5.1f} %  crash {st['crash']*100:4.1f} %  "
                f"det feas {row['det_feasible']*100:5.1f} %  back {st['backtracks']:.2f} climb {st['climbs']:.2f}  "
                f"modes {np.round(row['mode_frac'], 3).tolist()}  kl {row['kl']:.3f}  v {row['value_loss']:.3f}  "
                f"plr {len(plr.levels)}  {row['sec']:.1f} s")
        hist.append(row)
        if mf:
            mf.write(json.dumps(row) + "\n")
        if ckpt and (it % cfg.ckpt_every == 0 or it == cfg.iters - 1):
            save_checkpoint(ckpt, state(it))
        if snapshot and cfg.snapshot_every and it > 0 and it % cfg.snapshot_every == 0:
            save_checkpoint(snapshot.format(it=it), state(it))
    if ckpt and it >= it0:
        save_checkpoint(ckpt, state(it))
    if mf:
        mf.close()
    return actor, crit, hist, plr
