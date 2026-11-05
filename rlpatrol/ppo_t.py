"""PPO for vT (rlpatrol/env_t.py): the hybrid actor, privileged critic, world pool, PLR and checkpointing of
the generalizable policy (rlpatrol/ppo_g.py), with the step-wise, time-minimising environment."""
import json
import math
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn

from .env_g import DR
from .env_t import CTX, OBS_BASE, PRIV, T_MAX, TEnv, sample_pref3
from .ppo_g import PLR, ActorG, CtxMLP, _finite, batch_from_levels, load_pool, save_checkpoint


@dataclass
class ConfigT:
    iters: int = 1500
    batch: int = 8192
    epochs: int = 10
    minibatch: int = 16384
    lr: float = 3e-4
    lr_v: float = 1e-3
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    target_kl: float = 0.02
    ent: float = 0.003
    hidden: int = 256
    log_std0: float = -1.0
    noise_decay: bool = True
    log_std_end: float = -2.2
    act_pen: float = 0.02
    c_prox: float = 0.5
    d_safe: float = 0.5
    box_margin: float = 0.5
    c_back_time: float = 1.0          # unused since the review fix (a backtrack is timed by the time model); kept
                                      # so old checkpoints' configs still load
    plr: bool = False
    p_new: float = 0.3
    plr_capacity: int = 4000
    plr_beta: float = 0.1
    plr_rho: float = 0.3
    levels_per_iter: int = 64
    p_extra: float = 0.3
    dr: int = 0
    pool: str = "data/worlds/train.pkl"
    pool_size: int = 2000
    lmin: float = 1.5
    lmax: float = 12.0
    seed: int = 1
    threads: int = 0
    device: str = "cpu"
    ckpt_every: int = 10
    snapshot_every: int = 0
    probe_every: int = 20


RESUMABLE = {"iters", "device", "threads", "ckpt_every", "snapshot_every", "probe_every"}


def _get(cfg, k, d=None):
    return cfg.get(k, d) if isinstance(cfg, dict) else getattr(cfg, k, d)


def make_actor_t(cfg):
    return ActorG(OBS_BASE + CTX, CTX, _get(cfg, "hidden", 256), _get(cfg, "log_std0", -1.0))


def make_critic_t(cfg):
    return CtxMLP(OBS_BASE + PRIV + CTX, _get(cfg, "hidden", 256), 1, CTX)


def env_options_t(cfg):
    g = lambda k, d: _get(cfg, k, d)  # noqa: E731
    return dict(act_pen=g("act_pen", 0.02), c_prox=g("c_prox", 0.5), d_safe=g("d_safe", 0.5),
                box_margin=g("box_margin", 0.5))


def rollout_t(actor, batch, pref, cfg, deterministic=False, priv=True, generator=None, domain=None):
    env = TEnv(batch, pref, domain=domain or DR[_get(cfg, "dr", 0)], generator=generator, **env_options_t(cfg))
    O, OC, MO, U, LP, R, AL, LG = [], [], [], [], [], [], [], []
    for _ in range(T_MAX):
        alive = env.alive.clone()
        o, oc, legal = env.obs_pair(priv)
        with torch.no_grad():
            mode, a, u, lp = actor.act(o, legal, deterministic)
        r, done = env.step(mode, a)
        O.append(o); OC.append(oc); MO.append(mode); U.append(u); LP.append(lp); R.append(r); AL.append(alive); LG.append(legal)
        if done:
            break
    st = lambda xs: torch.stack(xs, 1)                     # noqa: E731
    return dict(O=st(O), OC=st(OC) if priv else None, MODE=st(MO), U=st(U), LP=st(LP), R=st(R), ALIVE=st(AL),
                LEGAL=st(LG)), env


def episode_stats_t(env):
    s = env.success
    t_ratio = (env.time / env.t_st)[s]
    return dict(success=s.float().mean().item(), crash=env.crashed.float().mean().item(),
                timeout=(~s & ~env.crashed).float().mean().item(),
                time_ratio=t_ratio.mean().item() if s.any() else math.nan,
                backtracks=env.backs.float().mean().item(), climbs=env.n_climb.float().mean().item(),
                steps=env.k.float().mean().item())


def train_t(cfg: ConfigT, log=print, ckpt=None, resume=None, metrics=None, snapshot=None, worlds=None):
    from .ppo import resolve_device
    dev = resolve_device(cfg.device)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    worlds = worlds if worlds is not None else load_pool(cfg.pool, cfg.pool_size, log=log)
    rng = np.random.default_rng(cfg.seed + 777)
    plr = PLR(len(worlds), cfg.plr_capacity, cfg.p_new, cfg.plr_beta, cfg.plr_rho, np.random.default_rng(cfg.seed + 99))
    prng = np.random.default_rng(999)
    probe_levels = [{"world": int(prng.integers(len(worlds))), "legs_seed": int(prng.integers(2 ** 62)), "score": 0.0, "last": -1}
                    for _ in range(16)]
    actor, crit = make_actor_t(cfg).to(dev), make_critic_t(cfg).to(dev)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    opt_v = torch.optim.Adam(crit.parameters(), lr=cfg.lr_v)
    ret_scale, hist, it0 = None, [], 0
    if resume is not None:
        dflt = {f.name: f.default for f in fields(ConfigT)}
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
        return {"format": "vT1", "actor": actor.state_dict(), "critic": crit.state_dict(), "opt_a": opt_a.state_dict(),
                "opt_v": opt_v.state_dict(), "cfg": asdict(cfg), "hist": hist, "it": it, "ret_scale": ret_scale,
                "plr": plr.state(), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "torch": torch.__version__}

    per_level = max(1, cfg.batch // cfg.levels_per_iter)
    it = it0 - 1
    for it in range(it0, cfg.iters):
        t0 = time.time()
        levels = plr.sample(cfg.levels_per_iter, it, cfg.plr)
        batch = batch_from_levels(levels, worlds, per_level, cfg, dev)
        B = len(batch["legs"])
        roll, env = rollout_t(actor, batch, sample_pref3(B), cfg)
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
            nm = m[:, k + 1] if k + 1 < T else torch.zeros(B, device=dev)
            nv = V[:, k + 1] * nm if k + 1 < T else torch.zeros(B, device=dev)
            delta = (Rn[:, k] + cfg.gamma * nv - V[:, k]) * m[:, k]
            last = delta + cfg.gamma * cfg.lam * last * nm
            adv[:, k] = last
        ret = adv + V
        absadv = (adv.abs() * m).sum(1) / m.sum(1).clamp_min(1)
        scores = torch.zeros(len(levels), device=dev).index_add_(0, batch["level"], absadv) / per_level
        plr.update(levels, scores.cpu().numpy(), it)
        keep = AL.reshape(-1)
        o, oc, mo, uu, lp, lg = (x.reshape(B * T, *x.shape[2:])[keep] for x in (O, OC, MO, U, LP, LG))
        ad, rt = adv.reshape(-1)[keep], ret.reshape(-1)[keep]
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        frac = it / max(cfg.iters - 1, 1)
        ent_coef = cfg.ent * (1 - frac) if cfg.noise_decay else cfg.ent
        kls, pls, vls = [], [], []
        ep = 0
        for ep in range(cfg.epochs):
            ek = []
            for idx in torch.randperm(len(ad), device=dev).split(cfg.minibatch):
                dm, dc = actor.dist(o[idx], lg[idx])
                lr_ = ActorG.logp(dm, dc, mo[idx], uu[idx]) - lp[idx]
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
                pls.append(pl.item()); vls.append(vl.item())
            kl = float(np.mean(ek)) if ek else math.nan
            kls.append(kl)
            if not ek or kl > cfg.target_kl:
                break
        if cfg.noise_decay:
            with torch.no_grad():
                actor.log_std.clamp_(max=cfg.log_std0 + frac * (cfg.log_std_end - cfg.log_std0))
        st = episode_stats_t(env)
        modes = torch.bincount(mo, minlength=3).float() / max(len(mo), 1)
        row = dict(it=it, ret=(R * m).sum(1).mean().item(), **st, mode_frac=modes.tolist(), kl=kls[-1] if kls else math.nan,
                   epochs=ep + 1, policy_loss=float(np.mean(pls)) if pls else math.nan,
                   value_loss=float(np.mean(vls)) if vls else math.nan, std=actor.log_std.exp().tolist(),
                   plr_levels=len(plr.levels), steps_T=T, sec=time.time() - t0)
        if cfg.probe_every and (it % cfg.probe_every == 0 or it == cfg.iters - 1):
            pb = batch_from_levels(probe_levels, worlds, 16, cfg, dev)
            g = torch.Generator().manual_seed(5)
            _, penv = rollout_t(actor, pb, sample_pref3(len(pb["legs"]), g), cfg, deterministic=True, priv=False)
            row.update({f"det_{k}": v for k, v in episode_stats_t(penv).items()})
            log(f"it {it:4d}  ret {row['ret']:+.3f}  succ {st['success']*100:5.1f} %  crash {st['crash']*100:4.1f} %  "
                f"timeout {st['timeout']*100:4.1f} %  det succ {row['det_success']*100:5.1f} %  "
                f"det time/straight {row['det_time_ratio']:.2f}  back {st['backtracks']:.2f} climb {st['climbs']:.2f}  "
                f"steps {st['steps']:.1f}  kl {row['kl']:.3f}  v {row['value_loss']:.3f}  {row['sec']:.1f} s")
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
