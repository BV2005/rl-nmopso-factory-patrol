"""Self-test of the sensor-input policy pipeline (x500_patrol lidar + depth camera). Minutes on a CPU.
  1. sensor geometry: sim_scan against hand-computed cases (wall ahead, blind sector, a machine
     below the lidar plane seen only by the depth camera, overflown clutter, lidar-only side return)
  2. flight converters: bins_from_laserscan / bins_from_points / to_obs give the bins sim_scan
     gives for the same geometry; segment_clear
  3. environment: the default observation is unchanged (regression against the pre-sensor code);
     sensor_obs and priv shapes, finiteness, noise, scan_override; oracle features refused
  4. training: a few PPO iterations with sensor_obs + priv_critic + budget + dense, checkpoint
     round trip, and the online LegPlanner on the sensor checkpoint
  5. speed: sim_scan on a training-size batch
  6. GRU actor (recurrent): rollout log-probabilities = the whole-episode recomputation of the PPO
     update, the episode-state guard, training with blocked legs, resume of an older checkpoint,
     the planner; blocked_legs still reproduces data/legs_blocked.mat
  python scripts/test_sensors.py [--device auto] [--batch 512]"""
import argparse
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol import sensors as S  # noqa: E402
from rlpatrol.env import N_NODES, N_RAYS, OBS_DIM, RAY_MAX, SENSOR_OBS, PathEnv, sample_pref  # noqa: E402
from rlpatrol.legs import random_legs  # noqa: E402
from rlpatrol.ppo import (Actor, Config, RecurrentActor, critic_extra, env_kwargs, load_checkpoint, make_actor,  # noqa: E402
                          net_shape, resolve_device, rollout, sample_budget_pref, train)
from rlpatrol.world import load_world  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


HALL = [-5.0, 5.0, -5.0, 5.0]


def walls(hall=HALL):
    x0, x1, y0, y1 = hall
    T, H = S.WALL_T, S.WALL_TOP
    return [[x0 - T, x0, y0 - T, y1 + T, H], [x1, x1 + T, y0 - T, y1 + T, H],
            [x0 - T, x1 + T, y0 - T, y0, H], [x0 - T, x1 + T, y1, y1 + T, H]]


def scan_m(boxes, pos=(0.0, 0.0, 1.8), yaw=0.0):
    b = torch.tensor(boxes, dtype=torch.float64)
    return S.sim_scan(b, torch.tensor([pos], dtype=torch.float64), torch.tensor([yaw], dtype=torch.float64),
                      S.SensorSpec(noise=False), metres=True)[0].numpy()


def geometry_checks():
    print("[1] sensor geometry (ideal sensors)")
    mid = S.SCAN_BINS // 2
    check(close(S.bin_centres()[mid], 0.0, 1e-12), f"bin {mid} is centred on the heading")
    r = scan_m(walls())
    # the forward bin is inside the depth camera's view: its range is from the optical centre
    check(close(r[mid], 5.0 - S.DEPTH_X), f"wall 5 m ahead: forward bin {r[mid]:.3f} m = 5 - DEPTH_X (depth camera)")
    side = int(np.argmin(np.abs(S.bin_centres().numpy() - math.pi / 2)))
    check(close(S.bin_centres()[side], math.pi / 2, 1e-5) and close(r[side], 5.0),
          f"wall 5 m to the left (outside the depth view): lidar bin {r[side]:.3f} m")
    r = scan_m(walls([-50, 50, -50, 50]) + [[-3, -2, -0.5, 0.5, 3.5]])
    check(np.isinf(r).all(), "a rack only behind the drone (blind sector) and far walls: no return in any bin")
    z_cam = 1.8 + S.DEPTH_Z
    t_top = (z_cam - 1.2) / math.tan(S.DEPTH_VFOV / 2)
    r = scan_m(walls([-50, 50, -50, 50]) + [[3.0, 4.0, -0.5, 0.5, 1.2]])
    check(close(r[mid], 3.0 - S.DEPTH_X), f"1.2 m machine 3 m ahead, below the lidar plane: depth sees it at {r[mid]:.3f} m")
    r = scan_m(walls([-50, 50, -50, 50]) + [[0.8, 2.5, -0.5, 0.5, 1.2]])
    check(close(r[mid], t_top), f"the same machine close: seen once its top enters the view, {r[mid]:.3f} = {t_top:.3f} m")
    r = scan_m(walls([-50, 50, -50, 50]) + [[2.0, 3.0, -0.5, 0.5, 1.0]])
    check(np.isinf(r[mid]), "1.0 m clutter (an overflown conveyor height) is not an obstacle")
    r = scan_m(walls([-50, 50, -50, 50]) + [[-0.5, 0.5, 2.0, 3.0, 1.2]])
    check(np.isinf(r).all(), "1.2 m machine to the side (outside the depth view, below the lidar plane): unseen")
    r = scan_m(walls([-50, 50, -50, 50]) + [[-0.5, 0.5, 2.0, 3.0, 3.5]])
    check(close(r[side], 2.0), f"rack 2 m to the left: lidar {r[side]:.3f} m")
    r = scan_m(walls(), pos=(0.0, 0.0, 1.8), yaw=math.pi / 2)
    check(close(r[mid], 5.0 - S.DEPTH_X), "heading north: the forward bin follows the heading")


def converter_checks():
    print("[2] flight converters")
    c = S.bin_centres().numpy()
    # LaserScan of a wall 4 m ahead (x = 4), the lidar's 1080 beams
    n, a0 = 1080, -S.LIDAR_FOV / 2
    inc = S.LIDAR_FOV / (n - 1)
    ang = a0 + inc * np.arange(n)
    ranges = np.where(np.cos(ang) > 1e-3, 4.0 / np.maximum(np.cos(ang), 1e-3), np.inf)
    ranges[ranges > 30] = np.inf
    b = S.bins_from_laserscan(ranges, a0, inc)
    mid = S.SCAN_BINS // 2
    check(close(b[mid], 4.0, 1e-4), "LaserScan wall 4 m ahead -> forward bin 4 m")
    lo = np.abs(c) - S.BIN / 2
    exp = np.where(np.abs(c) < math.pi / 2 - S.BIN / 2, 4.0 / np.cos(np.clip(lo, 0, None)), np.nan)
    m = np.isfinite(exp) & (np.abs(c) < 1.2)
    check(np.allclose(b[m], exp[m], rtol=1e-2), "LaserScan bins = the nearest return of each 10 deg sector")
    check(np.isinf(S.bins_from_laserscan(np.full(n, np.nan), a0, inc)).all(), "NaN ranges (no return) -> no return")
    # depth points of a wall 3 m ahead, heights -1..1 m around the camera
    yy, zz = np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-1, 1, 41))
    P = np.c_[np.full(yy.size, 3.0), yy.ravel(), zz.ravel()]
    bd = S.bins_from_points(P, drone_z=1.8)
    check(close(bd[mid], 3.0, 1e-9), "depth points of a wall 3 m ahead -> forward bin 3 m")
    check(np.isinf(bd[np.abs(c) > S.DEPTH_HFOV / 2 + S.BIN]).all(), "no depth returns outside the camera's field of view")
    low = P[P[:, 2] < S.OBST_MIN_H - 1.8 - S.DEPTH_Z - 1e-9]
    check(np.isinf(S.bins_from_points(low, drone_z=1.8)).all(), "depth points below OBST_MIN_H are ignored")
    o = S.to_obs(b, bd)
    check(o.shape == (S.SCAN_BINS,) and o.dtype == np.float32 and close(o[mid], 3.0 / S.SCAN_MAX, 1e-6)
          and (o <= 1).all() and (o >= 0).all(), "to_obs fuses by the minimum and normalises to [0, 1]")
    # the converters agree with the simulator on the same geometry
    boxes = walls([-6, 6, -6, 6]) + [[2.0, 3.0, -2.0, -1.0, 3.5], [-1.0, 0.0, 2.0, 2.5, 3.5]]
    sim = scan_m(boxes)
    bt = torch.tensor(boxes, dtype=torch.float64)
    d = torch.stack([torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))], -1)[None]
    t_in, t_out = S.ray_boxes(torch.zeros(1, 2, dtype=torch.float64), d, bt)
    t0 = t_in.clamp_min(0)
    rr = torch.where(t_out >= t0, t0, torch.full_like(t0, math.inf)).min(-1).values[0].numpy()
    lid = S.bins_from_laserscan(rr, a0, inc)
    out = np.abs(c) > S.DEPTH_HFOV / 2 + S.BIN                       # lidar-only bins
    err = np.abs(lid[out] - sim[out]) / lid[out]
    check(np.isfinite(lid[out]).all() and err.max() < 0.05,
          f"LaserScan bins of 1080 cast beams = sim_scan's 3 beams per bin (lidar-only sectors, max error {err.max() * 100:.1f} %)")
    sc = S.to_obs(np.full(S.SCAN_BINS, np.inf))
    sc[mid] = 2.0 / S.SCAN_MAX
    check(not S.segment_clear(sc, 0.0, 0.0, 3.0, 0.4), "segment_clear: 3 m segment into a return at 2 m -> blocked")
    check(S.segment_clear(sc, 0.0, 0.0, 1.0, 0.4), "segment_clear: 1 m segment short of a return at 2 m -> clear")
    check(S.segment_clear(sc, 0.0, math.pi / 2, 3.0, 0.4), "segment_clear: 3 m segment to the left of it -> clear")


def old_obs(env):
    """PathEnv.obs before the sensor option (verbatim), for the regression check."""
    P = env.pose()
    R, p = P[:, :3, :3], P[:, :3, 3]
    D = env.legs.dist[:, None]
    gv = env.legs.goal - p
    g = (R.transpose(1, 2) @ gv[..., None])[..., 0] / D
    reach = gv.norm(dim=-1, keepdim=True) / (env.segs_left() * env.legs.rmax[:, None])
    zc = ((p[:, 2:3] - env.zmid) / env.zhalf).clamp(-5, 5)
    off = ((env.q - p) / 10).clamp(-2, 2)
    left = torch.full_like(D, (N_NODES - env.i) / N_NODES)
    bear = torch.atan2(env.legs.goal[:, 1] - env.q[:, 1], env.legs.goal[:, 0] - env.q[:, 0])
    head = torch.atan2(R[:, 1, 0], R[:, 0, 0]) - bear
    o = torch.cat([g, reach, zc, off, left, D / 100, env.legs.rmax[:, None] / RAY_MAX,
                   torch.cos(head)[:, None], torch.sin(head)[:, None],
                   env.rays(env.q, bear)] + env.route_state(bear) + env.budget_state()
                  + ([env.u[:, None]] if env.u is not None else [])
                  + [env.pref], -1)
    return o.float()


def env_checks(world, dev):
    print("[3] environment")
    B = 128
    legs = random_legs(world, B, np.random.default_rng(0))
    st, gl = world.to_units(legs[:, :3]), world.to_units(legs[:, 3:])
    u = torch.full((B,), 0.6, dtype=torch.float64)
    for kw in ({}, {"route_obs": True}, {"budget": u, "budget_obs": True, "route_obs": True}):
        env = PathEnv(world, st, gl, sample_pref(B), **kw)
        same = True
        for _ in range(N_NODES):
            same &= bool(torch.equal(env.obs(), old_obs(env)))
            env.step(torch.rand(B, 3, device=dev) * 2 - 1)
        check(same, f"default observation unchanged ({', '.join(kw) or 'no options'})")
    env = PathEnv(world, st, gl, sample_pref(B), sensor_obs=True)
    o, oc = env.obs_pair()
    check(o.shape == (B, OBS_DIM + SENSOR_OBS) and torch.isfinite(o).all() and oc is o,
          f"sensor_obs: actor observation [{B}, {OBS_DIM + SENSOR_OBS}], finite; critic = actor without priv")
    env = PathEnv(world, st, gl, sample_pref(B), budget=u, sensor_obs=True, priv=True)
    o, oc = env.obs_pair()
    cfg = Config(budget=True, sensor_obs=True, priv_critic=True)
    ctx, extra = net_shape(cfg)
    check(o.shape[1] == OBS_DIM + extra + ctx - 4 and oc.shape[1] == OBS_DIM + critic_extra(cfg) + ctx - 4,
          f"priv: actor {o.shape[1]} / critic {oc.shape[1]} entries match net_shape / critic_extra")
    check(torch.equal(o[:, -ctx:], oc[:, -ctx:]) and torch.equal(o[:, :-ctx], oc[:, :o.shape[1] - ctx]),
          "priv: the critic sees the actor's observation, the privileged features, then the same conditioning")
    a1 = PathEnv(world, st, gl, sample_pref(B), sensor_obs=True, sensor_spec=S.SensorSpec(noise=False)).obs()
    a2 = PathEnv(world, st, gl, sample_pref(B), sensor_obs=True, sensor_spec=S.SensorSpec(noise=False)).obs()
    n1 = PathEnv(world, st, gl, sample_pref(B), sensor_obs=True).obs()
    k0 = OBS_DIM - N_RAYS - 4                                    # the shared core entries before the scan
    k1 = k0 + S.SCAN_BINS
    check(torch.equal(a1[:, k0:k1], a2[:, k0:k1]) and not torch.equal(a1[:, k0:k1], n1[:, k0:k1]),
          "ideal sensors are deterministic; sensor noise changes the scan")
    env = PathEnv(world, st, gl, sample_pref(B), sensor_obs=True)
    env.scan_override = np.full((B, S.SCAN_BINS), 0.25, np.float32)
    check(bool((env.obs()[:, k0:k1] == 0.25).all()), "scan_override replaces the simulated scan (flight)")
    tot = torch.zeros(B, device=dev)
    for _ in range(N_NODES):
        r, done, info = env.step(torch.rand(B, 3, device=dev) * 2 - 1)
        tot += r
    check(torch.isfinite(tot).all() and done, "sensor_obs episode: rewards finite, terminates after n nodes")
    for bad in ({"route_obs": True}, {"budget": u, "budget_obs": True}):
        try:
            PathEnv(world, st, gl, sample_pref(B), sensor_obs=True, **bad)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"sensor_obs refuses the oracle feature(s) {sorted(k for k in bad if k != 'budget')}")


def train_checks(world, dev, batch):
    print("[4] training (sensor_obs + priv_critic + budget + dense)")
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "s.pt"
        cfg = Config(iters=3, batch=batch, minibatch=batch * 2, budget=True, dense=True, budget_pen=50.0,
                     act_pen=0.02, sensor_obs=True, priv_critic=True, device=str(dev), probe_every=1, ckpt_every=1)
        actor, crit, hist = train(world, cfg, log=lambda s: None, ckpt=ck)
        check(len(hist) == 3 and all(math.isfinite(h["ret"]) and math.isfinite(h["value_loss"]) for h in hist),
              f"3 iterations, finite returns and value losses (feasible {hist[-1]['feasible'] * 100:.0f} %)")
        c = load_checkpoint(ck)
        check(c["cfg"]["sensor_obs"] and c["cfg"]["priv_critic"], "checkpoint carries the sensor options")
        check(crit.l1.in_features == OBS_DIM + critic_extra(cfg) + 1 and actor.mu.l1.in_features == OBS_DIM + SENSOR_OBS + 1,
              "critic is wider than the actor by the privileged features")
        from rlpatrol.planner import LegPlanner
        lp = LegPlanner(ck, n_samples=1, refine=False, device=str(dev))
        leg = random_legs(world, 1, np.random.default_rng(3))[0]
        plan = lp.plan(leg[:2], leg[3:5])
        check(isinstance(plan.feasible, bool) and (not plan.feasible or np.isfinite(plan.F).all()),
              f"LegPlanner plans a leg with the sensor checkpoint ({plan.method}, feasible {plan.feasible})")


def speed_checks(world, dev):
    print("[5] speed")
    B = 8192
    legs = random_legs(world, B, np.random.default_rng(1))
    q = torch.tensor(legs[:, :3], dtype=torch.float64, device=dev)
    yaw = torch.rand(B, dtype=torch.float64, device=dev) * 2 * math.pi
    boxes = S.scene_boxes(world)
    S.sim_scan(boxes, q[:16], yaw[:16])
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    s = S.sim_scan(boxes, q, yaw)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    check(s.shape == (B, S.SCAN_BINS) and torch.isfinite(s).all(), f"sim_scan of {B} drones: {dt * 1000:.0f} ms on {dev}")


def gru_checks(world, dev, batch):
    print("[6] GRU actor (recurrent) and blocked training legs")
    from scipy.io import loadmat
    from rlpatrol.legs import blocked_legs
    from rlpatrol.route import seg_clear
    B = blocked_legs(world, 200, np.random.default_rng(24680))
    ref = loadmat(str(ROOT / "data" / "legs_blocked.mat"))["legs"]
    check(B.shape == ref.shape and np.abs(B - ref).max() < 1e-12,
          f"legs.blocked_legs reproduces data/legs_blocked.mat (max diff {np.abs(B - ref).max():.0e}, float rounding)")
    q = torch.tensor(world.to_units(B[:, :3])[:, :2], dtype=world.threats.dtype, device=dev)
    g = torch.tensor(world.to_units(B[:, 3:])[:, :2], dtype=world.threats.dtype, device=dev)
    check(not bool(seg_clear(world.threats, q, g).any()), "every blocked leg's straight line is blocked")

    cfg = Config(budget=True, dense=True, sensor_obs=True, priv_critic=True, recurrent=True, device=str(dev))
    actor = make_actor(cfg).to(dev)
    check(isinstance(actor, RecurrentActor) and isinstance(make_actor(Config()), Actor),
          "make_actor: recurrent -> RecurrentActor, default -> Actor (old checkpoints)")
    with torch.no_grad():                                    # a non-zero policy, so the check means something
        for p in actor.parameters():
            p.add_(0.05 * torch.randn_like(p))
    n = 64
    legs = random_legs(world, n, np.random.default_rng(7))
    w, u = sample_budget_pref(n, cfg)
    O, U, LP, R, info, env = rollout(actor, world, legs, w, budget=u, **env_kwargs(cfg))
    with torch.no_grad():
        lp2 = Actor.logp(actor.dist_seq(O), U)
    err = (lp2 - LP).abs().max().item()
    check(err < 1e-4, f"rollout log-probs = whole-episode recomputation (max diff {err:.1e}): PPO ratios start at 1")
    o1 = env.obs()
    actor.reset(n)
    actor.act(o1)
    same = actor.act(o1)                                     # the same observation again: same step
    try:
        actor.act(o1.clone())
        guarded = False
    except RuntimeError:
        guarded = True
    check(guarded and same is not None, "a new observation before commit()/discard() is refused")
    actor.discard()
    actor.act(o1.clone()); actor.commit()
    check(actor.h is not None and actor.h.shape == (n, cfg.gru_hidden), "discard() then commit(): the state moves on")

    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "g.pt"
        cfg = Config(iters=3, batch=batch, minibatch=batch * 2, budget=True, dense=True, budget_pen=50.0, act_pen=0.02,
                     sensor_obs=True, priv_critic=True, recurrent=True, blocked_frac=0.3, device=str(dev),
                     probe_every=1, ckpt_every=1)
        _, _, hist = train(world, cfg, log=lambda s: None, ckpt=ck)
        check(len(hist) == 3 and all(math.isfinite(h["ret"]) and math.isfinite(h["value_loss"]) and math.isfinite(h["kl"])
                                     for h in hist),
              f"3 GRU iterations with 30 % blocked legs: finite (feasible {hist[-1]['feasible'] * 100:.0f} %)")
        c = load_checkpoint(ck)
        for k in ("recurrent", "gru_hidden", "blocked_frac", "sensor_obs", "priv_critic", "sensor_noise"):
            c["cfg"].pop(k, None)                            # as an older checkpoint would have it
        old = Config(**{k: v for k, v in c["cfg"].items()})
        c2 = load_checkpoint(ck)
        ck0 = Path(d) / "old.pt"
        torch.save({**c2, "cfg": c["cfg"], "actor": c2["actor"]}, ck0)
        try:
            # the MLP fields of this GRU checkpoint do not fit an MLP actor: only the config check matters here
            train(world, Config(**{**old.__dict__, "iters": 4}), log=lambda s: None, resume=load_checkpoint(ck0))
            resumed = True
        except ValueError as e:
            resumed = "config differs" not in str(e)
        except RuntimeError:                                 # state_dict mismatch: past the config check
            resumed = True
        check(resumed, "resume accepts a checkpoint written before the new Config fields (defaults)")
        from rlpatrol.planner import LegPlanner
        lp = LegPlanner(ck, n_samples=1, refine=False, device=str(dev))
        leg = random_legs(world, 1, np.random.default_rng(3))[0]
        plan = lp.plan(leg[:2], leg[3:5])
        check(isinstance(plan.feasible, bool) and lp.ckpt_info["recurrent"],
              f"LegPlanner plans with the GRU checkpoint ({plan.method}, feasible {plan.feasible})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    dev = resolve_device(a.device)
    world = load_world(device=dev)
    geometry_checks()
    converter_checks()
    env_checks(world, dev)
    train_checks(world, dev, a.batch)
    speed_checks(world, dev)
    gru_checks(world, dev, a.batch)
    print(f"all {len(OK)} checks passed")


if __name__ == "__main__":
    main()
