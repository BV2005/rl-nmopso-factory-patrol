"""Device test: every entry point of the vG and vT environments, judge and trainers, with the env on the GPU and
the caller's tensors made on the CPU (how scripts, tests and the follower call them). Catches "Expected all
tensors to be on the same device" before a long run does. About a minute on a GPU.
  1. GenEnv (vG): step / _proposal / _wall_hit / executed / obs_pair with CPU inputs; outputs on the env's device
  2. TEnv (vT): the same, plus flight_metrics / reversals / segment_time with mixed CPU / GPU inputs
  3. rollouts (rollout_g, rollout_t, the vT3 validation) and 2 training iterations of train_g and of train_t
     (vT3 settings: lr decay, KL stop, floors, validation + best checkpoint) on the GPU
  python scripts/test_devices.py [--device cuda]      (without a GPU it says so and exits 0: nothing to test)"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.env_g import ADVANCE as G_ADV, BACK as G_BACK, EVAL, GenEnv, build_batch, sample_conditions  # noqa: E402
from rlpatrol.env_t import ADVANCE, BACK, TEnv, flight_metrics, reversals, sample_pref3, segment_time  # noqa: E402
from rlpatrol.ppo_g import ConfigG, batch_from_levels, make_actor_g, rollout_g, train_g  # noqa: E402
from rlpatrol.ppo_t import ConfigT, make_actor_t, rollout_t, train_t, validate, validation_set  # noqa: E402
from rlpatrol.worldgen import cylinders, generate  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from tensors(v)


def on(dev, *outs):
    return all(t.device.type == dev.type for o in outs for t in tensors(o))


def gen_checks(dev):
    print("[1] GenEnv (vG)")
    W = [generate(s) for s in range(2)]
    b = batch_from_levels([{"world": i, "legs_seed": i} for i in range(2)], W, 4, ConfigG(modes="all"), dev)
    B = len(b["legs"])
    w, u = sample_conditions(B, True)                                           # CPU
    env = GenEnv(b, w, u, domain=EVAL, collision="box")
    o, oc, L = env.obs_pair()
    check(on(dev, o, oc, L), "obs_pair on the env's device")
    r, _ = env.step(torch.full((B,), G_ADV), torch.zeros(B, 3))                 # CPU mode / action
    q, d = env._proposal(torch.full((B,), G_ADV), torch.rand(B, 3) * 2 - 1)     # CPU
    wh = env._wall_hit(torch.zeros(B, 3, dtype=torch.float64), torch.ones(B, 3, dtype=torch.float64))   # CPU points
    r2, _ = env.step(torch.full((B,), G_BACK), torch.zeros(B, 3))
    P, n = env.executed()
    check(on(dev, r, q, d, wh, r2, P, n, env.heading), "step / _proposal / _wall_hit / executed with CPU inputs -> env's device")
    corner = np.array([[-9.0, -8.5, -9.0, -8.5, 1.0]])
    ob = build_batch([(np.array([-10, 10, -10, 10.0]), corner, cylinders(corner), np.array([-6, 0, 1.8, 6, 0, 1.8]), None)], dev)
    e = GenEnv(ob, torch.tensor([[1 / 3, 0, 1 / 3, 1 / 3]]), None, domain=EVAL)
    for _ in range(20):
        _, done = e.step(torch.full((1,), G_ADV), torch.tensor([[-0.3, 0.0, 0.0]]))
        if done:
            break
    check(on(dev, e.F, e.F_exec, e.feasible) and bool(e.feasible.all()), "a whole episode with CPU actions; F / F_exec on the env's device")


def t_checks(dev):
    print("[2] TEnv (vT) and the judge")
    W = [generate(s) for s in range(2)]
    b = batch_from_levels([{"world": i, "legs_seed": 3 + i} for i in range(2)], W, 4, ConfigT(), dev)
    B = len(b["legs"])
    env = TEnv(b, sample_pref3(B), domain=EVAL)                                  # CPU preference
    o, oc, L = env.obs_pair()
    q, turn, r, snap = env._proposal(torch.full((B,), ADVANCE), torch.rand(B, 3, dtype=torch.float64) * 2 - 1)
    rw, _ = env.step(torch.full((B,), ADVANCE), torch.zeros(B, 3))
    rb, _ = env.step(torch.full((B,), BACK), torch.zeros(B, 3))
    P, n = env.executed()
    check(on(dev, o, oc, L, q, turn, r, snap, rw, rb, P, n, env.time, env.r_time),
          "obs_pair / _proposal / step (advance, backtrack) with CPU inputs -> env's device")
    m = flight_metrics(P.cpu(), n.cpu(), b["boxes"], env.obst, walls=b["kmask"] & ~env.obst)    # CPU path, GPU world
    m2 = flight_metrics(P, n.cpu(), b["boxes"].cpu(), env.obst.cpu())                            # GPU path, CPU world
    rv = reversals(P, n.cpu())
    st = segment_time(P[:, 0], P[:, 1], torch.zeros(B, dtype=torch.float64, device=dev))
    check(on(torch.device("cpu"), m) and on(dev, m2, rv, st) and
          bool(torch.allclose(m["time_s"], m2["time_s"].cpu())),
          "flight_metrics / reversals / segment_time with mixed CPU / GPU inputs (results on the path's device, equal)")


def train_checks(dev):
    print("[3] rollouts and training on the device")
    W = [generate(s) for s in range(4)]
    cg = ConfigG(modes="all", collision="box", device=str(dev))
    ag = make_actor_g(cg).to(dev)
    bg = batch_from_levels([{"world": i, "legs_seed": i} for i in range(2)], W, 4, cg, dev)
    w, u = sample_conditions(len(bg["legs"]), True)
    roll, env = rollout_g(ag, bg, w, u, cg)
    check(on(dev, roll["R"], env.F), "rollout_g (CPU preference / budget)")
    ct = ConfigT(device=str(dev), val_worlds=2, val_legs=4)
    at = make_actor_t(ct).to(dev)
    bt = batch_from_levels([{"world": i, "legs_seed": i} for i in range(2)], W, 4, ct, dev)
    roll, env = rollout_t(at, bt, sample_pref3(len(bt["legs"])), ct)
    vb = validation_set(ct, dev)
    sc, vs, vt = validate(at, vb, ct)
    check(roll["R"].device.type == dev.type and np.isfinite(sc), f"rollout_t and the vT3 validation (score {sc:.2f})")
    with tempfile.TemporaryDirectory() as d:
        c = ConfigG(iters=2, batch=64, minibatch=256, levels_per_iter=8, device=str(dev), probe_every=1, ckpt_every=1,
                    modes="all", collision="box")
        _, _, h, _ = train_g(c, log=lambda s: None, ckpt=Path(d) / "g.pt", worlds=W)
        check(len(h) == 2 and np.isfinite(h[-1]["ret"]), "train_g: 2 iterations on the device")
        c = ConfigT(iters=2, batch=64, minibatch=512, levels_per_iter=8, device=str(dev), probe_every=1, ckpt_every=1,
                    lr_end=3e-5, kl_stop=1.5, log_std_min=-1.9, ent_min=0.001, val_every=1, val_worlds=2, val_legs=4)
        _, _, h, _ = train_t(c, log=lambda s: None, ckpt=Path(d) / "t.pt", worlds=W, best=Path(d) / "t_best.pt")
        check(len(h) == 2 and np.isfinite(h[-1]["ret"]) and (Path(d) / "t_best.pt").exists(),
              "train_t (vT3 settings): 2 iterations on the device, best checkpoint saved")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("no CUDA device: nothing to test (run this on the GPU pod)")
        return
    gen_checks(dev)
    t_checks(dev)
    train_checks(dev)
    print(f"all {len(OK)} checks passed")


if __name__ == "__main__":
    main()
