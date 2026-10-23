"""Smoke test of the training pipeline (minutes, not a training run). Checks, in order:
  1. environment: reset, observation shape/finiteness, action bounds, rewards finite and in range
  2. device: the requested device is used (cuda if available with --device auto)
  3. training: a few PPO iterations; JSONL metrics rows with the expected keys
  4. checkpoint: save, load, resume; on CPU a resumed run must equal an uninterrupted one exactly
  5. CLI: scripts/train.py fresh run, overwrite protection, --resume extension, bad-resume refusal
  6. planner: the online planner loads the smoke checkpoint and returns a (feasible or explicitly
     infeasible) plan without errors
Nothing is written under results/ except the CLI test's run, which is removed at the end.
  python scripts/smoke_test.py [--device auto] [--batch 512]"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.env import INFEASIBLE, N_NODES, OBS_DIM, PathEnv, sample_pref  # noqa: E402
from rlpatrol.legs import random_legs  # noqa: E402
from rlpatrol.ppo import Config, load_checkpoint, resolve_device, train  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def env_checks(world, dev):
    print("[1] environment")
    legs = random_legs(world, 256, np.random.default_rng(0))
    env = PathEnv(world, world.to_units(legs[:, :3]), world.to_units(legs[:, 3:]), sample_pref(256))
    o = env.obs()
    check(o.shape == (256, OBS_DIM) and torch.isfinite(o).all(), f"reset: observation [{256}, {OBS_DIM}], finite")
    check(o.device.type == dev.type, f"observation on {dev}")
    tot = torch.zeros(256, device=dev)
    for i in range(N_NODES):
        a = torch.rand(256, 3, device=dev) * 2 - 1
        r, psi, phi = env.act(a)
        check(bool(((r >= env.legs.rmin * 1.001 - 1e-9) & (r <= env.legs.rmax + 1e-9)).all()
                   and (psi.abs() <= math.pi / 4 + 1e-9).all() and (phi.abs() <= math.pi / 4 + 1e-9).all()),
              f"step {i}: action decodes inside NMOPSO's bounds") if i in (0, N_NODES - 1) else None
        rew, done, info = env.step(a)
        check(torch.isfinite(rew).all(), f"step {i}: rewards finite") if i in (0, N_NODES - 1) else None
        tot += rew
        check(done == (i == N_NODES - 1), "episode terminates after exactly n = 10 nodes") if done else None
    check(bool((tot >= -(INFEASIBLE + 3.0) - 1e-6).all() and (tot <= 1e-6).all()),
          f"episode returns in [-(INFEASIBLE + 3), 0]: min {tot.min().item():.3f} max {tot.max().item():.3f}")
    check(info["F"].shape == (256, 4), "terminal info carries the 4 MyCost objectives")
    big = torch.full((4, 3), 50.0, device=dev)                       # out-of-range actions are clipped
    env2 = PathEnv(world, world.to_units(legs[:4, :3]), world.to_units(legs[:4, 3:]), sample_pref(4))
    r, _, _ = env2.act(big)
    check(bool(torch.isfinite(r).all()), "out-of-range actions are clipped to [-1, 1] (finite decode)")


def train_checks(world, dev, batch, tmp):
    print(f"[2-4] training, checkpoint, resume ({dev})")
    tmp = tmp / dev.type
    tmp.mkdir(exist_ok=True)
    base = dict(batch=batch, epochs=2, act_pen=0.02, device=str(dev), probe_every=2, ckpt_every=1, seed=3)
    m = tmp / "m.jsonl"
    t0 = time.time()
    _, _, hist = train(world, Config(iters=4, **base), log=print, ckpt=tmp / "full.pt", metrics=m)
    check(len(hist) == 4, f"4 PPO iterations completed ({time.time() - t0:.1f} s)")
    rows = [json.loads(x) for x in m.read_text().splitlines()]
    keys = {"it", "ret", "feasible", "policy_loss", "value_loss", "entropy", "kl", "ep_len", "env_steps", "sec"}
    check(len(rows) == 4 and keys <= set(rows[0]), f"JSONL metrics: 4 rows with {sorted(keys)}")
    check(all(math.isfinite(r["ret"]) and math.isfinite(r["value_loss"]) for r in rows), "logged losses finite")
    if dev.type == "cuda":
        check(rows[-1].get("gpu_mem_peak_mb", 0) > 0, f"GPU used: peak {rows[-1]['gpu_mem_peak_mb']:.0f} MB")
    ck = load_checkpoint(tmp / "full.pt")
    check(ck["it"] == 3 and {"actor", "critic", "opt_a", "opt_v", "rng", "ret_scale"} <= set(ck),
          "checkpoint holds networks, optimisers, RNG states, return scale, iteration")
    # interrupted after 2 iterations, then resumed to 4
    train(world, Config(iters=2, **base), log=lambda s: None, ckpt=tmp / "part.pt")
    part = load_checkpoint(tmp / "part.pt")
    check(part["it"] == 1, "partial run checkpointed at iteration 1")
    _, _, h2 = train(world, Config(iters=4, **base), log=print, ckpt=tmp / "part.pt", resume=part)
    check([r["it"] for r in h2] == [0, 1, 2, 3], "resumed run continues at iteration 2 and keeps the history")
    a, b = [r["ret"] for r in hist], [r["ret"] for r in h2]
    if dev.type == "cpu":
        check(a == b, "resumed run reproduces the uninterrupted run exactly (CPU)")
    else:
        check(all(math.isfinite(x) for x in b), f"resumed run finite (GPU kernels are not bitwise deterministic): "
              f"returns {np.round(a, 3).tolist()} vs {np.round(b, 3).tolist()}")
    try:
        train(world, Config(iters=4, **{**base, "lr": 1e-3}), log=lambda s: None, resume=part)
        check(False, "resume with a changed learning rate must be refused")
    except ValueError:
        check(True, "resume refuses a configuration change in a non-resumable field")
    return tmp / "full.pt"


def cli_checks(dev, batch):
    print("[5] scripts/train.py command line")
    name = f"_smoke_cli_{int(time.time())}"
    py, tr = sys.executable, str(ROOT / "scripts" / "train.py")
    files = [ROOT / "results" / "ckpt" / f"{name}.pt", ROOT / "results" / "runs" / f"{name}.log",
             ROOT / "results" / "runs" / f"{name}.jsonl"]
    run = lambda *a: subprocess.run([py, tr, "--name", name, *a], capture_output=True, text=True, cwd=ROOT)
    try:
        r = run("--iters", "2", "--batch", str(batch), "--epochs", "1", "--device", str(dev.type), "--probe_every", "1")
        check(r.returncode == 0 and files[0].exists(), "fresh run writes results/ckpt/<name>.pt")
        r = run("--iters", "2")
        check(r.returncode != 0 and "already exists" in (r.stderr + r.stdout), "existing run is not overwritten")
        r = run("--resume", "--iters", "3")
        ck = load_checkpoint(files[0])
        check(r.returncode == 0 and ck["it"] == 2 and "resumed at iteration 2" in r.stdout,
              "--resume --iters 3 extends the run from iteration 2")
        r = run("--resume", "--batch", str(batch * 2))
        check(r.returncode != 0 and "cannot change on resume" in (r.stderr + r.stdout), "--resume refuses a changed --batch")
        lines = files[2].read_text().splitlines()
        check(len(lines) == 3, "JSONL appended on resume (3 rows)")
    finally:
        for f in files:
            f.unlink(missing_ok=True)


def planner_checks(ckpt):
    print("[6] online planner with the smoke checkpoint")
    from rlpatrol.planner import LegPlanner
    p = LegPlanner(ckpt, threads=2)
    r = p.plan_route((-8.0, 7.0), (7.0, -4.0))
    check(r.feasible or r.reason, f"plan_route returns a plan or an explicit failure ({r.method}, "
          f"{r.plan_time_s:.2f} s; an untrained policy may need route decomposition/refinement)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    dev = resolve_device(a.device)
    print(f"torch {torch.__version__}, cuda available {torch.cuda.is_available()}, device {dev}"
          + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == "cuda" else ""))
    world = load_world(device=dev)
    env_checks(world, dev)
    with tempfile.TemporaryDirectory() as td:
        ck = train_checks(world, dev, a.batch, Path(td))
        if dev.type == "cuda":                                   # the exact-resume check needs CPU determinism
            print("[4b] exact resume on CPU")
            train_checks(load_world(device=torch.device("cpu")), torch.device("cpu"), 256, Path(td))
        cli_checks(dev, a.batch)
        planner_checks(ck)
    print(f"SMOKE TEST PASSED ({len(OK)} checks)")


if __name__ == "__main__":
    main()
