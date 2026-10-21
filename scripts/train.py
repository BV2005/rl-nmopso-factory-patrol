"""Trains the preference-conditioned PPO leg planner.
  python scripts/train.py --name v0 [--iters 300 --batch 4096 ...any Config field]
Writes results/ckpt/<name>.pt and results/runs/<name>.log."""
import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.ppo import Config, train  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    for f in fields(Config):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = ap.parse_args()
    cfg = Config(**{f.name: getattr(a, f.name) for f in fields(Config)})
    (ROOT / "results" / "ckpt").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "runs").mkdir(parents=True, exist_ok=True)
    logf = open(ROOT / "results" / "runs" / f"{a.name}.log", "w", buffering=1)

    def log(s):
        print(s, flush=True); logf.write(s + "\n")

    log(f"config {cfg}")
    train(load_world(), cfg, log=log, ckpt=ROOT / "results" / "ckpt" / f"{a.name}.pt")
