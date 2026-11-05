"""Trains the time-minimising, encoding-free policy vT (rlpatrol/ppo_t.py) on procedurally generated worlds.
  python scripts/train_t.py --name vT_seed1 --config configs/vT.yaml --seed 1
  python scripts/train_t.py --name vT_seed1 --resume [--iters 3000]
Writes results/ckpt/<name>.pt, results/runs/<name>.log and .jsonl (as scripts/train.py)."""
import argparse
import json
import platform
import sys
from dataclasses import asdict, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from rlpatrol.ppo import resolve_device  # noqa: E402
from rlpatrol.ppo_g import load_checkpoint  # noqa: E402
from rlpatrol.ppo_t import RESUMABLE, ConfigT, train_t  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    for f in fields(ConfigT):
        t = str2bool if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=t, default=None)
    a = ap.parse_args()
    ckpt = ROOT / "results" / "ckpt" / f"{a.name}.pt"
    runs = ROOT / "results" / "runs"
    ckpt.parent.mkdir(parents=True, exist_ok=True); runs.mkdir(parents=True, exist_ok=True)
    logp, metp = runs / f"{a.name}.log", runs / f"{a.name}.jsonl"
    cli = {f.name: getattr(a, f.name) for f in fields(ConfigT) if getattr(a, f.name) is not None}
    resume = None
    if a.resume:
        if not ckpt.exists():
            sys.exit(f"--resume: {ckpt} does not exist")
        resume = load_checkpoint(ckpt)
        bad = {k: v for k, v in cli.items() if k not in RESUMABLE and resume["cfg"].get(k) != v}
        if bad:
            sys.exit(f"--resume: cannot change {sorted(bad)}")
        known = {f.name for f in fields(ConfigT)}
        cfg = ConfigT(**{**{k: v for k, v in resume["cfg"].items() if k in known}, **cli})
    else:
        if (ckpt.exists() or metp.exists()) and not a.overwrite:
            sys.exit(f"run {a.name} exists; --resume, a new --name or --overwrite")
        base = yaml.safe_load(Path(a.config).read_text()) if a.config else {}
        unknown = set(base or {}) - {f.name for f in fields(ConfigT)}
        if unknown:
            sys.exit(f"{a.config}: unknown fields {sorted(unknown)}")
        cfg = ConfigT(**{**(base or {}), **cli})
        if metp.exists():
            metp.unlink()
    dev = resolve_device(cfg.device)
    logf = open(logp, "a" if resume else "w", buffering=1)

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    log(f"python {platform.python_version()}  torch {torch.__version__}  device {dev}"
        + (f"  gpu {torch.cuda.get_device_name(dev)}" if dev.type == "cuda" else ""))
    log(f"config_json {json.dumps(asdict(cfg))}")
    try:
        train_t(cfg, log=log, ckpt=ckpt, resume=resume, metrics=metp, snapshot=str(ckpt.parent / f"{a.name}_it{{it:04d}}.pt"),
                best=ckpt.parent / f"{a.name}_best.pt")
    except KeyboardInterrupt:
        log(f"interrupted; resume with: python scripts/train_t.py --name {a.name} --resume")
        sys.exit(130)
    log(f"done: {ckpt}")


if __name__ == "__main__":
    main()
