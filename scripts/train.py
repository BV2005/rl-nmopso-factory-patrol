"""Trains the preference-conditioned PPO leg planner.
  python scripts/train.py --name v0 [--config configs/<file>.yaml] [--iters 300 --batch 4096 ...any Config field]
  python scripts/train.py --name v0 --resume [--iters 600]       continue results/ckpt/v0.pt
Precedence: Config defaults < --config file < command-line flags.
Writes (never overwriting an existing run unless --overwrite):
  results/ckpt/<name>.pt               resumable checkpoint (latest; every --ckpt_every iterations)
  results/ckpt/<name>_it<NNNN>.pt      snapshots (every --snapshot_every iterations, if > 0)
  results/runs/<name>.log              text log (appended on --resume)
  results/runs/<name>.jsonl            one JSON row per iteration: return, feasibility, losses, KL, timing"""
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

from rlpatrol.ppo import RESUMABLE, Config, load_checkpoint, resolve_device, train  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("1", "true", "yes", "on"):
        return True
    if v.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def env_info(dev):
    s = (f"python {platform.python_version()}  torch {torch.__version__}  cuda build {torch.version.cuda}  "
         f"cuda available {torch.cuda.is_available()}  device {dev}")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        s += f"  gpu {p.name} ({p.total_memory / 2 ** 30:.1f} GiB)"
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="run name: results/ckpt/<name>.pt, results/runs/<name>.*")
    ap.add_argument("--config", help="YAML file with Config fields (flat mapping)")
    ap.add_argument("--resume", action="store_true", help="continue results/ckpt/<name>.pt")
    ap.add_argument("--overwrite", action="store_true", help="allow a fresh run to replace an existing <name>")
    for f in fields(Config):
        t = str2bool if isinstance(f.default, bool) else type(f.default)
        ap.add_argument(f"--{f.name}", type=t, default=None, help=f"default {f.default!r}")
    a = ap.parse_args()

    ckpt = ROOT / "results" / "ckpt" / f"{a.name}.pt"
    runs = ROOT / "results" / "runs"
    logp, metp = runs / f"{a.name}.log", runs / f"{a.name}.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    cli = {f.name: getattr(a, f.name) for f in fields(Config) if getattr(a, f.name) is not None}

    resume = None
    if a.resume:
        if not ckpt.exists():
            sys.exit(f"--resume: {ckpt} does not exist")
        if a.config:
            sys.exit("--resume takes the configuration from the checkpoint; do not pass --config")
        resume = load_checkpoint(ckpt)
        if "opt_a" not in resume:
            sys.exit(f"{ckpt} was written before resumable checkpoints (no optimiser state); "
                     "it can be evaluated but not resumed. Start a new --name instead.")
        saved = resume["cfg"]
        bad = {k: (saved.get(k), v) for k, v in cli.items() if k not in RESUMABLE and saved.get(k) != v}
        if bad:
            sys.exit(f"--resume: these flags differ from the checkpoint and cannot change on resume: {bad}. "
                     f"Only {sorted(RESUMABLE)} may change.")
        known = {f.name for f in fields(Config)}
        cfg = Config(**{**{k: v for k, v in saved.items() if k in known}, **cli})
    else:
        if (ckpt.exists() or metp.exists()) and not a.overwrite:
            sys.exit(f"run '{a.name}' already exists ({ckpt}). Use --resume to continue it, a new --name, "
                     "or --overwrite to replace it.")
        base = {}
        if a.config:
            base = yaml.safe_load(Path(a.config).read_text()) or {}
            unknown = set(base) - {f.name for f in fields(Config)}
            if unknown:
                sys.exit(f"{a.config}: unknown Config field(s) {sorted(unknown)}")
        cfg = Config(**{**base, **cli})
        if metp.exists():
            metp.unlink()                                    # --overwrite: start the metrics log afresh

    dev = resolve_device(cfg.device)
    logf = open(logp, "a" if resume else "w", buffering=1)

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    log(env_info(dev))
    log(f"config {cfg}")
    log(f"config_json {json.dumps(asdict(cfg))}")
    try:
        train(load_world(device=dev), cfg, log=log, ckpt=ckpt, resume=resume, metrics=metp,
              snapshot=str(ckpt.parent / f"{a.name}_it{{it:04d}}.pt"))
    except KeyboardInterrupt:
        log(f"interrupted; resume with: python scripts/train.py --name {a.name} --resume")
        sys.exit(130)
    log(f"done: {ckpt}")


if __name__ == "__main__":
    main()
