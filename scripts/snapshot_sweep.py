"""Leg-planning quality along a training run: evaluate.py on every snapshot of the named runs (round 5b, v5).
  python scripts/snapshot_sweep.py v5_seed1 v5_seed2 v5_seed3 [--extra v2]
For each run, its snapshots results/ckpt/<run>_itNNNN.pt and its final checkpoint (plus --extra checkpoints)
are evaluated on the real and the random legs (evaluate.py defaults: K = 50, S = 3). Only the summaries are
kept, in results/snapshot_sweep.json (merged with an existing file); evaluate.py's per-checkpoint outputs
(~25 MB per checkpoint) are written to temporary tags and deleted."""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEEP = ("hv_rl", "hv_straight", "rl_legs_with_feasible", "legs")


def evaluate(ck, legs):
    tag = f"_sweep_{ck}_{legs}"
    outs = [ROOT / "results" / f"eval_{tag}.json", ROOT / "results" / f"rl_solutions_{tag}.mat"]
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--ckpt", f"results/ckpt/{ck}.pt",
                        "--legs", legs, "--tag", tag, "--overwrite"], cwd=ROOT, check=True, capture_output=True)
        s = json.loads(outs[0].read_text())["summary"]
        return {k: s[k] for k in KEEP}
    finally:
        for o in outs:
            o.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--extra", nargs="*", default=[], help="further checkpoints to evaluate, e.g. v2")
    ap.add_argument("--out", default=str(ROOT / "results" / "snapshot_sweep.json"))
    a = ap.parse_args()
    out_p = Path(a.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    cks = list(a.extra)
    for run in a.runs:
        snaps = sorted(p.stem for p in (ROOT / "results" / "ckpt").glob(f"{run}_it[0-9][0-9][0-9][0-9].pt"))
        cks += snaps + [run]
    for ck in cks:
        res[ck] = {legs: evaluate(ck, legs) for legs in ("real", "random")}
        m = re.search(r"_it(\d+)$", ck)
        print(f"{ck:22} {'it ' + str(int(m.group(1))) if m else 'final':>8}  HV real {res[ck]['real']['hv_rl']:.3f}"
              f"  feasible random legs {res[ck]['random']['rl_legs_with_feasible']}/{res[ck]['random']['legs']}", flush=True)
        out_p.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
