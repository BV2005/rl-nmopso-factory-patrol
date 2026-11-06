"""Builds the world pools of the generalizable policy (rlpatrol/worldgen.py) and caches them:
  data/worlds/train.pkl     training pool (seeds 0 ..)
  data/worlds/heldout.pkl   same generator, seeds never used in training (10_000_000 ..)
  data/worlds/ood.pkl       out-of-distribution generator (narrower aisles, larger halls, L-shaped stacks)
  python scripts/make_worlds.py [--train 2000] [--heldout 200] [--ood 200] [--sdf 3]
--sdf N also writes the first N held-out worlds as Gazebo worlds (data/worlds/heldout_<i>.sdf)."""
import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.worldgen import OOD, TRAIN, generate  # noqa: E402

SEEDS = {"train": 0, "heldout": 10_000_000, "ood": 20_000_000}


def build(name, n, params):
    t0 = time.time()
    W = [generate(SEEDS[name] + i, params) for i in range(n)]
    out = ROOT / "data" / "worlds" / f"{name}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(W, f)
    nb = sum(len(w.boxes) for w in W) / max(n, 1)
    print(f"{name}: {n} worlds, {nb:.1f} boxes/world, {time.time() - t0:.0f} s -> {out}")
    return W


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--ood", type=int, default=200)
    ap.add_argument("--sdf", type=int, default=0)
    a = ap.parse_args()
    build("train", a.train, TRAIN)
    H = build("heldout", a.heldout, TRAIN)
    build("ood", a.ood, OOD)
    for i in range(min(a.sdf, len(H))):
        p = ROOT / "data" / "worlds" / f"heldout_{i}.sdf"
        H[i].to_sdf(p, ROOT / "worlds" / "factory2.sdf")
        print("wrote", p)
