"""Round-5 comparison on the 73 real legs x 3 seeds (all costs from the verified MyCost port; the
NMOPSO fronts are MATLAB MyCost values).
Methods (fronts are the feasible non-dominated set):
  straight         the straight, level path (1 evaluation)
  rl               RL proposals: straight + <= 49 non-dominated v2 policy paths (data/warm_v2_real.mat)
  nm500            authors' NMOPSO, MaxIt 500 (unmodified)
  nm500+straight   nm500 front plus the straight path
  cold50           NMOPSO variant, MaxIt 50, random start (same seeds)
  cold50+straight  cold50 plus the straight path
  warm50           NMOPSO variant, MaxIt 50, swarm seeded with the rl paths (its final archive only)
  hybrid           rl  U  warm50  (external archive of the warm-started run)       <- proposed
  rl+cold50        rl  U  cold50  (ablation: proposals without the warm start)
Time per leg: NMOPSO wall time of the run (one core), plus the RL proposal time for rl-based methods.
  python scripts/final_compare.py [--warm data/warm_v2_seed2_real.mat --warmdir nmopso_warm50_seed2 --tag seed2]
Without the warm-started runs (--warmdir missing or incomplete) the warm50 / hybrid rows are left out."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import SCALE, TIE, feasible_front, hv  # noqa: E402
from rlpatrol.metrics import igd_plus  # noqa: E402
from rlpatrol.nmopso_cost import Legs  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

SEEDS = (1, 2, 3)
DIRS = {"nm500": "nmopso_real", "cold50": "nmopso_cold50", "warm50": "nmopso_warm50"}


def run(k, s, name):
    M = loadmat(ROOT / "results" / DIRS[name] / f"leg{k + 1:03d}_seed{s}.mat")
    return np.atleast_2d(M["front"]).T, float(M["time"].squeeze())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default="warm_v2_real.mat")
    ap.add_argument("--warmdir", default="nmopso_warm50")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    DIRS["warm50"] = a.warmdir
    torch.set_num_threads(2)
    w = load_world()
    legs = loadmat(DATA / "legs_real.mat")["legs"]
    W = loadmat(DATA / Path(a.warm).name)
    t_rl = float(W["wall_s"].squeeze()) / len(legs)
    have_warm = all((ROOT / "results" / a.warmdir / f"leg{k + 1:03d}_seed{s}.mat").exists() for k in range(len(legs)) for s in SEEDS)
    methods = ["straight", "rl", "nm500", "nm500+straight", "cold50", "cold50+straight"] + (["warm50", "hybrid"] if have_warm else []) + ["rl+cold50"]
    HV = {m: np.zeros((len(legs), len(SEEDS))) for m in methods}
    IGD = {m: np.zeros((len(legs), len(SEEDS))) for m in methods}
    T = {m: np.zeros((len(legs), len(SEEDS))) for m in methods}
    for k in range(len(legs)):
        m = int(W["cnt"][k, 0])
        lg = Legs(w, w.to_units(np.repeat(legs[k:k + 1, :3], m, 0)), w.to_units(np.repeat(legs[k:k + 1, 3:], m, 0)))
        Fr = lg.cost(lg.decode(*(torch.tensor(W[n][k, :m]) for n in ("r", "psi", "phi")))).numpy()
        st = Fr[:1]
        for j, s in enumerate(SEEDS):
            (fn, tn), (fc, tc) = run(k, s, "nm500"), run(k, s, "cold50")
            fw, tw = run(k, s, "warm50") if have_warm else (np.zeros((0, 4)), 0.0)
            fronts = {"straight": st, "rl": Fr, "nm500": fn, "nm500+straight": np.vstack([fn, st]),
                      "cold50": fc, "cold50+straight": np.vstack([fc, st]), "warm50": fw,
                      "hybrid": np.vstack([Fr, fw]), "rl+cold50": np.vstack([Fr, fc])}
            times = {"straight": 0.0, "rl": t_rl, "nm500": tn, "nm500+straight": tn, "cold50": tc,
                     "cold50+straight": tc, "warm50": tw + t_rl, "hybrid": tw + t_rl, "rl+cold50": tc + t_rl}
            fronts = {n: feasible_front(f) for n, f in fronts.items() if n in methods}
            ref = feasible_front(np.vstack([f for f in fronts.values() if len(f)])) / SCALE
            for n, f in fronts.items():
                HV[n][k, j] = hv(f)
                IGD[n][k, j] = igd_plus(f / SCALE, ref)
                T[n][k, j] = times[n]
    out = {"legs": len(legs), "seeds": list(SEEDS), "rl_proposal_s_per_leg": t_rl, "methods": {}}
    for n in methods:
        row = {"hv_mean": float(HV[n].mean()), "hv_std_over_seeds": float(HV[n].mean(0).std()),
               "igd_mean": float(IGD[n].mean()), "igd_median": float(np.median(IGD[n])),
               "time_median_s": float(np.median(T[n]))}
        for base in ("nm500", "nm500+straight", "cold50+straight"):
            if n == base:
                continue
            d = HV[n] - HV[base]
            hm, hb = HV[n].mean(1), HV[base].mean(1)
            nz = np.abs(hm - hb) > TIE
            row[f"vs_{base}"] = {"win_tie_loss": [float((d > TIE).mean()), float((abs(d) <= TIE).mean()), float((d < -TIE).mean())],
                                 "hv_rel_pct": float(100 * (hm.mean() / hb.mean() - 1)),
                                 "wilcoxon_p": float(wilcoxon(hm[nz], hb[nz]).pvalue) if nz.sum() >= 6 else None}
        out["methods"][n] = row
    (ROOT / "results" / f"final_compare{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(out, indent=1))
    print(f"{'method':16s} {'HV':>7s} {'IGD+':>7s} {'time/leg':>9s}   vs nm500+straight (W/T/L, rel, p)     vs cold50+straight")
    for n, r in out["methods"].items():
        a = r.get("vs_nm500+straight"); b = r.get("vs_cold50+straight")
        fa = f"{a['win_tie_loss'][0]:.2f}/{a['win_tie_loss'][1]:.2f}/{a['win_tie_loss'][2]:.2f} {a['hv_rel_pct']:+5.1f}% p={a['wilcoxon_p'] if a['wilcoxon_p'] is None else round(a['wilcoxon_p'], 4)}" if a else "-"
        fb = f"{b['win_tie_loss'][0]:.2f}/{b['win_tie_loss'][1]:.2f}/{b['win_tie_loss'][2]:.2f} {b['hv_rel_pct']:+5.1f}% p={b['wilcoxon_p'] if b['wilcoxon_p'] is None else round(b['wilcoxon_p'], 4)}" if b else "-"
        print(f"{n:16s} {r['hv_mean']:7.4f} {r['igd_mean']:7.4f} {r['time_median_s']:8.1f}s   {fa:38s} {fb}")


if __name__ == "__main__":
    main()
