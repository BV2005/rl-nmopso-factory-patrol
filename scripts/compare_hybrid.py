"""Round 5: RL-warm-started NMOPSO vs cold NMOPSO at the same MaxIt vs the authors' NMOPSO (MaxIt 500).
  python scripts/compare_hybrid.py --dirs nm500=results/nmopso_real cold50=results/nmopso_cold50 warm50=results/nmopso_warm50
         [--rl results/ckpt/v2.pt]
Per leg and seed (legs with every method and seed present): hypervolume (fixed normalisation), IGD+
against the union of all fronts of the leg, attainment of nm500's deployed (short-safe) path, time.
Time of a warm run includes the RL proposal time per leg (data/warm_*.mat wall_s / legs).
Pairwise: win / tie / loss of every method against nm500 at a 1e-3 HV tolerance, and a Wilcoxon
signed-rank test on the per-leg (seed-averaged) HV."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import SCALE, TIE, feasible_front, hv  # noqa: E402
from rlpatrol.metrics import add_eps, igd_plus, short_safe  # noqa: E402


def load_dir(d):
    out = {}
    for f in Path(d).glob("leg*_seed*.mat"):
        k, s = int(f.stem[3:6]) - 1, int(f.stem.split("seed")[1])
        M = loadmat(f)
        out[(k, s)] = (feasible_front(np.atleast_2d(M["front"]).T), float(M["time"].squeeze()))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name=dir; the first is the reference (nm500)")
    ap.add_argument("--warm", default=str(ROOT / "data" / "warm_v2_real.mat"))
    ap.add_argument("--out", default=str(ROOT / "results" / "compare_hybrid.json"))
    a = ap.parse_args()
    D = {n: load_dir(p) for n, p in (x.split("=", 1) for x in a.dirs)}
    names = list(D)
    refn = names[0]
    W = loadmat(a.warm)
    rl_time = float(W["wall_s"].squeeze()) / W["r"].shape[0]
    seeds = sorted({s for (_, s) in D[refn]})
    legs = sorted({k for (k, _) in D[refn]})
    legs = [k for k in legs if all((k, s) in D[n] for n in names for s in seeds)]
    res = {n: {"hv": [], "igd": [], "eps": [], "time": []} for n in names}
    for k in legs:
        ref_front = feasible_front(np.vstack([D[n][(k, s)][0] for n in names for s in seeds if len(D[n][(k, s)][0])])) / SCALE
        picks = [D[refn][(k, s)][0][short_safe(D[refn][(k, s)][0])] / SCALE for s in seeds if len(D[refn][(k, s)][0])]
        for n in names:
            fr = [D[n][(k, s)][0] for s in seeds]
            res[n]["hv"].append([hv(f) for f in fr])
            res[n]["igd"].append([igd_plus(f / SCALE, ref_front) for f in fr])
            res[n]["eps"].append([min(add_eps(f / SCALE, p) for p in picks) if picks else np.nan for f in fr])
            res[n]["time"].append([D[n][(k, s)][1] + (rl_time if n.startswith("warm") else 0) for s in seeds])
    summ = {"legs": len(legs), "seeds": seeds, "rl_proposal_s_per_leg": rl_time}
    H0 = np.array(res[refn]["hv"]).mean(1)
    for n in names:
        H = np.array(res[n]["hv"]); I = np.array(res[n]["igd"]); E = np.array(res[n]["eps"]); T = np.array(res[n]["time"])
        row = {"hv_mean": float(H.mean()), "igd_mean": float(I.mean()), "igd_median": float(np.median(I)),
               "attains_nm500_pick": float((E <= 1e-9).mean()), "eps_median": float(np.nanmedian(E)),
               "time_median_s": float(np.median(T)), "time_mean_s": float(T.mean())}
        if n != refn:
            d = H - np.array(res[refn]["hv"])
            row["vs_ref_win_tie_loss"] = [float((d > TIE).mean()), float((abs(d) <= TIE).mean()), float((d < -TIE).mean())]
            hm = H.mean(1); diff = hm - H0
            row["hv_rel_to_ref_pct"] = float(100 * (hm.mean() / H0.mean() - 1))
            nz = np.abs(diff) > TIE
            row["wilcoxon_p"] = float(wilcoxon(hm[nz], H0[nz]).pvalue) if nz.sum() >= 6 else None
            row["speedup_vs_ref_median"] = float(np.median(np.array(res[refn]["time"])) / np.median(T))
        summ[n] = row
    Path(a.out).write_text(json.dumps({"summary": summ, "legs": legs, "per_leg": res}, indent=1))
    print(json.dumps(summ, indent=1))
