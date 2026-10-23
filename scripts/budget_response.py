"""Threat-budget response and the low-threat end of the front on the 73 real legs (round 6, round A).
  python scripts/budget_response.py v2 v4 vA_seed1 vA_ctrl_seed1 ...      (checkpoint names in results/ckpt)
Per checkpoint, adds or replaces its entry in results/budget_response.json and prints a table:
  budget-conditioned checkpoints: deterministic rollouts with the balanced preference (1/3, 0, 1/3, 1/3)
    at every budget u of evaluate.py's U_SWEEP:
      F2_over_straight   median F2 / F2 of the straight path (legs whose straight path is feasible, F2 > 0)
      within_budget      share of those legs with F2 <= u * straight F2
  every checkpoint: the lowest F2 over all of its rollouts (evaluate.py's conditions, S = 3) vs the lowest F2
    of NMOPSO-500's front (mean over its seeds):
      min_F2_rl_minus_nm_median   median difference, in F_SCALE[1] units (> 0: NMOPSO reaches lower threat)
      legs_rl_min_F2_le_nm        share of legs where RL reaches NMOPSO's lowest threat (0.1 % tolerance)
  and once: nm_min_F2_over_straight_median, how far NMOPSO-500 itself takes the threat below the straight path."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import conditions, load_nmopso, pref_set, rl_solve  # noqa: E402
from rlpatrol.env import F_SCALE  # noqa: E402
from rlpatrol.ppo import Actor, net_shape  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402

BALANCED = [1 / 3, 0, 1 / 3, 1 / 3]


def min_f2(F):
    ok = np.isfinite(F).all(1)
    return F[ok, 1].min() if ok.any() else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", help="checkpoint names (results/ckpt/<name>.pt)")
    ap.add_argument("--out", default=str(ROOT / "results" / "budget_response.json"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    world = load_world()
    legs = loadmat(DATA / "legs_real.mat")["legs"]
    nL = len(legs)
    st, _ = rl_solve(None, world, legs, pref_set(1), None, 0)
    f2s = np.array([s[0][0, 1] for s in st])                    # straight path's F2 (inf if it collides)
    ok_leg = np.isfinite(f2s) & (f2s > 0)
    nm = load_nmopso(ROOT / "results" / "nmopso_real", nL)
    nm_min = np.array([np.mean([min_f2(F) for F, _ in nm[k].values()]) if k in nm else np.nan for k in range(nL)])
    f2u = F_SCALE[1].item()

    out_p = Path(a.out)
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    both_nm = ok_leg & np.isfinite(nm_min)
    res["_nmopso"] = {"nm_min_F2_over_straight_median": float(np.median(nm_min[both_nm] / f2s[both_nm])),
                      "legs": int(both_nm.sum())}
    print(f"NMOPSO-500 lowest-threat path: F2 / straight median {res['_nmopso']['nm_min_F2_over_straight_median']:.2f}"
          f" over {res['_nmopso']['legs']} legs")
    for name in a.ckpts:
        ck = torch.load(ROOT / "results" / "ckpt" / f"{name}.pt", map_location="cpu", weights_only=False)
        cfg = ck["cfg"]; ctx, extra = net_shape(cfg)
        actor = Actor(cfg["hidden"], ctx=ctx, extra=extra); actor.load_state_dict(ck["actor"]); actor.eval()
        budget, bobs, robs = cfg.get("budget", False), cfg.get("budget_obs", False), cfg.get("route_obs", False)
        prefs, budgets = conditions(budget, 50)
        sols, _ = rl_solve(actor, world, legs, prefs, budgets, 3, budget_obs=bobs, route_obs=robs)
        r = {"budget": budget, "dense": cfg.get("dense", False), "budget_obs": bobs, "route_obs": robs}
        if budget:                                              # the first C rollouts of a leg are deterministic
            bal = [i for i in range(len(prefs)) if np.allclose(prefs[i], BALANCED)]
            r["u"], r["F2_over_straight"], r["within_budget"] = [], [], []
            for i in bal:
                F = np.array([sols[k][0][i] for k in range(nL)])
                good = ok_leg & np.isfinite(F).all(1)
                r["u"].append(float(budgets[i]))
                r["F2_over_straight"].append(float(np.median(F[good, 1] / f2s[good])))
                r["within_budget"].append(float(np.mean(F[good, 1] <= budgets[i] * f2s[good] + 1e-12)))
        rl_min = np.array([min_f2(F) for F, _ in sols])
        both = np.isfinite(rl_min) & np.isfinite(nm_min)
        r["min_F2_rl_minus_nm_median"] = float(np.median(rl_min[both] - nm_min[both]) / f2u)
        r["legs_rl_min_F2_le_nm"] = float(np.mean(rl_min[both] <= nm_min[both] * 1.001))
        r["legs_compared"] = int(both.sum())
        res[name] = r
        if budget:
            w = dict(zip(r["u"], r["within_budget"]))
            print(f"{name:15} F2/straight by u: " + " ".join(f"{u:.1f}:{x:.2f}" for u, x in zip(r["u"], r["F2_over_straight"]))
                  + f"   within budget: u=0.3 {w[0.3] * 100:.0f}%  u=1.0 {w[1.0] * 100:.0f}%")
        print(f"{name:15} low-threat end: RL min F2 - NMOPSO min F2, median {r['min_F2_rl_minus_nm_median']:+.3f} (F_SCALE units);"
              f" RL reaches NMOPSO's lowest threat on {r['legs_rl_min_F2_le_nm'] * 100:.0f}% of {r['legs_compared']} legs")
    out_p.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
