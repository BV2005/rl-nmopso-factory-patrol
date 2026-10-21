"""Gazebo plans with pad-safe transits (scripts/padsafe_routes.py): same pads, zones and patrol loops as
factory2_auto_full; every transit leg planned around the other drones' pads (keep-out cylinders of
2.5 m + MyCost's 0.1 m drone size), loop legs as in scripts/export_plan.py.
  rl     loop: v2 policy proposals (data/warm_v2_real.mat); transits: v2 policy in the world with the
         keep-out cylinders (straight path + K x (1 + S) rollouts, short-safe pick)
  nm500  loop: authors' NMOPSO-500 seed 1 (results/nmopso_real); transits: the same NMOPSO (variant runner
         at MaxIt 500 = authors' code) with the keep-out cylinders (results/padsafe_nm500), from
         matlab -batch "cd matlab; run_nmopso_variant('../data/legs_padsafe.mat','../results/padsafe_nm500',1,4,500,'',1500)"
  python scripts/export_padsafe_plan.py rl|nm500  -> results/plans/<name>_padsafe/patrol_paths.json, top.png"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import conditions, rl_solve  # noqa: E402
from export_plan import chosen_paths, plot  # noqa: E402
from padsafe_routes import min_dist_to  # noqa: E402
from robustness import nm_front, polyline  # noqa: E402
from rlpatrol.metrics import short_safe  # noqa: E402
from rlpatrol.ppo import Actor  # noqa: E402
from rlpatrol.world import DATA, load_world, with_extra  # noqa: E402


def transit_paths(name, w, legs, extra):
    out, times = [], []
    if name == "rl":
        ck = torch.load(ROOT / "results" / "ckpt" / "v2.pt", weights_only=False)
        actor = Actor(ck["cfg"]["hidden"]); actor.load_state_dict(ck["actor"]); actor.eval()
        prefs, budgets = conditions(False, 50)
    for k, L in enumerate(legs):
        wk = with_extra(w, extra[k].reshape(-1, 3))
        if name == "rl":
            t0 = time.perf_counter()
            sols, _ = rl_solve(actor, wk, L[None], prefs, budgets, 3)
            st, _ = rl_solve(None, wk, L[None], prefs[:1], None, 0)
            F, E = np.vstack([sols[0][0], st[0][0]]), np.concatenate([sols[0][1], st[0][1]])
            times.append(time.perf_counter() - t0)
        else:
            F, E = nm_front("padsafe_nm500", k, 1)
            times.append(float(loadmat(ROOT / "results" / "padsafe_nm500" / f"leg{k + 1:03d}_seed1.mat")["time"].squeeze()))
        ok = np.where(np.isfinite(F).all(1))[0]
        if len(ok) == 0:
            raise RuntimeError(f"{name}: no feasible path for transit leg {k + 1}")
        out.append(polyline(wk, L, E[ok[short_safe(F[ok])]]))
    return out, times


def main(name):
    torch.set_num_threads(3)
    w = load_world()
    J = json.loads((DATA / "padsafe_preview.json").read_text())
    real = loadmat(DATA / "legs_real.mat")["legs"]
    loop_paths, loop_costs = chosen_paths(name, w, real)
    P = loadmat(DATA / "legs_padsafe.mat"); tlegs, extra = P["legs"], P["extra"]
    tpaths, times = transit_paths(name, w, tlegs, extra)
    homes = {d["id"]: np.array(d["home"][:2], float) for d in J["drones"]}
    kt = 0
    for d in J["drones"]:
        d["altitude"] = w.z_patrol; d["depart_delay"] = 0
        for part, key in (("transit_in", "transit_in"), ("patrol_points", "waypoints"), ("transit_out", "transit_out")):
            Q = np.asarray(d[part], float)
            wp = [Q[0]]
            for a, b in zip(Q[:-1], Q[1:]):
                if part == "patrol_points":
                    k = int(np.flatnonzero(np.all(np.abs(real - np.r_[a, b]) < 1e-9, axis=1))[0])
                    wp.extend(loop_paths[k][1:])
                else:
                    assert np.allclose(tlegs[kt], np.r_[a, b]); wp.extend(tpaths[kt][1:]); kt += 1
            d[key] = np.round(np.array(wp), 3).tolist()
        others = [homes[i] for i in homes if i != d["id"]]
        for key in ("transit_in", "transit_out"):
            dist = min_dist_to(np.array(d[key]), others)
            print(f"{name} drone {d['id']} {key}: {np.linalg.norm(np.diff(np.array(d[key])[:, :2], axis=0), axis=1).sum():.1f} m, "
                  f"min distance to other pads {dist:.2f} m")
            assert dist >= 2.55, "transit too close to another pad"
        del d["keepout"]
    J["planner"] = {"rl": "RL proposals (PPO v2 + straight path), short-safe; transits around other pads",
                    "nm500": "authors' NMOPSO (MaxIt 500) seed 1, short-safe; transits around other pads"}[name]
    J["transit_planning_time_s"] = {"total": float(np.sum(times)), "median_per_leg": float(np.median(times))}
    out = ROOT / "results" / "plans" / f"{name}_padsafe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "patrol_paths.json").write_text(json.dumps(J, indent=1))
    plot(J, w, out / "top.png", name)
    print(f"{name}: transit planning {np.sum(times):.1f} s for {len(tlegs)} legs; wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
