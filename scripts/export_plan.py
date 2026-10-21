"""Gazebo patrol plans (patrol_paths.json, the format of plan_factory_patrol.m that the demo's
patrol_follower.py flies) for the latest plan (factory2_auto_full: 3 drones, its zones, tours and
transits), with every leg replaced by the path a planner chose (its short-safe pick, as
plan_factory_patrol.m selects with select = 'short-safe'):
  rl     the v2 policy's proposals + the straight path (data/warm_v2_real.mat)
  nm500  the authors' NMOPSO, MaxIt 500, seed 1 (results/nmopso_real)
  hybrid rl U RL-warm-started NMOPSO-50, seed 1 (results/nmopso_warm50)
The legs are those of data/legs_real.mat, in the order scripts/make_legs.py built them from the
preview. depart_delay is 0: the demo branch's departure scheduler fills it in before flying.
  python scripts/export_plan.py rl|nm500|hybrid  -> results/plans/<name>/patrol_paths.json, top.png"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from make_legs import PREVIEW  # noqa: E402
from robustness import nm_front, polyline, quality  # noqa: E402
from rlpatrol.metrics import short_safe  # noqa: E402
from rlpatrol.nmopso_cost import Legs  # noqa: E402
from rlpatrol.world import DATA, load_world  # noqa: E402


def chosen_paths(name, w, legs):
    W = loadmat(DATA / "warm_v2_real.mat")
    out, F_out = [], []
    for k, L in enumerate(legs):
        c = int(W["cnt"][k, 0]); Er = np.stack([W[n][k, :c] for n in ("r", "psi", "phi")], 1)
        lg = Legs(w, w.to_units(np.repeat(L[None, :3], c, 0)), w.to_units(np.repeat(L[None, 3:], c, 0)))
        Fr = lg.cost(lg.decode(*(torch.as_tensor(Er[:, i]) for i in range(3)))).numpy()
        if name == "rl":
            F, E = Fr, Er
        elif name == "nm500":
            F, E = nm_front("nmopso_real", k, 1)
        else:
            Fw, Ew = nm_front("nmopso_warm50", k, 1)
            F, E = np.vstack([Fr, Fw]), np.concatenate([Er, Ew])
        ok = np.where(np.isfinite(F).all(1))[0]
        j = ok[short_safe(F[ok])]
        out.append(polyline(w, L, E[j])); F_out.append(F[j])
    return out, F_out


def main(name):
    w = load_world()
    legs = loadmat(DATA / "legs_real.mat")["legs"]
    J = json.loads(PREVIEW.read_text())
    paths, costs = chosen_paths(name, w, legs)
    k = 0
    for d in J["drones"]:
        d["altitude"] = w.z_patrol
        d["depart_delay"] = 0
        d["leg_costs"] = []
        for part, key in (("transit_in", "transit_in"), ("patrol_points", "waypoints"), ("transit_out", "transit_out")):
            P = np.asarray(d[part], float)
            wp = [P[0]]
            for a, b in zip(P[:-1], P[1:]):
                if np.linalg.norm(a[:2] - b[:2]) < 0.05:          # vertical: flown straight (none in this plan)
                    wp.append(b); continue
                assert np.allclose(legs[k], np.r_[a, b]), f"leg order mismatch at {k}"
                wp.extend(paths[k][1:]); d["leg_costs"].append(costs[k].tolist()); k += 1
            d[key] = np.round(np.array(wp), 3).tolist()
    assert k == len(legs)
    J["planner"] = {"rl": "RL proposals (PPO v2 policy + straight path), short-safe pick",
                    "nm500": "authors' NMOPSO.m (unmodified, MaxIt 500), seed 1, short-safe pick",
                    "hybrid": "RL proposals U RL-warm-started NMOPSO (MaxIt 50), seed 1, short-safe pick"}[name]
    J["layout"] = "factory2_auto_full"
    out = ROOT / "results" / "plans" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "patrol_paths.json").write_text(json.dumps(J, indent=1))
    rects = w.rects[:, :4]
    for d in J["drones"]:
        loop = np.array(d["waypoints"]); tr = [np.array(d["transit_in"]), np.array(d["transit_out"])]
        L = np.linalg.norm(np.diff(loop, axis=0), axis=1).sum()
        T = sum(np.linalg.norm(np.diff(t, axis=0), axis=1).sum() for t in tr)
        clr = min(quality(p, np.r_[p[0], p[-1]], rects)["clearance_m"] for p in [loop, *tr])
        print(f"{name} drone {d['id']}: loop {L:.1f} m + transit {T:.1f} m, min clearance to boxes {clr:.2f} m")
    plot(J, w, out / "top.png", name)


def plot(J, w, file, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    for r in w.rects:
        ax.add_patch(plt.Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], color="0.45"))
    for d in J["drones"]:
        c = col[d["id"] - 1]
        P = np.array(d["waypoints"]); ax.plot(P[:, 0], P[:, 1], "-", color=c, lw=1.8, label=f"drone {d['id']}")
        for t in ("transit_in", "transit_out"):
            T = np.array(d[t]); ax.plot(T[:, 0], T[:, 1], "--", color=c, lw=1.2)
        ax.plot(*d["home"][:2], "s", color=c, mec="k", ms=10)
    h = w.hall
    ax.add_patch(plt.Rectangle((h[0], h[2]), h[1] - h[0], h[3] - h[2], fill=False, lw=1.5))
    ax.set_aspect("equal"); ax.set_xlim(h[0] - 1, h[1] + 1); ax.set_ylim(h[2] - 1, h[3] + 1)
    ax.set_title(f"factory2_auto_full patrol, legs planned by: {J['planner']}", fontsize=10)
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(file, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1])
