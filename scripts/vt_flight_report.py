"""Summary of a Gazebo + PX4 SITL flight with vT (gazebo/nmopso_demo/vt_follower.py) from its run folder.

Per drone and overall: PATROL legs planned online and their outcomes (success / fallback straight to the goal /
skipped), decisions, backtracks, climbs, how often the policy's own decision was not clear in the scan (fallbacks),
and the time-model check - the leg's flight time in vT's time model (env_t: 1 m/s, 0.5 m/s vertical, 0.5 s per 45
deg of turn) against the time actually flown in the simulator (PX4 sim time, first decision to arrival at the goal),
plus the time spent hovering at waypoints before deciding (settling, turning, rescanning, blocked). The collision
verdict comes from scripts/flight_check.py (fly_plan.sh writes flight_check.json; it is run here if missing).
  python scripts/vt_flight_report.py <run dir> [--out results/flight_vt_<name>.json]"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def summarise(rows):
    if not rows:
        return {}
    ok = [r for r in rows if r["outcome"] == "success"]
    f = lambda k, rs=ok: float(np.sum([float(r[k]) for r in rs])) if rs else float("nan")  # noqa: E731
    model, flown, straight = f("model_time_s"), f("flown_time_s"), f("straight_time_s")
    per_leg = np.array([float(r["flown_time_s"]) / float(r["model_time_s"]) for r in ok if float(r["model_time_s"]) > 0])
    return dict(legs=len(rows), success=len(ok) / len(rows),
                outcomes={o: sum(r["outcome"] == o for r in rows) for o in sorted({r["outcome"] for r in rows})},
                decisions=int(sum(int(r["decisions"]) for r in rows)), backtracks=int(sum(int(r["backtracks"]) for r in rows)),
                climbs=int(sum(int(r["climbs"]) for r in rows)), fallbacks=int(sum(int(r["fallbacks"]) for r in rows)),
                model_time_s=model, flown_time_s=flown, straight_time_s=straight,
                flown_over_model=flown / model if model else float("nan"),
                flown_over_model_per_leg_median=float(np.median(per_leg)) if len(per_leg) else float("nan"),
                model_over_straight=model / straight if straight else float("nan"),
                flown_over_straight=flown / straight if straight else float("nan"),
                wait_s=f("wait_s", rows), blocked_s=f("blocked_s", rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    files = sorted(run.glob("trace_drone*_vt_legs.csv"))
    if not files:
        sys.exit(f"{run}: no trace_drone*_vt_legs.csv (was it flown with VT_CKPT=...?)")
    drones, every = {}, []
    for p in files:
        d = p.name[len("trace_drone"):].split("_")[0]
        rows = list(csv.DictReader(open(p)))
        drones[d] = summarise(rows)
        every += rows
    fc = run / "flight_check.json"
    if not fc.exists() and (run / "drone_states.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts" / "flight_check.py"), str(run), "--out", str(fc)], check=False,
                       capture_output=True)
    check = json.loads(fc.read_text()) if fc.exists() else None
    res = dict(run=str(run), drones=drones, overall=summarise(every), flight_check=check)
    for name, s in [*((f"drone {k}", v) for k, v in drones.items()), ("all", res["overall"])]:
        if not s:
            continue
        print(f"{name:8s} legs {s['legs']:3d}  success {s['success'] * 100:5.1f} %  {s['outcomes']}  decisions {s['decisions']}  "
              f"backtracks {s['backtracks']}  climbs {s['climbs']}  fallbacks {s['fallbacks']}")
        print(f"         time: model {s['model_time_s']:.1f} s, flown {s['flown_time_s']:.1f} s -> flown / model "
              f"{s['flown_over_model']:.2f} (per-leg median {s['flown_over_model_per_leg_median']:.2f}); "
              f"model / straight {s['model_over_straight']:.2f}, flown / straight {s['flown_over_straight']:.2f}; "
              f"waiting at waypoints {s['wait_s']:.1f} s (blocked {s['blocked_s']:.1f} s)")
    if check is not None:
        print("flight_check:", {k: v for k, v in check.items() if k.startswith("any_")})
    else:
        print("flight_check: no drone_states.csv / flight_check.json in the run folder")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
