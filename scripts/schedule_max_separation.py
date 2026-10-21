#!/usr/bin/env python3
"""Departure delays that maximise the smallest predicted same-level separation between drones, for
plans where schedule_departures.py (the demo's timed conflict check) finds no conflict-free schedule.
Uses the demo's timeline() unchanged: each drone hovers over its start for TAKEOFF + delay, then flies
its plan at the patrol speed; after landing it no longer counts. Separation is horizontal distance at
|dz| < DZ; delays 0..MAX s in STEP s steps, the earliest drone at 0. Also reports the predicted
separation for the delays chosen.
  python3 schedule_max_separation.py <demo dir> <patrol_paths.json> [--write]"""
import argparse
import itertools
import json
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("demo")
ap.add_argument("plan")
ap.add_argument("--laps", type=int, default=1)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--max", type=int, default=120)
ap.add_argument("--step", type=int, default=2)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.demo)
import schedule_departures as sd  # noqa: E402

plan = json.load(open(a.plan))
ids = sorted(d["id"] for d in plan["drones"])
tl = {d["id"]: sd.timeline(d, a.laps, a.speed) for d in plan["drones"]}


def track(i, delay):
    path, start = tl[i]
    return [start] * int((sd.TAKEOFF + delay) / sd.DT) + path


def min_sep(delays):
    tr = {i: track(i, delays[i]) for i in ids}
    best = math.inf
    for i, j in itertools.combinations(ids, 2):
        A, B = tr[i], tr[j]
        for p, q in zip(A, B):                     # both airborne (zip stops when one has landed)
            if abs(p[2] - q[2]) < sd.DZ:
                best = min(best, math.hypot(p[0] - q[0], p[1] - q[1]))
    return best


cands = []
grid = range(0, a.max + 1, a.step)
for first in ids:
    others = [i for i in ids if i != first]
    for ds in itertools.product(grid, repeat=len(others)):
        delays = {first: 0, **dict(zip(others, ds))}
        cands.append((min_sep(delays), -max(delays.values()), delays))
cands.sort(key=lambda c: (c[0], c[1]), reverse=True)
sep, _, delays = cands[0]
print(f"best: delays {delays}, predicted min same-level separation {sep:.2f} m "
      f"(conflict radius of schedule_departures.py: {sd.RADIUS} m)")
print(f"all-zero delays: predicted min separation {min_sep({i: 0 for i in ids}):.2f} m")
if a.write:
    for d in plan["drones"]:
        d["depart_delay"] = delays[d["id"]]
    plan["departures"] = {"method": "max predicted same-level separation (schedule_max_separation.py; "
                                    "no conflict-free schedule exists for this layout)",
                          "predicted_min_separation_m": round(sep, 3), "laps": a.laps, "speed_mps": a.speed}
    json.dump(plan, open(a.plan, "w"), indent=2)
    print(f"wrote depart_delay into {a.plan}")
