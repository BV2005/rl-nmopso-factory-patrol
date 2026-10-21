#!/usr/bin/env python3
"""Departure delays like the demo's schedule_departures.py (its walk / timeline / conflict functions,
imported unchanged), but trying every order in which the drones are scheduled instead of id order:
with a shared take-off area, a drone whose transit passes over another drone's pad must leave first.
Picks the order with the smallest largest delay (ties: smallest total delay).
  python3 schedule_any_order.py <demo dir> <patrol_paths.json> [--laps 1] [--speed 1.0] [--slack 3] [--write]
Standard library only (runs in the demo's WSL environment)."""
import argparse
import itertools
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("demo")
ap.add_argument("plan")
ap.add_argument("--laps", type=int, default=1)
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--slack", type=float, default=3.0)
ap.add_argument("--max-delay", type=int, default=240)
ap.add_argument("--write", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.demo)
import schedule_departures as sd  # noqa: E402

plan = json.load(open(a.plan))
tl = {d["id"]: sd.timeline(d, a.laps, a.speed) for d in plan["drones"]}
best = None
for order in itertools.permutations(sorted(tl)):
    delay = {}
    for i in order:
        for dd in range(0, a.max_delay + 1):
            if not any(sd.conflict(tl[i], tl[j], dd, delay[j], a.slack) for j in delay):
                delay[i] = dd
                break
        else:
            delay = None
            break
    if delay is None:
        print(f"order {order}: no conflict-free schedule within {a.max_delay} s")
        continue
    key = (max(delay.values()), sum(delay.values()))
    print(f"order {order}: delays {delay}")
    if best is None or key < best[0]:
        best = (key, order, delay)
if best is None:
    raise SystemExit("no order gives a conflict-free schedule")
_, order, delay = best
print(f"chosen order {order}: " + ", ".join(f"drone {i} {delay[i]} s (mission {len(tl[i][0]) * sd.DT:.0f} s)" for i in sorted(delay)))
if a.write:
    for d in plan["drones"]:
        d["depart_delay"] = delay[d["id"]]
    plan["departures"] = {"method": "timed conflict check, every scheduling order (schedule_any_order.py using "
                                    "schedule_departures.py)", "order": list(order), "radius_m": sd.RADIUS, "dz_m": sd.DZ,
                          "slack_s": a.slack, "laps": a.laps, "speed_mps": a.speed}
    json.dump(plan, open(a.plan, "w"), indent=2)
    print(f"wrote depart_delay into {a.plan}")
