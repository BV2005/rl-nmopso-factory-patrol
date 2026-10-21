#!/usr/bin/env python3
"""Departure delays from a timed conflict check, instead of hand-set take-off delays.

    python3 schedule_departures.py <patrol_paths.json> [--laps 1] [--speed 1.0] [--slack 3] [--write]

Each drone's plan (transit_in, the patrol loop LAPS times, transit_out) is walked at the patrol speed
after a nominal take-off, giving its position over time. Two drones conflict when they are both in the
air, closer than RADIUS metres horizontally and within DZ metres in height (the follower's yield rule).
Drones are scheduled in id order: each gets the smallest departure delay (1 s steps) for which it has
no conflict with any drone already scheduled, for every relative timing error in [-slack, +slack]
seconds (arming and take-off times vary by a few seconds between runs). --write stores the delays as
`depart_delay` in the plan. Standard library only.
"""
import argparse
import json
import math

RADIUS, DZ = 2.5, 1.0          # conflict: horizontal distance and height difference
DT = 0.2                       # time step of the walk (s)
TAKEOFF = 8.0                  # nominal s from arming to leaving the start (climb + settle)


def walk(points, speed):
    """positions every DT seconds along a polyline at constant speed"""
    out = [tuple(points[0])]
    for a, b in zip(points, points[1:]):
        n = max(1, int(math.dist(a, b) / (speed * DT)))
        out += [tuple(a[k] + (b[k] - a[k]) * i / n for k in range(3)) for i in range(1, n + 1)]
    return out


def timeline(d, laps, speed):
    """(positions from leaving the start, hover position before it) for one drone"""
    path = list(d.get('transit_in') or [])
    loop = d['waypoints']
    for _ in range(laps):
        path += loop if not path else loop[1:]
    tout = d.get('transit_out') or []
    path += tout[1:] if path and tout else tout
    start = (d['home'][0], d['home'][1], d.get('altitude', path[0][2]))
    return walk(path, speed), start


def conflict(a, b, da, db, slack):
    """True if drones a, b (timelines with departure delays da, db) come too close for any timing error"""
    (pa, sa), (pb, sb) = a, b
    na, nb = int((TAKEOFF + da) / DT), int((TAKEOFF + db) / DT)
    ta = [sa] * na + pa                      # hover over the start until departure, then the path
    tb = [sb] * nb + pb
    for shift in range(-int(slack / DT), int(slack / DT) + 1, max(1, int(0.5 / DT))):
        for i, p in enumerate(ta):
            j = i + shift
            if 0 <= j < len(tb):
                q = tb[j]
                if abs(p[2] - q[2]) < DZ and math.hypot(p[0] - q[0], p[1] - q[1]) < RADIUS:
                    return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('--laps', type=int, default=1)
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--slack', type=float, default=3.0)
    ap.add_argument('--max-delay', type=int, default=240)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    drones = sorted(plan['drones'], key=lambda d: d['id'])
    tl = {d['id']: timeline(d, a.laps, a.speed) for d in drones}
    delay = {}
    for d in drones:
        i = d['id']
        for dd in range(0, a.max_delay + 1):
            if not any(conflict(tl[i], tl[j], dd, delay[j], a.slack) for j in delay):
                delay[i] = dd
                break
        else:
            raise SystemExit(f'drone {i}: no conflict-free departure within {a.max_delay} s')
        print(f'drone {i}: depart_delay {delay[i]} s  (mission {len(tl[i][0]) * DT:.0f} s at {a.speed} m/s)')
    if a.write:
        for d in plan['drones']:
            d['depart_delay'] = delay[d['id']]
        plan['departures'] = {'method': 'timed conflict check (schedule_departures.py)', 'radius_m': RADIUS,
                              'dz_m': DZ, 'slack_s': a.slack, 'laps': a.laps, 'speed_mps': a.speed}
        json.dump(plan, open(a.plan, 'w'), indent=2)
        print(f'wrote depart_delay into {a.plan}')


if __name__ == '__main__':
    main()
