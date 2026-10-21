#!/usr/bin/env python3
"""Checks a finished NMOPSO patrol run: python3 check_patrol.py <run_logs/...>

Per drone (PATROL phase): cross-track error from the NMOPSO polyline, closest approach to any
obstacle box, altitude range, lap time; the same for the transit legs of a shared launch area.
Swarm: minimum horizontal distance between any two drones at the same time (overall, and with
both drones airborne outside the launch area), the minimum 3D distance and the minimum horizontal
distance between drones at similar altitude (a transit layer passes over other drones' zones),
and the share of the free factory floor that came
within the 2 m sensor reach of a patrol trace. Obstacles and hall come from the plan JSON when it
has them (plans for other worlds), else the patrol_drone factory below. Standard library only.
"""
import csv
import json
import math
import sys
from pathlib import Path

# obstacle footprints of sim/worlds/factory.sdf (world ENU, metres): xmin xmax ymin ymax
RECTS = [(-10.5, -9.5, -8.0, -2.0), (-10.5, -9.5, 2.0, 8.0), (4.5, 5.5, -8.0, -2.0), (4.5, 5.5, 2.0, 8.0),
         (12.0, 16.0, -12.0, -8.0), (12.0, 16.0, 8.0, 12.0), (-16.5, -13.5, 8.5, 11.5),
         (-6.6, -5.4, -10.6, -9.4), (8.4, 9.6, 3.4, 4.6)]
HALL = (-20.0, 20.0, -15.0, 15.0)
SENSOR = 2.0
LEVEL_DZ = 0.6     # drones closer than this vertically count as flying at the same level


def load_world(plan):
    global RECTS, HALL
    if 'obstacles' in plan:
        RECTS = [tuple(o['rect']) for o in plan['obstacles']]
        HALL = tuple(plan['hall'])


def rect_dist(x, y):
    return min(math.hypot(max(r[0] - x, 0, x - r[1]), max(r[2] - y, 0, y - r[3])) for r in RECTS)


def seg_dist(p, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    L2 = ex * ex + ey * ey
    t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / L2))
    return math.hypot(a[0] + t * ex - p[0], a[1] + t * ey - p[1])


def main(run):
    run = Path(run)
    plan = json.load(open(run / 'patrol_paths.json'))
    load_world(plan)
    traces = {}
    for d in plan['drones']:
        f = run / f"trace_drone{d['id']}.csv"
        if f.exists():
            traces[d['id']] = [r for r in csv.DictReader(open(f))]
    out = [f'== NMOPSO patrol check: {run.name}']
    patrol_pts = []
    for d in plan['drones']:
        tr = traces.get(d['id'], [])
        for phase, key in (('TRANSIT_IN', 'transit_in'), ('PATROL', 'waypoints'), ('TRANSIT_OUT', 'transit_out')):
            rows = [r for r in tr if r['phase'] == phase]
            wp = [tuple(p) for p in (d.get(key) or [])]
            if len(wp) < 2:
                continue
            if not rows:
                out.append(f"drone {d['id']}: no {phase} samples")
                continue
            pts = [(float(r['x']), float(r['y']), float(r['z'])) for r in rows]
            xte = [min(seg_dist(p, a, b) for a, b in zip(wp, wp[1:])) for p in pts]
            clr = min(rect_dist(p[0], p[1]) for p in pts)
            t = [float(r['t']) for r in rows]
            what = f"{max(int(r['lap']) for r in tr)} lap(s) completed" if phase == 'PATROL' else 'leg'
            out.append(f"drone {d['id']} {phase.lower():11s}: {t[-1] - t[0]:4.0f} s, {what}; cross-track error "
                       f"mean {sum(xte) / len(xte):.2f} m, max {max(xte):.2f} m; closest obstacle {clr:.2f} m; "
                       f"altitude {min(p[2] for p in pts):.2f}-{max(p[2] for p in pts):.2f} m")
            if phase == 'PATROL':
                patrol_pts += [(p[0], p[1]) for p in pts]
    # inter-drone separation, samples matched at 0.5 s resolution
    buckets = {}
    for i, rows in traces.items():
        for r in rows:
            if r['phase'] in ('TAKEOFF', 'TRANSIT_IN', 'PATROL', 'TRANSIT_OUT', 'LAND'):
                buckets.setdefault(round(float(r['t']) * 2), {})[i] = (float(r['x']), float(r['y']), float(r['z']))
    homes = [tuple(d['home']) for d in plan['drones']]
    near_pads = lambda p: min(math.dist(p[:2], h) for h in homes) < 1.0 or p[2] < 1.0
    sep = [(math.dist(a[:2], b[:2]), near_pads(a) or near_pads(b), math.dist(a, b), abs(a[2] - b[2]))
           for b_ in buckets.values() for i, a in b_.items() for j, b in b_.items() if i < j]
    if sep:
        out.append(f'swarm: minimum horizontal separation between drones {min(s[0] for s in sep):.2f} m (whole flight)')
        away = [s for s in sep if not s[1]]
        if away:
            out.append(f'swarm: minimum horizontal separation away from the launch pads {min(s[0] for s in away):.2f} m')
            out.append(f'swarm: minimum 3D separation away from the launch pads {min(s[2] for s in away):.2f} m')
            level = [s[0] for s in away if s[3] < LEVEL_DZ]
            if level:
                out.append(f'swarm: minimum horizontal separation at similar altitude (|dz| < {LEVEL_DZ} m) '
                           f'away from the launch pads {min(level):.2f} m')
    # coverage of the free floor by the flown traces
    g = 0.5
    free = cov = 0
    pts = patrol_pts[::5]
    x = HALL[0] + g / 2
    while x < HALL[1]:
        y = HALL[2] + g / 2
        while y < HALL[3]:
            if rect_dist(x, y) > 0:
                free += 1
                if any((px - x) ** 2 + (py - y) ** 2 <= SENSOR ** 2 for px, py in pts):
                    cov += 1
            y += g
        x += g
    out.append(f'coverage: {100 * cov / max(free, 1):.1f} % of the free floor came within {SENSOR:.0f} m of a drone')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
