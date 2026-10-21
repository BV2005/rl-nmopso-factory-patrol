#!/usr/bin/env python3
"""Per-drone and swarm metrics from a recorded patrol run: python3 metrics_summary.py <run_logs/...>

Reads drone_states.csv and events.csv (patrol_recorder.py, 10 Hz) and patrol_paths.json, writes
metrics_summary.csv (one row per drone plus an 'all' row) and prints it as a table.

Validity filters, applied before any metric: a sample is dropped if its position message is older
than 0.5 s, if it implies a jump faster than 15 m/s from the drone's previous valid sample, or if
it lies outside the hall (0.5 m tolerance). Airborne = PX4 land detector says not landed.
Battery: PX4 publishes battery_status at 1 Hz, so the 10 Hz rows repeat it between updates;
energy is the time integral of voltage x current over the airborne samples, and only computed when
the current is valid. In PX4 SITL it is not: the battery simulator (SIM_BAT_DRAIN, default 60 s)
drains the percentage linearly with armed time down to SIM_BAT_MIN_PCT (default 50 %), derives the
voltage from it and reports current -1 ("no current sensor in simulation"). Battery % there
reflects armed time, not flight effort.
Standard library only.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

MAX_AGE_S = 0.5
MAX_JUMP_MPS = 15.0
DT = 0.1
LEVEL_DZ = 0.6          # drones closer than this vertically fly at the same level
PHASES = ['TAKEOFF', 'TRANSIT_IN', 'PATROL', 'TRANSIT_OUT', 'LAND']


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def rect_dist(rects, x, y):
    return min(math.hypot(max(r[0] - x, 0, x - r[1]), max(r[2] - y, 0, y - r[3])) for r in rects)


def main(run):
    run = Path(run)
    plan = json.load(open(run / 'patrol_paths.json'))
    hall = plan.get('hall', [-1e9, 1e9, -1e9, 1e9])
    rects = [o['rect'] for o in plan.get('obstacles', [])]
    homes = {d['id']: d['home'] for d in plan['drones']}
    rows = list(csv.DictReader(open(run / 'drone_states.csv')))
    events = list(csv.DictReader(open(run / 'events.csv')))

    # validity filters
    valid, dropped, last = defaultdict(list), defaultdict(int), {}
    for r in rows:
        i = int(r['drone'])
        t, x, y, z = fnum(r['t']), fnum(r['x']), fnum(r['y']), fnum(r['z'])
        bad = fnum(r['pos_age_s']) > MAX_AGE_S
        bad |= not (hall[0] - 0.5 <= x <= hall[1] + 0.5 and hall[2] - 0.5 <= y <= hall[3] + 0.5)
        if i in last and not bad:
            tp, p = last[i]
            bad |= t > tp and math.dist(p, (x, y, z)) / (t - tp) > MAX_JUMP_MPS
        if bad:
            dropped[i] += 1
            continue
        last[i] = (t, (x, y, z))
        valid[i].append(r)

    out = []
    for i in sorted(homes):
        v = valid[i]
        air = [r for r in v if r['landed'] == '0']
        m = {'drone': i, 'samples_valid': len(v), 'samples_dropped': dropped[i]}
        if not air:
            out.append(m)
            continue
        t = [fnum(r['t']) for r in air]
        p = [(fnum(r['x']), fnum(r['y']), fnum(r['z'])) for r in air]
        dist = sum(math.dist(a, b) for a, b in zip(p, p[1:]))
        m['flight_time_s'] = t[-1] - t[0]
        m['distance_m'] = dist
        m['mean_speed_mps'] = dist / max(t[-1] - t[0], 1e-9)
        m['max_speed_mps'] = max(fnum(r['speed']) for r in air)
        for ph in PHASES:
            m[f'time_{ph.lower()}_s'] = DT * sum(r['phase'] == ph for r in v)
        m['max_tilt_deg'] = max(max(abs(fnum(r['roll_deg'])), abs(fnum(r['pitch_deg']))) for r in air)
        high = [q for q in p if q[2] > 0.5]
        if rects and high:
            m['min_obstacle_clearance_m'] = min(rect_dist(rects, q[0], q[1]) for q in high)
        bat = [(fnum(r['t']), fnum(r['battery_pct']), fnum(r['voltage_v']), fnum(r['current_a']),
                fnum(r['discharged_mah'])) for r in air]
        bat = [b for b in bat if not math.isnan(b[1])]
        if bat:
            m['battery_start_pct'], m['battery_end_pct'] = bat[0][1], bat[-1][1]
            m['battery_used_pct'] = bat[0][1] - bat[-1][1]
            m['voltage_start_v'], m['voltage_end_v'] = bat[0][2], bat[-1][2]
            # current and discharged mAh are -1 when invalid (PX4 BatteryStatus); PX4's SITL battery
            # simulator never measures current, so energy stays blank there instead of being invented
            if all(b[3] >= 0 for b in bat):        # discharged mAh is integrated from the current too
                wh = sum(0.5 * (a[2] * a[3] + b[2] * b[3]) * (b[0] - a[0]) for a, b in zip(bat, bat[1:])) / 3600
                m['energy_wh'] = wh
                m['energy_j_per_m'] = wh * 3600 / max(dist, 1e-9)
                m['mean_current_a'] = sum(b[3] for b in bat) / len(bat)
                if bat[0][4] >= 0 and bat[-1][4] >= 0:
                    m['discharged_mah'] = bat[-1][4] - bat[0][4]
        ev = [e for e in events if e['drone'] == str(i)]
        m['laps'] = sum(e['event'] == 'lap_done' for e in ev)
        m['holds'] = sum(e['event'] == 'hold_start' for e in ev)
        m['hold_time_s'] = sum(fnum(e['detail'].split()[0]) for e in ev if e['event'] == 'hold_end')
        m['failsafe_events'] = sum(e['event'] == 'failsafe' and e['detail'].endswith('True') for e in ev)
        out.append(m)

    # swarm: separation from the synchronous 10 Hz rows (every sample tick writes all drones)
    by_t = defaultdict(dict)
    for i, v in valid.items():
        for r in v:
            if r['landed'] == '0':
                by_t[r['t']][i] = (fnum(r['x']), fnum(r['y']), fnum(r['z']))
    near_pad = lambda q: q[2] < 1.0 or min(math.dist(q[:2], h) for h in homes.values()) < 1.0
    hsep, sep3, lsep, hall_sep = [], [], [], []
    for pts in by_t.values():
        ids = sorted(pts)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                p, q = pts[ids[a]], pts[ids[b]]
                h = math.dist(p[:2], q[:2])
                hall_sep.append(h)
                if near_pad(p) or near_pad(q):
                    continue
                hsep.append(h)
                sep3.append(math.dist(p, q))
                if abs(p[2] - q[2]) < LEVEL_DZ:
                    lsep.append(h)
    per = [m for m in out if 'distance_m' in m]
    tot = {'drone': 'all', 'samples_valid': sum(m['samples_valid'] for m in out),
           'samples_dropped': sum(m['samples_dropped'] for m in out)}
    if per:
        tot['flight_time_s'] = max(m['flight_time_s'] for m in per)
        tot['distance_m'] = sum(m['distance_m'] for m in per)
        tot['distance_balance'] = max(m['distance_m'] for m in per) / max(min(m['distance_m'] for m in per), 1e-9)
        for k in ('energy_wh', 'discharged_mah', 'laps', 'holds', 'hold_time_s', 'failsafe_events'):
            if all(k in m for m in per):
                tot[k] = sum(m[k] for m in per)
        if 'energy_wh' in tot:
            tot['energy_j_per_m'] = tot['energy_wh'] * 3600 / tot['distance_m']
    if hall_sep:
        tot['min_sep_horizontal_all_m'] = min(hall_sep)
    if hsep:
        tot['min_sep_horizontal_m'] = min(hsep)
        tot['min_sep_3d_m'] = min(sep3)
    if lsep:
        tot['min_sep_same_level_m'] = min(lsep)
    out.append(tot)

    cols = ['drone', 'flight_time_s', 'distance_m', 'mean_speed_mps', 'max_speed_mps'] + \
           [f'time_{ph.lower()}_s' for ph in PHASES] + \
           ['laps', 'holds', 'hold_time_s', 'battery_start_pct', 'battery_end_pct', 'battery_used_pct',
            'voltage_start_v', 'voltage_end_v', 'discharged_mah', 'energy_wh', 'energy_j_per_m', 'mean_current_a', 'max_tilt_deg',
            'min_obstacle_clearance_m', 'failsafe_events', 'distance_balance', 'min_sep_horizontal_all_m',
            'min_sep_horizontal_m', 'min_sep_3d_m', 'min_sep_same_level_m', 'samples_valid', 'samples_dropped']
    fmt = lambda v: f'{v:.3f}' if isinstance(v, float) else v
    with open(run / 'metrics_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in out:
            w.writerow([fmt(m.get(c, '')) for c in cols])
    # console table: metrics as rows, drones as columns
    print(f'== metrics summary: {run.name}  (drone_states.csv at 10 Hz; separation: airborne, away from the pads)')
    if rows and all(fnum(r['current_a']) < 0 for r in rows if r['current_a'] not in ('', 'nan')):
        print('   battery: PX4 SITL simulator (time-based drain, no current sensor) - no energy figures')
    print(f"{'metric':28s}" + ''.join(f'{str(m["drone"]):>12s}' for m in out))
    for c in cols[1:]:
        if any(c in m for m in out):
            print(f'{c:28s}' + ''.join(f'{fmt(m[c]) if c in m else "":>12}' for m in out))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '.')
