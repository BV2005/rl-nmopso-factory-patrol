#!/usr/bin/env python3
"""Adds an NMOPSO run to the 'Preliminary results' chart of the patrol_drone benchmark (Auction, APF).

    python3 plot_benchmark_comparison.py <run_logs/<run>> [out.png]

NMOPSO bars are computed from the run's recorder files, with the benchmark chart's definitions:
  * distance flown in [start, end]: 3D path length of drone_states.csv (10 Hz) between the drone's
    TAKEOFF and DONE events (events.csv), samples with an implied jump > 15 m/s skipped;
  * battery consumed, time-based model: 0.01 % per 100 ms over the same window (the benchmark's
    model; PX4's own SITL battery is not used, see metrics_summary.py).
Auction and APF values are not recomputed here: their runs (run_20260929_214901, run_20260929_215220)
are not on this machine, so the values printed on the benchmark chart are used (BENCHMARK below).
NMOPSO flies a different mission (patrol of its whole zone in factory2, not the benchmark's 6 tasks).
"""
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# values printed on the benchmark chart (patrol_drone, n = 1): distance [m], battery [% points]
BENCHMARK = {
    'Auction': {'distance': [101, 88, 65], 'battery': [4.44, 4.44, 4.43], 'run': 'run_20260929_214901'},
    'APF':     {'distance': [34, 33, 31],  'battery': [4.48, 4.48, 4.48], 'run': 'run_20260929_215220'},
}
DRAIN_PCT_PER_S = 0.01 / 0.1          # benchmark's time-based battery model: 0.01 % per 100 ms
COLORS = ['#3a7bd5', '#e3662f', '#3cb27d']


def nmopso_values(run):
    events = list(csv.DictReader(open(run / 'events.csv')))
    window = {}
    for e in events:
        if e['source'] != 'follower' or e['event'] != 'phase':
            continue
        d = int(e['drone'])
        if e['detail'] == 'TAKEOFF':
            window.setdefault(d, [float(e['t']), None])
        elif e['detail'] == 'DONE' and d in window:
            window[d][1] = float(e['t'])
    track = defaultdict(list)
    for r in csv.DictReader(open(run / 'drone_states.csv')):
        track[int(r['drone'])].append((float(r['t']), float(r['x']), float(r['y']), float(r['z'])))
    dist, bat, span = [], [], []
    for d in sorted(window):
        t0, t1 = window[d]
        pts = [p for p in track[d] if t0 <= p[0] <= t1]
        s = 0.0
        for a, b in zip(pts, pts[1:]):
            step = math.dist(a[1:], b[1:])
            if b[0] > a[0] and step / (b[0] - a[0]) <= 15.0:
                s += step
        dist.append(s)
        bat.append(DRAIN_PCT_PER_S * (t1 - t0))
        span.append(t1 - t0)
    return dist, bat, span


def bars(ax, groups, key, fmt):
    w = 0.26
    for g, (name, vals) in enumerate(groups):
        for i, v in enumerate(vals[key]):
            x = g + (i - 1) * (w + 0.01)
            ax.bar(x, v, w, color=COLORS[i], label=f'D{i}' if g == 0 else None)
            ax.text(x, v, fmt(v), ha='center', va='bottom', fontsize=9, color='#333333')
    ax.set_xticks(range(len(groups)), [n for n, _ in groups], fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#eeeeee'); ax.set_axisbelow(True)


def main(run, out):
    run = Path(run)
    dist, bat, span = nmopso_values(run)
    groups = [(k, v) for k, v in BENCHMARK.items()] + \
             [('NMOPSO', {'distance': dist, 'battery': bat})]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    bars(a1, groups, 'distance', lambda v: f'{v:.0f}')
    a1.set_title('Per-UAV flight distance', loc='left', fontweight='bold', fontsize=12)
    a1.set_ylabel('distance flown in [start, end] (m)')
    a1.legend(frameon=False)
    bars(a2, groups, 'battery', lambda v: f'{v:.2f}')
    a2.set_title('Per-UAV battery consumed (time-based model)', loc='left', fontweight='bold', fontsize=12)
    a2.set_ylabel('battery consumed (% points)')
    fig.suptitle('Preliminary results — 3 UAVs, one run per algorithm (n = 1)', x=0.01, ha='left',
                 fontweight='bold', fontsize=13)
    fig.text(0.01, 0.01,
             f"{BENCHMARK['Auction']['run']} (auction), {BENCHMARK['APF']['run']} (APF): values as printed on the benchmark "
             f"chart (6-task benchmark scenario).\nNMOPSO: run {run.name} of branch nmopso-demo-factory2-voronoi — a different "
             f"mission: one full patrol lap of each drone's Voronoi zone in factory2 (take-off to landing: "
             f"{', '.join(f'{s:.0f}' for s in span)} s).\nBattery for all three: 0.01 % per 100 ms regardless of motion, "
             f"so it tracks mission time, not flight effort (NMOPSO recomputed with this model from its events.csv).",
             fontsize=8, color='#777777', va='bottom')
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=150)
    print(f'NMOPSO {run.name}: distance {[round(v, 1) for v in dist]} m, battery {[round(v, 2) for v in bat]} %, '
          f'window {[round(v, 1) for v in span]} s -> {out}')


if __name__ == '__main__':
    r = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(r, sys.argv[2] if len(sys.argv) > 2 else str(Path(r) / 'benchmark_with_nmopso.png'))
