#!/usr/bin/env python3
"""Top-down factory map with the paths the drones actually flew in one run.

    python3 plot_flown_paths.py <run_logs/<run>> [out.png]

Map: obstacles, heights and names from worlds/factory.sdf (collision boxes; floor, walls, ceiling
and floor markings left out), drawn like the patrol_drone benchmark map. Paths: drone_states.csv
(patrol_recorder.py, 10 Hz) of the run, airborne samples only; solid in the 1.6-2.0 m patrol band,
dotted in the 2.9-3.3 m transit layer. For reference, the benchmark's UAV start positions and tasks
T1-T6 are drawn faintly: their coordinates are read off the benchmark map image (tasks/benchmark.yaml
is not on this machine); the NMOPSO drones neither start there nor visit those tasks.
"""
import csv
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

COLORS = ['#3a7bd5', '#e3662f', '#3cb27d']
TRANSIT_Z = 2.5
# benchmark reference, read off the patrol_drone map image
BENCH_STARTS = [(-8.0, 7.0), (-4.0, 7.0), (0.0, 7.0)]
BENCH_TASKS = [('T1', -10.8, -1.5, 4.0), ('T2', -8.2, 2.0, 4.0), ('T3', -5.8, -4.0, 4.0),
               ('T4', 4.0, 5.0, 3.5), ('T5', 10.0, 4.0, 3.5), ('T6', 7.0, -4.0, 3.5)]
LABELS = {'loading_dock': 'Loading dock', 'central_partition': 'Partition', 'cnc_machine': 'CNC',
          'lathe_machine': 'Lathe', 'assembly_workbench': 'Workbench', 'conveyor_belt': 'Conveyor',
          'packing_station': 'Packing', 'shipping_crate_stack': 'Crates'}


def sdf_boxes(sdf):
    """[(name, xmin, xmax, ymin, ymax, top)] of the obstacle boxes (yaw-free world assumed)"""
    out = []
    for m in ET.parse(sdf).getroot().iter('model'):
        name = m.get('name')
        pose = [float(v) for v in (m.findtext('pose') or '0 0 0 0 0 0').split()]
        box = m.find('.//collision/geometry/box/size')
        if box is None:
            continue
        sx, sy, sz = (float(v) for v in box.text.split())
        top = pose[2] + sz / 2
        if sz <= 0.05 or 'wall' in name or name in ('floor', 'ceiling', 'ceiling_barrier') or top > 10:
            continue
        out.append((name, pose[0] - sx / 2, pose[0] + sx / 2, pose[1] - sy / 2, pose[1] + sy / 2, top))
    return out


def main(run, out):
    run = Path(run)
    sdf = Path(__file__).parent / 'worlds' / 'factory.sdf'
    tracks = defaultdict(list)
    for r in csv.DictReader(open(run / 'drone_states.csv')):
        if r['landed'] == '0':
            tracks[int(r['drone'])].append((float(r['x']), float(r['y']), float(r['z'])))

    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.add_patch(Rectangle((-15, -10), 30, 20, fill=False, lw=6, ec='#4a4a4a'))
    for name, x0, x1, y0, y1, top in sdf_boxes(sdf):
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='#e6e6e2', ec='#b5b5b0', lw=1, zorder=1))
        if name.startswith('rack_row'):
            continue
        label = f'{LABELS.get(name, name)} ({top:g} m)'
        if name == 'central_partition':
            ax.text(x1 + 0.15, y1 - 0.2, label, rotation=90, va='top', fontsize=8, color='#888888')
        else:
            ax.text((x0 + x1) / 2, y0 - 0.25, label, ha='center', va='top', fontsize=8, color='#888888')
    ax.text(-9.2, -6.3, 'Rack rows 1–4 (3.5 m)', ha='center', va='top', fontsize=8, color='#888888')

    # benchmark reference (faint)
    for i, (x, y) in enumerate(BENCH_STARTS):
        ax.plot(x, y, '^', ms=8, mfc='none', mec=COLORS[i], alpha=0.45, zorder=2)
        ax.text(x, y + 0.35, f'D{i} bench start', ha='center', fontsize=7, color='#999999')
    for t, x, y, h in BENCH_TASKS:
        ax.plot(x, y, 'o', ms=11, mfc='none', mec='#aaaaaa', mew=1.2, zorder=2)
        ax.plot(x, y, 'o', ms=3, color='#aaaaaa', zorder=2)
        ax.text(x + 0.35, y, f'{t}', va='center', fontsize=8, color='#999999')

    # flown paths
    for d in sorted(tracks):
        c = COLORS[(d - 1) % 3]
        P = tracks[d]
        for (a, b) in zip(P, P[1:]):
            ls = ':' if max(a[2], b[2]) > TRANSIT_Z else '-'
            ax.plot([a[0], b[0]], [a[1], b[1]], ls, color=c, lw=1.8, zorder=4, solid_capstyle='round')
        ax.plot(P[0][0], P[0][1], '^', ms=10, color=c, mec='k', zorder=5)
        ax.plot([], [], '-', color=c, lw=2.5,
                label=f'D{d - 1} (drone {d}): {sum(((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2) ** 0.5 for a, b in zip(P, P[1:])):.0f} m flown')
    ax.plot([], [], 'k:', lw=1.8, label='transit layer (2.9–3.3 m)')
    ax.plot([], [], '^', color='#777777', mec='k', ms=9, lw=0, label='NMOPSO launch pads')
    ax.plot([], [], 'o', mfc='none', mec='#aaaaaa', ms=10, lw=0, label='benchmark tasks / starts (reference)')

    ax.set_xlim(-15.8, 15.8); ax.set_ylim(-10.8, 10.8); ax.set_aspect('equal')
    ax.set_xlabel('x (m, Gazebo east)'); ax.set_ylabel('y (m, Gazebo north)')
    ax.grid(color='#f0f0f0'); ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_title(f'Factory world (30 m × 20 m, top-down) — paths flown by the NMOPSO drones, run {run.name}',
                 loc='left', fontweight='bold', fontsize=12)
    fig.text(0.01, 0.01, 'Paths: drone_states.csv (10 Hz, airborne samples), one patrol lap per drone. '
             'Faint markers: Auction / APF benchmark starts and tasks, read off the benchmark map (not part of the NMOPSO mission).',
             fontsize=8, color='#777777')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    r = sys.argv[1] if len(sys.argv) > 1 else '.'
    main(r, sys.argv[2] if len(sys.argv) > 2 else str(Path(r) / 'flown_paths.png'))
