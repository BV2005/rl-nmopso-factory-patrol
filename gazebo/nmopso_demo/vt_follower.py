#!/usr/bin/env python3
"""Flies one x500_patrol drone with its PATROL legs planned online by vT (rlpatrol/env_t.py), from its sensors.

patrol_follower.py flies tracks planned offline on the known map; sensor_follower.py plans NMOPSO nodes with vS.
Here the transits keep their offline (pad-safe, deconflicted) tracks and every PATROL leg between two consecutive
patrol points of the plan is flown one vT waypoint at a time, planned from what the drone senses
(rlpatrol/vt_pilot.py):

  at the leg start  the drone turns in place to face the leg's goal (vT's first heading)
  at a waypoint     once it is within arrive_dist and has settled for `settle` s, facing the last segment's
                    direction (the heading the policy assumes): the lidar returns (/droneN/scan) and the levelled
                    depth points (/droneN/depth/points) give the flight scan (scan3d.flight_scan: ranges to the
                    zone box_margin around every return, obstacle-height channel), the policy (deterministic) picks
                    the next waypoint; VTPilot.decide() checks it against the scan and falls back to the policy's
                    samples, an action sweep, then a backtrack. If nothing is clear the drone hovers and rescans;
                    after max_block s it flies straight to the leg's goal if the scan shows that clear, else the
                    leg is skipped (logged)
  the last step     the final approach to the goal (or, out of decisions, a straight segment to it if clear)

The carrot of patrol_follower.py flies every segment at `speed` m/s (1.0 = vT's V_CRUISE) facing the segment.
Logs, next to <log_file>:
  <log>_vt.csv       every decision: pose, scan (27 ranges, 27 heights), candidate source / tries, mode, target
  <log>_vt_legs.csv  every leg: outcome, decisions, backtracks, climbs, the time model's flight time and the flight
                     time measured in the simulator (PX4 timestamps: sim time) from the first decision to arrival
                     at the goal, the time spent waiting (settling, turning, blocked)
  <log>_track.json   the PATROL track as flown (for check_patrol.py)
Summarise a run with scripts/vt_flight_report.py. Needs PyTorch and the repository on PYTHONPATH
(run_demo.sh VT_CKPT=... sets it up)."""
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from px4_msgs.msg import VehicleAttitude
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # the repository root
import torch  # noqa: E402

from patrol_follower import PatrolFollower  # noqa: E402
from rlpatrol import sensors as S  # noqa: E402
from rlpatrol.scan3d import flight_scan  # noqa: E402
from rlpatrol.vt_pilot import MODES, VTPilot  # noqa: E402


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class VTFollower(PatrolFollower):
    def __init__(self):
        super().__init__()
        for name, default in (('ckpt', ''), ('pref', ''), ('scan_topic', ''), ('cloud_topic', ''), ('cloud_frame', 'flu'),
                              ('cloud_stride', 1), ('radius', 0.1), ('samples', 16), ('zmax_local', 2.6),
                              ('arrive_dist', 0.25), ('settle', 0.3), ('yaw_tol', 0.15), ('max_scan_age', 0.5),
                              ('max_block', 10.0), ('use_depth', True)):
            self.declare_parameter(name, default)
        P = lambda k: self.get_parameter(k).value                          # noqa: E731
        torch.set_num_threads(1)
        with open(self.get_parameter('paths_file').value) as f:
            plan = json.load(f)
        hall = plan.get('hall') or json.loads((Path(__file__).resolve().parents[2] / 'data' / 'factory2_model.json').read_text())['hall']
        pref = [float(v) for v in str(P('pref')).split(',')] if P('pref') else [1 / 3] * 3
        self.pilot = VTPilot(P('ckpt'), hall, pref=pref, zmax_local=float(P('zmax_local')), radius=float(P('radius')),
                             samples=int(P('samples')), seed=self.id)
        self.arrive, self.settle, self.yaw_tol = float(P('arrive_dist')), float(P('settle')), float(P('yaw_tol'))
        self.max_age, self.max_block = float(P('max_scan_age')), float(P('max_block'))
        self.cloud_frame, self.use_depth, self.stride = P('cloud_frame'), bool(P('use_depth')), max(1, int(P('cloud_stride')))

        me = next(d for d in plan['drones'] if d['id'] == self.id)
        pts = [tuple(float(v) for v in p) for p in me['patrol_points']]
        if math.dist(pts[0][:2], pts[-1][:2]) >= 0.05:                    # close the loop (transit_out starts at pts[0])
            pts.append(pts[0])
        self.legs = [(a, b) for a, b in zip(pts, pts[1:]) if math.dist(a[:2], b[:2]) >= 0.05]
        self.tracks['PATROL'] = self.make_track([pts[0], pts[0]])          # grows while it is flown
        self.set_track('PATROL')
        self.reset_loop()

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        ns = f'/drone{self.id}'
        self.lidar_xy = self.cloud = None
        self.t_lidar = self.t_depth = -1.0
        self.att = None
        self.create_subscription(LaserScan, P('scan_topic') or f'{ns}/scan', self.on_scan, qos)
        if self.use_depth:
            self.create_subscription(PointCloud2, P('cloud_topic') or f'{ns}/depth/points', self.on_cloud, qos)
        self.create_subscription(VehicleAttitude, f'{ns}/fmu/out/vehicle_attitude', lambda m: setattr(self, 'att', m), qos)

        log = self.get_parameter('log_file').value or f'/tmp/nmopso_drone{self.id}.csv'
        base = str(Path(log).with_suffix(''))
        self.dec_f = open(base + '_vt.csv', 'w', newline='')
        self.dec = csv.writer(self.dec_f)
        self.dec.writerow(['t', 'sim_t', 'leg', 'decision', 'x', 'y', 'z', 'yaw', 'outcome', 'source', 'tries', 'mode', 'snap',
                           'tx', 'ty', 'tz', 'a0', 'a1', 'a2', 'lidar_age', 'depth_age',
                           *[f'r{k}' for k in range(S.SCAN_BINS)], *[f'h{k}' for k in range(S.SCAN_BINS)]])
        self.leg_f = open(base + '_vt_legs.csv', 'w', newline='')
        self.legw = csv.writer(self.leg_f)
        self.legw.writerow(['lap', 'leg', 'ax', 'ay', 'az', 'bx', 'by', 'bz', 'outcome', 'decisions', 'backtracks', 'climbs',
                            'fallbacks', 'model_time_s', 'flown_time_s', 'straight_time_s', 'wait_s', 'blocked_s'])
        self.get_logger().info(f'drone {self.id}: vT checkpoint {P("ckpt")} (it {self.pilot.it}), {len(self.legs)} patrol '
                               f'legs planned online from the sensors, pref {pref}, margin {self.pilot.margin} m')

    # --- sensors --------------------------------------------------------------------------------------------
    def on_scan(self, m):
        self.lidar_xy = S.laserscan_points(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max)
        self.t_lidar = time.monotonic()

    def on_cloud(self, m):
        p = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'), skip_nans=True).astype(float)
        if self.cloud_frame == 'optical':                                  # z forward, x right, y down -> FLU
            p = np.c_[p[:, 2], -p[:, 0], -p[:, 1]]
        self.cloud = p[::self.stride]
        self.t_depth = time.monotonic()

    def roll_pitch(self):
        """roll and pitch (rad) of the body in the FLU convention, from PX4's FRD attitude."""
        if self.att is None:
            return 0.0, 0.0
        w, x, y, z = (float(v) for v in self.att.q)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, -pitch                                                # FRD pitch up is negative in FLU

    def yaw_enu(self):
        return math.pi / 2 - float(self.lpos.heading)                      # PX4 heading: NED yaw

    def sim_s(self):
        return float(self.lpos.timestamp) * 1e-6                          # PX4 time (sim time in SITL)

    def scan(self, now, pos):
        """the flight scan (ranges, heights, full-resolution beams) at the drone's pose, or None while a sensor is stale."""
        if self.lidar_xy is None or now - self.t_lidar > self.max_age:
            return None
        depth = None
        if self.use_depth:
            if self.cloud is None or now - self.t_depth > self.max_age:
                return None
            depth = S.level(self.cloud, *self.roll_pitch())
        return flight_scan(self.lidar_xy, depth, drone_z=pos[2], margin=self.pilot.margin, yaw=self.yaw_enu(), beams=True)

    # --- the online PATROL track --------------------------------------------------------------------------------
    def reset_loop(self):
        self.leg_i, self.leg_on, self.loop_done = 0, False, False
        self.arrived = self.blocked_since = None
        self.lap_seen = self.lap

    def append(self, p):
        p = tuple(float(v) for v in p)
        self.cum.append(self.cum[-1] + math.dist(self.wp[-1], p))
        self.wp.append(p)
        self.loop_len = self.cum[-1]

    def start_leg(self):
        a, b = self.wp[-1], self.legs[self.leg_i][1]                     # from where the flown track ends
        self.pilot.start_leg(a, b)
        self.leg_on, self.leg_a, self.leg_b = True, a, b
        self.t_first = None                                                # sim time of the first decision
        self.fallbacks, self.wait_s, self.blocked_s, self.n_dec = 0, 0.0, 0.0, 0
        self.end_outcome = None
        d = np.array(b[:2]) - np.array(a[:2])                              # turn in place to face the goal:
        d = d / max(np.linalg.norm(d), 1e-9)                               # a 1 cm step sets the setpoint yaw
        self.append((a[0] + 0.01 * d[0], a[1] + 0.01 * d[1], a[2]))
        self.emit('leg_start', f'{self.leg_i}: {a[:2]} -> {b[:2]}')

    def finish_leg(self, outcome):
        rep = self.pilot.report()
        flown = (self.sim_s() - self.t_first) if self.t_first is not None else float('nan')
        self.legw.writerow([self.lap, self.leg_i, *(f'{v:.3f}' for v in self.leg_a), *(f'{v:.3f}' for v in self.leg_b), outcome,
                            rep['decisions'], rep['backtracks'], rep['climbs'], self.fallbacks, f"{rep['model_time_s']:.2f}",
                            f'{flown:.2f}', f"{rep['straight_time_s']:.2f}", f'{self.wait_s:.2f}', f'{self.blocked_s:.2f}'])
        self.leg_f.flush()
        self.emit('leg_done', f'{self.leg_i}: {outcome}')
        self.get_logger().info(f'leg {self.leg_i}: {outcome}, {rep["decisions"]} decisions, model {rep["model_time_s"]:.1f} s, '
                               f'flown {flown:.1f} s (straight {rep["straight_time_s"]:.1f} s)')
        self.leg_i += 1
        self.leg_on = False

    def log_decision(self, now, pos, outcome, d, scan):
        r, h = scan[:2] if scan is not None else (np.full(S.SCAN_BINS, np.nan), np.full(S.SCAN_BINS, np.nan))
        tgt = d.target if d is not None else (math.nan,) * 3
        a = d.action[0].tolist() if d is not None else [math.nan] * 3
        self.dec.writerow([f'{now - self.t0:.2f}', f'{self.sim_s():.2f}', self.leg_i, self.n_dec, *(f'{v:.3f}' for v in pos),
                           f'{self.yaw_enu():.3f}', outcome, d.source if d else '', d.tries if d else 0,
                           MODES[d.mode] if d else '', int(d.snap) if d else 0, *(f'{v:.3f}' for v in tgt),
                           *(f'{v:.4f}' for v in a), f'{now - self.t_lidar:.2f}', f'{now - self.t_depth:.2f}',
                           *(f'{v:.4f}' for v in r), *(f'{v:.4f}' for v in h)])

    def give_up(self, now, pos, scan, why):
        """no way on (blocked for max_block s, or out of decisions): straight to the goal if the scan shows it clear,
        else skip the leg (the next one starts where the drone is)."""
        if scan is not None and self.pilot.clear(self.leg_b, *scan):
            self.append(self.leg_b)
            self.end_outcome = f'{why}_straight'
            self.fallbacks += 1
        else:
            self.log_decision(now, pos, f'{why}_skipped', None, scan)
            self.finish_leg(f'{why}_skipped')
        self.get_logger().warn(f'leg {self.leg_i}: {why} -> {self.end_outcome or "skipped"}')

    def extend_track(self, now, pos):
        if self.phase != 'PATROL':
            return False
        if self.lap != self.lap_seen:                                      # a new lap: plan it afresh
            self.tracks['PATROL'] = self.make_track([self.wp[-1], self.wp[-1]])
            self.set_track('PATROL')
            self.reset_loop()
        if self.loop_done:
            return False
        self.s = min(self.s, self.loop_len)
        if math.dist(pos, self.wp[-1]) > self.arrive:                     # still flying to the current end
            self.arrived = None
            return True
        if self.arrived is None:
            self.arrived = now
            self.sim_arrived = self.sim_s()
        if not self.leg_on:
            if self.leg_i >= len(self.legs):
                self.loop_done = True
                return False
            self.start_leg()
            self.arrived = None
            return True
        if self.end_outcome is not None:                                   # arrived at the goal by a fallback segment
            self.finish_leg(self.end_outcome)
            return True
        if self.pilot.over:
            if self.pilot.success:                                         # arrived at the goal
                self.finish_leg('success')
                return True
        settled = now - self.arrived >= self.settle
        facing = abs(wrap(self.yaw_enu() - self.pilot.heading)) <= self.yaw_tol or now - self.arrived > 3.0
        if not (settled and facing):
            return True                                                    # settling / still turning to the heading
        scan = self.scan(now, pos)
        if scan is None:
            return True                                                    # hover until the sensors are fresh
        if self.pilot.over:                                                # out of decisions, not at the goal
            self.give_up(now, pos, scan, 'timeout')
            return True
        d = self.pilot.decide(*scan)
        if d is None:
            if self.blocked_since is None:
                self.blocked_since = now
                self.log_decision(now, pos, 'blocked', None, scan)
            if now - self.blocked_since >= self.max_block:
                self.blocked_s += now - self.blocked_since
                self.blocked_since = None
                self.give_up(now, pos, scan, 'blocked')
            return True
        if self.blocked_since is not None:
            self.blocked_s += now - self.blocked_since
            self.blocked_since = None
        if self.t_first is None:
            self.t_first = self.sim_s()
        else:
            self.wait_s += self.sim_s() - self.sim_arrived                 # hovering at the waypoint before deciding
        self.fallbacks += d.source != 'policy'
        target = self.pilot.commit(d)
        self.n_dec += 1
        self.append(target)
        self.log_decision(now, pos, 'ok', d, scan)
        self.arrived = None
        return True


def main():
    rclpy.init()
    node = VTFollower()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info(f'patrol finished (held {node.hold_total:.1f} s for the vertical-clearance rule)')
    finally:
        node.log_f.close()
        node.dec_f.close()
        node.leg_f.close()
        log = node.get_parameter('log_file').value or f'/tmp/nmopso_drone{node.id}.csv'
        with open(str(Path(log).with_suffix('')) + '_track.json', 'w') as f:
            json.dump({'PATROL': [list(p) for p in node.tracks['PATROL'][0]]}, f)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
