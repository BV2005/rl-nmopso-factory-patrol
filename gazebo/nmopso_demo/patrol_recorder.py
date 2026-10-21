#!/usr/bin/env python3
"""Records the state of every patrol drone at 10 Hz, plus an event log, as CSV files.

    python3 patrol_recorder.py --ros-args -p paths_file:=<patrol_paths.json> -p out_dir:=<run_logs/...>

Per drone it subscribes to PX4's uXRCE-DDS topics (namespace droneN):
  fmu/out/vehicle_local_position_v1   position, velocity, heading (local NED, origin = the drone's pad)
  fmu/out/vehicle_attitude            attitude quaternion
  fmu/out/battery_status_v1           battery (PX4 publishes it at 1 Hz: rows in between repeat it)
  fmu/out/vehicle_status_v4           arming state, navigation state, failsafe
  fmu/out/vehicle_land_detected       landed flag
  patrol/event                        patrol_follower.py events (JSON): phase, laps, holds, landing
and writes into out_dir:
  drone_states.csv   one row per drone every 0.1 s: factory-frame (world ENU) position and velocity,
                     attitude, battery, PX4 state, phase, and the age of the newest position message
  events.csv         follower events plus changes the recorder sees in PX4 state (arming, nav state,
                     failsafe, landed, battery warning)
It stops by itself 3 s after every drone has reported phase DONE, or on Ctrl-C / SIGINT.
"""
import csv
import json
import math
import os
import time

import rclpy
from px4_msgs.msg import BatteryStatus, VehicleAttitude, VehicleLandDetected, VehicleLocalPosition, VehicleStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

RATE_HZ = 10.0
STATE_COLUMNS = ['t', 'drone', 'phase', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'speed', 'roll_deg', 'pitch_deg',
                 'yaw_deg', 'battery_pct', 'voltage_v', 'current_a', 'discharged_mah', 'battery_warning',
                 'arming_state', 'nav_state', 'failsafe', 'landed', 'pos_age_s']


def euler_deg(q):
    """roll, pitch, yaw (deg) from PX4's FRD->NED quaternion [w, x, y, z]; yaw as ENU heading (0 = east)"""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw_ned = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw_enu = math.pi / 2 - yaw_ned
    yaw_enu = math.atan2(math.sin(yaw_enu), math.cos(yaw_enu))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw_enu)


class PatrolRecorder(Node):
    def __init__(self):
        super().__init__('patrol_recorder')
        self.declare_parameter('paths_file', '')
        self.declare_parameter('out_dir', '.')
        with open(self.get_parameter('paths_file').value) as f:
            plan = json.load(f)
        out = self.get_parameter('out_dir').value
        os.makedirs(out, exist_ok=True)

        self.drones = {d['id']: {'home': d['home'], 'lpos': None, 't_lpos': None, 'att': None, 'bat': None,
                                 'status': None, 'landed': None, 'phase': 'WAIT_POSITION', 'last': {}}
                       for d in plan['drones']}
        px4 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        for i in self.drones:
            ns = f'/drone{i}/fmu/out'
            self.create_subscription(VehicleLocalPosition, f'{ns}/vehicle_local_position_v1',
                                     lambda m, i=i: self.on_lpos(i, m), px4)
            self.create_subscription(VehicleAttitude, f'{ns}/vehicle_attitude',
                                     lambda m, i=i: self.drones[i].__setitem__('att', m), px4)
            self.create_subscription(BatteryStatus, f'{ns}/battery_status_v1',
                                     lambda m, i=i: self.drones[i].__setitem__('bat', m), px4)
            self.create_subscription(VehicleStatus, f'{ns}/vehicle_status_v4',
                                     lambda m, i=i: self.drones[i].__setitem__('status', m), px4)
            self.create_subscription(VehicleLandDetected, f'{ns}/vehicle_land_detected',
                                     lambda m, i=i: self.drones[i].__setitem__('landed', m), px4)
            self.create_subscription(String, f'/drone{i}/patrol/event', self.on_event, 50)

        self.t0 = time.monotonic()
        self.states_f = open(os.path.join(out, 'drone_states.csv'), 'w', newline='')
        self.states = csv.writer(self.states_f)
        self.states.writerow(STATE_COLUMNS)
        self.events_f = open(os.path.join(out, 'events.csv'), 'w', newline='')
        self.events = csv.writer(self.events_f)
        self.events.writerow(['t', 'drone', 'source', 'event', 'phase', 'detail'])
        self.done_at = None
        self.rows = 0
        self.create_timer(1.0 / RATE_HZ, self.sample)
        self.get_logger().info(f'recording {len(self.drones)} drones at {RATE_HZ:.0f} Hz into {out}')

    def now(self):
        return time.monotonic() - self.t0

    def on_lpos(self, i, m):
        self.drones[i]['lpos'] = m
        self.drones[i]['t_lpos'] = time.monotonic()

    def event(self, drone, source, event, phase, detail):
        self.events.writerow([f'{self.now():.2f}', drone, source, event, phase, detail])
        self.events_f.flush()

    def on_event(self, msg):
        e = json.loads(msg.data)
        d = self.drones.get(e['drone'])
        if d is None:
            return
        if e['event'] == 'phase':
            d['phase'] = e['detail']
        self.event(e['drone'], 'follower', e['event'], d['phase'], e.get('detail', ''))

    def watch(self, i, key, value, label):
        """logs an event when a PX4 state value changes"""
        last = self.drones[i]['last']
        if key in last and last[key] != value:
            self.event(i, 'px4', label, self.drones[i]['phase'], f'{last[key]} -> {value}')
        last[key] = value

    def sample(self):
        t = self.now()
        for i, d in self.drones.items():
            m = d['lpos']
            if m is None:
                continue
            hx, hy = d['home']
            x, y, z = hx + m.y, hy + m.x, -m.z                  # NED (origin = pad) -> factory ENU
            vx, vy, vz = m.vy, m.vx, -m.vz
            roll = pitch = yaw = float('nan')
            if d['att'] is not None:
                roll, pitch, yaw = euler_deg(d['att'].q)
            b, s, ld = d['bat'], d['status'], d['landed']
            bat = ([100 * b.remaining if b.remaining >= 0 else float('nan'), b.voltage_v, b.current_a,
                    b.discharged_mah, b.warning] if b is not None else [float('nan')] * 4 + [''])
            st = [s.arming_state, s.nav_state, int(s.failsafe)] if s is not None else ['', '', '']
            landed = int(ld.landed) if ld is not None else ''
            self.states.writerow([f'{t:.2f}', i, d['phase'], *(f'{v:.3f}' for v in (x, y, z, vx, vy, vz)),
                                  f'{math.sqrt(vx * vx + vy * vy + vz * vz):.3f}',
                                  *(f'{v:.2f}' for v in (roll, pitch, yaw)),
                                  *(f'{v:.3f}' if isinstance(v, float) else v for v in bat), *st, landed,
                                  f'{time.monotonic() - d["t_lpos"]:.3f}'])
            self.rows += 1
            if s is not None:
                self.watch(i, 'arming', s.arming_state, 'arming_state')
                self.watch(i, 'nav', s.nav_state, 'nav_state')
                self.watch(i, 'failsafe', bool(s.failsafe), 'failsafe')
            if ld is not None:
                self.watch(i, 'landed', bool(ld.landed), 'landed')
            if b is not None:
                self.watch(i, 'bwarn', b.warning, 'battery_warning')
        if int(t * RATE_HZ) % int(RATE_HZ) == 0:
            self.states_f.flush()
        if self.done_at is None and all(d['phase'] == 'DONE' for d in self.drones.values()):
            self.done_at = t
            self.get_logger().info('all drones DONE; stopping in 3 s')
        if self.done_at is not None and t - self.done_at > 3.0:
            raise SystemExit

    def close(self):
        for f in (self.states_f, self.events_f):
            f.flush()
            f.close()
        self.get_logger().info(f'wrote {self.rows} state rows')


def main():
    rclpy.init()
    node = PatrolRecorder()
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):   # done, or SIGINT / SIGTERM
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
