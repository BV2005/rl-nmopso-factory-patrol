#!/usr/bin/env python3
"""Flies one PX4 SITL drone around its NMOPSO patrol loop in the factory world.

The loop (world ENU metres, z = height above the floor) comes from
nmopso_demo/paths/patrol_paths.json, written by NMOPSO_repro/factory_patrol/plan_factory_patrol.m
(every leg planned by the authors' unmodified NMOPSO.m).

Control follows the conventions of patrol_bridge's planner:
  * setpoints and offboard heartbeats over ROS 2 (px4_msgs, best-effort QoS);
  * mode switch, arming and landing over MAVLink (pymavlink, udpin 14540+N, target system N+1),
    because arming over the ROS 2 vehicle_command topic was rejected in that project.
Unlike patrol_bridge (UWB, world-origin frame), this demo uses PX4's stock GPS x500, whose local
frame starts at the spawn point, so world -> NED is taken relative to the drone's home.

A carrot moves along the loop at `speed` m/s and waits whenever the drone falls more than
`max_lead` metres behind it, so the drone tracks the planned polyline instead of cutting corners.
Every tick is logged to a CSV for check_patrol.py.

Shared launch area (plan with transit_in / transit_out): after take-off the drone hovers for its
depart_delay, flies transit_in (pad -> its zone), patrols the zone loop, flies transit_out back to
its pad and lands. Without transit legs the loop starts and ends at home, as before.

Vertical-clearance rule (transit layer): before a climb or descent (a vertical segment of a track, or
the landing), the drone holds until no other airborne drone is within clear_radius horizontally in
the column it will pass through (column_blocked()), at most max_hold seconds.
"""
import csv
import json
import math
import time

import rclpy
from pymavlink import mavutil
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleLocalPosition, VehicleStatus
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

ARMED = VehicleStatus.ARMING_STATE_ARMED
OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD
CMD = mavutil.mavlink


class PatrolFollower(Node):
    def __init__(self):
        super().__init__('patrol_follower')
        self.declare_parameter('drone_id', 1)
        self.declare_parameter('paths_file', '')
        self.declare_parameter('log_file', '')
        self.declare_parameter('speed', 1.0)
        self.declare_parameter('max_lead', 1.5)
        self.declare_parameter('laps', 1)                 # 0 = patrol until stopped
        self.declare_parameter('clear_radius', 2.0)       # vertical-clearance rule, see column_blocked()
        self.declare_parameter('max_hold', 30.0)
        self.declare_parameter('yield_radius', 2.5)       # benchmark rule: yield to a lower-ID peer this close
        self.id = self.get_parameter('drone_id').value
        self.yield_radius = float(self.get_parameter('yield_radius').value)
        self.clear_radius = float(self.get_parameter('clear_radius').value)
        self.max_hold = float(self.get_parameter('max_hold').value)
        self.speed = float(self.get_parameter('speed').value)
        self.max_lead = float(self.get_parameter('max_lead').value)
        self.laps = int(self.get_parameter('laps').value)

        with open(self.get_parameter('paths_file').value) as f:
            data = json.load(f)
        me = next(d for d in data['drones'] if d['id'] == self.id)
        self.home = me['home']
        self.alt = float(me['altitude'])
        self.depart_delay = float(me.get('depart_delay', 0.0))
        self.tracks = {k: self.make_track(me.get(key) or [])
                       for k, key in (('TRANSIT_IN', 'transit_in'), ('PATROL', 'waypoints'),
                                      ('TRANSIT_OUT', 'transit_out'))}
        self.set_track('PATROL')
        # task mode: stops on the PATROL track (task points) where the drone hovers `hover` seconds
        wp, cum, _ = self.tracks['PATROL']
        self.hover = float(me.get('hover', 0.0))
        self.stops = []
        for st in me.get('stops') or []:
            k = min(range(len(wp)), key=lambda i: math.dist(wp[i], st['point']))
            self.stops.append((cum[k], st['task'], tuple(wp[k])))
        self.stops.sort()
        self.stop_i = 0
        self.hover_since = None
        self.yield_since = None
        self.get_logger().info(f'drone {self.id}: {len(self.wp)} waypoints, loop {self.loop_len:.1f} m, '
                               f'transit {self.tracks["TRANSIT_IN"][2]:.1f} + {self.tracks["TRANSIT_OUT"][2]:.1f} m, '
                               f'{self.laps or "endless"} lap(s) at {self.speed} m/s')

        ns = f'/drone{self.id}'
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.lpos = None
        self.vstatus = None
        self.create_subscription(VehicleLocalPosition, f'{ns}/fmu/out/vehicle_local_position_v1',
                                 lambda m: setattr(self, 'lpos', m), qos)
        self.create_subscription(VehicleStatus, f'{ns}/fmu/out/vehicle_status_v4',
                                 lambda m: setattr(self, 'vstatus', m), qos)
        # the other drones' positions (world ENU), for the vertical-clearance rule
        self.others = {}
        for d in data['drones']:
            if d['id'] != self.id:
                self.create_subscription(
                    VehicleLocalPosition, f"/drone{d['id']}/fmu/out/vehicle_local_position_v1",
                    lambda m, i=d['id'], h=d['home']: self.others.__setitem__(i, (h[0] + m.y, h[1] + m.x, -m.z)), qos)
        self.hold_since = None
        self.hold_total = 0.0
        # events for patrol_recorder.py (JSON strings): phase changes, laps, holds, landing
        self.event_pub = self.create_publisher(String, f'{ns}/patrol/event', 50)
        self.offboard_pub = self.create_publisher(OffboardControlMode, f'{ns}/fmu/in/offboard_control_mode', 10)
        self.sp_pub = self.create_publisher(TrajectorySetpoint, f'{ns}/fmu/in/trajectory_setpoint', 10)

        self.mav = mavutil.mavlink_connection(f'udpin:0.0.0.0:{14540 + self.id}', source_system=255)
        self.target = (self.id + 1, 1)

        log = self.get_parameter('log_file').value or f'/tmp/nmopso_drone{self.id}.csv'
        self.log_f = open(log, 'w', newline='')
        self.log = csv.writer(self.log_f)
        self.log.writerow(['t', 'phase', 'x', 'y', 'z', 'sp_x', 'sp_y', 'sp_z', 'lap', 's'])

        self.phase = 'WAIT_POSITION'
        self.t_phase = time.monotonic()
        self.t0 = time.monotonic()
        self.s = 0.0
        self.lap = 0
        self.sp = (self.home[0], self.home[1], 0.0)
        self.last_cmd = 0.0
        self.last_hb = 0.0
        self.create_timer(0.05, self.tick)                # 20 Hz

    # --- helpers -------------------------------------------------------------------------------
    @staticmethod
    def make_track(points):
        """(waypoints, cumulative arc length, length) of a polyline"""
        wp = [tuple(p) for p in points]
        cum = [0.0]
        for a, b in zip(wp, wp[1:]):
            cum.append(cum[-1] + math.dist(a, b))
        return wp, cum, cum[-1]

    def set_track(self, name):
        self.wp, self.cum, self.loop_len = self.tracks[name]
        self.s = 0.0

    def now_us(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def world(self):
        """current position in world ENU (NED local frame starts at home)"""
        return (self.home[0] + self.lpos.y, self.home[1] + self.lpos.x, -self.lpos.z)

    def publish_sp(self, x, y, z, yaw):
        m = TrajectorySetpoint()
        m.timestamp = self.now_us()
        m.position = [float(y - self.home[1]), float(x - self.home[0]), float(-z)]
        m.yaw = float(yaw)
        self.sp_pub.publish(m)
        self.sp = (x, y, z)

    def command(self, cmd, p1=0.0, p2=0.0):
        self.mav.mav.command_long_send(*self.target, cmd, 0, p1, p2, 0, 0, 0, 0, 0)

    def point_at(self, s):
        """position and heading (NED yaw) at arc length s along the current track"""
        return self.point_on((self.wp, self.cum, self.loop_len), s)

    @staticmethod
    def point_on(track, s):
        wp, cum, length = track
        s = max(0.0, min(s, length))
        i = max(0, min(len(cum) - 2, next((k for k in range(len(cum) - 1) if cum[k + 1] >= s), len(cum) - 2)))
        a, b = wp[i], wp[i + 1]
        seg = cum[i + 1] - cum[i]
        t = 0.0 if seg < 1e-9 else (s - cum[i]) / seg
        p = tuple(a[k] + t * (b[k] - a[k]) for k in range(3))
        yaw = math.atan2(b[0] - a[0], b[1] - a[1]) if seg > 1e-9 else 0.0
        return p, yaw

    def column_blocked(self, pos, z_to):
        """Vertical-clearance rule for a climb or descent from pos to altitude z_to: the ids of airborne
        drones within clear_radius horizontally in the column the move passes through (from 0.5 m past
        the current level to 0.5 m beyond the target). Drones at the current level do not block, so two
        drones hovering side by side over their pads cannot wait for each other."""
        z0 = pos[2]
        lo, hi = (z_to - 0.5, z0 - 0.5) if z_to < z0 else (z0 + 0.5, z_to + 0.5)
        return [i for i, (x, y, z) in self.others.items()
                if z > 0.3 and lo < z < hi and math.hypot(x - pos[0], y - pos[1]) < self.clear_radius]

    def vertical_ahead(self, pos):
        """target altitude if the carrot sits at the start of a vertical segment the drone has reached"""
        i = next((k for k in range(len(self.cum) - 1) if self.cum[k + 1] > self.s), None)
        if i is None or self.s - self.cum[i] > self.speed * 0.05 + 1e-6:
            return None
        a, b = self.wp[i], self.wp[i + 1]
        if math.hypot(b[0] - a[0], b[1] - a[1]) < 0.05 and abs(b[2] - a[2]) > 0.3 and math.dist(pos, a) < 0.3:
            return b[2]
        return None

    def holding(self, now, pos, z_to):
        """applies the vertical-clearance rule; True while the drone has to wait (at most max_hold s)"""
        blockers = self.column_blocked(pos, z_to)
        if blockers and (self.hold_since is None or now - self.hold_since < self.max_hold):
            if self.hold_since is None:
                self.hold_since = now
                self.get_logger().info(f'{self.phase}: holding at {pos[2]:.1f} m, drone(s) {blockers} in the column')
                self.emit('hold_start', f'z={pos[2]:.1f} m, blocked by {blockers}')
            return True
        if self.hold_since is not None:
            waited = now - self.hold_since
            self.hold_total += waited
            self.get_logger().info(f'{self.phase}: column clear after {waited:.1f} s'
                                   + (' (max_hold reached)' if blockers else ''))
            self.emit('hold_end', f'{waited:.1f} s' + (' (max_hold reached)' if blockers else ''))
            self.hold_since = None
        return False

    def yielding(self, now, pos):
        """benchmark proximity rule (patrol_controller.c): hold while a LOWER-ID airborne drone is
        within yield_radius horizontally at a similar height (|dz| < 1 m); the lower ID keeps going"""
        if self.yield_radius <= 0:
            return False
        peers = [i for i, (x, y, z) in self.others.items()
                 if i < self.id and z > 0.3 and abs(z - pos[2]) < 1.0
                 and math.hypot(x - pos[0], y - pos[1]) < self.yield_radius]
        if peers and self.yield_since is None:
            self.yield_since = now
            self.emit('yield_start', f'to drone(s) {peers}')
        elif not peers and self.yield_since is not None:
            self.emit('yield_end', f'{now - self.yield_since:.1f} s')
            self.yield_since = None
        return bool(peers)

    def emit(self, event, detail=''):
        """publishes an event for patrol_recorder.py"""
        m = String()
        m.data = json.dumps({'drone': self.id, 'event': event, 'phase': self.phase, 'detail': detail})
        self.event_pub.publish(m)

    def set_phase(self, p):
        self.get_logger().info(f'{self.phase} -> {p}')
        self.phase = p
        self.t_phase = time.monotonic()
        self.emit('phase', p)

    # --- control loop --------------------------------------------------------------------------
    def tick(self):
        now = time.monotonic()
        self.mav.recv_match(blocking=False)               # lets udpin learn PX4's address
        if now - self.last_hb > 1.0:
            self.mav.mav.heartbeat_send(CMD.MAV_TYPE_GCS, CMD.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self.last_hb = now
        m = OffboardControlMode()
        m.timestamp = self.now_us()
        m.position = True
        self.offboard_pub.publish(m)
        if self.lpos is None or not (self.lpos.xy_valid and self.lpos.z_valid):
            return
        pos = self.world()
        armed = self.vstatus is not None and self.vstatus.arming_state == ARMED
        offboard = self.vstatus is not None and self.vstatus.nav_state == OFFBOARD

        if self.phase == 'WAIT_POSITION':
            self.publish_sp(self.home[0], self.home[1], self.alt, 0.0)
            if now - self.t_phase > 2.0:
                self.set_phase('ARM')
        elif self.phase == 'ARM':
            self.publish_sp(self.home[0], self.home[1], self.alt, 0.0)
            if now - self.last_cmd > 1.0:
                if not offboard:
                    self.command(CMD.MAV_CMD_DO_SET_MODE, 1.0, 6.0)
                elif not armed:
                    self.command(CMD.MAV_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_cmd = now
            if armed and offboard:
                self.set_phase('TAKEOFF')
        elif self.phase == 'TAKEOFF':
            # climb over the pad, then hover for depart_delay (shared launch area: lets neighbours leave)
            first = 'TRANSIT_IN' if self.tracks['TRANSIT_IN'][0] else 'PATROL'
            _, yaw0 = self.point_on(self.tracks[first], 0.0)
            self.publish_sp(self.home[0], self.home[1], self.alt, yaw0)
            if abs(pos[2] - self.alt) < 0.15 and now - self.t_phase > 3.0 + self.depart_delay:
                self.set_track(first)
                self.set_phase(first)
        elif self.phase in ('TRANSIT_IN', 'PATROL', 'TRANSIT_OUT'):
            carrot, _ = self.point_at(self.s)
            z_to = self.vertical_ahead(pos)
            wait = z_to is not None and self.holding(now, pos, z_to)
            wait = self.yielding(now, pos) or wait
            # task stop ahead on the PATROL track: the carrot stops there; hover `hover` s once reached
            stop = self.stops[self.stop_i] if self.phase == 'PATROL' and self.stop_i < len(self.stops) else None
            if stop is not None and self.s >= stop[0]:
                self.s = stop[0]
                wait = True
                if math.dist(pos, stop[2]) < 0.3:
                    self.hover_since = self.hover_since or now
                    if now - self.hover_since >= self.hover:
                        self.emit('task_completed', stop[1])
                        self.get_logger().info(f'task {stop[1]} completed ({now - self.t0:.1f} s since start)')
                        self.stop_i += 1
                        self.hover_since = None
            if math.dist(pos, carrot) < self.max_lead and not wait:
                self.s += self.speed * 0.05
            # the next phase starts only once the drone has reached the end of the track (not just
            # the carrot, which leads by up to max_lead) and every task stop on it is done
            pending = self.phase == 'PATROL' and self.stop_i < len(self.stops)
            if self.s >= self.loop_len and math.dist(pos, self.wp[-1]) < 0.3 and not pending:
                if self.phase == 'TRANSIT_IN':
                    self.set_track('PATROL')
                    self.set_phase('PATROL')
                elif self.phase == 'TRANSIT_OUT':
                    self.set_phase('LAND')
                    return
                else:
                    self.lap += 1
                    self.get_logger().info(f'lap {self.lap} done ({now - self.t0:.0f} s since start)')
                    self.emit('lap_done', str(self.lap))
                    self.s = 0.0
                    if self.laps and self.lap >= self.laps:
                        if self.tracks['TRANSIT_OUT'][0]:
                            self.set_track('TRANSIT_OUT')
                            self.set_phase('TRANSIT_OUT')
                        else:
                            self.set_phase('LAND')
                            return
            p, yaw = self.point_at(self.s)
            self.publish_sp(*p, yaw)
        elif self.phase == 'LAND':
            # hover over home, then auto-land; once on the ground for 3 s, disarm and finish
            # (PX4's own disarm-after-landing is not relied on: it was not observed in the demo)
            self.publish_sp(self.home[0], self.home[1], self.alt, 0.0)
            over_home = math.dist(pos[:2], self.home) < 0.5 or now - self.t_phase > 20.0
            # vertical-clearance rule before the descent starts (neighbouring pads are 1.5 m apart)
            if not getattr(self, 'land_sent', False) and over_home and self.holding(now, pos, 0.0):
                over_home = False
            if over_home and pos[2] > 0.15 and now - self.last_cmd > 2.0:
                if not getattr(self, 'land_sent', False):
                    self.emit('land_command', f'z={pos[2]:.1f} m')
                self.land_sent = True
                self.command(CMD.MAV_CMD_NAV_LAND)
                self.last_cmd = now
            if pos[2] < 0.15:
                self.ground_since = getattr(self, 'ground_since', None) or now
                if now - self.ground_since > 3.0:
                    if armed and now - self.last_cmd > 1.0:
                        self.command(CMD.MAV_CMD_COMPONENT_ARM_DISARM, 0.0)
                        self.last_cmd = now
                    if not armed or now - self.ground_since > 10.0:
                        self.set_phase('DONE')
            else:
                self.ground_since = None
        elif self.phase == 'DONE':
            self.log_f.flush()
            raise SystemExit
        self.log.writerow([f'{now - self.t0:.2f}', self.phase, *(f'{v:.3f}' for v in pos),
                           *(f'{v:.3f}' for v in self.sp), self.lap, f'{self.s:.2f}'])


def main():
    rclpy.init()
    node = PatrolFollower()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info(f'patrol finished (held {node.hold_total:.1f} s for the vertical-clearance rule)')
    finally:
        node.log_f.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
