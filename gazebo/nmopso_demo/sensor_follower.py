#!/usr/bin/env python3
"""Flies one x500_patrol drone with its PATROL legs planned online by the sensor-input policy.

patrol_follower.py flies tracks planned offline on the known map. Here only the transits keep their
offline (pad-safe, deconflicted) tracks; every PATROL leg between two consecutive patrol points of
the plan is planned while it is flown, one NMOPSO node at a time, from what the drone senses:

  at the leg start  the drone turns in place to face the leg's goal (the policy's first heading);
  at every node     it takes the fused scan (2D lidar /droneN/scan + depth points /droneN/depth/points,
                    rlpatrol/sensors.py: the same 27 bins the policy was trained on), the policy
                    (a sensor_obs checkpoint, deterministic) proposes the next node, the segment is
                    checked against the scan (sensors.segment_clear, drone radius + margin); if it is
                    not clear, up to `samples` stochastic proposals are tried, then a sweep of the
                    action space (heading offsets up to +-90 deg from the goal bearing, shorter
                    steps), nearest to the policy's own action first: a trained policy's samples
                    cluster around its mean, i.e. around the blocked direction. If nothing is clear
                    the drone hovers and scans again; after `max_block` s it flies straight on to
                    the leg's goal if that is clear, else it retreats `retreat_dist` back along the
                    flown (known free) track and plans the leg again as a new episode (at most
                    `max_retries` times), else the leg is skipped and logged
  after node n      it flies to the leg's goal, as MyCost.m closes every path - only once the scan
                    shows that closing segment clear too (else the same blocked handling).

The policy's chain (PathEnv) is advanced with the accepted actions, so its observation of the goal,
reach and heading is exactly the training one; only the scan comes from the real sensors
(PathEnv.scan_override). Every decision is logged to <log_file>_sensor.csv (scan, action, node,
safety outcome) for offline analysis.

Needs PyTorch and the repository on PYTHONPATH (run_demo.sh SENSOR_CKPT=... sets it up)."""
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
from rlpatrol.env import PathEnv  # noqa: E402
from rlpatrol.nmopso_cost import N_NODES, tmat  # noqa: E402
from rlpatrol.ppo import env_kwargs, load_checkpoint, make_actor  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402

BALANCED = [1 / 3, 0.0, 1 / 3, 1 / 3]             # w over F1, F3, F4 for threat-budget checkpoints


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SensorFollower(PatrolFollower):
    def __init__(self):
        super().__init__()
        for name, default in (('ckpt', ''), ('budget', 0.6), ('pref', ''), ('scan_topic', ''),
                              ('cloud_topic', ''), ('cloud_frame', 'flu'), ('radius', 0.45), ('samples', 16),
                              ('arrive_dist', 0.3), ('yaw_tol', 0.15), ('max_scan_age', 0.5),
                              ('max_block', 10.0), ('use_depth', True), ('max_retries', 2), ('retreat_dist', 1.0)):
            self.declare_parameter(name, default)
        P = lambda k: self.get_parameter(k).value                          # noqa: E731
        ck = load_checkpoint(P('ckpt'))
        cfg = ck['cfg']
        if not cfg.get('sensor_obs', False):
            raise SystemExit(f"{P('ckpt')} is not a sensor-input checkpoint (sensor_obs is off)")
        torch.set_num_threads(1)
        self.actor = make_actor(cfg)                                       # MLP, or the GRU actor (recurrent)
        self.actor.load_state_dict(ck['actor'])
        self.actor.eval()
        self.cfg = cfg
        self.world = load_world()
        self.u = float(P('budget')) if cfg.get('budget', False) else None
        pref = [float(v) for v in str(P('pref')).split(',')] if P('pref') else (
            BALANCED if self.u is not None else [0.25] * 4)
        self.pref = torch.tensor([pref], dtype=torch.float64)
        self.radius, self.samples = float(P('radius')), int(P('samples'))
        self.arrive, self.yaw_tol = float(P('arrive_dist')), float(P('yaw_tol'))
        self.max_age, self.max_block = float(P('max_scan_age')), float(P('max_block'))
        self.max_retries, self.retreat_dist = int(P('max_retries')), float(P('retreat_dist'))
        self.cloud_frame, self.use_depth = P('cloud_frame'), bool(P('use_depth'))

        with open(self.get_parameter('paths_file').value) as f:
            me = next(d for d in json.load(f)['drones'] if d['id'] == self.id)
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
        self.lidar_bins = self.depth_bins = None
        self.t_lidar = self.t_depth = -1.0
        self.att = None
        self.create_subscription(LaserScan, P('scan_topic') or f'{ns}/scan', self.on_scan, qos)
        if self.use_depth:
            self.create_subscription(PointCloud2, P('cloud_topic') or f'{ns}/depth/points', self.on_cloud, qos)
        self.create_subscription(VehicleAttitude, f'{ns}/fmu/out/vehicle_attitude', lambda m: setattr(self, 'att', m), qos)

        log = self.get_parameter('log_file').value or f'/tmp/nmopso_drone{self.id}.csv'
        self.dec_f = open(str(Path(log).with_suffix('')) + '_sensor.csv', 'w', newline='')
        self.dec = csv.writer(self.dec_f)
        self.dec.writerow(['t', 'leg', 'node', 'x', 'y', 'z', 'yaw', 'outcome', 'tries', 'nx', 'ny', 'nz',
                           'a_r', 'a_yaw', 'a_z', 'lidar_age', 'depth_age', *[f's{k}' for k in range(S.SCAN_BINS)]])
        self.get_logger().info(f'drone {self.id}: sensor-input policy {P("ckpt")} (it {ck.get("it")}), '
                               f'{len(self.legs)} patrol legs planned online, budget {self.u}, pref {pref}')

    # --- sensors --------------------------------------------------------------------------------
    def on_scan(self, m):
        self.lidar_bins = S.bins_from_laserscan(m.ranges, m.angle_min, m.angle_increment, m.range_min, m.range_max)
        self.t_lidar = time.monotonic()

    def roll_pitch(self):
        """roll and pitch (rad) of the body in the FLU convention, from PX4's FRD attitude."""
        if self.att is None:
            return 0.0, 0.0
        w, x, y, z = (float(v) for v in self.att.q)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, -pitch                                                # FRD pitch up is negative in FLU

    def on_cloud(self, m):
        if self.lpos is None:
            return
        p = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'), skip_nans=True).astype(float)
        if self.cloud_frame == 'optical':                                  # z forward, x right, y down -> FLU
            p = np.c_[p[:, 2], -p[:, 0], -p[:, 1]]
        p = p[::4]                                                         # 19 k of 77 k points is plenty
        self.depth_bins = S.bins_from_points(p, -self.lpos.z, *self.roll_pitch())
        self.t_depth = time.monotonic()

    def scan_obs(self, now):
        """the fused scan observation, or None while the lidar is stale (the depth camera is optional)."""
        if self.lidar_bins is None or now - self.t_lidar > self.max_age:
            return None
        bins = [self.lidar_bins]
        if self.use_depth and self.depth_bins is not None and now - self.t_depth <= self.max_age:
            bins.append(self.depth_bins)
        return S.to_obs(*bins)

    def yaw_enu(self):
        return math.pi / 2 - float(self.lpos.heading)                      # PX4 heading: NED yaw

    # --- online planning -------------------------------------------------------------------------
    def reset_loop(self):
        self.leg_i, self.env, self.node_i, self.retries = 0, None, 0, 0
        self.blocked_since = self.arrived = None
        self.loop_done = False
        self.lap_seen = self.lap

    def append(self, p):
        p = tuple(float(v) for v in p)
        self.cum.append(self.cum[-1] + math.dist(self.wp[-1], p))
        self.wp.append(p)
        self.loop_len = self.cum[-1]

    def start_leg(self):
        # the leg starts where the flown track ends: its planned start, unless the previous leg was
        # skipped (then the drone is still where it gave up)
        b = self.legs[self.leg_i][1]
        a = self.wp[-1]
        w = self.world
        st, gl = w.to_units(np.array([a])), w.to_units(np.array([b]))
        kw = env_kwargs(self.cfg, noise=False, critic=False)
        self.env = PathEnv(w, st, gl, self.pref, budget=None if self.u is None else torch.tensor([self.u], dtype=torch.float64), **kw)
        self.node_i = 0
        self.actor.reset(1)                                                # a new episode: fresh GRU memory
        self.goal = b
        # turn in place to face the goal: a 1 cm step towards it sets the setpoint yaw
        d = np.array(b[:2]) - np.array(a[:2])
        d = d / max(np.linalg.norm(d), 1e-9)
        self.append((self.wp[-1][0] + 0.01 * d[0], self.wp[-1][1] + 0.01 * d[1], self.wp[-1][2]))
        self.arrived = None                                                # the turn's timeout starts now
        self.emit('leg_start', f'{self.leg_i}: {a[:2]} -> {b[:2]}')

    def next_leg(self):
        self.leg_i += 1
        self.env, self.retries = None, 0

    def clear_to(self, scan, heading, here, p):
        v = np.asarray(p[:2], float) - here[:2]
        return S.segment_clear(scan, heading, math.atan2(v[1], v[0]), float(np.linalg.norm(v)), self.radius)

    def retreat(self, pos):
        """fly back along the flown track (known free) until retreat_dist from pos."""
        back = []
        for p in reversed(self.wp[:-1]):
            back.append(p)
            if math.dist(p[:2], pos[:2]) >= self.retreat_dist:
                break
        for p in back:
            self.append(p)

    def blocked(self, now, pos, scan, heading, here, tries, node, a, closing=False):
        """no clear way on: hover and rescan for max_block s, then (1) fly straight to the leg's goal if the
        scan shows that clear, else (2) retreat along the flown track and plan the leg again from there as
        a new episode (fresh memory, a full set of nodes; at most max_retries times), else (3) skip it."""
        if self.blocked_since is None:                                     # log once per blocked episode
            self.blocked_since = now
            self.log_decision(now, pos, 'blocked_closing' if closing else 'blocked', tries, node, a, scan)
        if now - self.blocked_since < self.max_block:
            return True
        self.blocked_since = None
        if not closing and self.clear_to(scan, heading, here, self.goal):
            outcome = 'straight_to_goal'
            self.node_i = N_NODES                                           # closed (and checked again) next call
        elif self.retries < self.max_retries:
            outcome = 'retreat_replan'
            self.retries += 1
            self.retreat(pos)
            self.env = None                                                 # the same leg, from the retreat point
        else:
            outcome = 'leg_skipped'
            self.next_leg()
        self.log_decision(now, pos, outcome, tries, self.goal, [0.0, 0.0, 0.0], scan)
        self.emit('sensor_blocked', f'leg {self.leg_i} node {self.node_i}: {outcome}')
        self.get_logger().warn(f'leg {self.leg_i} node {self.node_i}: blocked for {self.max_block:.0f} s -> {outcome}')
        return True

    def track_heading(self):
        """ENU yaw of the last track segment: the setpoint yaw the drone turns to (the policy's chain
        heading, unless the node was clamped to the map bounds)."""
        (x0, y0, _), (x1, y1, _) = self.wp[-2], self.wp[-1]
        return math.atan2(y1 - y0, x1 - x0)

    def node_for(self, a):
        """the node (world metres) action a [1, 3] leads to, without advancing the chain."""
        e = self.env
        r, psi, phi = e.act(a)
        q = e.legs.clamp((e.legs.T0 @ (e.T @ tmat(r, phi, psi)))[:, :3, 3])
        return self.world.to_metres(q[0].numpy())

    def candidates(self, o):
        """actions to try, in order: the policy's (deterministic), `samples` stochastic ones, then
        the action-space sweep sorted by distance to the policy's action."""
        with torch.no_grad():
            a0 = self.actor.act(o, True)[0]
            yield a0
            for _ in range(self.samples):
                yield self.actor.act(o)[0]
        grid = [torch.tensor([[ar, ay, float(a0[0, 2])]], dtype=a0.dtype)
                for ar in (0.0, -0.5, -1.0) for ay in np.linspace(-1, 1, 17)]
        yield from sorted(grid, key=lambda a: float((a - a0).norm()))

    def log_decision(self, now, pos, outcome, tries, node, a, scan):
        self.dec.writerow([f'{now - self.t0:.2f}', self.leg_i, self.node_i, *(f'{v:.3f}' for v in pos),
                           f'{self.yaw_enu():.3f}', outcome, tries, *(f'{v:.3f}' for v in node),
                           *(f'{v:.4f}' for v in a), f'{now - self.t_lidar:.2f}', f'{now - self.t_depth:.2f}',
                           *(f'{v:.4f}' for v in scan)])

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
        self.arrived = self.arrived or now
        if self.env is None:
            if self.leg_i >= len(self.legs):
                self.loop_done = True
                return False
            self.start_leg()
            return True
        if abs(wrap(self.yaw_enu() - self.track_heading())) > self.yaw_tol and now - self.arrived < 3.0:
            return True                                                    # still turning to the segment's heading
        scan = self.scan_obs(now)
        if scan is None:
            return True                                                    # hover until the lidar is fresh
        heading = self.yaw_enu()
        here = np.array(pos)
        if self.node_i >= N_NODES:                                         # nodes done: close the leg to its goal,
            if self.clear_to(scan, heading, here, self.goal):              # checked against the scan like every node
                self.blocked_since = None
                self.append(self.goal)
                self.emit('leg_done', str(self.leg_i))
                self.next_leg()
                return True
            return self.blocked(now, pos, scan, heading, here, 0, self.goal, [0.0, 0.0, 0.0], closing=True)
        self.env.scan_override = scan[None]
        o = self.env.obs()
        tries, accepted = 0, None
        for a in self.candidates(o):
            node = self.node_for(a)
            tries += 1
            if self.clear_to(scan, heading, here, node):
                accepted = (a, node)
                break
        if accepted is None:
            self.actor.discard()                                           # no step: this scan is not remembered
            return self.blocked(now, pos, scan, heading, here, tries, node, a[0].tolist())
        self.blocked_since = None
        a, node = accepted
        self.actor.commit()                                                # the GRU remembers this node's scan
        self.env.step(a)                                                   # advance the policy's chain
        self.node_i += 1
        self.append(node)
        self.log_decision(now, pos, 'ok', tries, node, a[0].tolist(), scan)
        return True


def main():
    rclpy.init()
    node = SensorFollower()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info(f'patrol finished (held {node.hold_total:.1f} s for the vertical-clearance rule)')
    finally:
        node.log_f.close()
        node.dec_f.close()
        # the PATROL track as planned online (the last lap), for check_patrol.py's cross-track error
        with open(str(Path(node.get_parameter('log_file').value or f'/tmp/nmopso_drone{node.id}.csv').with_suffix(''))
                  + '_track.json', 'w') as f:
            json.dump({'PATROL': [list(p) for p in node.tracks['PATROL'][0]]}, f)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
