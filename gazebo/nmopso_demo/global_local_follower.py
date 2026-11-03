#!/usr/bin/env python3
"""Global-local patrol follower: the planned (global) track, a local VFH+ planner for what the map does not
know, and global replanning with an obstacle memory.

GLOBAL  the PATROL track of the plan file (planned on the prior map by vB / NMOPSO). On a replan,
        LegPlanner(global_ckpt).plan_route (A* route + the policy per sub-leg, grid="physical") from the
        drone's position to the current leg's goal, in the prior world plus the obstacle memory.
LOCAL   every tick (20 Hz), on the fused lidar + depth scan (rlpatrol/sensors.py), modes:
  FOLLOW  the track; the segment to the lookahead point is checked against the scan
  AVOID   it is blocked: VFH+ (rlpatrol/local_planner.py) steers `step` m at a time towards the track
          further on, and rejoins the track at the first point ahead that is visible and clear
  BRAKE   clearance below d_brake: hold, then AVOID
  REPLAN  VFH+ finds no free direction, or AVOID lasts longer than avoid_timeout: the obstacle in front
          (scan returns the prior map does not explain) goes into the obstacle memory and the rest of
          the leg is replanned; on failure the drone holds and retries every replan_retry s, and after
          max_replans failures the leg's goal is dropped (the next patrol point is planned to instead).
Transits keep their planned (pad-safe) tracks. Every mode change and replan goes to
<log_file>_local.csv; the final PATROL track to <log_file>_track.json (check_patrol.py).
local = vs: the AVOID steps come from the sensor-input policy vS (a short episode from the drone to the rejoin
target, one node per scan, proposals checked against the scan, VFH+ for a step when none is clear) instead of
VFH+ - learned vs classical local avoidance under the same safety gate.
Options: adaptive_u: a replan whose route passes near a freshly remembered obstacle is planned with
threat budgets <= 0.3 only (more cautious where the map was just found wrong)."""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # the repository root
from patrol_follower import PatrolFollower  # noqa: E402
from sensor_follower import SensorFollower  # noqa: E402
from rlpatrol import local_planner as LP  # noqa: E402
from rlpatrol.obstacle_memory import ObstacleMemory  # noqa: E402
from rlpatrol.planner import LegPlanner  # noqa: E402
from rlpatrol.env import PathEnv  # noqa: E402
from rlpatrol.nmopso_cost import N_NODES  # noqa: E402
from rlpatrol.ppo import env_kwargs, load_checkpoint, make_actor  # noqa: E402
import torch  # noqa: E402


class GlobalLocalFollower(PatrolFollower):
    # the sensor plumbing of the sensor-input follower (lidar + depth -> fused bins, heading)
    on_scan, on_cloud, roll_pitch = SensorFollower.on_scan, SensorFollower.on_cloud, SensorFollower.roll_pitch
    scan_obs, yaw_enu = SensorFollower.scan_obs, SensorFollower.yaw_enu
    candidates, node_for = SensorFollower.candidates, SensorFollower.node_for       # local = vs

    def __init__(self):
        super().__init__()
        for name, default in (('global_ckpt', ''), ('scan_topic', ''), ('cloud_topic', ''), ('cloud_frame', 'flu'),
                              ('use_depth', True), ('max_scan_age', 0.5), ('lookahead', 1.5), ('d_act', 2.5),
                              ('d_brake', 0.6), ('radius', 0.45), ('step', 0.5), ('avoid_timeout', 15.0),
                              ('rejoin_window', 4.0), ('replan_retry', 5.0), ('max_replans', 3),
                              ('adaptive_u', False), ('memory_ttl', 300.0), ('local', 'vfh'),
                              ('local_ckpt', ''), ('budget', 0.6), ('samples', 16)):
            self.declare_parameter(name, default)
        P = lambda k: self.get_parameter(k).value                          # noqa: E731
        self.cloud_frame, self.use_depth, self.max_age = P('cloud_frame'), bool(P('use_depth')), float(P('max_scan_age'))
        self.vfh = LP.VFHParams(d_act=float(P('d_act')), d_brake=float(P('d_brake')), radius=float(P('radius')),
                                step=float(P('step')))
        self.lookahead, self.rejoin_window = float(P('lookahead')), float(P('rejoin_window'))
        self.avoid_timeout, self.replan_retry = float(P('avoid_timeout')), float(P('replan_retry'))
        self.max_replans, self.adaptive_u = int(P('max_replans')), bool(P('adaptive_u'))
        self.planner = LegPlanner(P('global_ckpt'), n_samples=1, refine=True, grid='physical')
        self.memory = ObstacleMemory(self.planner.world, ttl=float(P('memory_ttl')))
        self.local = P('local')
        if self.local not in ('vfh', 'vs'):
            raise SystemExit(f"local must be vfh or vs, not {self.local!r}")
        self.vs_env = None
        if self.local == 'vs':
            ck = load_checkpoint(P('local_ckpt') or str(Path(__file__).resolve().parents[2] / 'results' / 'ckpt' / 'vS_seed1.pt'))
            if not ck['cfg'].get('sensor_obs', False):
                raise SystemExit('local=vs needs a sensor_obs checkpoint')
            self.cfg = ck['cfg']
            self.actor = make_actor(self.cfg)
            self.actor.load_state_dict(ck['actor'])
            self.actor.eval()
            self.samples = int(P('samples'))
            self.world = self.planner.world                                 # node_for's metres conversion
            self.vs_u = float(P('budget')) if self.cfg.get('budget', False) else None
            self.vs_pref = torch.tensor([[1 / 3, 0.0, 1 / 3, 1 / 3]] if self.vs_u is not None else [[0.25] * 4],
                                        dtype=torch.float64)
        with open(self.get_parameter('paths_file').value) as f:
            me = next(d for d in json.load(f)['drones'] if d['id'] == self.id)
        self.goals = [tuple(float(v) for v in p) for p in me['patrol_points']]

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        ns = f'/drone{self.id}'
        self.lidar_bins = self.depth_bins = self.att = None
        self.t_lidar = self.t_depth = -1.0
        self.create_subscription(LaserScan, P('scan_topic') or f'{ns}/scan', self.on_scan, qos)
        if self.use_depth:
            self.create_subscription(PointCloud2, P('cloud_topic') or f'{ns}/depth/points', self.on_cloud, qos)
        self.create_subscription(VehicleAttitude, f'{ns}/fmu/out/vehicle_attitude', lambda m: setattr(self, 'att', m), qos)

        self.mode, self.t_mode, self.hold_pos = 'FOLLOW', 0.0, None
        self.avoid_sp, self.avoid_yaw, self.prev = None, None, None
        self.replans_failed, self.t_replan, self.t_replan_ok = 0, -1e9, -1e9
        self.turn_to, self.look_turns, self.recent, self.n_blocked = None, 0, [], 0
        self.counts = {k: 0 for k in ('avoid', 'rejoin', 'brake', 'look_around', 'replan_ok', 'replan_fail', 'goal_dropped')}
        log = self.get_parameter('log_file').value or f'/tmp/nmopso_drone{self.id}.csv'
        self.log_base = str(Path(log).with_suffix(''))
        self.loc_f = open(self.log_base + '_local.csv', 'w', newline='')
        self.loc = csv.writer(self.loc_f)
        self.loc.writerow(['t', 'event', 'mode', 'x', 'y', 'z', 'clearance', 'bearing', 'detail'])
        self.get_logger().info(f'drone {self.id}: global-local follower, global {P("global_ckpt")}, '
                               f'{len(self.goals)} patrol points, VFH+ d_act {self.vfh.d_act} m')

    # --- helpers -----------------------------------------------------------------------------------
    def event(self, now, kind, pos, clearance=float('nan'), bearing=float('nan'), detail=''):
        if kind in self.counts:
            self.counts[kind] += 1
        self.loc.writerow([f'{now - self.t0:.2f}', kind, self.mode, *(f'{v:.3f}' for v in pos),
                           f'{clearance:.2f}', f'{bearing:.3f}', detail])
        self.emit('local_' + kind, detail)

    def set_mode(self, now, mode):
        self.mode, self.t_mode = mode, now

    @staticmethod
    def in_view(heading, pos, q, half=math.radians(100)):
        """q is inside the sensors' field of view from pos (the 90 deg behind the drone is blind)."""
        return abs(LP.wrap(math.atan2(q[1] - pos[1], q[0] - pos[0]) - heading)) <= half

    def track_point(self, s):
        return self.point_on((self.wp, self.cum, self.loop_len), s)[0]

    def seg_index(self, s):
        return max(0, min(len(self.cum) - 2, next((k for k in range(len(self.cum) - 1) if self.cum[k + 1] >= s), len(self.cum) - 2)))

    def next_goal(self, s):
        """index into self.goals of the first patrol point on the track beyond arc length s."""
        for k, g in enumerate(self.goals):
            i = min(range(len(self.wp)), key=lambda j: math.dist(self.wp[j], g))
            if self.cum[i] > s + 0.3:
                return k
        return None

    def set_patrol_track(self, wp):
        self.tracks['PATROL'] = self.make_track(wp)
        self.wp, self.cum, self.loop_len = self.tracks['PATROL']

    # --- the local planner (PatrolFollower hooks) ---------------------------------------------------
    def local_wait(self, now, pos):
        if self.phase != 'PATROL':
            self.mode = 'FOLLOW'
            return False
        scan = self.scan_obs(now)
        if scan is None:                                                   # no sensing: hold
            if self.mode != 'HOLD':
                self.hold_pos = pos
                self.set_mode(now, 'HOLD')
            return True
        if self.mode == 'HOLD':
            self.set_mode(now, 'FOLLOW')
        heading = self.yaw_enu()
        c = LP.clearance(scan, self.vfh)
        if c < self.vfh.d_brake and self.mode in ('FOLLOW', 'AVOID'):
            self.hold_pos = pos
            self.set_mode(now, 'BRAKE')
            self.event(now, 'brake', pos, c)
        if self.mode == 'FOLLOW':
            look = self.track_point(self.s + self.lookahead)
            if math.dist(look, pos) > 0.2 and LP.path_blocked(scan, heading, pos, look, self.vfh):
                self.set_mode(now, 'AVOID')
                self.prev, self.look_turns, self.vs_env = None, 0, None
                self.event(now, 'avoid', pos, c)
            else:
                return False
        if self.mode == 'BRAKE':
            if now - self.t_mode < 1.0:
                return True
            self.set_mode(now, 'AVOID')
        if self.mode == 'AVOID':
            # rejoin: the first track point ahead that is visible and clear
            # (hysteresis: also the point FOLLOW will check from there, and at least 0.5 s in AVOID)
            for d in np.arange(self.lookahead, self.rejoin_window + 1e-9, 0.25) if now - self.t_mode >= 0.5 else []:
                q = self.track_point(self.s + d)
                q2 = self.track_point(self.s + d + self.lookahead)
                if not (self.in_view(heading, pos, q) and self.in_view(heading, pos, q2)):
                    continue
                if not LP.path_blocked(scan, heading, pos, q, self.vfh) and not LP.path_blocked(
                        scan, heading, pos, self.track_point(self.s + d + self.lookahead), self.vfh):
                    self.s = min(self.s + d, self.loop_len)
                    self.set_mode(now, 'FOLLOW')
                    self.event(now, 'rejoin', pos, c, detail=f'{d:.2f} m ahead')
                    return False
            target = self.track_point(self.s + self.rejoin_window)
            tb = math.atan2(target[1] - pos[1], target[0] - pos[0])
            if self.local == 'vs' and self.turn_to is None and self.vs_step(now, pos, scan, heading, target):
                return True
            if self.turn_to is not None:                                   # looking around (B): finish the turn
                if abs(LP.wrap(self.turn_to - heading)) > math.radians(15):
                    return True
                self.turn_to = None
            th = LP.vfh_heading(scan, heading, tb, self.prev, self.vfh)
            step_ok = th is not None and not LP.path_blocked(
                scan, heading, pos, (pos[0] + self.vfh.step * math.cos(th), pos[1] + self.vfh.step * math.sin(th)), self.vfh)
            if not step_ok and self.look_turns < 2:                         # B: turn 90 deg to the more open side
                r = LP.ranges(scan)
                cc = LP.S.bin_centres().numpy()
                left = r[cc > 0].mean() >= r[cc < 0].mean()
                self.turn_to = float(LP.wrap(heading + (1 if left else -1) * math.pi / 2))
                self.look_turns += 1
                self.hold_pos = pos
                self.avoid_sp, self.avoid_yaw = (pos[0], pos[1], self.alt), self.turn_to
                self.event(now, 'look_around', pos, c, self.turn_to)
                return True
            if not step_ok or now - self.t_mode > self.avoid_timeout:
                self.hold_pos = pos
                self.set_mode(now, 'REPLAN')
                self.event(now, 'stuck', pos, c, detail='no free direction' if not step_ok else 'avoid timeout')
            else:
                self.prev = th
                self.avoid_yaw = th
                if abs(LP.wrap(th - heading)) > math.radians(30):           # A: face the new direction first
                    self.avoid_sp = (pos[0], pos[1], self.alt)              # (no sideways motion into the blind rear)
                else:
                    self.avoid_sp = (pos[0] + self.vfh.step * math.cos(th), pos[1] + self.vfh.step * math.sin(th), self.alt)
                return True
        if self.mode == 'REPLAN':
            if now - self.t_replan >= self.replan_retry:
                self.t_replan = now
                self.replan(now, pos, scan, heading)
            return self.mode != 'FOLLOW'
        return True

    def vs_step(self, now, pos, scan, heading, target):
        """local = vs: one decision of the short vS episode to target; False -> VFH+ decides this tick."""
        if self.vs_env is None:
            if math.dist(pos[:2], target[:2]) < 0.5:
                return False
            w = self.planner.world
            st = w.to_units(np.array([[pos[0], pos[1], self.alt]]))
            gl = w.to_units(np.array([[target[0], target[1], self.alt]]))
            u = None if self.vs_u is None else torch.tensor([self.vs_u], dtype=torch.float64)
            self.vs_env = PathEnv(w, st, gl, self.vs_pref, budget=u, **env_kwargs(self.cfg, noise=False, critic=False))
            self.actor.reset(1)
            self.vs_node, self.vs_sp = 0, None
        if self.vs_sp is not None and math.dist(pos[:2], self.vs_sp[:2]) > 0.3:
            return True                                                    # flying to the current local node
        if self.vs_node >= N_NODES:
            self.vs_env = None
            return False
        self.env = self.vs_env                                             # SensorFollower.node_for works on self.env
        self.vs_env.scan_override = scan[None]
        o = self.vs_env.obs()
        for a in self.candidates(o):
            node = self.node_for(a)
            v = node[:2] - np.asarray(pos[:2])
            if np.linalg.norm(v) > 0.05 and not LP.path_blocked(scan, heading, pos, node, self.vfh):
                b = math.atan2(v[1], v[0])
                if abs(LP.wrap(b - heading)) > math.radians(30):           # face it first (blind rear)
                    self.actor.discard()
                    self.avoid_sp, self.avoid_yaw = (pos[0], pos[1], self.alt), b
                    return True
                self.actor.commit()
                self.vs_env.step(a)
                self.vs_node += 1
                self.vs_sp = (float(node[0]), float(node[1]), self.alt)
                self.avoid_sp, self.avoid_yaw, self.prev = self.vs_sp, b, b
                return True
        self.actor.discard()
        return False

    def adjust_setpoint(self, now, pos, p, yaw):
        if self.phase != 'PATROL' or self.mode == 'FOLLOW':
            return p, yaw
        if self.mode == 'AVOID' and self.avoid_sp is not None:
            return self.avoid_sp, math.pi / 2 - self.avoid_yaw             # ENU bearing -> NED yaw
        hp = self.hold_pos or pos
        return (hp[0], hp[1], self.alt), math.pi / 2 - self.yaw_enu()

    # --- the global planner -------------------------------------------------------------------------
    def replan(self, now, pos, scan, heading):
        for b in self.memory.boxes_from_scan(scan, pos, heading, sector=(-math.pi / 3, math.pi / 3)):
            self.memory.add(b, now)
        world = self.memory.world(now)
        self.planner.set_world(world)
        k = self.next_goal(self.s)
        if k is None:
            self.set_mode(now, 'FOLLOW')
            return
        if self.replans_failed >= self.max_replans and k + 1 < len(self.goals):
            k += 1                                                          # give up this patrol point
            self.event(now, 'goal_dropped', pos, detail=str(self.goals[k - 1][:2]))
        goal = self.goals[k]
        t0 = time.perf_counter()
        plan = self.planner.plan_route(pos[:2], goal[:2])
        if plan.feasible and self.adaptive_u and self.memory.near(plan.path[:, :2], now):
            keep = self.planner.budgets <= 0.3                              # more cautious near fresh obstacles
            prefs, budgets = self.planner.prefs, self.planner.budgets
            self.planner.prefs, self.planner.budgets = prefs[keep], budgets[keep]
            cautious = self.planner.plan_route(pos[:2], goal[:2])
            self.planner.prefs, self.planner.budgets = prefs, budgets
            plan = cautious if cautious.feasible else plan
        ms = (time.perf_counter() - t0) * 1000
        self.recent = [(t, p) for t, p in self.recent if now - t < 20.0] + [(now, pos)]
        if plan.feasible and sum(math.dist(p[:2], pos[:2]) < 1.0 for _, p in self.recent) >= 3:
            plan.feasible, plan.reason = False, 'no progress after repeated replans here'
            self.recent = []
        if not plan.feasible:
            self.replans_failed += 1
            self.event(now, 'replan_fail', pos, detail=f'{ms:.0f} ms: {plan.reason}')
            return
        gi = min(range(len(self.wp)), key=lambda j: math.dist(self.wp[j], goal))
        i = self.seg_index(self.s)
        new = [tuple(p) for p in self.wp[:i + 1]] + [tuple(pos)] + \
              [(float(x), float(y), self.alt) for x, y, _ in plan.path[1:]] + [tuple(p) for p in self.wp[gi + 1:]]
        self.set_patrol_track(new)
        self.s = self.cum[i + 1]                                           # at pos
        self.replans_failed = 0
        self.t_replan_ok = now
        self.set_mode(now, 'FOLLOW')
        self.event(now, 'replan_ok', pos, detail=f'{ms:.0f} ms, {plan.method}, {len(self.memory.items)} remembered')


def main():
    rclpy.init()
    node = GlobalLocalFollower()
    try:
        rclpy.spin(node)
    except SystemExit:
        node.get_logger().info(f'patrol finished; local events {node.counts}')
    finally:
        node.log_f.close()
        node.loc_f.close()
        with open(node.log_base + '_track.json', 'w') as f:
            json.dump({'PATROL': [list(p) for p in node.tracks['PATROL'][0]], 'counts': node.counts}, f)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
