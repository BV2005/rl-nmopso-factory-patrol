"""Offline test of the vT Gazebo pipeline (no ROS, no Gazebo; CPU, ~15 min): everything between the drone's raw
sensor data and its setpoints, with synthetic sensors (rlpatrol/synth_sensors.py: ray-cast 1080-beam lidar and
320 x 240 depth camera of x500_patrol in the factory2 world, low overflown boxes included).
  1. flight scan parity: scan3d.flight_scan of the raw data == the scan vT was trained on (sim_scan_b, margin)
  2. sensor helpers: LaserScan -> points, levelling with roll / pitch; a scan taken short of a waypoint re-expressed
     there from the pose it was taken at; the corner check of a step from where the drone is
  3. VTPilot on the 73 factory2 real legs (perfect tracking): success, no collision, env time == judge time
  4. tracking error (0.1 m per waypoint): still no collision
  5. unexpected boxes across legs (sensed, never in any map): avoided
  6. gazebo/nmopso_demo/vt_follower.py itself under a fake ROS 2 / PX4 layer with a PX4-like drone (position P-loop
     0.95 /s + velocity feed-forward, 3 m/s^2, yaw 45 deg/s): drone 1 of the pad-safe plan flies take-off, transit, one
     patrol lap planned online, transit back, landing - flowing through the waypoints (flow: feed-forward + decisions
     a waypoint ahead), with the drone tilting as it accelerates and the sensor data arriving 50 ms late; and, without
     --quick, stopping at every waypoint (as first flown in Gazebo: ~3x the time model) for comparison
  python scripts/test_vt_flight.py [--ckpt results/ckpt/vT3_seed1_best.pt] [--quick]"""
import argparse
import importlib
import json
import math
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol import sensors as S  # noqa: E402
from rlpatrol.env_t import flight_metrics  # noqa: E402
from rlpatrol.scan3d import H_MAX, flight_scan, reframe, sim_scan_b, walls  # noqa: E402
from rlpatrol.synth_sensors import depth_cloud, lidar_scan  # noqa: E402
from rlpatrol.vt_pilot import VTPilot  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402

OK = []
PLAN = ROOT / "gazebo" / "nmopso_demo" / "paths_rl_padsafe" / "patrol_paths.json"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def factory2():
    J = json.loads((ROOT / "data" / "factory2_model.json").read_text())
    w = load_world()
    hall = np.asarray(w.hall, float)
    W = walls(torch.tensor(hall)[None])[0].numpy()
    rects = np.asarray(w.rects, float)
    over = np.atleast_2d(np.array(J["overflown"], float))
    return hall, rects, W, np.vstack([rects, over, W])                  # sensed world: obstacles, low boxes, walls


def sense(boxes, pos, yaw, margin, beams=False):
    rs, a0, inc = lidar_scan(boxes, pos, yaw)
    return flight_scan(S.laserscan_points(rs, a0, inc), depth_cloud(boxes, pos, yaw), pos[2], margin, yaw, beams=beams)


def collides(path, boxes, margin):
    if len(path) < 2:                                                    # nothing flown
        return False
    P = torch.tensor(np.asarray(path, float))[None]
    kb = torch.tensor(np.asarray(boxes, float))[None]
    return bool(flight_metrics(P, torch.tensor([len(path)]), kb, torch.ones(1, len(boxes), dtype=torch.bool), margin=margin)["collides"][0])


def fly(pilot, a, b, boxes, track_sigma=0.0, rng=None):
    """one leg, offline: sense where the drone is, decide, fly to the target (+ tracking error) -> report, path."""
    pilot.start_leg(a, b)
    pos, path, why, own = np.asarray(a, float).copy(), [np.asarray(a, float).copy()], "timeout", 0
    while not pilot.over:
        d = pilot.decide(*sense(boxes, pos, pilot.heading, pilot.margin, beams=True))
        if d is None:
            why = "blocked"
            break
        own += d.source == "policy"
        pos = pilot.commit(d).copy()
        if track_sigma and rng is not None and not d.snap:
            pos[:2] += rng.normal(0, track_sigma, 2)
        path.append(pos.copy())
    rep = pilot.report()
    rep.update(why="success" if rep["success"] else why, own=own, path=path)
    return rep


def parity_checks(n_poses):
    print("[1] flight scan == training scan (synthetic raw sensors in factory2)")
    hall, rects, W, sensed = factory2()
    train = np.vstack([rects, W])
    rng = np.random.default_rng(0)
    M = 0.5
    dr, agree = [], []
    n = 0
    while n < n_poses:
        p = np.array([rng.uniform(hall[0] + 0.6, hall[1] - 0.6), rng.uniform(hall[2] + 0.6, hall[3] - 0.6), rng.uniform(1.6, 2.0)])
        if any(b[0] - M < p[0] < b[1] + M and b[2] - M < p[1] < b[3] + M for b in rects):
            continue
        yaw = rng.uniform(-math.pi, math.pi)
        rf, hf = sense(sensed, p, yaw, M)
        rs, hs = sim_scan_b(torch.tensor(train)[None], torch.ones(1, len(train), dtype=torch.bool), torch.tensor(p)[None],
                            torch.tensor([yaw]), S.SensorSpec(noise=False), margin=M)
        rs, hs = rs[0].numpy(), hs[0].numpy()
        dr.append((rf - rs) * S.SCAN_MAX)
        cls = lambda h: np.where(h == 0, 0, np.where(h >= 0.999, 2, 1))  # noqa: E731
        agree.append(cls(hf) == cls(hs))
        n += 1
    dr, agree = np.concatenate(dr), np.concatenate(agree)
    close, farther = (np.abs(dr) < 0.25).mean(), (dr > 0.25).mean()
    check(close >= 0.95 and farther <= 0.03,
          f"ranges: {close * 100:.1f} % of bins within 0.25 m of training's, flight farther by > 0.25 m on {farther * 100:.1f} % "
          f"(obstacle zones the sensors cannot see: outside the camera's view or occluded; mean diff {dr.mean():+.3f} m)")
    check(agree.mean() >= 0.95, f"height channel (none / known top / tall-or-unknown) agrees on {agree.mean() * 100:.1f} % of bins")


def helper_checks():
    print("[2] sensor helpers")
    xy = S.laserscan_points([1.0, np.inf, 0.05, 2.0], -math.pi / 2, math.pi / 2, 0.1, 30.0)
    check(xy.shape == (2, 2) and np.allclose(xy[0], [0, -1], atol=1e-12) and np.allclose(xy[1], [-2, 0], atol=1e-12),
          "laserscan_points: valid returns only, angle_min on the right (y < 0)")
    p = np.array([[2.0, 0.0, 0.0]])
    lv = S.level(p, 0.0, -math.radians(10))                              # nose up 10 deg (FLU pitch < 0)
    check(abs(lv[0, 2] - 2 * math.sin(math.radians(10))) < 1e-12 and abs(np.linalg.norm(lv) - 2) < 1e-12,
          "level: a point straight ahead seen with the nose 10 deg up is above the camera after levelling")
    r, h = flight_scan(None, None, 1.8, 0.5, 0.0)
    check(bool((r == 1).all() and (h == 0).all()), "flight_scan with no returns: all clear, no heights")


def reframe_checks(ckpt):
    print("[2b] deciding ahead: the scan re-expressed at the waypoint, the corner check")
    hall, rects, W, sensed = factory2()
    rng = np.random.default_rng(5)
    good, bad, n = [], [], 0
    while n < 30:
        w = np.array([rng.uniform(hall[0] + 1.5, hall[1] - 1.5), rng.uniform(hall[2] + 1.5, hall[3] - 1.5), 1.8])
        yaw = rng.uniform(-math.pi, math.pi)
        u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        a = w - 0.6 * u                                                  # the drone 0.6 m short of the waypoint
        if any(b[0] - 0.8 < q[0] < b[1] + 0.8 and b[2] - 0.8 < q[1] < b[3] + 0.8 for b in rects for q in (a, w)):
            continue
        rs, a0, inc = lidar_scan(sensed, a, yaw)
        lid, dep = S.laserscan_points(rs, a0, inc), depth_cloud(sensed, a, yaw)
        ref = flight_scan(*reframe(lid, dep, (*a, yaw), (*w, yaw)), w[2], 0.5, yaw)[0]
        late = flight_scan(*reframe(lid, dep, (*(a + 0.4 * u), yaw), (*w, yaw)), w[2], 0.5, yaw)[0]   # pose 0.4 m later
        truth = sense(sensed, w, yaw, 0.5)[0]
        good.append(np.abs(ref - truth) * S.SCAN_MAX)
        bad.append(np.abs(late - truth) * S.SCAN_MAX)
        n += 1
    good, bad = np.concatenate(good), np.concatenate(bad)
    check(np.median(good) < 0.05 and (good < 0.25).mean() >= 0.9 and np.mean(good) < np.mean(bad),
          f"a scan taken 0.6 m short of the waypoint, re-expressed there from the pose it was taken at: median bin error "
          f"{np.median(good):.3f} m, {(good < 0.25).mean() * 100:.0f} % within 0.25 m (from the pose 0.4 m later, as at the "
          f"decision: mean {np.mean(bad):.2f} m vs {np.mean(good):.2f} m)")
    # the corner: heading +x, the next step turns 90 deg left; a box inside the corner is clear of the step from the
    # waypoint but not of the chord the drone flies from 0.6 m short of it
    pilot = VTPilot(ckpt, hall)
    c = np.array([-3.0, -4.0, 1.8])
    pilot.start_leg(c, c + np.array([6.0, 0.0, 0.0]))
    box = np.array([c[0] - 1.6, c[0] - 0.72, c[1] + 0.68, c[1] + 2.0, 3.0])
    world = np.vstack([W, box[None]])
    sc = sense(world, c, 0.0, pilot.margin, beams=True)
    q = c + np.array([0.0, 0.6, 0.0])
    check(pilot.clear(q, *sc) and not pilot.clear(q, *sc, start=c - np.array([0.6, 0.0, 0.0])),
          "corner check: a box inside a 90 deg turn is clear of the step from the waypoint, not of the corner cut from 0.6 m short of it")


def pilot_checks(ckpt, n_legs, n_noisy, n_boxes):
    hall, rects, W, sensed = factory2()
    legs = loadmat(ROOT / "data" / "legs_real.mat")["legs"]
    pilot = VTPilot(ckpt, hall)
    print(f"[3] VTPilot on factory2's real legs, sensors only, perfect tracking ({n_legs} legs)")
    t0 = time.time()
    R = [fly(pilot, legs[i, :3], legs[i, 3:], sensed) for i in range(n_legs)]
    succ = np.mean([r["success"] for r in R])
    phys = sum(collides(r["path"], sensed, 0.3) for r in R)
    zone = sum(collides(r["path"], sensed, 0.5) for r in R)
    tr = [r["model_time_ratio"] for r in R if r["success"]]
    own = sum(r["own"] for r in R) / max(1, sum(r["decisions"] for r in R))
    check(succ >= 0.95 and phys == 0,
          f"success {succ * 100:.1f} %, physical collisions (drone r 0.3 m) {phys}, margin-zone entries {zone}; time / straight "
          f"{np.mean(tr):.3f}; the policy's own decision accepted {own * 100:.0f} % ({time.time() - t0:.0f} s)")
    P = torch.tensor(np.array(R[0]["path"]))[None]
    m = flight_metrics(P, torch.tensor([P.shape[1]]), torch.zeros(1, 1, 5, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.bool))
    check(abs(float(m["time_s"][0]) - R[0]["model_time_s"]) < 1e-6, "the pilot's flight time == the judge's time of the flown path")
    print(f"[4] tracking error 0.1 m per waypoint ({n_noisy} legs)")
    rng = np.random.default_rng(3)
    Rn = [fly(pilot, legs[i, :3], legs[i, 3:], sensed, 0.1, rng) for i in range(n_noisy)]
    succ = np.mean([r["success"] for r in Rn])
    phys = sum(collides(r["path"], sensed, 0.3) for r in Rn)
    check(succ >= 0.9 and phys == 0, f"success {succ * 100:.1f} %, physical collisions {phys}")
    print(f"[5] unexpected boxes across the legs ({n_boxes} legs; sensed only, in no map)")
    out, k = [], 0
    for i in range(len(legs)):
        if len(out) >= n_boxes:
            break
        a, b = legs[i, :3], legs[i, 3:]
        L = np.linalg.norm(b[:2] - a[:2])
        if L < 4.0:
            continue
        c = (a[:2] + b[:2]) / 2
        box = np.array([c[0] - 0.3, c[0] + 0.3, c[1] - 0.3, c[1] + 0.3, 2.5])
        world = np.vstack([sensed, box[None]])
        if any(not (bb[1] < box[0] - 1.0 or bb[0] > box[1] + 1.0 or bb[3] < box[2] - 1.0 or bb[2] > box[3] + 1.0) for bb in rects):
            continue                                                     # keep it in open space (a way round exists)
        r = fly(pilot, a, b, world)
        r["collide"] = collides(r["path"], world, 0.3)
        out.append(r)
        k += 1
    succ = np.mean([r["success"] for r in out])
    phys = sum(r["collide"] for r in out)
    check(len(out) >= min(3, n_boxes) and succ >= 0.75 and phys == 0,
          f"{len(out)} legs with a 0.6 x 0.6 x 2.5 m box on the straight line: success {succ * 100:.0f} %, physical collisions "
          f"{phys}, time / straight {np.mean([r['model_time_ratio'] for r in out if r['success']]):.2f}")


# --- 6. the ROS node under a fake ROS 2 / PX4 layer ------------------------------------------------------------------
class _Clock:
    t = 0.0


def _fake_ros(params):
    """stub modules for rclpy, px4_msgs, sensor_msgs, std_msgs, pymavlink: the node's publications, subscriptions,
    timer and MAVLink commands are recorded; time.monotonic of the follower modules is the simulated clock."""
    class Msg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class VehicleStatus(Msg):
        ARMING_STATE_ARMED = 2
        NAVIGATION_STATE_OFFBOARD = 14

    class Param:
        def __init__(self, v):
            self.value = v

    class Logger:
        def info(self, s):
            pass

        def warn(self, s):
            print(f"      [follower] {s}", flush=True)

    class Node:
        def __init__(self, name):
            self._p, self.subs, self.pubs, self.timers = {}, {}, {}, []

        def declare_parameter(self, k, d):
            self._p[k] = params.get(k, d)

        def get_parameter(self, k):
            return Param(self._p[k])

        def create_subscription(self, t, topic, cb, qos):
            self.subs[topic] = cb

        def create_publisher(self, t, topic, depth):
            pub = types.SimpleNamespace(msgs=[])
            pub.publish = pub.msgs.append
            self.pubs[topic] = pub
            return pub

        def create_timer(self, period, cb):
            self.timers.append(cb)

        def get_logger(self):
            return Logger()

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=int(_Clock.t * 1e9)))

        def destroy_node(self):
            pass

    class Mav:
        def __init__(self):
            self.cmds = []
            self.mav = types.SimpleNamespace(heartbeat_send=lambda *a: None, command_long_send=lambda *a: self.cmds.append(a))

        def recv_match(self, blocking=False):
            return None

    mavlink = types.SimpleNamespace(MAV_TYPE_GCS=6, MAV_AUTOPILOT_INVALID=8, MAV_CMD_DO_SET_MODE=176,
                                    MAV_CMD_COMPONENT_ARM_DISARM=400, MAV_CMD_NAV_LAND=21)
    mods = {
        "rclpy": types.SimpleNamespace(init=lambda: None, spin=lambda n: None, try_shutdown=lambda: None),
        "rclpy.node": types.SimpleNamespace(Node=Node),
        "rclpy.qos": types.SimpleNamespace(QoSProfile=lambda **kw: None, DurabilityPolicy=types.SimpleNamespace(VOLATILE=0),
                                           HistoryPolicy=types.SimpleNamespace(KEEP_LAST=0), ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=0)),
        "px4_msgs": types.SimpleNamespace(), "px4_msgs.msg": types.SimpleNamespace(
            OffboardControlMode=Msg, TrajectorySetpoint=Msg, VehicleLocalPosition=Msg, VehicleStatus=VehicleStatus, VehicleAttitude=Msg),
        "std_msgs": types.SimpleNamespace(), "std_msgs.msg": types.SimpleNamespace(String=Msg),
        "sensor_msgs": types.SimpleNamespace(), "sensor_msgs.msg": types.SimpleNamespace(LaserScan=Msg, PointCloud2=Msg),
        "sensor_msgs_py": types.SimpleNamespace(point_cloud2=types.SimpleNamespace(
            read_points_numpy=lambda m, field_names=None, skip_nans=True: m.pts.astype(np.float32))),
        "pymavlink": types.SimpleNamespace(mavutil=types.SimpleNamespace(mavlink=mavlink, mavlink_connection=lambda *a, **k: Mav())),
    }
    mods["sensor_msgs_py.point_cloud2"] = mods["sensor_msgs_py"].point_cloud2
    mods["pymavlink.mavutil"] = mods["pymavlink"].mavutil
    mods["rclpy"].node, mods["rclpy"].qos = mods["rclpy.node"], mods["rclpy.qos"]
    sys.modules.update(mods)
    return Msg, VehicleStatus


def follower_checks(ckpt, drone=1, max_sim_s=1200.0, flow=True):
    mode = "flowing through the waypoints" if flow else "stopping at every waypoint"
    print(f"[6] vt_follower.py under a fake ROS 2 / PX4 layer: drone {drone} of the pad-safe plan, one lap, {mode}")
    hall, rects, W, sensed = factory2()
    plan = json.loads(PLAN.read_text())
    me = next(d for d in plan["drones"] if d["id"] == drone)
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / f"trace_drone{drone}.csv"
        Msg, VS = _fake_ros(dict(drone_id=drone, paths_file=str(PLAN), log_file=str(log), laps=1, speed=1.0, ckpt=str(ckpt),
                                 max_scan_age=1.0, yield_radius=0.0, flow=flow))
        sys.path.insert(0, str(ROOT / "gazebo" / "nmopso_demo"))
        for m in ("patrol_follower", "vt_follower"):
            sys.modules.pop(m, None)
        pf = importlib.import_module("patrol_follower")
        vf = importlib.import_module("vt_follower")
        fake_time = types.SimpleNamespace(monotonic=lambda: _Clock.t)
        pf.time = vf.time = fake_time
        _Clock.t = 0.0
        node = vf.VTFollower()
        home = me["home"]
        p = np.array([home[0], home[1], 0.0])                            # the drone, world ENU
        vel = np.zeros(3)
        yaw = 0.0                                                        # ENU
        armed = offboard = landing = False
        states, dt = [], 0.05
        last_sense = -1e9
        acc = np.zeros(3)
        hist = []                                                        # (p, yaw) per tick: sensor data 1 tick (50 ms) late
        ffs = []                                                         # (t, phase, |feed-forward|) per setpoint
        ns = f"/drone{drone}"
        tick = node.timers[0]
        done = False
        while _Clock.t < max_sim_s:
            ned_yaw = math.pi / 2 - yaw
            node.subs[f"{ns}/fmu/out/vehicle_local_position_v1"](Msg(x=p[1] - home[1], y=p[0] - home[0], z=-p[2], heading=ned_yaw,
                                                                     xy_valid=True, z_valid=True, timestamp=int(_Clock.t * 1e6)))
            node.subs[f"{ns}/fmu/out/vehicle_status_v4"](Msg(arming_state=VS.ARMING_STATE_ARMED if armed else 1,
                                                             nav_state=VS.NAVIGATION_STATE_OFFBOARD if offboard else 0))
            # attitude: the tilt that gives the acceleration (FRD: nose down to accelerate forward, right wing down right)
            af = acc[0] * math.cos(yaw) + acc[1] * math.sin(yaw)
            al = -acc[0] * math.sin(yaw) + acc[1] * math.cos(yaw)
            pr, rr = -math.atan2(af, 9.81) / 2, -math.atan2(al, 9.81) / 2
            node.subs[f"{ns}/fmu/out/vehicle_attitude"](Msg(q=[math.cos(rr) * math.cos(pr), math.sin(rr) * math.cos(pr),
                                                                math.cos(rr) * math.sin(pr), -math.sin(rr) * math.sin(pr)]))
            hist = (hist + [(p.copy(), yaw)])[-2:]
            # sensors at 10 Hz near the end of the online track (where the drone decides), taken one tick before they arrive
            if node.phase == "PATROL" and _Clock.t - last_sense > 0.099 and math.dist(p, node.wp[-1]) < 1.2:
                ps, ys = hist[0]
                rs, a0, inc = lidar_scan(sensed, ps, ys)
                node.subs[f"{ns}/scan"](Msg(ranges=rs.tolist(), angle_min=a0, angle_increment=inc, range_min=0.1, range_max=30.0))
                node.subs[f"{ns}/depth/points"](Msg(pts=depth_cloud(sensed, ps, ys)))
                last_sense = _Clock.t
            try:
                tick()
            except SystemExit:
                done = True
                break
            for c in node.mav.cmds:                                      # MAVLink commands
                cmd, p1 = c[2], c[4]
                if cmd == 176:
                    offboard = True
                elif cmd == 400:
                    armed = p1 > 0.5
                elif cmd == 21:
                    landing = True
            node.mav.cmds.clear()
            sp = node.pubs[f"{ns}/fmu/in/trajectory_setpoint"].msgs
            if armed and sp:
                m = sp[-1]
                tgt = np.array([home[0] + m.position[1], home[1] + m.position[0], -m.position[2]])
                if landing:
                    tgt[2] = 0.0
                # PX4-like multicopter position control: velocity setpoint = feed-forward + P * position error
                # (MPC_XY_P 0.95, MPC_Z_P 1.0), limited to 5 m/s horizontally / 1 m/s vertically, acceleration
                # 3 m/s^2 (MPC_ACC_HOR), yaw rate 45 deg/s (MPC_YAWRAUTO_MAX)
                ffv = getattr(m, "velocity", None)
                ff = np.array([ffv[1], ffv[0], -ffv[2]]) if ffv is not None and not landing else np.zeros(3)
                ffs.append((_Clock.t, node.phase, None if ffv is None else float(np.linalg.norm(ff))))
                e = tgt - p
                vsp = ff + np.array([0.95 * e[0], 0.95 * e[1], 1.0 * e[2]])
                hn = float(np.linalg.norm(vsp[:2]))
                if hn > 5.0:
                    vsp[:2] *= 5.0 / hn
                vsp[2] = float(np.clip(vsp[2], -1.0, 1.0))
                dv = vsp - vel
                dn = float(np.linalg.norm(dv))
                if dn > 3.0 * dt:
                    dv *= 3.0 * dt / dn
                vel = vel + dv
                acc = dv / dt
                p = p + vel * dt
                ey = (math.pi / 2 - m.yaw) - yaw                         # setpoint NED yaw -> ENU
                yaw += float(np.clip(math.atan2(math.sin(ey), math.cos(ey)), -math.radians(45) * dt, math.radians(45) * dt))
                sp[:] = sp[-1:]
            states.append((_Clock.t, *p, node.phase))
            _Clock.t += dt
        node.log_f.close(); node.dec_f.close(); node.leg_f.close()
        rows = [l.split(",") for l in (Path(tmp) / f"trace_drone{drone}_vt_legs.csv").read_text().splitlines()[1:]]
        import csv
        dec_all = list(csv.DictReader(open(Path(tmp) / f"trace_drone{drone}_vt.csv")))
        dec_rows = [r for r in dec_all if r["outcome"] == "ok"]
        import subprocess
        rep = subprocess.run([sys.executable, str(ROOT / "scripts" / "vt_flight_report.py"), tmp], capture_output=True, text=True)
    outcomes = [r[8] for r in rows]
    path = np.array([s[1:4] for s in states if s[4] == "PATROL"], float)
    phys = collides(path[::4], sensed, 0.3) if len(path) > 1 else True
    model = sum(float(r[13]) for r in rows if r[8] == "success")
    flown = sum(float(r[14]) for r in rows if r[8] == "success")
    check(done and len(rows) == len(node.legs) and all(o == "success" for o in outcomes) and model > 0 and flown > 0,
          f"take-off, transit, {len(rows)} patrol legs planned online (all '{outcomes[0] if outcomes else '-'}'), transit back and "
          f"landing in {_Clock.t:.0f} s simulated; {len(dec_rows)} decisions logged")
    check(not phys, "the flown patrol (PX4-like drone) never comes within 0.3 m of an obstacle")
    check(rep.returncode == 0 and "success 100.0 %" in rep.stdout,
          "scripts/vt_flight_report.py summarises the run's logs" + ("" if rep.returncode == 0 else f": {rep.stderr[-400:]}"))
    print("\n".join("      " + l for l in rep.stdout.splitlines()[:2]))
    dyaw = max(float(r["scan_dyaw"]) for r in dec_rows)
    tilt = max(float(r["scan_tilt"]) for r in dec_rows)
    check(tilt <= node.max_tilt + 1e-9 and (not flow or dyaw <= node.yaw_tol + 1e-9),
          f"every decision from sensor data taken level (tilt <= {tilt:.3f} rad, max_tilt {node.max_tilt})"
          + (f" and facing the heading (<= {math.degrees(dyaw):.1f} deg)" if flow else "")
          + f"; {sum(r['outcome'] == 'corner_stop' for r in dec_all)} corner stops")
    if flow:
        pat = [(t, v) for t, ph, v in ffs if ph == "PATROL" and v is not None]
        # braking to an undecided end: the carrot slows at brake_acc (the last ticks of the profile, below ~0.35 m/s, are
        # coarser); the first flow version dropped `speed` (1 m/s) to 0 in one tick there
        step = max((v0 - v1 for (_, v0), (_, v1) in zip(pat, pat[1:])), default=0.0)
        fast = max((v0 - v1 for (_, v0), (_, v1) in zip(pat, pat[1:]) if v1 >= 0.35), default=0.0) / 0.05
        tr = [v for t, ph, v in ffs if ph.startswith("TRANSIT")]
        check(len(pat) > 0 and step <= 0.4 and fast <= node.brake_acc + 0.2 and all(v is None for v in tr),
              f"feed-forward only on the PATROL legs; it never drops by more than {step:.2f} m/s in a tick (speed {node.speed}), "
              f"and above 0.35 m/s at {fast:.2f} m/s^2 at most (brake_acc {node.brake_acc})")
    print(f"      time model {model:.1f} s vs flown {flown:.1f} s over the successful legs: ratio {flown / model:.2f}")
    return flown / model if model > 0 else math.inf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "ckpt" / "vT3_seed1_best.pt"))
    ap.add_argument("--quick", action="store_true", help="fewer poses and legs (~3 min)")
    a = ap.parse_args()
    torch.set_num_threads(4)
    q = a.quick
    parity_checks(40 if q else 150)
    helper_checks()
    reframe_checks(a.ckpt)
    pilot_checks(a.ckpt, 15 if q else 73, 8 if q else 25, 3 if q else 8)
    flow = follower_checks(a.ckpt, flow=True)
    stop = math.nan if q else follower_checks(a.ckpt, flow=False)
    check(math.isfinite(flow) and flow < 1.6 and (q or (math.isfinite(stop) and flow < stop)),
          f"flowing through the waypoints flies the patrol at {flow:.2f} x the time model"
          + ("" if q else f", stopping at each {stop:.2f} x") + " (Gazebo + PX4 measured 2.93 x stopping)")
    print(f"all {len(OK)} checks passed")


if __name__ == "__main__":
    main()
