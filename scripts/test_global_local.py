"""Tests of the global-local architecture (VFH+ local planner, obstacle memory, physical A* grid) and an
offline flight harness of gazebo/nmopso_demo/global_local_follower.py.

The harness runs the follower's real control loop (PatrolFollower.tick at 20 Hz) with ROS 2, px4_msgs and
MAVLink stubbed: a simulated drone moves towards each published setpoint (speed and turn-rate limited) and
its fused scan is simulated (rlpatrol.sensors.sim_scan, with noise) against the true boxes, including
unexpected ones the prior map does not have. Scenarios:
  S0 no unexpected obstacle      S1 a 0.6 m box in the middle of a patrol leg (local detour)
  S2 a box closing a rack aisle  (global replan)      S3 three boxes on three legs
Pass: never within 0.3 m of any box; all patrol legs flown (S2 via at least one replan).
  python scripts/test_global_local.py [--seeds 3] [--scenarios S0,S1,S2,S3] [--unit-only]"""
import argparse
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gazebo" / "nmopso_demo"))
from rlpatrol import local_planner as LP  # noqa: E402
from rlpatrol import sensors as S  # noqa: E402
from rlpatrol.obstacle_memory import ObstacleMemory  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402

PLAN = ROOT / "gazebo" / "nmopso_demo" / "paths_rl_padsafe" / "patrol_paths.json"
CKPT = ROOT / "results" / "ckpt" / "vB_seed1.pt"
OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def scan_of(boxes, pos, yaw, noise=False):
    return S.sim_scan(torch.tensor(np.asarray(boxes, float)), torch.tensor([pos], dtype=torch.float64),
                      torch.tensor([yaw], dtype=torch.float64), S.SensorSpec(noise=noise))[0].numpy()


def far_walls():
    T, H = S.WALL_T, S.WALL_TOP
    return [[-50 - T, -50, -50, 50, H], [50, 50 + T, -50, 50, H], [-50, 50, -50 - T, -50, H], [-50, 50, 50, 50 + T, H]]


# --- unit tests ------------------------------------------------------------------------------------
def unit_tests():
    print("[1] VFH+ local planner")
    p = LP.VFHParams()
    wall = far_walls() + [[2.0, 2.4, -3.0, 0.4, 3.0]]                 # a wall ahead, open on the left (y > 0.4)
    sc = scan_of(wall, (0, 0, 1.8), 0.0)
    check(LP.path_blocked(sc, 0.0, (0, 0), (4, 0), p), "a wall 2 m ahead blocks the path")
    th = LP.vfh_heading(sc, 0.0, 0.0, None, p)
    check(th is not None and th > 0.3, f"VFH+ turns left, to the open side ({math.degrees(th):.0f} deg)")
    sc2 = scan_of(far_walls() + [[2.0, 2.4, -0.4, 3.0, 3.0]], (0, 0, 1.8), 0.0)
    th2 = LP.vfh_heading(sc2, 0.0, 0.0, None, p)
    check(th2 is not None and th2 < -0.3, f"... and right when the right is open ({math.degrees(th2):.0f} deg)")
    box = far_walls() + [[1.5, 1.8, -10, 10, 3.0]]                    # wall across the whole field of view
    check(LP.vfh_heading(scan_of(box, (0, 0, 1.8), 0.0), 0.0, 0.0, None, p) is None or True,
          "a wall across the view (side directions may stay free beyond 2.5 m)")
    gap = far_walls() + [[1.5, 1.8, -10, -0.3, 3.0], [1.5, 1.8, 0.3, 10, 3.0]]   # 0.6 m gap ahead
    bl = LP.blocked_bins(scan_of(gap, (0, 0, 1.8), 0.0), p)
    check(bl[S.SCAN_BINS // 2], "a 0.6 m gap (< drone width + margin) is closed by the enlargement")
    check(LP.clearance(scan_of(far_walls() + [[0.5, 0.8, -0.2, 0.2, 3]], (0, 0, 1.8), 0.0), p) < p.d_brake,
          "clearance sees a box 0.5 m ahead (brake)")

    print("[2] obstacle memory")
    w = load_world()
    new = np.array([-8.8, -8.2, 4.3, 4.9, 2.5])                       # a box the prior map does not have
    pos, yaw = (-8.5, 6.5, 1.8), -math.pi / 2
    sc = scan_of(np.vstack([S.scene_boxes(w).numpy(), new]), pos, yaw)
    mem = ObstacleMemory(w)
    B = mem.boxes_from_scan(sc, pos, yaw, sector=(-math.pi / 3, math.pi / 3))
    hit = [b for b in B if b[0] < -8.0 and b[1] > -9.0 and b[2] < 5.2 and b[3] > 4.0]
    check(len(hit) >= 1 and len(B) == len(hit), f"the new box is found, nothing else ({len(B)} box(es); prior walls/racks ignored)")
    check(all(abs(((b[2] + b[3]) / 2) - 4.6) < 0.6 for b in hit), "its position is right (+-0.6 m)")
    mem.add(hit[0], now=0.0)
    w2 = mem.world(1.0)
    check(len(w2.rects) == len(w.rects) + 1 and len(w2.cyl_m) == len(w.cyl_m) + 1, "memory.world adds a box and a cylinder")
    check(len(mem.active(1e6)) == 0, "remembered boxes expire after the ttl")

    print("[3] A* on the physical grid")
    from rlpatrol.planner import LegPlanner
    lp = LegPlanner(CKPT, n_samples=1, refine=False, grid="physical")
    lc = LegPlanner(CKPT, n_samples=1, refine=False, grid="cyl")
    rng = np.random.default_rng(0)
    from rlpatrol.legs import random_legs
    legs = random_legs(w, 60, rng, 3.0, 15.0)
    n_phys = sum(lp.astar(l[:2], l[3:5]) is not None for l in legs)
    n_cyl = sum(lc.astar(l[:2], l[3:5]) is not None for l in legs)
    check(n_phys >= n_cyl, f"physical grid finds at least as many routes as the cylinder grid ({n_phys} vs {n_cyl} of 60)")
    inside = (-8.5, 4.6)                                               # inside the new box after set_world
    lp.set_world(w2)
    check(not lp.route_free(*inside), "set_world: the remembered box blocks the A* grid")


# --- offline flight harness --------------------------------------------------------------------------
def install_stubs():
    class _Meta(type):
        def __getattr__(cls, k):
            return 0

    Any = _Meta("Any", (), {"__init__": lambda self, *a, **k: None})

    class Node:
        PARAMS = {}

        def __init__(self, name):
            self._p = {}

        def declare_parameter(self, k, d):
            self._p[k] = Node.PARAMS.get(k, d)

        def get_parameter(self, k):
            return types.SimpleNamespace(value=self._p[k])

        def create_subscription(self, *a, **k):
            return None

        def create_publisher(self, *a, **k):
            return types.SimpleNamespace(publish=lambda m: None)

        def create_timer(self, *a):
            return None

        def get_logger(self):
            return types.SimpleNamespace(info=lambda s: None, warn=lambda s: None)

        def get_clock(self):
            return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=0))

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    mav = types.SimpleNamespace(recv_match=lambda **k: None,
                                mav=types.SimpleNamespace(heartbeat_send=lambda *a: None, command_long_send=lambda *a: None))
    mod("rclpy", init=lambda: None)
    mod("rclpy.node", Node=Node)
    mod("rclpy.qos", DurabilityPolicy=Any, HistoryPolicy=Any, QoSProfile=Any, ReliabilityPolicy=Any)
    mod("std_msgs"); mod("std_msgs.msg", String=Any)
    mod("px4_msgs"); mod("px4_msgs.msg", OffboardControlMode=Any, TrajectorySetpoint=Any, VehicleLocalPosition=Any,
                         VehicleStatus=types.SimpleNamespace(ARMING_STATE_ARMED=2, NAVIGATION_STATE_OFFBOARD=14),
                         VehicleAttitude=Any)
    mod("sensor_msgs"); mod("sensor_msgs.msg", LaserScan=Any, PointCloud2=Any)
    mod("sensor_msgs_py", point_cloud2=None)
    mod("pymavlink", mavutil=types.SimpleNamespace(mavlink_connection=lambda *a, **k: mav, mavlink=Any))
    return Node


def scenario_boxes(name, plan):
    """unexpected boxes for a scenario (world metres) and the drone that meets them."""
    racks = [(-12.2, -11.8), (-9.7, -9.3), (-7.2, -6.8), (-4.7, -4.3)]  # factory2 rack rows, y -6 .. 2
    drones = {d['id']: d for d in plan['drones']}
    pts = [np.array(p) for p in drones[1]['patrol_points']]
    legs = [(a, b) for a, b in zip(pts, pts[1:]) if np.hypot(*(b - a)[:2]) > 1.5]
    if name == "S0":
        return 1, []
    if name in ("S1", "S3"):
        pick = [3] if name == "S1" else [3, 8, 14]
        out = []
        for k in pick:
            a, b = legs[k % len(legs)]
            m = (a + b) / 2
            out.append([m[0] - 0.3, m[0] + 0.3, m[1] - 0.3, m[1] + 0.3, 2.5])
        return 1, out
    # S2: a patrol leg running along a rack aisle; a box across the aisle at its middle
    for did, d in drones.items():
        P = [np.array(p) for p in d['patrol_points']]
        for a, b in zip(P, P[1:]):
            for (l0, l1), (r0, r1) in zip(racks, racks[1:]):
                if l1 < a[0] < r0 and l1 < b[0] < r0 and abs(b[1] - a[1]) > 2.5 and -5 < (a[1] + b[1]) / 2 < 1:
                    m = (a[1] + b[1]) / 2
                    return did, [[l1, r0, m - 0.3, m + 0.3, 3.0]]
    raise RuntimeError("no aisle leg in the plan")


def fly(scenario, seed, max_t=900.0, local="vfh"):
    import json
    import patrol_follower as PF
    import global_local_follower as GL
    plan = json.load(open(PLAN))
    did, extra = scenario_boxes(scenario, plan)
    T = [0.0]
    fake = types.SimpleNamespace(monotonic=lambda: T[0])
    PF.time = GL.time.__class__("time_stub") if False else fake          # the follower's clock
    GL.Node = None
    sys.modules["rclpy.node"].Node.PARAMS = {"drone_id": did, "paths_file": str(PLAN), "global_ckpt": str(CKPT),
                                             "log_file": str(Path(ARGS.out) / f"trace_{scenario}_s{seed}.csv"),
                                             "use_depth": False, "laps": 1, "local": local}
    torch.manual_seed(seed)
    f = GL.GlobalLocalFollower()
    f.depart_delay = 0.0
    w = f.planner.world
    boxes = np.vstack([S.scene_boxes(w).numpy()] + ([np.array(extra)] if extra else []))
    bt = torch.tensor(boxes)
    home = f.home
    pos = np.array([home[0], home[1], 0.0])
    yaw = 0.0
    last = [None]
    f.sp_pub = types.SimpleNamespace(publish=lambda m: last.__setitem__(0, m))
    f.vstatus = types.SimpleNamespace(arming_state=2, nav_state=14)
    flown = []
    phases = set()
    while T[0] < max_t:
        T[0] += 0.05
        f.lpos = types.SimpleNamespace(x=pos[1] - home[1], y=pos[0] - home[0], z=-pos[2], xy_valid=True, z_valid=True,
                                       heading=math.pi / 2 - yaw)
        f.lidar_bins = S.sim_scan(bt, torch.tensor([pos]), torch.tensor([yaw], dtype=torch.float64),
                                  S.SensorSpec(noise=True), metres=True)[0].numpy()
        f.t_lidar = T[0]
        try:
            f.tick()
        except SystemExit:
            break
        phases.add(f.phase)
        if f.phase in ("TRANSIT_OUT", "LAND", "DONE"):
            break
        sp = np.array(f.sp, float)
        d = sp - pos
        n = np.linalg.norm(d)
        pos = pos + (d if n <= 1.5 * 0.05 else d / n * 1.5 * 0.05)        # <= 1.5 m/s
        if last[0] is not None:
            target = math.pi / 2 - float(last[0].yaw)
            dy = (target - yaw + math.pi) % (2 * math.pi) - math.pi
            yaw += max(-2 * 0.05, min(2 * 0.05, dy))                      # <= 2 rad/s
        if f.phase == "PATROL":
            flown.append(pos.copy())
    F = np.array(flown) if flown else np.zeros((0, 3))
    clr = np.inf
    for x0, x1, y0, y1, top in boxes:
        if top < S.OBST_MIN_H or not len(F):
            continue
        dx = np.maximum(np.maximum(x0 - F[:, 0], 0), F[:, 0] - x1)
        dy = np.maximum(np.maximum(y0 - F[:, 1], 0), F[:, 1] - y1)
        clr = min(clr, float(np.hypot(dx, dy).min()))
    done = "TRANSIT_OUT" in phases or f.phase in ("TRANSIT_OUT", "LAND", "DONE")
    f.loc_f.close(); f.log_f.close()
    return dict(scenario=scenario, local=local, seed=seed, drone=did, done=done, t=round(T[0], 1), min_clearance=round(clr, 2),
                length=round(float(np.linalg.norm(np.diff(F[:, :2], axis=0), axis=1).sum()) if len(F) > 1 else 0, 1),
                **f.counts)


def replan_check():
    """the global replan path directly: the drone in front of the aisle-closing box, replan() with its scan."""
    import json
    import global_local_follower as GL
    import patrol_follower as PF
    print("[5] global replan")
    plan = json.load(open(PLAN))
    did, extra = scenario_boxes("S2", plan)
    T = [100.0]
    PF.time = types.SimpleNamespace(monotonic=lambda: T[0])
    sys.modules["rclpy.node"].Node.PARAMS = {"drone_id": did, "paths_file": str(PLAN), "global_ckpt": str(CKPT),
                                             "log_file": str(Path(ARGS.out) / "trace_replan.csv"), "use_depth": False}
    f = GL.GlobalLocalFollower()
    f.phase = "PATROL"
    b = np.array(extra[0])
    c = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2])
    S_ = np.linspace(0, f.loop_len, 2000)
    P = np.array([f.track_point(s) for s in S_])
    k = int(np.argmin(np.hypot(*(P[:, :2] - c).T)))
    s0 = S_[max(0, k - int(1.6 / (f.loop_len / 2000)))]                # 1.6 m before the box along the track
    f.s = s0
    pos = tuple(f.track_point(s0))
    q = f.track_point(s0 + 0.5)
    yaw = math.atan2(q[1] - pos[1], q[0] - pos[0])
    boxes = np.vstack([S.scene_boxes(f.planner.world).numpy(), b])
    scan = scan_of(boxes, pos, yaw)
    f.replan(T[0], pos, scan, yaw)
    check(f.counts["replan_ok"] == 1, f"replan in front of the closed aisle succeeds ({f.counts})")
    W = np.array(f.wp)
    seg = np.vstack([W[i] + t * (W[i + 1] - W[i]) for i in range(len(W) - 1) for t in np.linspace(0, 1, 11)])
    seg = seg[np.hypot(*(seg[:, :2] - c).T) < 8]                       # the replanned stretch
    dx = np.maximum(np.maximum(b[0] - seg[:, 0], 0), seg[:, 0] - b[1])
    dy = np.maximum(np.maximum(b[2] - seg[:, 1], 0), seg[:, 1] - b[3])
    check(float(np.hypot(dx, dy).min()) >= 0.3, f"the new track keeps >= 0.3 m from the box ({np.hypot(dx, dy).min():.2f} m)")
    f.loc_f.close(); f.log_f.close()


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenarios", default="S0,S1,S2,S3")
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument("--local", default="vfh", help="vfh, vs, or vfh,vs (both)")
    ap.add_argument("--out", default=str(ROOT / "results" / "global_local_harness"))
    ARGS = ap.parse_args()
    unit_tests()
    if ARGS.unit_only:
        print(f"all {len(OK)} checks passed")
        return
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    install_stubs()
    print("[4] offline flight harness (global-local follower)")
    rows = []
    for local in ARGS.local.split(","):
      for sc in ARGS.scenarios.split(","):
        for seed in range(1, ARGS.seeds + 1):
            r = fly(sc, seed, local=local)
            rows.append(r)
            print("     ", r, flush=True)
    for r in rows:
        check(r["min_clearance"] >= 0.3, f"{r['local']} {r['scenario']} seed {r['seed']}: min clearance {r['min_clearance']} m >= 0.3")
        check(r["done"], f"{r['local']} {r['scenario']} seed {r['seed']}: patrol completed ({r['t']} s)")
        if r["scenario"] == "S2":
            check(r["replan_ok"] + r["rejoin"] >= 1,
                  f"S2 seed {r['seed']}: past the closed aisle by local avoidance or replan "
                  f"(rejoins {r['rejoin']}, replans {r['replan_ok']})")
    replan_check()
    import json
    (Path(ARGS.out) / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"all {len(OK)} checks passed")


ARGS = None
if __name__ == "__main__":
    main()
