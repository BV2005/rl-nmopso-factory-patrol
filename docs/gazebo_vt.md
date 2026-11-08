# Testing vT in Gazebo + PX4 SITL

vT3 was trained and evaluated in a batched PyTorch simulation (`rlpatrol/env_t.py`). This test flies it with the
real flight stack: PX4 SITL (EKF2, position controller, offboard setpoints over uXRCE-DDS), the `x500_patrol` drone's
rendered 2D lidar and depth camera in the factory2 world, and the patrol of `main` (take-off, transits, landing,
vertical-clearance and yield rules). Only the PATROL legs change: every leg between two patrol points is planned
online, one vT waypoint at a time, **from the sensors only** (no map).

It answers three questions the simulation cannot:
1. **Does the policy work on real sensor data?** The scan comes from `/droneN/scan` and `/droneN/depth/points`, not
   from ray casting against known boxes.
2. **Are the paths flyable by PX4?** The flown trajectory (10 Hz states) is checked against the true obstacle boxes.
3. **Is the time model right?** Every leg logs vT's model flight time next to the time flown in the simulator.

## How it fits together

```
 /droneN/scan (LaserScan)  ──► sensors.laserscan_points ──┐
 /droneN/depth/points      ──► sensors.level(roll, pitch) ─┼─► scan3d.flight_scan ──► (27 ranges, 27 heights)
 /droneN/fmu/out/vehicle_attitude, vehicle_local_position ─┘    (ranges to the zone box + 0.5 m, as in training)
                                                                          │
                     rlpatrol/vt_pilot.VTPilot: the vT policy on its training observation (walls-only TEnv as
                     bookkeeping, scans overridden), every step checked against the full-resolution scan (round
                     zones); fallbacks: the policy's samples, an action sweep, a backtrack                │
                                                                          ▼  next waypoint (world ENU)
 gazebo/nmopso_demo/vt_follower.py (a PatrolFollower): the carrot flies each segment at 1 m/s facing it ──► PX4
```

**Two scans from the same data.**
- *The policy's observation* reproduces the training scan from raw data (`scripts/test_vt_flight.py` [1], 150 random
  poses in factory2 with synthetic 1080-beam lidar and 320 × 240 depth images): 97 % of range bins within 0.25 m,
  the height channel equal on 98 % of bins. The zone around each return is a square in world axes
  (`scan3d._square_beams`), so a box's returns give exactly training's grown box. The remaining differences are
  zones the sensors cannot see: obstacle parts outside the camera's ±36.5° view or below the lidar plane, and
  occluded parts. In flight these are not known; the physical obstacle is still ≥ 0.5 m away there.
- *The safety check* of every step uses the full-resolution scan (1080 beams of 0.25°, `flight_scan(beams=True)`)
  with **round** zones (distance < 0.5 m to a return): training's collision model (`segment_check_box`). The square
  zones of the observation over-approximate obstacle corners, and the policy learned to cut those corners (they
  were never collisions); a square-zone or 10°-bin check refused such steps (thin partition, short legs). Beam
  points are taken from the sensor that saw them (depth: the camera, 0.132 m ahead of base_link).

## 1. Offline test first (any machine, CPU, ~15 min; `--quick` ~3 min)

```bash
python scripts/test_vt_flight.py            # [1] scan parity [2] helpers [3]-[5] pilot flights [6] vt_follower.py itself
```

[6] runs `vt_follower.py` unmodified under a fake ROS 2 / PX4 layer (stub `rclpy`, `px4_msgs`, `pymavlink`; a
kinematic drone following the setpoints; synthetic sensors through the node's own callbacks). Drone 1 of the
pad-safe plan flies take-off, transit, one lap of 22 PATROL legs planned online, transit back and landing.

## 2. The pod (Ubuntu 24.04 with an NVIDIA GPU; graphics capability for headless rendering)

```bash
# RunPod: an Ubuntu 24.04 image; containers need NVIDIA_DRIVER_CAPABILITIES=all (graphics, not just compute)
apt-get update && apt-get install -y sudo git python3-pip         # root containers often lack sudo
cd /workspace && git clone -b time-optimal-rl https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
NO_PIP=1 bash gazebo/setup_sim.sh            # ROS 2 Jazzy, Gazebo Harmonic, PX4 (tested commit), XRCE agent, px4_msgs, x500_patrol
# the flight code's Python packages in a venv (system site packages visible: the ROS 2 Python packages need them)
python3 -m venv --system-site-packages .venv && . .venv/bin/activate && export PYTHONNOUSERSITE=1
pip install -r requirements.txt pymavlink==2.4.49 --extra-index-url https://download.pytorch.org/whl/cpu
python3 -c "import sys; sys.path.insert(0, '.'); import rlpatrol.vt_pilot; print('vT pilot imports OK')"
python3 scripts/test_vt_flight.py --quick    # the offline test on the pod too
ls /usr/share/glvnd/egl_vendor.d/            # 10_nvidia.json: headless GPU rendering of the sensors
```

## 3. Check the sensor drone (once)

Fly the planned paths with `x500_patrol` (no policy) and verify the sensor assumptions of `docs/cloud_gpu.md` §4:
topic names, the point-cloud frame (`CLOUD_FRAME=flu` or `optical`), no lidar self-hits, real-time factor.
```bash
MODEL=x500_patrol bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
```

## 4. Fly vT

One drone first (fastest, no traffic), then all three:
```bash
python3 -c "import json; d = json.load(open('gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json')); \
d['drones'] = [x for x in d['drones'] if x['id'] == 1]; json.dump(d, open('/tmp/plan_drone1.json', 'w'))"
MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt TIMEOUT=1800 bash gazebo/fly_plan.sh /tmp/plan_drone1.json
run=$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)
python3 scripts/vt_flight_report.py "$run" --out results/flight_vt3_seed1_drone1.json

MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt TIMEOUT=2400 bash gazebo/fly_plan.sh \
    gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
run=$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)
python3 scripts/vt_flight_report.py "$run" --out results/flight_vt3_seed1.json
```
While it flies (another shell, `source /opt/ros/jazzy/setup.bash`):
```bash
tail -f "$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)"/follower_drone1.log
gz topic -e -t /world/factory/stats -n 1 | grep real_time_factor          # near 1; times are sim time either way
```

**Unexpected obstacle** (in no map; the drone has to see it): while drone 1 patrols, drop a box on its loop, e.g. on
the first PATROL leg ((−6.85, 3.65) → (−13.35, 3.65)):
```bash
bash gazebo/spawn_box.sh "$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)" -10.0 3.65     # 0.6 x 0.6 x 2.5 m
```
`flight_check.py` then counts collisions with it too (`<run>/extra_boxes.json`).

## 5. What to look at

| | Where | Good |
|---|---|---|
| legs succeeded | `vt_flight_report.py`: `success`, `outcomes` | all `success`; `blocked_*` / `timeout_*` legs need a look in `trace_droneN_vt.csv` |
| collisions | `flight_check.json` (`any_obstacle_collision`, `any_drone_collision`) | false |
| time model | `flown / model` | ≈ 1.0–1.3: above 1 by the stop at every waypoint (`settle`, the yaw to the next segment) and PX4's acceleration; much higher means the model misses something |
| fallbacks | `fallbacks` / `decisions` | small: the policy's own decision was clear in the real scan |
| scans | `trace_droneN_vt.csv` (`r0..r26`, `h0..h26`) | hovering in the open: no ranges < 0.1; walls appear where expected |

Bring the results home: `tar czf flight_vt.tgz gazebo/nmopso_demo/run_logs/<run> results/flight_vt3_*.json`.

## Parameters (`vt_follower.py`, `--ros-args -p name:=value`)

`flow` (true: fly through the waypoints), `lookahead` (0.6 m: decide this far before a waypoint),
`ckpt` (vT checkpoint), `pref` (3 weights over time, threat, altitude; default 1/3 each), `radius` (0.1 m: scan check
around the segment, on top of the 0.5 m margin), `samples` (16), `arrive_dist` (0.25 m), `settle` (0.3 s at a
waypoint before scanning, only without flow), `yaw_tol` (0.15 rad, only without flow), `max_scan_age` (0.5 s), `max_block` (10 s), `cloud_frame`
(`flu` / `optical`), `cloud_stride` (1), `use_depth` (true), `zmax_local` (2.6 m: highest climb).

## Known limits

- **Flying through the waypoints (`flow`, default on).** The first Gazebo + PX4 flight stopped at every waypoint: 22/22
  legs, 0 collisions, 0 fallbacks, but 2.93 × the time model. 80 % of the time the drone crawled at 0.15–0.6 m/s (mean
  0.33 m/s): with position-only setpoints PX4's position loop (`MPC_XY_P` ≈ 0.95 /s) trails a 1 m/s carrot by ~1 m and
  then approaches each short (0.4–1.5 m) segment's end at a speed proportional to the remaining distance. With `flow`
  the carrot's velocity is sent as feed-forward and the next decision is taken once the drone is within `lookahead`
  (0.6 m) of the waypoint, on the scan re-expressed at the waypoint (`scan3d.reframe`), so the drone never stops. In the
  offline test with a PX4-like drone: 2.21 × the time model stopping, 0.96 × flowing. `flow:=false` restores the
  first behaviour; the report's `wait_s` is the time spent hovering at waypoints.
- **Unobservable zones.** See above: parts of obstacles outside the sensors' view are not known in flight.
- **One policy per drone, no coordination.** The drones keep `main`'s zoning, transit layer and yield rule; vT does
  not see the other drones (multi-drone RL is the next step).
- **Simulated sensors.** Gazebo's lidar and depth camera are ideal ray casts (no stereo noise, no dropouts); the
  training scan had noise and dropout, so this is not the hardest case yet.
