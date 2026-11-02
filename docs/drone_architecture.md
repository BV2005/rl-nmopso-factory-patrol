# x500_patrol: the sensor drone and the sensor-input policy

Until this branch the patrol drones were PX4's stock `gz_x500` with simulated GPS and **no perception
sensors**: every leg was planned offline on the known map (`data/factory2_model.json`), and Gazebo
only checked that the planned paths are flyable. Nothing ever read a sensor. This branch adds:

1. **The drone**: `gazebo/models/x500_patrol`, an x500 with a 2D lidar and a forward stereo-depth +
   RGB camera.
2. **A policy whose inputs come from those sensors**: `PathEnv(sensor_obs=True)`, trained with
   `configs/vS.yaml`. The actor sees only what the drone can sense; the simulated scan used in
   training and the real scan in flight share one definition (`rlpatrol/sensors.py`).
3. **Closed-loop flight**: `gazebo/nmopso_demo/sensor_follower.py` plans every PATROL leg while
   flying it, one node per scan.

## Hardware architecture

```
                 +--------------------------- companion computer (Jetson-class, ROS 2 Jazzy) ---------------------------+
 2D lidar  --->  |  /droneN/scan  --+                                                                                    |
 (270 deg)       |                  +--> rlpatrol.sensors: 27-bin fused scan --> sensor policy (MLP) --> next node       |
 depth cam --->  |  /droneN/depth/points                    ^                       |                                     |
 (OAK-D Lite)    |                                          |                       v                                     |
 RGB cam   --->  |  /droneN/camera/image (later: VLM)        |      segment_clear (scan safety check) --> setpoint track    |
                 +------------------------------------------|---------------------------------------------|---------------+
                                                            | pose, attitude (uXRCE-DDS)                  | TrajectorySetpoint (offboard)
                                                    +-------+---------------------------------------------v-------+
                                                    |  PX4 flight controller (Pixhawk-class): EKF2, position control |
                                                    +-----------------------------------------------+---------------+
                                                                                                    v
                                                                                           ESCs + motors (x500)
```

| Role | Simulation (`model.sdf`) | Real part (to be selected) |
|---|---|---|
| Frame, motors, FCU | PX4 `x500` (Holybro X500 V2 + Pixhawk) | Holybro X500 V2, Pixhawk 6C |
| 2D lidar | PX4 `lidar_2d_v2`: 1080 beams, +-135 deg, 0.1-30 m, 30 Hz; scan plane 0.315 m above base_link | a 270 deg 2D lidar, or a 360 deg one masked to the policy's 270 deg |
| Depth camera | OAK-D Lite optics: 73 x 58 deg, 0.2-19.1 m, 320 x 240 at 15 Hz; optical centre (0.132, 0, 0.261) m | Luxonis OAK-D Lite |
| RGB camera | 69 deg, 640 x 360 at 10 Hz (bridged with `RGB=1`) | the OAK-D Lite's colour camera |
| State estimate | PX4 stock simulated GPS | indoor positioning: UWB / VIO (not part of this branch) |
| Companion | the follower process | Jetson Orin Nano class, ROS 2 Jazzy |

**Why GPS still.** An earlier project on the development machine documented that Gazebo's optical-flow
plugin never produced usable data there and that rendered sensors stalled its physics clock
(airframe `4030_gz_patrol_x500`). This branch therefore changes perception only. GPS-denied
estimation (flow + range, UWB, VIO) is a separate step and can be tested on a GPU machine.

**Mounting rules** (checked in `model.sdf`): the lidar sits on a mast over base_link (x = y = 0), so
its scan plane (0.315 m) is above the props and the camera block (top at about 0.27 m). Lidar ranges
are measured from base_link, depth ranges from the camera's optical centre. **Keep the constants in
`rlpatrol/sensors.py` equal to the SDF poses.** The policy is trained on exactly that geometry.

Frames: world ENU (Gazebo), heading counter-clockwise from east. The scan bins are relative to the
drone's heading (bin 13 is straight ahead, bin 0 is 135 deg to the right). PX4 reports heading as NED
yaw; the follower converts it (`yaw_enu = pi/2 - heading`).

## What the policy observes

| | default (v2, vB) | sensor-input (vS) actor | vS critic (training only) |
|---|---|---|---|
| goal in the chain frame, reach, height, heading vs goal, nodes left | yes | yes | yes |
| obstacle geometry | 16 exact rays to the NMOPSO cylinders | 27 fused lidar + depth bins (`sensors.sim_scan`) | the scan **and** the 16 exact rays |
| whole-map shortest-route features (`route_obs`) | vB | **no**: not sensable | yes |
| threat-to-go of the known cylinders (`budget_obs`) | vB | **no** | yes |
| threat budget u, preference | yes | yes | yes |

The critic is privileged (asymmetric actor-critic). It uses the oracle features to judge returns
better during training. Only the actor is deployed, and it never needs them.

**Fused scan** (`rlpatrol/sensors.py`). Each 10 deg bin holds the nearest return from either source:
- the **lidar** sees obstacles taller than its scan plane;
- the **depth camera** sees every point at least 1.1 m above the floor (`OBST_MIN_H`), inside its
  73 deg view, once the obstacle's top enters its 58 deg vertical view.

So a 1.2 m machine below the lidar plane is seen by the camera from about 1.55 m out. Closer than that it
is invisible, which is real physics the policy must learn to respect. The 90 deg behind the drone is
blind. Ranges are capped at 6 m and normalised.

**Domain randomisation** (`SensorSpec`, on with `sensor_noise`):
- lidar noise 2 cm;
- stereo noise 0.0067 x range^2;
- 2 % bin dropout;
- pose error 5 cm;
- heading error about 1 deg.

**Why the simulated beams sit on the bin edges.** A real bin is the minimum over its 40 lidar beams,
and the nearest point of a flat face inside a sector is at a sector edge (or at the face normal). The
simulator therefore casts 3 beams per bin, on both edges and the centre, so its bins never read
farther than the real ones. `scripts/test_sensors.py` checks the two against each other.

## Flying the policy (`sensor_follower.py`)

Transits keep their offline routes, which carry the pad-safe routing and departure deconfliction.
Each PATROL leg between consecutive `patrol_points` is planned while it is flown:

1. Turn in place to face the leg's goal, which is the policy's first heading.
2. At each node:
   1. Fuse the latest lidar scan and depth cloud (stale after 0.5 s; lidar-only if the depth stream is missing).
   2. The policy proposes the next node.
   3. `segment_clear` checks the segment against the scan, with a 0.45 m radius.
   4. If the segment isn't clear, up to 16 stochastic proposals are tried, then a sweep of the
      action space (heading offsets up to +-90 deg from the goal bearing, shorter steps), nearest
      to the policy's own action first. A trained policy's samples cluster around its mean, which
      is the blocked direction, so the sweep is what finds the way around.
   5. If none is clear, the drone hovers and rescans. After 10 s it flies straight to the goal if
      that segment is clear; otherwise it skips the leg (the event is logged).
3. After 10 nodes, close the leg to its goal (NMOPSO's encoding).

Offline check of this loop (ROS stubbed, a fake drone that reaches each setpoint, scans simulated
from the true boxes, an almost untrained policy):
- **No obstacle:** all 22 legs x 10 nodes were accepted, and the flown track stays at least 0.51 m
  from every box.
- **An unexpected 0.6 m box on leg 3:** the drone went around it. All decisions were still
  accepted and no leg was skipped, at 0.54 m minimum clearance.
- **Before the action-space sweep:** the same box made the drone skip 6 legs in a row.

Every decision is written to `run_logs/<run>/trace_droneN_sensor.csv`: pose, scan, action, node and
outcome. These logs are training data for the later perception or world-model step. The track as
flown goes to `trace_droneN_track.json`, which `check_patrol.py` uses for the cross-track error.
