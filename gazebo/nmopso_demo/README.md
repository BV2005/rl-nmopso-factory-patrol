> **In this repository:** this folder is the patrol demo of the patrol_drone project (written for its
> branch `nmopso-demo-factory2-voronoi`; the text below describes that branch). Fly the plans of this
> repository with `gazebo/fly_plan.sh` (see the top-level README); the flown plans are in
> `paths_rl_padsafe/` and `paths_nm500_padsafe/`, and `paths/` is the planner's own factory2_auto plan.

# NMOPSO patrol demo, factory2 with balanced power-diagram (Voronoi) zones, 3 drones, Gazebo + PX4 SITL

Branch `nmopso-demo-factory2-voronoi`. The world `worlds/factory.sdf` is `Downloads/factory2` as given
(storage racks, partition, assembly machines). The three drones take off from one shared launch area.
The factory is divided among them by patrol_drone's zoning algorithm, adapted for the shared launch
area. Every flown leg that is not a straight climb or descent is planned by the authors' unmodified
NMOPSO (Duong, Bui & Phung 2025; https://github.com/ngandng/NMOPSO).

The previous branch, `nmopso-demo-storage-assembly`, divides the same world into hand-drawn strips.

## Zoning (`NMOPSO_repro/factory_patrol/factory_zones.m`, layout `factory2_voronoi`)

The zoning is a **balanced power diagram** (a weighted Voronoi diagram), as in patrol_drone
`firmware/core/pd_zone.c`. Point p belongs to zone d if d minimises |p − seed_d|² − w_d. Two changes
for a shared launch area:

- **Spread-out generators.** The seeds are not the drones' homes, because all the homes sit together
  on the pads. They are k-means centroids of the patrol lane network, which covers the whole world:
  - the 4 rack aisles and both sides of the partition, running north–south;
  - 3 lanes between the machine rows, running east–west.
- **Balanced on lane length, not floor area.** The weights are updated as in `pd_zone_build()`:
  w_d −= k (load_d − mean). The load is the lane length inside each zone. Result: **39.4 / 39.1 / 40.7 m**
  of lane per zone, after 29 iterations.
- **The zones:**
  - Drone 1: the south half of the storage area.
  - Drone 2: the north half of the storage area and the west side of the partition.
  - Drone 3: the assembly area and the east side of the partition.

**Keeping each drone in its zone:**
- Lane pieces stay at least 1 m inside the zone boundaries, the same test as `pd_zone_contains()`.
- Connections between pieces are routed on a clearance grid, restricted to the zone (0.6 m margin).
- For every loop leg, NMOPSO also gets the zone's boundary lined with cylinders near the leg, the
  counterpart of `pd_zone_apply()` blocking cells. In the plan, every loop point is at least 1.0 m
  inside its own zone.

## Shared launch area

The pads are at (−2.5, −8), (−1, −8) and (0.5, −8), 1.5 m apart, and they all lie in drone 1's zone.
So:

- **Drone 1** takes off to the 1.6–2.0 m patrol band and reaches its lanes inside its own zone.
- **Drones 2 and 3** climb straight up over their pads to a **transit layer at 2.9–3.3 m** (below the
  3.5 m racks, which NMOPSO still avoids). They cross drone 1's zone there, at least 0.9 m above its
  patrol band, then descend vertically at their own zone's entry point. The return is the same in
  reverse.
- **Vertical-clearance rule** (`patrol_follower.py`): before any climb, descent or landing, a drone
  holds its position until no other airborne drone is within 2 m horizontally in the column it will
  pass through. This lasts at most 30 s.
- **Climbs, descents, take-off and landing** are straight vertical moves, not NMOPSO legs.

## Run

```sh
nmopso_demo/run_demo.sh                 # headless, one lap, then the check
TOPVIEW=1 nmopso_demo/run_demo.sh       # GUI looking straight down on the whole hall
HEADLESS=0 nmopso_demo/run_demo.sh      # normal Gazebo GUI
```

Planning (MATLAB, Windows): `plan_factory_patrol(6, [], 12, 'factory2_voronoi')` produces 76 legs, 72
of them planned by NMOPSO in about 50 minutes on 6 workers, with no timeouts.
`plan_factory_patrol(..., true)` is a dry run that writes only the zones and patrol points
(`paths/preview_top.png`).

## Data pipeline (`docs/data_pipeline.png`, `docs/architecture.png`)

Each run writes to `run_logs/<run>/`:

| File | Written by | Content |
|---|---|---|
| `drone_states.csv` | `patrol_recorder.py`, **10 Hz**, all drones | factory-frame position and velocity, speed, roll / pitch / yaw, battery %, voltage, current, mAh, battery warning, arming / nav state, failsafe, landed, phase, age of the position message |
| `events.csv` | `patrol_recorder.py` | follower events (phase changes, laps, clearance holds, land command) and PX4 state changes (arming, nav state, failsafe, landed, battery warning) |
| `trace_droneN.csv` | `patrol_follower.py`, 20 Hz | position, carrot setpoint, phase, lap, arc length |
| `metrics_summary.csv` | `metrics_summary.py` | one row per drone and one for the whole swarm: flight time, distance, speed, time per phase, laps, holds, battery, tilt, clearance, separation (horizontal, 3D, same level), valid / dropped samples |
| `summary.txt` | `check_patrol.py` + `metrics_summary.py` | tracking and coverage check plus the metrics table |

- **Sources:** the recorder subscribes to PX4's `vehicle_local_position_v1`, `vehicle_attitude`,
  `battery_status_v1`, `vehicle_status_v4` and `vehicle_land_detected` for every drone, and to the
  followers' `/droneN/patrol/event`.
- **Validity filters** in `metrics_summary.py`: a sample is dropped if its position message is older
  than 0.5 s, if it implies a jump faster than 15 m/s, or if it lies outside the hall.
- **Battery in SITL is not physical.** PX4's battery simulator drains the percentage linearly with
  armed time (`SIM_BAT_DRAIN` 60 s) down to `SIM_BAT_MIN_PCT` (50 %). It derives the voltage from that
  percentage and reports current −1 ("no current sensor in simulation"). So battery % reflects
  armed time, and energy / mAh stay blank. PX4 also publishes `battery_status` at only 1 Hz, so the
  10 Hz rows repeat the latest value.

Run 20260929_185412 (one lap each):
- **Rows:** 4 707 state rows at 10 Hz and 48 events; 0 samples dropped.
- **Distance flown:** 110.4 / 99.1 / 97.2 m (balance 1.14).
- **Clearance hold:** drone 2 held once, for 3.2 s.
- **Separation:** at least 2.76 m at the same level; at least 1.34 m in 3D.

## Result (run 20260929_180613, one lap each at 1 m/s)

| Drone | Total flown | Patrol loop flown | Transit in + out | Path tracking (mean / max) | Closest obstacle | Altitude |
|---|---|---|---|---|---|---|
| 1 | 106.9 m | 98.5 m | 4.2 m | 0.08 / 0.36 m | 0.75 m | 1.47–2.02 m |
| 2 | 96.8 m | 69.5 m | 19.8 m | 0.08 / 0.34 m | 0.82 m | 1.52–3.14 m |
| 3 | 97.0 m | 72.1 m | 17.1 m | 0.07 / 0.37 m | 0.95 m | 1.66–3.20 m |

- **Distance flown is now balanced within 10 %.** With the hand-drawn strips on the same world it was
  107.9 / 46.1 / 89.4 m.
- **Separation away from the pads:**
  - at similar altitude (|dz| < 0.6 m): at least **2.81 m**;
  - in 3D: at least 1.37 m, which is a transit passing about 1.3 m above a patrolling drone.
- **Separation at the pads:** 0.94 m. Drone 3 touched down 0.38 m off its pad (PX4 auto-land drift),
  and drone 2 then landed next to it. The pads are 1.5 m apart.
- **Collisions:** none.
- **Coverage:** 85.4 % of the free floor came within 2 m of a drone. The clearance rule never had to
  hold a drone in this run.
- **First attempt:** drone 1 also used the transit layer. It descended at its zone entry while drone 2
  passed over it (0.96 m horizontally, 1.1 m in 3D). That plan and its flight are kept in
  `out_factory2_voronoi/v1_drone1_transit_layer/`.

## Limitations

- **Kinks remain.** NMOPSO still adds 16–41 % to each loop (drone 1's 81 m of straight lines becomes
  114 m), because the flown solution is the lowest-threat one on each Pareto front. The zones are
  balanced on lane length, so this extra length is what keeps drone 1 about 10 % longer.
- **Split aisles become dead ends.** The zone boundary between drones 1 and 2 crosses the racks, so
  each drone flies its half of an aisle out and back.
- **Landing precision.** The 1.5 m pads and PX4's landing drift of about 0.4 m leave under 1 m between
  landed drones. Landing in offboard position control, or pads 2 m apart, would fix this.
- **Offline planning**, and the drones use simulated GPS and a magnetometer.
