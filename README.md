# Global-local planner (branch `global-local-planner`, work in progress)

Part of [rl-nmopso-factory-patrol](https://github.com/BV2005/rl-nmopso-factory-patrol): fast RL path planning for indoor factory-patrol drones.
The sensor-input policy vS (`physical-ai-robust-rl`) is good on familiar legs but behind the map policy on
detours: a local planner cannot see the far side of an obstacle. This branch splits planning into two tiers: a
**global route** on a prior map that remembers what the sensors saw, and a **local tier** that flies it from the
live scan and avoids unexpected obstacles.

**Status: work in progress, not evaluated.** At the last commit the local tier gets stuck in narrow aisles and
behind wide obstacles.

## What this branch adds

- **Global tier**: vB (the map policy of `main`) plus an A* route split (`LegPlanner.plan_route`, on a physical
  occupancy grid of the prior map) and an **obstacle memory** (`rlpatrol/obstacle_memory.py`) that adds what the
  sensors saw in flight; the route is replanned when the memory changes.
- **Local tier**: VFH+ (`rlpatrol/local_planner.py`) or vS, from the live scan.
- **Follower** `gazebo/nmopso_demo/global_local_follower.py`; `scripts/spawn_obstacle.sh` drops unexpected
  obstacles into the running Gazebo world.
- **Tests** `scripts/test_global_local.py`: unit tests and an offline flight harness that runs the follower's real
  control loop with ROS 2 / PX4 / MAVLink stubbed, in four scenarios (no unexpected obstacle; a box mid-leg; a box
  closing a rack aisle, which needs a global replan; three boxes on three legs). Pass: never within 0.3 m of a box,
  every patrol leg flown.

## Set-up

| Tier | Needed for | Requirements |
|---|---|---|
| 1. Python | tests, training, evaluation, every table | Python 3.10+ (tested 3.13), any OS; a CUDA GPU for training (tests and evaluation run on CPU) |
| 2. Simulation | Gazebo flights with the sensor drone | Ubuntu 24.04 (native or WSL2), ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL; a GPU that renders headless |
| (MATLAB) | only the NMOPSO baseline runs of `main` | see the `main` README |

```bash
git clone -b global-local-planner https://github.com/BV2005/rl-nmopso-factory-patrol.git
cd rl-nmopso-factory-patrol
python3 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# training on a GPU: install the CUDA build of torch for your driver first (pytorch.org), then the rest
bash gazebo/setup_sim.sh      # tier 2, once (sudo, ~15 GB, PX4 build 10-30 min): ROS 2 Jazzy, Gazebo Harmonic,
                              # PX4 at the tested commit, XRCE-DDS agent, px4_msgs, pymavlink, the x500_patrol model
```
A complete cloud-GPU walk-through (drivers, headless rendering, training, flights, copying results back):
`docs/cloud_gpu.md`.

## Run

```bash
python scripts/test_global_local.py                    # [--seeds 3] [--scenarios S0,S1,S2,S3] [--unit-only]
python scripts/test_sensors.py --device cuda           # the sensor code of physical-ai-robust-rl
# fly (tier 2): global route by vB on the prior map + obstacle memory, local tier from the live scan
MODEL=x500_patrol FOLLOWER=global_local GLOBAL_CKPT=results/ckpt/vB_seed1.pt bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
# while it flies: an unexpected obstacle
bash scripts/spawn_obstacle.sh -8.5 4.6                    # a 0.6 x 0.6 x 2.5 m pallet stack in drone 1's patrol leg
bash scripts/spawn_obstacle.sh -10.75 -2 2.1 0.6 3 aisle   # closes the aisle between rack rows 1 and 2 (global replan)
```
Design: `docs/drone_architecture.md`.

## Next steps

1. Use **vT3** (`time-optimal-rl`: 95–100 % success, flyable turns, sensors only) as the local tier instead of
   VFH+ / vS. Its one weakness, wrong-way detours around blockages it cannot see yet, is exactly what the global
   route fixes.
2. Start the obstacle memory **empty** instead of from a prior map, so the drones learn the building while they
   patrol: compare no memory, memory built over laps 1–3, and a full prior map, on the blocked and held-out legs.
3. Evaluate with the same judge as `time-optimal-rl` (`scripts/eval_t.py`) and in Gazebo.
4. The drone's position comes from PX4's **simulated GPS** in every flight: the planners are map-free for obstacles, not GPS-free. Indoor localisation (VIO / lidar SLAM / UWB) is not part of this repository (`docs/drone_architecture.md`).

## Branches of this repository

`main` → `physical-ai-robust-rl` → `generalizable-rl` → `time-optimal-rl`; `global-local-planner` branches off `physical-ai-robust-rl`. Each branch contains the code of the branches it builds on.

| Branch | Focus | Headline result |
|---|---|---|
| [`main`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/main) | PPO on NMOPSO's own problem (same encoding, bounds and cost), planning on the known map | RL proposals in 0.07–0.2 s per leg; vB within 1 % of NMOPSO-500's hypervolume at ~1/5000 of its time; collision-free 3-drone Gazebo + PX4 flights |
| [`physical-ai-robust-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/physical-ai-robust-rl) | sensor drone `x500_patrol` (2D lidar, depth + RGB camera) and a policy whose inputs are those sensors (vS) | vS close to the map policy on real legs (HV 1.011 vs 1.03–1.05), behind on blocked legs (0.458 vs 0.52–0.57) |
| [`generalizable-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/generalizable-rl) | vG: one sensor-input policy trained on procedurally generated factories; climb / backtrack actions | vG4: 88 % success on held-out worlds, 94 % out-of-distribution, 76–80 % on blocked legs |
| [`time-optimal-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/time-optimal-rl) | vT: encoding-free planner that minimises flight time with a flyable turn bound | vT3: 95–100 % success on every set, above vB with the full map; 0 % turn-rate violations; flew 22/22 patrol legs in Gazebo + PX4 from its own sensors |
| **[`global-local-planner`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/global-local-planner)** (this branch) | global route on a prior map + a local tier from the live scan + obstacle memory | work in progress, not evaluated |

## The base project (`main`)

Three PX4 x500 drones patrol a 30 × 20 m factory (Gazebo world `worlds/factory2.sdf`: four 3.5 m storage racks, a
partition, assembly machines). Every leg between two patrol points is a four-objective path-planning problem
(length, threat, altitude, smoothness) that **NMOPSO** (Duong, Bui & Phung 2025) solves well but in ~6 minutes per
leg. `main` trains PPO on NMOPSO's exact problem (its cost function ported to PyTorch and verified to 1e-15 against
the authors' `MyCost.m`), compares it fairly with the authors' unmodified code, and flies the resulting plans:

- RL proposals in 0.07–0.2 s per leg; the budget-conditioned policy **vB** comes within 1 % of NMOPSO-500's
  hypervolume at ~1/5000 of its time; RL-warm-started NMOPSO-50 beats budget-matched NMOPSO by 11.4 %;
- both plans flown collision-free with 3 drones in Gazebo + PX4 SITL after pad-safe transits and robust departure
  times.

That code (world model, cost port, NMOPSO wrappers, MATLAB factory planner, flight-plan export, departure
scheduling) is all in this branch too. Its results tables, the MATLAB tier and the reproduction of every `main`
result are in the [`main` README](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/main#readme); the improvement rounds are in `LOOP.md`.

## Licenses and citation

- **Code in this repository:** Apache-2.0 (`LICENSE`).
- **`third_party/NMOPSO`, and the variant derived from it:** keep the authors' BSD-style license
  (`third_party/NMOPSO/license.txt`).
- **`gazebo/nmopso_demo`:** comes from the `patrol_drone` project.

See `NOTICE` for all three. If you use the NMOPSO baseline, cite: T.T.N. Duong, D.N. Bui, M.D. Phung,
*Navigation variable-based multi-objective particle swarm optimization for UAV path planning with kinematic
constraints*, Neural Computing and Applications (2025), https://doi.org/10.1007/s00521-024-10945-1.
