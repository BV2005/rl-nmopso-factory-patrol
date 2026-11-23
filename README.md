# Sensor drone and a sensor-input RL leg planner (branch `physical-ai-robust-rl`)

Part of [rl-nmopso-factory-patrol](https://github.com/BV2005/rl-nmopso-factory-patrol): fast RL path planning for indoor factory-patrol drones.
`main` plans every patrol leg offline on the **known map**, for stock GPS x500s with no perception sensors. This
branch makes the planner physical: a drone with real perception sensors, and a policy (**vS**) that plans from
those sensors alone, with no map.

## What this branch adds

- **The drone** `gazebo/models/x500_patrol` (installed into PX4 by `gazebo/install_patrol_drone.sh`):

| Part | Simulated (`gazebo/models/x500_patrol`) | Real counterpart |
|---|---|---|
| 2D lidar | PX4 `lidar_2d_v2`: 1080 beams, ±135°, 0.1–30 m, 30 Hz, on a mast above base_link | a 270° 2D lidar |
| Depth camera | OAK-D Lite optics: 73 × 58°, 0.2–19.1 m, 320 × 240 at 15 Hz | Luxonis OAK-D Lite |
| RGB camera | 69°, 640 × 360 at 10 Hz | the OAK-D Lite's colour camera |
| State estimate | PX4's stock simulated GPS | indoor positioning (UWB / VIO), not part of this repository |
| Companion computer | the follower process | Jetson Orin Nano class, ROS 2 Jazzy, PX4 offboard over uXRCE-DDS |

- **Sensor fusion** `rlpatrol/sensors.py`: lidar and depth fused into a 27-bin range scan around the heading
  (with an obstacle-height channel), the same in training (ray-cast, batched on the GPU) and in flight (from
  `sensor_msgs/LaserScan` and `PointCloud2`).
- **Sensor-input policy vS**: `PathEnv(sensor_obs=True)` gives the actor only the scan and the leg geometry (no
  map) and the critic the exact geometry during training (privileged critic); `configs/vS.yaml`. Ablations:
  a GRU actor (`configs/vSG.yaml`, `--recurrent`), blocked training legs (`--blocked_frac`).
- **Closed-loop flight** `gazebo/nmopso_demo/sensor_follower.py`: plans each leg in flight, one NMOPSO node per
  scan, behind a `segment_clear` safety check.
- **Tests** `scripts/test_sensors.py` (sensor geometry, converters, environment, training, planner);
  **docs** `docs/drone_architecture.md` (design), `docs/cloud_gpu.md` (GPU pod, training, flights).

## Results

NMOPSO's objectives, hypervolume (HV) of the proposals with `main`'s fixed normalisation (`scripts/evaluate.py`;
seed 1 unless noted):

| Policy | Inputs | real legs (73) HV | random legs (200) HV, feasible | blocked legs (200) HV, feasible |
|---|---|---|---|---|
| vB (seeds 1–3, `main`) | the map | 1.030–1.049 | 0.948–0.974, 192–193 | 0.521–0.567, 185–186 |
| **vS** | sensors only | **1.011** (1.023 with sensor noise) | **0.909**, 190 | 0.458, 184 |
| vSb (vS + 30 % blocked training legs) | sensors | 0.984 | 0.894, 185 | 0.429, 178 |
| vSr (GRU memory) | sensors | 1.003 | 0.913, 190 | 0.450, 181 |
| vSG (GRU + blocked legs) | sensors | 0.973 | 0.896, 184 | 0.440, 176 |

- From sensors alone, vS is 2–4 % behind the map policy on the real patrol legs and 4–7 % on random legs; sensor
  noise does not hurt it.
- On blocked legs (detours) it is 12–19 % behind: the scan cannot show the far side of an obstacle.
- Memory (GRU) and blocked training legs did **not** help. That led to `generalizable-rl` (train on many worlds)
  and `time-optimal-rl` (drop NMOPSO's 10-node encoding, which cannot express long detours).
- vS trains in 10.7 min on an RTX 5090 (2000 iterations) and plans a leg in ~0.5 s.
- Checkpoints `results/ckpt/vS*_seed1.pt`; logs `results/runs/vS*_seed1.*`; evaluations `results/eval_vS*_seed1_*.json`.

## Trained models, results and configs

This branch holds **code only** (plus the world, the drone model, the flight plans and the input data in `data/`).
The trained checkpoints, training logs, evaluations, NMOPSO runs, flight records and training configs behind every
number above are published as the release [`results-v1`](https://github.com/BV2005/rl-nmopso-factory-patrol/releases/tag/results-v1): one archive for all
branches, which share these files (`MANIFEST.tsv` lists the files of each branch). Unpack it into a clone;
`.gitignore` keeps `configs/` and `results/` out of commits:

```bash
curl -L https://github.com/BV2005/rl-nmopso-factory-patrol/releases/download/results-v1/configs-and-results.tar.gz | tar -xz
```
The training (`--config configs/...`), evaluation and flight commands below expect them; the scripts write their
own outputs into `results/`.

## Set-up

| Tier | Needed for | Requirements |
|---|---|---|
| 1. Python | tests, training, evaluation, every table | Python 3.10+ (tested 3.13), any OS; a CUDA GPU for training (tests and evaluation run on CPU) |
| 2. Simulation | Gazebo flights with the sensor drone | Ubuntu 24.04 (native or WSL2), ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL; a GPU that renders headless |
| (MATLAB) | only the NMOPSO baseline runs of `main` | see the `main` README |

```bash
git clone -b physical-ai-robust-rl https://github.com/BV2005/rl-nmopso-factory-patrol.git
cd rl-nmopso-factory-patrol
python3 -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# training on a GPU: install the CUDA build of torch for your driver first (pytorch.org), then the rest
bash gazebo/setup_sim.sh      # tier 2, once (sudo, ~15 GB, PX4 build 10-30 min): ROS 2 Jazzy, Gazebo Harmonic,
                              # PX4 at the tested commit, XRCE-DDS agent, px4_msgs, pymavlink, the x500_patrol model
```
A complete cloud-GPU walk-through (drivers, headless rendering, training, flights, copying results back):
`docs/cloud_gpu.md`.

## Reproduce

```bash
python scripts/test_sensors.py --device cuda     # this branch's checks
python scripts/smoke_test.py --device cuda       # main's training pipeline still passes
python scripts/train.py --name vS_seed1 --config configs/vS.yaml --seed 1            # ~11 min on an RTX 5090
python scripts/evaluate.py --ckpt results/ckpt/vS_seed1.pt --legs real                # also --legs random / blocked
python scripts/evaluate.py --ckpt results/ckpt/vS_seed1.pt --legs real --sensor_noise --tag vS_seed1_real_noise
python scripts/train.py --name vSG_seed1 --config configs/vSG.yaml --seed 1           # GRU + blocked legs
python scripts/train.py --name vSb_seed1 --config configs/vS.yaml --seed 1 --blocked_frac 0.3
python scripts/train.py --name vSr_seed1 --config configs/vS.yaml --seed 1 --recurrent true
# fly (tier 2): the sensor drone, PATROL legs planned in flight by vS
MODEL=x500_patrol SENSOR_CKPT=results/ckpt/vS_seed1.pt bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
MODEL=x500_patrol bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json   # sensors only, planned paths
```
Each flight writes `gazebo/nmopso_demo/run_logs/<run>/` (states and events at 10 Hz, `summary.txt`,
`flight_check.json`).

## Limitations

- Behind the map policy on detours (above); one training world (factory2), seed 1 only for the vS variants.
- vS keeps NMOPSO's 10-node encoding, so long detours cannot be expressed at all.
- The drone's position comes from PX4's **simulated GPS** in every flight: the planners are map-free for obstacles, not GPS-free. Indoor localisation (VIO / lidar SLAM / UWB) is not part of this repository (`docs/drone_architecture.md`).

## Branches of this repository

`main` → `physical-ai-robust-rl` → `generalizable-rl` → `time-optimal-rl`; `global-local-planner` branches off `physical-ai-robust-rl`. Each branch contains the code of the branches it builds on.

| Branch | Focus | Headline result |
|---|---|---|
| [`main`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/main) | PPO on NMOPSO's own problem (same encoding, bounds and cost), planning on the known map | RL proposals in 0.07–0.2 s per leg; vB within 1 % of NMOPSO-500's hypervolume at ~1/5000 of its time; collision-free 3-drone Gazebo + PX4 flights |
| **[`physical-ai-robust-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/physical-ai-robust-rl)** (this branch) | sensor drone `x500_patrol` (2D lidar, depth + RGB camera) and a policy whose inputs are those sensors (vS) | vS close to the map policy on real legs (HV 1.011 vs 1.03–1.05), behind on blocked legs (0.458 vs 0.52–0.57) |
| [`generalizable-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/generalizable-rl) | vG: one sensor-input policy trained on procedurally generated factories; climb / backtrack actions | vG4: 88 % success on held-out worlds, 94 % out-of-distribution, 76–80 % on blocked legs |
| [`time-optimal-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/time-optimal-rl) | vT: encoding-free planner that minimises flight time with a flyable turn bound | vT3: 95–100 % success on every set, above vB with the full map; 0 % turn-rate violations; flew 22/22 patrol legs in Gazebo + PX4 from its own sensors |
| [`global-local-planner`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/global-local-planner) | global route on a prior map + a local tier from the live scan + obstacle memory | work in progress, not evaluated |

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
