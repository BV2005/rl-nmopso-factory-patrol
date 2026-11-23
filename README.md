# Generalizable sensor-input RL planner vG (branch `generalizable-rl`)

Part of [rl-nmopso-factory-patrol](https://github.com/BV2005/rl-nmopso-factory-patrol): fast RL path planning for indoor factory-patrol drones.
The sensor-input policy vS (`physical-ai-robust-rl`) was trained in one world, factory2. **vG** is one policy
trained to **generalize** to factories it has never seen, still planning from the `x500_patrol` drone's lidar and
depth camera with no map.

## What this branch adds

- **Procedural worlds** (`rlpatrol/worldgen.py`, `scripts/make_worlds.py`): halls 20–40 × 15–25 m, rack rows with
  aisles and dead ends, machines of random size and height (some can be flown over), partitions, unexpected boxes
  on the leg; every leg end checked clear of the collision zone; separate training, held-out and
  out-of-distribution (OOD) world sets.
- **Batched multi-world environment** (`rlpatrol/env_g.py`): 27-bin range scan + obstacle-height scan, hybrid
  actions **advance / climb / backtrack** (tabu directions after a backtrack), privileged critic.
- **Training** (`rlpatrol/ppo_g.py`, `scripts/train_g.py`, `configs/vG*.yaml`): PPO with prioritized level replay
  (PLR) and domain randomisation.
- **Collision model**: obstacle boxes + 0.5 m (the scan reports ranges to that zone); paths are also re-scored
  with NMOPSO's own cylinder model, so vG stays comparable with vS and vB.
- **Review fixes** (vG4): heading after a backtrack, no clamping to the hall (walls are collisions),
  executed-flight objectives (`F_exec`).
- **Tests** `scripts/test_generalizable.py` (world generator, 3-D cost, scan, environment, PPO, PLR, the fixes);
  **evaluation** `scripts/eval_worlds.py`; **design and every round** `docs/generalizable_rl.md`.

## Results

vG4 = box collision model + review fixes, 2 seeds (`scripts/eval_worlds.py`):

| Set | vG4 seed 1: success, crash | vG4 seed 2: success, crash | before the review fixes (vG3_box seed 1 / 2) |
|---|---|---|---|
| held-out worlds (200 worlds never trained on) | 88.7 %, 5.7 % | 88.6 %, 5.5 % | 88.7 / 89.1 %, crash 7.9 % |
| out-of-distribution worlds | 94.0 %, 1.1 % | 93.9 %, 1.6 % | 94.5 / 93.6 %, crash 3.5–4.3 % |
| factory2 real legs | 97.3 %, 0 % | 100 %, 0 % | 100 / 100 % |
| factory2 random legs | 87.5 %, 6.0 % | 87.5 %, 3.8 % | 87.5 / 86.5 % |
| factory2 blocked legs | **79.5 %**, 8.5 % | **76.0 %**, 8.7 % | 75.0 / 69.5 %, crash 12–13 % |

- vG generalizes: held-out worlds score as well as training worlds, and the OOD set higher.
- The review fixes halve the crashes and add 4.5–6.5 points on blocked legs.
- vG stays behind vS / vB on blocked legs: NMOPSO's 10-node encoding (step ≤ 0.3 × leg) cannot express long
  detours. `time-optimal-rl` removes the encoding (vT3: 94.5 % on blocked legs).
- Checkpoints `results/ckpt/vG4_box_seed*.pt` (earlier rounds: `vG_p*`, `vG2*`, `vG3_*`); logs
  `results/runs/vG*`; evaluations `results/eval_g_*.json`.

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
git clone -b generalizable-rl https://github.com/BV2005/rl-nmopso-factory-patrol.git
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
python scripts/test_generalizable.py --device cuda     # this branch's checks
python scripts/test_sensors.py --device cuda           # the sensor code of physical-ai-robust-rl still passes
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200
python scripts/train_g.py --name vG4_box_seed1 --config configs/vG3_box.yaml --seed 1     # seed 2: --seed 2
python scripts/eval_worlds.py --ckpt results/ckpt/vG4_box_seed1.pt --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt
```
Pod commands (GPU sizes, run times, copying results back): `docs/generalizable_rl.md` and `docs/cloud_gpu.md`.

## Limitations

- The 10-node encoding limits detours (above); 2 seeds; simulation only.
- vG has no Gazebo follower of its own: the sensor drone and `sensor_follower.py` (vS) come from
  `physical-ai-robust-rl`, and vT3 (`time-optimal-rl`) is the planner that was flown.
- The drone's position comes from PX4's **simulated GPS** in every flight: the planners are map-free for obstacles, not GPS-free. Indoor localisation (VIO / lidar SLAM / UWB) is not part of this repository (`docs/drone_architecture.md`).

## Branches of this repository

`main` → `physical-ai-robust-rl` → `generalizable-rl` → `time-optimal-rl`; `global-local-planner` branches off `physical-ai-robust-rl`. Each branch contains the code of the branches it builds on.

| Branch | Focus | Headline result |
|---|---|---|
| [`main`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/main) | PPO on NMOPSO's own problem (same encoding, bounds and cost), planning on the known map | RL proposals in 0.07–0.2 s per leg; vB within 1 % of NMOPSO-500's hypervolume at ~1/5000 of its time; collision-free 3-drone Gazebo + PX4 flights |
| [`physical-ai-robust-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/physical-ai-robust-rl) | sensor drone `x500_patrol` (2D lidar, depth + RGB camera) and a policy whose inputs are those sensors (vS) | vS close to the map policy on real legs (HV 1.011 vs 1.03–1.05), behind on blocked legs (0.458 vs 0.52–0.57) |
| **[`generalizable-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/generalizable-rl)** (this branch) | vG: one sensor-input policy trained on procedurally generated factories; climb / backtrack actions | vG4: 88 % success on held-out worlds, 94 % out-of-distribution, 76–80 % on blocked legs |
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
