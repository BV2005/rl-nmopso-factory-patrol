# vT: a time-minimising, encoding-free RL leg planner (branch `time-optimal-rl`)

Part of [rl-nmopso-factory-patrol](https://github.com/BV2005/rl-nmopso-factory-patrol): fast RL path planning for indoor factory-patrol drones.
Every earlier planner (NMOPSO, vB, vS, vG) uses NMOPSO's path encoding: exactly 10 nodes, step ≤ 0.3 × leg, and
geometric objectives. **vT** drops the encoding and optimises what a patrol cares about, **flight time**, from the
`x500_patrol` drone's lidar and depth camera, with no map. vT3 is the best planner of the project, and the one
flown in Gazebo + PX4. This branch also contains `generalizable-rl` (vG) and `physical-ai-robust-rl` (vS).

## What this branch adds

- **Step-wise waypoints** until the goal (≤ 60 decisions, 0.4–1.5 m steps), turn ≤ 45° per step and per 0.5 m flown
  (a minimum turn radius, so every path is flyable at cruise speed); climb and backtrack modes
  (`rlpatrol/env_t.py`).
- **Time model**: max(horizontal / 1 m/s, vertical / 0.5 m/s) + 0.5 s per 45° of turn; a backtrack is a timed hover
  yaw plus the flight back.
- **Reward**: flight time charged per step (capped at 3 × the straight-line time), a linear success term (threat,
  altitude); failures pay the full time cap.
- **Training** (vT3; `scripts/train_t.py`, `configs/vT3.yaml`): PPO with lr decay, a per-minibatch KL stop,
  std / entropy floors, and a separate validation world set that selects the reported checkpoint
  (`<name>_best.pt`), so no test set is used to choose a model.
- **One judge for every method** (`env_t.flight_metrics`, `scripts/eval_t.py`): success with no collision
  (obstacle boxes and walls + 0.5 m), flight time / straight-line time, threat, altitude, turn-rate violations, and a
  **paired** comparison on the legs every method completes.
- **Flight stack**: `rlpatrol/vt_pilot.py` (decision core: the policy, then its samples, an action sweep and a
  backtrack, each checked against the full-resolution scan), `rlpatrol/scan3d.py` (the flight scan from raw lidar
  and depth points), `gazebo/nmopso_demo/vt_follower.py` (the ROS 2 node; flies through the waypoints),
  `scripts/vt_flight_report.py`, `gazebo/spawn_box.sh` (unexpected obstacles).
- **Tests** `scripts/test_time.py` (time model, judge, environment, fixes, PPO, evaluation), `scripts/test_devices.py`
  (CPU inputs on a CUDA environment), `scripts/test_vt_flight.py` (the ROS node itself under a fake ROS 2 / PX4
  layer with synthetic sensors). **Docs** `docs/time_optimal_rl.md` (design, every fix, vT2 / vT3, paired times),
  `docs/gazebo_vt.md` (flights).

## Results

vT3: 3 seeds, validation-selected checkpoints; the same legs for every method. Success %, and time / straight-line
time on each method's own successes:

| Set | **vT3** (seeds 1 / 2 / 3) | vG4 (2 seeds) | vS | vB (full map) |
|---|---|---|---|---|
| held-out worlds | **95.3** (95.7 / 94.5 / 95.6) · 1.36 | 87.9 · 1.28 | 89.5 · 1.32 | 91.3 · 1.19 |
| OOD worlds | **96.8** (97.3 / 96.5 / 96.7) · 1.19 | 93.5 · 1.27 | 94.1 · 1.29 | 95.7 · 1.14 |
| factory2 real legs | **100** · **1.11** | 100 · 1.59 | 100 · 1.66 | 100 · 1.70 |
| factory2 random legs | **97.5** · 1.40 | 85.5 · 1.45 | 93.5 · 1.45 | 97.0 · 1.32 |
| factory2 blocked legs | **94.5** · 1.77 | 75.0 · 1.76 | 87.5 · 1.66 | 93.5 · 1.52 |
| turn-rate violations | **0 %** | 10–59 % | 4.5–37 % | 2–38 % |

**Paired flight time** (time / straight-line time on the legs **every** method completes):

| Set | legs | vT3 | vB | vS | vG4 |
|---|---|---|---|---|---|
| held-out | 691 / 800 | **1.09** | 1.16 | 1.29 | 1.27 |
| OOD | 741 / 800 | **1.08** | 1.12 | 1.28 | 1.27 |
| factory2 real | 73 / 73 | **1.08–1.16** | 1.70 | 1.66 | 1.63 |
| factory2 random | 168 / 200 | **1.16–1.17** | 1.27 | 1.40 | 1.47 |
| factory2 blocked | 140 / 200 | 1.45–1.53 | **1.44** | 1.59 | 1.72 |

- From sensors only, vT3 has the highest success on every set (above vB, which plans on the full map), with
  crash rates of 0–3.5 %; its 3 seeds agree within 1.5 points; with an unexpected box on the leg it keeps 93–96 %.
- It is faster than vB on 68–97 % of shared legs (median faster on every set) and ~35 % faster on the real patrol
  legs. vB is faster on average only on blocked legs, through a tail (~15 % of those legs) where vT3 detours the
  wrong way round a blockage it cannot see yet.
- Planning: ~2.7 ms of network time per decision on one CPU thread.
- History: vT (52.5 % on blocked legs) → vT2, environment and reward fixes (85.5 %) → vT3, training stability (94.5 %).
- Models `results/ckpt/vT3_seed{1,2,3}_best.pt`; evaluations `results/eval_t_vT3_*.json` (paired:
  `*_paired.json`).

### Gazebo + PX4 SITL

- **First flight** (drone 1, one lap, vT3 seed 1 from the rendered lidar and depth camera): **22/22 patrol legs,
  0 collisions**, 0 fallbacks (the policy's own step was clear in the real scan every time), 1 backtrack, but
  2.93 × the time model's flight time: the follower stopped at every waypoint and PX4's position loop crawled
  (mean 0.33 m/s). Details in `docs/gazebo_vt.md`.
- **Flow-through follower** (`flow`, the default now): velocity feed-forward and each decision taken 0.6 m before
  the waypoint, on a scan re-expressed there from the pose each sensor message was taken at; decisions only from
  data taken level and facing the heading; the corner cut is checked; the carrot brakes instead of stopping dead.
  Offline (`scripts/test_vt_flight.py`, PX4-like drone with tilt and 50 ms sensor latency, 21 checks):
  **0.89 ×** the time model flowing vs 2.30 × stopping, 22/22 legs, no collision. **The Gazebo rerun with flow is
  still to be done.**

## Trained models, results and configs

This branch holds **code only** (plus the world, the drone model, the flight plans and the input data in `data/`).
The trained checkpoints, training logs, evaluations, NMOPSO runs, flight records and training configs behind every
number above are kept in the tag [`time-optimal-rl-results`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/time-optimal-rl-results). To use them, restore them into a clone;
`.gitignore` keeps `configs/` and `results/` out of commits:

```bash
git fetch origin tag time-optimal-rl-results --no-tags
git archive time-optimal-rl-results configs results | tar -x
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
git clone -b time-optimal-rl https://github.com/BV2005/rl-nmopso-factory-patrol.git
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
python scripts/test_time.py --device cuda            # time model, judge, environment, fixes, PPO, evaluation
python scripts/test_devices.py                       # GPU: every entry point with CPU inputs on a CUDA environment
python scripts/test_vt_flight.py                     # the flight stack offline (~15 min CPU; --quick ~3 min)
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200
for s in 1 2 3; do python scripts/train_t.py --name vT3_seed$s --config configs/vT3.yaml --seed $s; done
python scripts/eval_t.py --ckpt results/ckpt/vT3_seed1_best.pt --vg results/ckpt/vG4_box_seed1.pt \
       --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt --worlds 50 --legs 16 --threads 8
```

Fly it (tier 2; `docs/gazebo_vt.md` has every step, the checks and what to look at):
```bash
# drone 1 only
python3 -c "import json; d=json.load(open('gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json')); d['drones']=[x for x in d['drones'] if x['id']==1]; json.dump(d, open('/tmp/plan_drone1.json','w'))"
MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt TIMEOUT=1800 bash gazebo/fly_plan.sh /tmp/plan_drone1.json
python3 scripts/vt_flight_report.py "$(ls -d gazebo/nmopso_demo/run_logs/* | tail -1)"
# stop at every waypoint instead (comparison), or other vt_follower parameters
VT_ARGS="-p flow:=false" MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt TIMEOUT=1800 bash gazebo/fly_plan.sh /tmp/plan_drone1.json
# all three drones
MODEL=x500_patrol VT_CKPT=results/ckpt/vT3_seed1_best.pt TIMEOUT=2400 bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
```

## Limitations and next steps

- Simulation only; the paired time comparison uses single-seed baselines; vB / vS / vG were trained on NMOPSO's
  objectives, not flight time; no classical planners (A* + smoothing, a reactive sensor-only planner) yet.
- The wrong-way detours on blocked legs: a remembered map of what the drone has sensed (an obstacle memory, as in
  `global-local-planner`) or a coarse prior route would pick the side.
- The flow-through follower is verified offline, not yet in Gazebo.
- The drone's position comes from PX4's **simulated GPS** in every flight: the planners are map-free for obstacles, not GPS-free. Indoor localisation (VIO / lidar SLAM / UWB) is not part of this repository (`docs/drone_architecture.md`).

## Branches of this repository

`main` → `physical-ai-robust-rl` → `generalizable-rl` → `time-optimal-rl`; `global-local-planner` branches off `physical-ai-robust-rl`. Each branch contains the code of the branches it builds on.

| Branch | Focus | Headline result |
|---|---|---|
| [`main`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/main) | PPO on NMOPSO's own problem (same encoding, bounds and cost), planning on the known map | RL proposals in 0.07–0.2 s per leg; vB within 1 % of NMOPSO-500's hypervolume at ~1/5000 of its time; collision-free 3-drone Gazebo + PX4 flights |
| [`physical-ai-robust-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/physical-ai-robust-rl) | sensor drone `x500_patrol` (2D lidar, depth + RGB camera) and a policy whose inputs are those sensors (vS) | vS close to the map policy on real legs (HV 1.011 vs 1.03–1.05), behind on blocked legs (0.458 vs 0.52–0.57) |
| [`generalizable-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/generalizable-rl) | vG: one sensor-input policy trained on procedurally generated factories; climb / backtrack actions | vG4: 88 % success on held-out worlds, 94 % out-of-distribution, 76–80 % on blocked legs |
| **[`time-optimal-rl`](https://github.com/BV2005/rl-nmopso-factory-patrol/tree/time-optimal-rl)** (this branch) | vT: encoding-free planner that minimises flight time with a flyable turn bound | vT3: 95–100 % success on every set, above vB with the full map; 0 % turn-rate violations; flew 22/22 patrol legs in Gazebo + PX4 from its own sensors |
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
