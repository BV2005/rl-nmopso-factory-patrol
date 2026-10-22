# RL proposals for multi-objective UAV path planning: making NMOPSO fast enough for factory patrol

Three PX4 x500 drones patrol a 30 × 20 m factory (Gazebo world `worlds/factory2.sdf`: four 3.5 m
storage racks, a partition, assembly machines). Every leg between two patrol points is a
four-objective path-planning problem: path length, threat (obstacle proximity), altitude and
smoothness. **NMOPSO** (Duong, Bui & Phung 2025) solves it well but takes about 6 minutes per leg.

This repository trains a **PPO policy** on NMOPSO's own problem: the same path encoding, the same
bounds and the same cost function. The policy proposes paths for an unseen leg in about 0.2 s. It is
compared fairly against the authors' **unmodified** NMOPSO, and the resulting patrol plans are flown
with 3 drones in **Gazebo + PX4 SITL**.

Everything needed to reproduce the results is here: code, the authors' NMOPSO (unmodified), the
factory planner, the world, trained models, every NMOPSO run and the flight logs.

## Results

All path costs are scored with the authors' `MyCost.m`. The PyTorch port used for training matches it
to 1e-15 on 8 000 random solutions (`scripts/verify_cost.py`). The benchmark is the 73 real patrol legs
of the `factory2_auto_full` plan, with 3 NMOPSO seeds each.

**Leg planning** (`scripts/final_compare.py`):
- Hypervolume (HV, higher is better) uses fixed normalisation: ideal point 0, nadir = the 95th percentile of NMOPSO's fronts, reference 1.1.
- IGD+ (lower is better) is measured against the union of all fronts of the leg.
- Time is single-core time per leg.

| Method | HV | IGD+ | Time per leg |
|---|---|---|---|
| Straight, level path only | 0.835 | – | 0 s |
| NMOPSO, 50 iterations | 0.437 | 0.452 | 37.7 s |
| NMOPSO, 50 iterations, plus the straight path | 0.901 | 0.149 | 37.7 s |
| NMOPSO, 500 iterations (authors' code) | 0.941 | 0.043 | 369 s |
| NMOPSO, 500 iterations, plus the straight path | **1.050** | **0.012** | 369 s |
| **RL proposals** (3 training seeds) | 0.975 ± 0.004 | 0.097 | **0.2 s** |
| **RL proposals ∪ RL-warm-started NMOPSO-50** | 1.003 | 0.064 | 37.5 s |

- **RL proposals alone** beat NMOPSO-50 + straight by about 8 % in HV, at 1/190 of its time.
- **The hybrid** beats it by 11.4 % (better on 68 % of runs, worse on 3 %, Wilcoxon p ≈ 0).
- **Full NMOPSO-500 given the straight path remains the best front,** 4.4 % above the hybrid, at 10× the time.

**Path quality and robustness** of the path each method would fly (`scripts/robustness.py`;
physical obstacle boxes, drone radius 0.3 m):

| | RL | NMOPSO-500 |
|---|---|---|
| Length / straight distance | **1.10** | 1.36 |
| Interior clearance, 5th percentile | 0.94 m | 0.90 m |
| Collisions: nominal, or obstacles 0.3 m larger than modelled | 0 | 0 |
| Collision probability, waypoint error σ = 0.1 / 0.2 / 0.3 m | 0 / 0.1 / 2.2 % | 0 / 0 / 0.6 % |
| Unexpected 0.6 m obstacle: replanning time | **1.2 s** | 38 s (NMOPSO-50) |

**Gazebo / PX4 SITL** (3 drones, one lap at 1 m/s, pads at (−8, 7), (−4, 7) and (0, 7);
`results/flights/`). After re-routing the transits around the other drones' pads and choosing
departure delays that are robust to ±6 s of timing error, both plans flew with **no collision**:

| | NMOPSO-500 | RL |
|---|---|---|
| Distance flown (3 drones) | 285.2 m | **253.0 m** |
| Drone 1 patrol time | 116 s | **77 s** |
| Minimum obstacle clearance / minimum drone separation | 0.64 m / 3.82 m | 0.72 m / 3.82 m |
| Maximum cross-track error | 0.38 m | 0.41 m |
| Coverage (floor within 2 m) | 91.8 % | 92.2 % |
| Failsafes / yield holds | 0 / 0 | 0 / 0 |

**What did not work** (see `LOOP.md` for every round of the improvement loop):
- PPO did not learn preference or threat-budget conditioning in this setting, so the low-threat end of the front comes from NMOPSO.
- RL finds no feasible path on 11 % of long random legs (8–15 m).
- The first flight of the plan as generated ended in a mid-air collision over a launch pad (`results/flights/rl_20261021_152127/FINDINGS.md`). That led to the pad-safe transits.

## Repository layout

```
rlpatrol/            Python package: world model, exact PyTorch port of NMOPSO's decoder and MyCost,
                     the leg-planning MDP, PPO, metrics, CEM refinement
scripts/             experiments (train, evaluate, compare, robustness, obstacle replanning, flight plans,
                     departure scheduling, flight collision check, cost-port verification)
matlab/              thin wrappers: world export, MyCost scoring, NMOPSO batch runs (unmodified, and the
                     marked variant with MaxIt / warm start in matlab/nmopso_variant/)
third_party/NMOPSO/  the authors' NMOPSO, unmodified (BSD-style license)
planner/             factory patrol planner (MATLAB): SDF parsing, free-space lanes, balanced power-diagram
                     zones, 2-opt tours, obstacle cylinders, NMOPSO model
worlds/factory2.sdf  the Gazebo world
data/                obstacle model, benchmark legs, plan preview, warm-start seeds, NMOPSO legs used for scaling
results/             trained policies (ckpt/), NMOPSO runs, evaluations, flight plans and flight records
gazebo/              ROS 2 / Gazebo / PX4 SITL demo, set-up and flight scripts
paper/main.tex       paper draft;  LOOP.md  research log of the improvement loop
```

## Set-up

Three tiers; each needs the ones above it only for what it runs.

| Tier | Needed for | Requirements |
|---|---|---|
| 1. Python | training, evaluation, all tables from saved results, flight plans | Python 3.10+ (tested 3.13), any OS |
| 2. MATLAB | the NMOPSO baseline runs, the world export, re-scoring with `MyCost.m` | MATLAB (tested R2026a) with the Parallel Computing Toolbox |
| 3. Simulation | Gazebo flights | Ubuntu 24.04 (native or WSL2), ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL |

### Tier 1: Python

```bash
git clone https://github.com/BV2005/rl-nmopso-factory-patrol.git
cd rl-nmopso-factory-patrol
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/verify_cost.py --offline   # port vs the authors' MyCost.m (saved MATLAB output): VERIFIED
```

### Tier 2: MATLAB

No set-up beyond MATLAB. `matlab/setup_paths.m` puts `planner/` and `third_party/NMOPSO/` on the path
and points the planner at `worlds/factory2.sdf`. Tell the Python scripts where MATLAB is if it is not
on the PATH:

```bash
export MATLAB_EXE="/usr/local/MATLAB/R2026a/bin/matlab"     # Windows: set MATLAB_EXE=C:\Program Files\MATLAB\R2026a\bin\matlab.exe
python scripts/verify_cost.py                              # runs MyCost.m on 8 000 solutions: VERIFIED
```

### Tier 3: Simulation

On Ubuntu 24.04, run once. It needs sudo and about 15 GB, and PX4's build takes 10–30 min:

```bash
bash gazebo/setup_sim.sh
```

It installs ROS 2 Jazzy with Gazebo Harmonic, then:
- PX4-Autopilot at the tested commit `cda3c3603f` in `~/PX4-Autopilot`, with `make px4_sitl`;
- the Micro XRCE-DDS Agent;
- `px4_msgs` at `ed9ddee` in `~/px4_ros_ws`;
- pymavlink.

Different locations can be set with `PX4_DIR` and `PX4_MSGS_WS`; `gazebo/nmopso_demo/run_demo.sh`
reads both.

## Reproducing the results

### From the saved runs (Python only, minutes)

```bash
python scripts/final_compare.py                       # leg-planning table (results/final_compare.json)
python scripts/final_compare.py --warm warm_v2_seed2_real.mat --warmdir nmopso_warm50_seed2 --tag seed2   # RL rows, seed 2
python scripts/robustness.py                          # path quality, tracking / map error, failures
python scripts/replan_obstacle.py analyse             # unexpected-obstacle replanning
python scripts/evaluate.py --ckpt results/ckpt/v2.pt --legs real --nmopso results/nmopso_real
python scripts/flight_check.py results/flights/rl_padsafe_20261021_162255   # collision check of a flight
```

### From scratch

Times are for a 12-core laptop CPU without a GPU.

| Step | Command | Time |
|---|---|---|
| World model from the SDF (MATLAB) | `matlab -batch "cd matlab; export_world"` | 1 min |
| Benchmark legs | `python scripts/make_legs.py` | s |
| Cost-port check (MATLAB) | `python scripts/verify_cost.py` | 2 min |
| NMOPSO-500 baseline, 73 legs × 3 seeds (MATLAB, 4 workers) | `matlab -batch "cd matlab; run_nmopso_legs('../data/legs_real.mat','../results/nmopso_real',1:3,4)"` | ~6 h |
| Train the policy (v2; seeds 2, 3 with `--seed`) | `python scripts/train.py --name v2 --iters 300 --batch 2048 --epochs 5 --act_pen 0.02 --threads 4` | 30 min |
| Evaluate RL | `python scripts/evaluate.py --ckpt results/ckpt/v2.pt --legs real --nmopso results/nmopso_real` | 5 min |
| Warm-start seeds | `python scripts/make_warm.py --ckpt results/ckpt/v2.pt --legs real` | 15 s |
| Warm and cold NMOPSO-50 (MATLAB) | `matlab -batch "cd matlab; run_nmopso_variant('../data/legs_real.mat','../results/nmopso_warm50',1:3,4,50,'../data/warm_v2_real.mat'); run_nmopso_variant('../data/legs_real.mat','../results/nmopso_cold50',1:3,4,50,'')"` | 75 min |
| Tables | `python scripts/final_compare.py`, `python scripts/robustness.py` | min |
| Obstacle replanning | `python scripts/replan_obstacle.py prep`, then the two MATLAB calls in its docstring, then `... analyse` | 45 min |
| Quality along a training run (`results/snapshot_sweep.json`) | `python scripts/snapshot_sweep.py v5_seed1 v5_seed2 v5_seed3 --extra v2` | 10 min |
| Threat-budget response, low-threat end vs NMOPSO (`results/budget_response.json`) | `python scripts/budget_response.py v2 v4 vA_seed1 vA_seed2 vA_seed3 vA_ctrl_seed1` | 2 min |
| Legs without a feasible RL path (`results/hard_legs_<eval>.json`) | `python scripts/hard_legs.py results/eval_v5_seed1_random.json` | 1 min |

The unmodified NMOPSO loops forever if no random start path is feasible. `run_nmopso_variant`
takes a timeout for that case; `run_nmopso_legs` (the plain baseline) has none, and none was needed
on the real legs.

### Flight plans and Gazebo flights

```bash
# plans: every leg of the factory2_auto_full patrol replaced by a planner's short-safe path
python scripts/padsafe_routes.py                 # transits routed around the other drones' pads
python scripts/export_padsafe_plan.py rl         # RL plan (transits planned in ~9 s)
python scripts/export_padsafe_plan.py nm500      # NMOPSO plan (needs results/padsafe_nm500, see its docstring)
# departure delays maximising the worst-case separation over +-6 s timing errors
cp results/plans/rl_padsafe/patrol_paths.json gazebo/nmopso_demo/paths_rl_padsafe/
python3 scripts/schedule_max_separation.py gazebo/nmopso_demo gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json \
        --max 60 --step 3 --slack 6 --write
# fly (Ubuntu, tier 3): 3 drones, one lap; prints the collision check
bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_rl_padsafe/patrol_paths.json
bash gazebo/fly_plan.sh gazebo/nmopso_demo/paths_nm500_padsafe/patrol_paths.json
HEADLESS=0 TOPVIEW=1 bash gazebo/fly_plan.sh <plan>    # watch it from above
```

The plans in `gazebo/nmopso_demo/paths_*_padsafe/` already contain the scheduled delays: they are
exactly the plans that were flown. Each run writes `gazebo/nmopso_demo/run_logs/<run>/`:
- `drone_states.csv` and `events.csv` at 10 Hz;
- `summary.txt`, with tracking, clearance, separation and coverage;
- `flight_check.json`, with the collision verdicts.

## Licenses and citation

- **Code in this repository:** Apache-2.0 (`LICENSE`).
- **`third_party/NMOPSO`, and the variant derived from it:** keep the authors' BSD-style license (`third_party/NMOPSO/license.txt`).
- **`gazebo/nmopso_demo`:** comes from the `patrol_drone` project.

See `NOTICE` for all three.

If you use the NMOPSO baseline, cite: T.T.N. Duong, D.N. Bui, M.D. Phung, *Navigation variable-based
multi-objective particle swarm optimization for UAV path planning with kinematic constraints*, Neural
Computing and Applications (2025), https://doi.org/10.1007/s00521-024-10945-1.
