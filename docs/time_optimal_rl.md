# vT: a time-minimising, encoding-free leg planner (branch `time-optimal-rl`)

## Why
Every planner so far (NMOPSO, vB, vS, vG) uses NMOPSO's path encoding: exactly 10 nodes, step ≤ 0.3 × leg
length, ≤ 45° per node. Long detours do not fit, which caps every method's success (on clean legs vB, which
gets the full map, still fails ~10 % of held-out legs, and more candidates do not help). The objectives
(path length, threat, altitude, smoothness) are geometric proxies; what a patrol cares about is how long
a leg takes and how safely it is flown.

## Design
| | vG (NMOPSO-compatible) | **vT** |
|---|---|---|
| path | 10 nodes, step ≤ 0.3 × leg, ≤ 45° per node | waypoints until the goal (≤ 60 decisions), step 0.4–1.5 m (absolute), turn ≤ min(45°, 45° per 0.5 m) |
| objectives | F1 length, F2 threat (budget), F3 altitude, F4 smoothness | **flight time**, threat, altitude + climb energy (preference over 3) |
| time model | — | per segment max(horizontal / 1 m/s, vertical / 0.5 m/s) + 0.5 s per 45° of turn; backtrack = 180° hover yaw (2 s) + fly back |
| collision | box + 0.5 m (vG3_box) | box + 0.5 m, scan ranges to that zone |
| recovery | climb, backtrack | climb, backtrack, final approach (goal within 1.5 m, inside the turn bound, clear → fly to it) |
| planning time | measured | measured, not optimised (one network pass per decision: ~constant) |

The turn bound per metre matters: "45° per step" alone would allow sharp turns made of many tiny steps; the
per-metre bound is a minimum turn radius (≈ 0.64 m at 1 m/s), so every vT path is flyable at cruise speed.

Reward (since the reward fix below; w = preference over time, threat, altitude):
- **time, every step**: −6 w_time × (this step's flight time / straight-line time), turns and backtrack reversals
  included, charged up to 3× the straight-line time — the episode total is exactly 6 w_time × min(executed time /
  straight time, 3);
- **success**: +2 − min(Σ_{threat, alt} w_i F_i / SCALE_i, 4) (linear, no squashing);
- **failure** (collision or timeout): −6 − the rest of the time cap, so every failure pays the full time charge
  (crashing early is never cheaper than flying slowly) and any success beats any failure by ≥ 4;
- per step a return-neutral potential on the distance to the goal (+1 in total on success), a proximity penalty
  0.5 (1 − clearance / 0.5 m) (obstacles and walls), −0.1 for a tabu direction, a small action penalty.
Constants in `env_t.py` / `configs/vT.yaml` (`c_time, t_cap, succ, fail, c_q, q_max`); the terms are logged per
iteration (`r[time prox term]`).
Observation: goal in the heading frame, distance / height to the goal, band and headroom, decisions and
backtracks left, heading vs goal bearing, the last turn and step, time used, the margin-aware range scan, the
obstacle-height scan, the tabu mask, the legal modes, the preference. Privileged critic: + ideal scans, goal
visibility, distance. PPO / PLR / domain randomisation / world pool / clean legs from the generalizable branch.

## The common judge (`env_t.flight_metrics`, `scripts/eval_t.py`)
Every method's path — vT's waypoints, vG's / vS's / vB's 10-node paths — is scored the same way: success =
reaches the goal with no collision in the box model; flight time (and time / straight-line time), length,
threat, altitude, the largest turn and turn-rate violations (turn per metre above the bound); plus the
method's planning time per leg on the same CPU. NMOPSO's own HV is no longer the yardstick for vT (different
objectives); flight outcomes are.

## Review fixes (after the first vT results; checkpoints trained before them must be retrained)
1. **Heading after a backtrack**: the drone faces the direction it flew back in (it used to keep the heading it
   once arrived with). The reversal is a real hover yaw, timed by the time model (180° = 2 s; `T_REVERSE` is
   gone), so the env's flight time equals `flight_metrics` on what was flown, backtracks included.
2. **Turn bound on the horizontal step**: |turn| ≤ min(45°, 45° per 0.5 m × horizontal step length), the length
   the judge divides by (a climbing step used to get the 3D length's larger bound); the final approach uses the
   same bound (it used to allow `KAPPA × max(d, R_MIN)`). Steps are no longer clamped to the hall, which bent
   them after the bound was applied.
3. **Executed objectives**: the reward scores the executed polyline (`TEnv.executed()`: every waypoint flown to,
   backtrack returns included; `F_exec`); the final route's objectives are reported in `F`. `eval_t.py` judges
   every method's executed flight (vT, vG) — backtrack reversals are timed but left out of the turn-rate check
   (`reversals()`: a stop-and-yaw, not a flown curve).
4. **Walls**: the hall walls are collisions with the same 0.5 m margin (the scan always saw them; the clamp
   used to hide them), in the env and in the judge (`flight_metrics(..., walls=)`). Threat counts obstacles only.
5. **Time model limits**: constant speeds, no acceleration / braking / jerk limits, a fixed time per degree of
   turn (no speed-dependent radius), no wind, battery or controller lag. It ranks paths; it does not predict
   flight time. Validate against PX4 SITL (logged setpoint tracking) before quoting absolute times.

6. **Reward fix.** The old success term −5 tanh(w·F / SCALE / 20) was nearly flat in time: measured with vT_seed1
   on 192 held-out legs, its time part averaged −0.04 for flights 25 % slower than straight, against −6.5 for a
   failure (~150 : 1), and a backtrack (2 s reversal + the flight back) cost ~0.05; time was only scored at the
   end of up to 60 decisions. Now time is charged per step (slope 2.0 per unit of time / straight at w = 1/3, was
   0.17; spread across the same successes 0.5, was 0.04), the success term is linear, and a failure pays the
   full time cap plus 6 (the double-counted remaining distance is gone: the shaping already charges it).
   Tests: `scripts/test_time.py` [2c] (time charges add up to the capped executed time ratio, with backtracks;
   straight flight return = 1 + 2 − 6 w_time; failures ≤ the bound; worst success > best failure).

Tests: `scripts/test_time.py` section [2b] (flown turn bound on whole random episodes, env time == judge time
with backtracks, reversal time and heading, F_exec vs F, wall collisions in env and judge).

## Files
`rlpatrol/env_t.py` (time model, judge, TEnv), `rlpatrol/ppo_t.py` (ConfigT, rollout_t, train_t; reuses
ActorG / CtxMLP / PLR / batch_from_levels from `ppo_g.py`), `scripts/train_t.py`, `scripts/eval_t.py`,
`scripts/test_time.py`, `configs/vT.yaml`.

## How to run (RunPod)
```bash
git clone -b time-optimal-rl https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 && pip install numpy scipy matplotlib pyyaml
python scripts/test_time.py --device cuda
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200
for s in 1 2; do nohup python scripts/train_t.py --name vT_seed$s --config configs/vT.yaml --seed $s > results/runs/vT_seed${s}_runner.out 2>&1 & done
python scripts/eval_t.py --ckpt results/ckpt/vT_seed1.pt --vg results/ckpt/vG3_box_seed1.pt \
       --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt --worlds 50 --legs 16 --threads 8
python scripts/eval_t.py --ckpt results/ckpt/vT_seed2.pt --worlds 50 --legs 16 --threads 8
```
Each step is up to 60 decisions (vG: 20), so expect ~2–3× vG's time per iteration.

## Retraining after the review and reward fixes (new pod)
Every checkpoint trained before the fixes is invalid (env and reward changed): retrain vG (box model) as
`vG4_box_*` and vT as `vT2_*`; vS / vB (NMOPSO environment) are unchanged and stay the baselines. This branch
contains the vG code (merged from `generalizable-rl`), so one clone trains both.
Commands:
```bash
# 1. setup (RTX PRO 4500 / 5090 Blackwell: cu130 wheels; A100 / H100: the same command works)
cd /workspace
git clone -b time-optimal-rl https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy scipy matplotlib pyyaml
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 2. self-tests (both must end with 'all N checks passed')
python scripts/test_generalizable.py --device cuda
python scripts/test_time.py --device cuda
# 3. world pool (train / held-out / OOD)
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200
# 4. training: 2 seeds each, 4 runs in parallel
mkdir -p results/runs
for s in 1 2; do
  nohup python scripts/train_g.py --name vG4_box_seed$s --config configs/vG3_box.yaml --seed $s > results/runs/vG4_box_seed${s}_runner.out 2>&1 &
  nohup python scripts/train_t.py --name vT2_seed$s --config configs/vT.yaml --seed $s > results/runs/vT2_seed${s}_runner.out 2>&1 &
done
tail -f results/runs/vT2_seed1.log            # watch: succ, det succ, det time/straight, r[time prox term]
nvidia-smi                                    # GPU use
# 5. evaluation (after training)
python scripts/eval_worlds.py --ckpt results/ckpt/vG4_box_seed1.pt --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt
python scripts/eval_worlds.py --ckpt results/ckpt/vG4_box_seed2.pt
python scripts/eval_t.py --ckpt results/ckpt/vT2_seed1.pt --vg results/ckpt/vG4_box_seed1.pt,results/ckpt/vG4_box_seed2.pt \
       --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt --worlds 50 --legs 16 --threads 8
python scripts/eval_t.py --ckpt results/ckpt/vT2_seed2.pt --worlds 50 --legs 16 --threads 8
# 6. bring results home (from your PC: scp -P <port> root@<pod-ip>:/workspace/rl-nmopso-factory-patrol/results_fix.tgz .)
tar czf results_fix.tgz results/ckpt/vG4_box_seed*.pt results/ckpt/vT2_seed*.pt results/runs/vG4_box_seed* results/runs/vT2_seed* \
        results/eval_*vG4_box* results/eval_t_vT2_*
```

## vT2 results and vT3 (training stability)
vT2 (fixed env + reward) peaked around it 500 and then degraded. Both seeds at it 500, shared judge, success %:
held-out 90.6, OOD 95.5, factory2 real 99.3, random 93.3, blocked 85.5 (seed 1 / 2 on blocked: 90.5 / 80.5),
0 % turn-rate violations; vS 89.5 / 94.1 / 100 / 93.5 / 87.5, vB (full map) 91.3 / 95.7 / 100 / 97.0 / 93.5.
Final checkpoints were worse (seed 1 blocked 81.0 %, seed 2 52.0 %). Seed 2's log shows why: the learned std
fell to 0.07–0.14 (the schedule only capped it from above, the entropy bonus decayed to 0), so the constant lr
made large policy steps — from it ~540 the KL exceeded its target after the first epoch on every iteration, with
a 0.21 spike at it 600 — and the backtrack mode crept from 3.5 % to 9 % of decisions. Picking the it-500
snapshot by its held-out / factory2 results would also mean choosing on the test sets.

vT3 = vT2 env and reward + (`configs/vT3.yaml`, all off by default in `ConfigT`): actor lr decaying linearly
3e-4 → 3e-5; the update stops at the first minibatch with KL > 1.5 × target; log std ≥ −1.9 (std ≥ 0.15);
entropy coefficient ≥ 0.001; 800 iterations; a validation world set (32 worlds, seeds 30 000 000.., disjoint from
train / held-out / OOD) scored every 20 iterations, the best one saved as `<name>_best.pt` (score = success −
0.25 (time / straight − 1)) — report `_best`, never a checkpoint chosen on a test set. 3 seeds for the spread.
```bash
cd /workspace && git clone -b time-optimal-rl https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
# (or, on the existing pod: cd /workspace/rl-nmopso-factory-patrol && git pull)
python3 -m venv .venv && . .venv/bin/activate          # skip on the existing pod; just: . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 && pip install numpy scipy matplotlib pyyaml
python scripts/test_time.py --device cuda
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200     # skip if data/worlds/*.pkl exist
mkdir -p results/runs
for s in 1 2 3; do
  nohup python scripts/train_t.py --name vT3_seed$s --config configs/vT3.yaml --seed $s > results/runs/vT3_seed${s}_runner.out 2>&1 &
done
tail -f results/runs/vT3_seed1.log            # every 20 it: ... lr ...  val succ .. % t/s .. (*best* when saved)
# after training: evaluate the validation-selected checkpoints (vG4 / vS / vB for comparison)
python scripts/eval_t.py --ckpt results/ckpt/vT3_seed1_best.pt --vg results/ckpt/vG4_box_seed1.pt,results/ckpt/vG4_box_seed2.pt \
       --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt --worlds 50 --legs 16 --threads 8
for s in 2 3; do python scripts/eval_t.py --ckpt results/ckpt/vT3_seed${s}_best.pt --worlds 50 --legs 16 --threads 8; done
tar czf results_vt3.tgz results/ckpt/vT3_seed* results/runs/vT3_seed* results/eval_t_vT3_*
```
