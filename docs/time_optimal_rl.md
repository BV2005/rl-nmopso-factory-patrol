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

Reward: on success −5 tanh(w·F / SCALE / 20) (F = [time / straight time − 1, threat, altitude + climb]);
a collision or timeout −(6 + remaining distance / leg); per step a return-neutral potential on the distance to
the goal, a proximity penalty 0.5 (1 − clearance / 0.5 m), −0.1 for a tabu direction, a small action penalty.
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
