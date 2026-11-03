# Plan: branch `generalizable-rl` — a sensor-driven local RL policy that generalizes to unseen factories

## Context
Findings so far: vS (sensor-only PPO) matches the map policy vB on familiar patrol legs but loses on detours
(blocked HV 0.458 vs 0.52–0.57); a GRU and 30 % blocked training legs did not help (the latter broke budget
conditioning); everything was trained in ONE world (factory2), so generalization is untested; the global-local
branch's local tier gets stuck in narrow aisles / behind wide obstacles. The user wants recovery behaviour
(try the policy's top-k actions, climb over low obstacles using height, back off to the previous node and try a
different direction) and a training method chosen for **generalization** — not teacher imitation or
search distillation. Intended outcome: a policy (vG) trained with RL on procedurally generated worlds with an
automatic curriculum, with climb/backtrack actions, evaluated on held-out and out-of-distribution worlds.
No code until this plan is approved; this branch starts as plan + docs.

## Overall project architecture (target)
```
 OFFLINE                        factory map (CAD / mapping flight) ── planner/factory_patrol: zones, patrol points
                                                     │
 ON THE DRONE ───────────────────────────────────────┼────────────────────────────────────────────────────────
  SENSORS        2D lidar 270°, depth+RGB camera (x500_patrol)                                  [exists]
  PERCEPTION     rlpatrol/sensors.py → 27-bin fused range scan  + NEW 27-bin obstacle-height scan
  MAP            prior map + obstacle memory (global-local branch)                              [WIP branch]
  GLOBAL         vB + A* route split (LegPlanner.plan_route) on prior map + memory   per leg / replan
  LOCAL          NEW vG policy (this branch): node / climb / backtrack from the scan              10–20 Hz decisions
  SAFETY GATE    segment_clear + 3D clearance check + brake/hover (deterministic, always on)    [exists, extended]
  EXECUTION      follower → PX4 offboard setpoints; zones + yield rules unchanged              [exists]
 ───────────────────────────────────────────────────────────────────────────────────────────────────────────
 TRAINING        world generator → PLR level buffer → batched multi-world PathEnv-3D → PPO (hybrid actions,
                 privileged critic, domain randomization)          (GPU, RunPod)
 EVALUATION      held-out worlds, OOD worlds, factory2 benchmark legs, offline flight harness, Gazebo (cloud)
```
vG replaces VFH+/vS as the local tier of the global-local follower; vB stays the global planner (prior map).

## RL architecture (vG)

**Episode** — one leg, start → goal, up to T = 20 decisions (≤ 10 kept path nodes, as NMOPSO's encoding);
the path is a stack of nodes: `advance` pushes, `backtrack` pops.

**Observation (egocentric only, no absolute map coordinates)**
- existing core (goal in chain frame, reach, height in band, heading vs goal, nodes left, step limits)
- 27-bin fused range scan (lidar + depth), noise as now
- NEW 27-bin obstacle-top height scan from the depth camera (max point height per bin, normalized; "unknown" flag
  where only the lidar plane saw it)
- NEW recovery state: decisions left, backtracks used, current altitude vs allowed max, tabu mask over the
  27 relative directions already tried from this node
- conditioning: threat budget u, preference w (as vS)

**Action (hybrid)**
- discrete mode ∈ {advance, climb, backtrack} (categorical head)
- continuous (a_r, a_yaw, a_z) residual as now (tanh-Gaussian head), used by advance and climb
  - climb: same step but altitude target up to z_max_local (above the patrol band), only legal if the height
    scan shows the obstacle top + clearance < z_max_local and below ceiling
  - backtrack: return to the previous node; the direction just tried becomes tabu there

**Reward** (all from the true world, privileged, training only)
- terminal: −MyCost-3D scalarized as now (budget/preference), collisions −INFEASIBLE
- MyCost-3D: NMOPSO's 4 objectives, but obstacles are boxes with heights (a segment above a box top + clearance
  is not a collision and pays threat only if within the danger margin vertically); identical to MyCost when no
  climb happens → NMOPSO comparability kept on 2D legs
- per decision: −c_back per backtrack (length flown twice), −c_climb·Δz (energy/time), small time cost
- dense threat shaping as vS (`dense`)

**Networks**: shared MLP encoder (obs → 256) → actor heads (mode logits + Gaussian mean, learned std),
conditioning injected per layer (PrefMLP pattern). Privileged critic: actor obs + exact rays, route features
(per-world Dijkstra), true obstacle heights near the path. PPO with summed log-probs of both heads;
illegal modes masked (no backtrack at node 0, no climb if not allowed).

**Training distribution (the generalization part)**
1. `rlpatrol/worldgen.py` procedural factories: hall 20–40 × 15–25 m; 1–6 rack rows (length, orientation,
   aisle 1.6–3.5 m, dead-end aisles); machines of random footprint and height 0.8–3.0 m (some overflyable);
   partitions with/without gaps; unexpected boxes placed on the start→goal line; validity: start/goal free,
   goal reachable (A*), hall coverage of free space ≥ 40 %.
2. Batched multi-world env: each episode indexes its own world; obstacles as padded tensors [W, K_max, 5]
   (boxes) and [W, M_max, 4] (cylinders); `sim_scan`, MyCost-3D and route tables vectorized over worlds;
   per-world route tables precomputed when a level is generated (cached).
3. **Prioritized Level Replay (PLR)**: level = (world seed, leg set seed); buffer of N = 4000 levels; score =
   mean |GAE advantage| of the level's last rollout; rank-based sampling with staleness mixing; new levels
   generated with p = 0.3; replayed with 0.7. Avoids both "easy-only" and "hard-only" collapse.
4. Domain randomization per episode: sensor noise/dropout (as now), lidar/camera mount height ±3 cm,
   z_max_local 2.2–3.0 m, ceiling height, node execution error N(0, 0.1 m) (tracking), missing returns.

**Phased training** (each phase a checkpoint + eval): P1 vS-architecture on generated worlds (no new actions)
→ measures what diversity alone gives; P2 + climb/backtrack actions; P3 + PLR; P4 full DR. Seeds 1–3 each.

## Evaluation protocol
- **Held-out worlds**: 200 generator seeds never used in training. **OOD**: narrower aisles (1.4 m), larger
  halls, obstacle shapes not generated (L-shaped stacks, diagonal racks). **factory2**: real/random/blocked legs
  (existing benchmark; never trained on).
- Baselines: vB (map, oracle), vS, VFH+ + backtracking (classical), vG phases P1–P4.
- Metrics: success, collisions, min clearance, HV (2D legs), MyCost-3D, backtracks/climbs per leg, extra length
  vs vB, first-choice agreement with the shorter side (from the true map), inference time.
- Then: vG as local tier in the global-local follower → offline flight harness (S0–S3 + low/tall/dead-end
  scenarios) → Gazebo on RunPod.

## Implementation steps (after approval)
0. Branch `generalizable-rl` from `origin/physical-ai-robust-rl`; `docs/generalizable_rl.md` = this plan.
1. `rlpatrol/worldgen.py` + `scripts/make_worlds.py` (train/held-out/OOD sets, validity checks, SDF export for Gazebo).
2. Multi-world batching: `rlpatrol/world.py` (WorldBatch), `nmopso_cost.py` (MyCost-3D, batched threats/boxes),
   `sensors.py` (per-episode boxes; height scan), `route.py` (per-world tables).
3. `rlpatrol/env.py`: decision stack (advance/climb/backtrack), tabu mask, legal-action masks, new obs entries.
4. `rlpatrol/ppo.py`: hybrid action head, masked categorical, privileged critic inputs; `rlpatrol/plr.py` level buffer.
5. `scripts/eval_worlds.py` (held-out/OOD/factory2), configs `configs/vG_p1..p4.yaml`.
6. Tests: `scripts/test_worldgen.py` (validity, determinism), `test_sensors.py` extended (height scan, batched
   scan = per-world scan), MyCost-3D == MyCost on 2D paths (regression to 1e-12), PPO hybrid log-prob consistency.
7. Integration: `LOCAL=vg` in the global-local follower; harness scenarios incl. low box (climb) and dead end (backtrack).

## Reuse
`rlpatrol/sensors.py` (sim_scan, bins, segment_clear), `env.py` (PathEnv core, budget/dense/priv), `ppo.py`
(PPO loop, PrefMLP, privileged critic, checkpoints), `legs.py` (random/blocked legs), `route.py` (Dijkstra tables),
`planner.py` (A* for validity checks), `test_global_local.py` harness pattern, factory2 benchmark legs.

## Compute (RunPod RTX 5090)
vS took 10.7 min / 2000 iters; multi-world scan + MyCost-3D + PLR estimated 2–4× per iteration and ~2× more
iterations → ~1–2 h per run; 4 phases × 3 seeds ≈ 12–24 GPU-hours. World pool generation + route tables:
minutes (cached).

## Risks
- MyCost-3D changes the objective for climbing legs → report 2D-comparable metrics separately.
- Hybrid actions + backtracking can make PPO unstable → phased training, action masks, entropy schedule.
- PLR scoring overhead; fallback to uniform levels if it does not help (ablation).
- Partial observability remains: success is "better on average", backtracking covers wrong guesses.

## Verification
- Unit: worldgen validity/determinism; MyCost-3D ≡ MyCost on 2D paths; batched sim_scan ≡ per-world;
  hybrid log-prob rollout = recomputation (as the GRU check); existing test_sensors / smoke_test pass.
- Short CPU training (10 iters) on generated worlds end-to-end; then GPU runs per phase.
- Generalization: held-out and OOD tables vs vS/vB/VFH+; factory2 benchmark not regressing vs vS.

---

## Implementation status (this branch)
Implemented as new modules; the existing vS / vB code paths are unchanged.

| Plan item | File | Status |
|---|---|---|
| World generator, cylinder cover (port of `factory_cylinders.m`), validity, SDF export | `rlpatrol/worldgen.py`, `scripts/make_worlds.py` | done |
| MyCost-3D, batched per-episode worlds (== MyCost on 2D paths, 1.6e-15) | `rlpatrol/cost3d.py` | done |
| Batched fused scan + obstacle-height scan (== `sensors.sim_scan`) | `rlpatrol/scan3d.py` | done |
| Env: decision stack, advance / climb / backtrack, tabu, legal masks, DR, privileged obs | `rlpatrol/env_g.py` | done |
| Hybrid-action PPO, privileged critic, PLR, vectorised level batches, resumable checkpoints | `rlpatrol/ppo_g.py`, `scripts/train_g.py` | done |
| Phase configs P1–P4 | `configs/vG_p1..p4.yaml` | done |
| Generalization benchmark (held-out, OOD, factory2; vS / vB baselines) | `scripts/eval_worlds.py` | done |
| Tests (23 checks) | `scripts/test_generalizable.py` | pass (CPU) |
| VFH+ + backtracking baseline in eval_worlds | — | not yet |
| Per-world route features in the privileged critic (uses goal visibility + threat-to-go instead) | — | simplified |
| `LOCAL=vg` in the global-local follower | `global-local-planner` branch | after merge |
| Training runs + results | RunPod | not yet |

Deviations from the plan, on purpose: the privileged critic uses true goal visibility, ideal scans and
the straight threat-to-go instead of per-world Dijkstra route features (cost); `dense` threat shaping
is not ported (terminal reward + per-decision costs); GAE uses lam = 0.95 (variable episode lengths).

## How to run (RunPod GPU)
```bash
git clone -b generalizable-rl https://github.com/BV2005/rl-nmopso-factory-patrol && cd rl-nmopso-factory-patrol
python3 -m venv .venv && . .venv/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 && pip install numpy scipy matplotlib pyyaml
python scripts/test_generalizable.py --device cuda            # 23 checks
python scripts/make_worlds.py --train 2000 --heldout 200 --ood 200 --sdf 3   # world pools (minutes)
for p in 1 2 3 4; do                                          # phases (seed 1); add --seed 2, 3 for the paper
  nohup python scripts/train_g.py --name vG_p${p}_seed1 --config configs/vG_p$p.yaml --seed 1 \
        > results/runs/vG_p${p}_seed1_runner.out 2>&1 &
done
python scripts/eval_worlds.py --ckpt results/ckpt/vG_p4_seed1.pt \
       --baselines results/ckpt/vS_seed1.pt,results/ckpt/vB_seed1.pt --worlds 50 --legs 16
```
The four phases fit on one RTX 5090 in parallel (each ~1–2 GB). The training log prints feasibility,
crash rate, backtracks/climbs per leg, the mode mix and the PLR buffer size every `probe_every` iterations.
