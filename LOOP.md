# Improvement loop

Each round:
1. **Run**: train a variant (`scripts/train.py --name vN ...`), evaluate it on the real and random legs
   (`scripts/evaluate.py`) against the authors' unmodified NMOPSO (`matlab/run_nmopso_legs.m`).
2. **Question**: a reviewer agent reads the numbers and the code and challenges them: is the
   comparison fair (same objectives, same leg, same bounds, same scoring code), is the gain real
   (seeds, variance, held-out legs), is the reward being gamed, what is the weakest part of the method.
3. **Decide**: one change per round, chosen from the reviewer's points; record it below with the result.
4. **GPU gate**: a variant that beats NMOPSO's hypervolume on held-out legs by >= 10 % on >= 3 training
   seeds, with the costs re-scored by MATLAB MyCost.m, is ready to move to a GPU for scaling up
   (bigger batches, longer training, the multi-agent patrol extension). The user decides where.

## Rounds

| Round | Variant | Change | Real legs: HV RL / NMOPSO | Feasible legs | Verdict |
|---|---|---|---|---|---|
| 0 | v0 (it 50 of 400, stopped) | absolute (r, psi, phi) actions, clipped Gaussian, linear scalarisation, PBRS | 0.15 / 1.45 (2 legs, union-normalised HV: flawed) | 73/73 | Reviewer: policy ignores w (front collapses), circles (phi saturated 28 %, paths 2.8x straight), F3 from exploration noise, clipped r = rmin is always infeasible, union-normalised HV saturates NMOPSO. |
| 1 | v1 | residual actions (zero = straight + level, exact NMOPSO encoding), tanh Gaussian, w in every layer, split optimisers, return norm, KL stop, lambda 1, no shaping, real-length legs, fixed-normalisation HV + straight / random-search baselines | it 75 (killed at ~90 by low memory): 0.49 / 0.69 (21 legs, fixed-norm HV); straight path 0.63, random 0.05 | 73/73 (99.9 % of rollouts) | Learns feasibility, but bends paths where straight is clear: its front is worse than the straight path. |
| 2 | v2 | straight prior: zero-action path always a front candidate (1 evaluation); -0.02 \|a\|^2 per step in training; batch 2048, 5 epochs (memory) | 0.743 / 0.732 (25 legs; straight 0.671, random 0.048); RL beats a NMOPSO seed on 36 % of legs; random legs: RL 0.880 vs straight 0.591 | real 73/73, random 178/200 | +1.5 % mean HV, below the 10 % gate; MATLAB MyCost re-scoring agrees to 5e-15; preference conditioning still weak; long random legs fail. Reviewer 2: the gain is the straight prior only - NMOPSO + straight path scores 0.809 (RL wins 1/28); RL loses the low-F2 end (NMOPSO lower min F2 on 22/28); front is ~2-D (F1 vs F2, F3 degenerate); 36 % win rate was ties counted as losses; failing random legs are mostly infeasible for NMOPSO's initialisation too. |
| 3 | v3 | threat-budget conditioning: F2 <= u x straight F2 is a penalty (5 per F_SCALE[1]), u observed, w over F1/F3/F4; evaluation: budget sweep, NMOPSO+straight baseline, IGD+, attainment of NMOPSO's pick, tolerant win/tie/loss, optional CEM refinement (RL+LS) | stopped at it 150 | | budget ignored: F2/straight 0.56 at u = 0.3 vs 0.60 at u = 1.0 - penalty (~0.17) below the detour's length cost (~0.2) |
| 4 | v4 | budget penalty x10 (50) | stopped at it 100 | | still ignores u: F1 0.118 vs 0.115 for u = 0.1 vs 1.2; mean-action change with u 0.005-0.03 vs action size 0.04-0.42; weights on u stay at init scale. Conclusion: PPO does not learn threat conditioning here (weak, poorly credited signal over 10 chained steps). |
| 5 | hybrid | RL-warm-started NMOPSO: straight path + up to 49 non-dominated v2 paths seed NMOPSO's swarm (matlab/nmopso_variant/NMOPSO.m = authors' script + 2 marked additions: MaxIt from env, warm particles); compare warm vs cold NMOPSO at MaxIt 50 (same seeds) and the authors' NMOPSO at 500 | pending | | |
