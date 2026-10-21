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
| 2 | v2 | straight prior: zero-action path always a front candidate (1 evaluation); -0.02 \|a\|^2 per step in training; batch 2048, 5 epochs (memory) | v1 it 75 + straight candidate already 0.696 / 0.692 (21 legs) | | running |
