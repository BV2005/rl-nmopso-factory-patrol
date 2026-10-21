# Flight rl_20261021_152127 (RL plan, factory2_auto_full, delays 0/6/0 s)

- Drone 1: full mission, cross-track error max 0.29 m, min obstacle clearance 0.69 m, landed at 127 s.
- Drones 2 and 3: **mid-air collision at t = 18.0-18.3 s** over drone 3's pad (0.05, 7.1, 1.8 m);
  distance closed 0.72 -> 0.30 m in 0.6 s. Before it, neither came within 0.89 m of an obstacle.
- Cause: drone 2's transit (planned from its pad to its zone) passes 0.1-0.3 m over drone 3's pad;
  drone 3, still climbing after take-off, met drone 2 and yielded under the follower's rule (the
  higher ID holds, the lower keeps going), so drone 2 flew into it. The departure model assumes all
  drones start together; drone 3 actually armed 4.3 s after drone 2 (sequential arming in run_demo).
  The NMOPSO plan has the same geometry (its transit passes about 0.5 m from drone 3's pad).
- Afterwards drones 2-3 thrashed in failsafe; landed drone 1 was knocked over at about 535 s.
