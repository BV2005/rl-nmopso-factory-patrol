"""Self-test of vT, the time-minimising encoding-free planner (docs/time_optimal_rl.md). Minutes on a CPU.
  1. time model and the common judge (flight_metrics) on hand-computed paths
  2. environment: turn bound and step length under random actions, straight flight in the open takes
     exactly the straight-line time, env time == judge time, collisions end the episode, backtracking
     restores the waypoint and costs time, observation shapes, legal modes
  2b. review fixes: the flown turn per horizontal metre (judge on the executed polyline) stays within the
     bound for whole random episodes; env time == judge time on the executed polyline with backtracks;
     a backtrack = reversal yaw (4 T_TURN) + the flight back, heading = the direction flown back; reward
     objectives are the executed flight's (F_exec), the route's are reported (F); steps are not clamped to
     the hall (a step into a wall's margin collides) and the judge counts wall collisions
  2c. reward: the per-step time charges add up to C_TIME w_time min(executed time / straight time, T_CAP)
     (with backtracks too); a straight open flight returns SHAPE + SUCC - C_TIME w_time; a failure pays FAIL
     + the full time cap; the worst success beats any failure
  3. PPO: rollout log-probs == recomputation, 3 training iterations, resume
  4. eval_t.py against vG, vS, vB on 2 held-out worlds and factory2's real legs
  python scripts/test_time.py [--device cpu] [--batch 128]"""
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol.env_g import EVAL, build_batch  # noqa: E402
from rlpatrol.env_t import (ADVANCE, BACK, C_TIME, CTX, FAIL, KAPPA, OBS_BASE, PRIV, Q_MAX, R_MAX, R_MIN, SHAPE,  # noqa: E402
                            SUCC, T_CAP, T_MAX, T_TURN, TURN_MAX, V_CRUISE, V_Z, TEnv, flight_metrics, sample_pref3,
                            segment_time)
from rlpatrol.ppo_g import ActorG, batch_from_levels, load_checkpoint  # noqa: E402
from rlpatrol.ppo_t import ConfigT, make_actor_t, rollout_t, train_t  # noqa: E402
from rlpatrol.worldgen import generate  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def open_items(legs, boxes=None):
    hall = np.array([-30, 30, -30, 30.0])
    b = np.zeros((0, 5)) if boxes is None else np.asarray(boxes, float)
    return [(hall, b, np.zeros((0, 4)), np.asarray(l, float), None) for l in legs]


def model_checks():
    print("[1] time model and judge")
    t = lambda a, b, turn=0.0: segment_time(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64),  # noqa: E731
                                            torch.tensor(turn, dtype=torch.float64)).item()
    check(abs(t([0, 0, 1.8], [3, 0, 1.8]) - 3 / V_CRUISE) < 1e-12, "3 m level: 3 s at cruise")
    check(abs(t([0, 0, 1.8], [0.5, 0, 2.8]) - 1.0 / V_Z) < 1e-12, "a 1 m climb over 0.5 m is limited by the vertical speed (2 s)")
    check(abs(t([0, 0, 1.8], [1, 0, 1.8], math.pi / 4) - (1 + T_TURN)) < 1e-12, "a 45 deg turn adds T_TURN")
    P = torch.tensor([[[0, 0, 1.8], [2, 0, 1.8], [2, 2, 1.8]]], dtype=torch.float64)
    m = flight_metrics(P, torch.tensor([3]), torch.zeros(1, 1, 5, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.bool))
    exp = 4 / V_CRUISE + 2 * T_TURN
    check(abs(m["time_s"].item() - exp) < 1e-9 and abs(m["length_m"].item() - 4) < 1e-9 and
          abs(math.degrees(m["max_turn"].item()) - 90) < 1e-6 and not bool(m["collides"]),
          f"L-shaped 2 + 2 m path: {m['time_s'].item():.2f} s (= 4 s + a 90 deg turn), length 4 m, turn 90 deg")
    box = torch.tensor([[[0.8, 1.2, -0.2, 0.2, 3.0]]], dtype=torch.float64)
    m2 = flight_metrics(P, torch.tensor([3]), box, torch.ones(1, 1, dtype=torch.bool))
    check(bool(m2["collides"]), "the judge's box model: a path through a box + 0.5 m collides")


def env_checks(dev):
    print("[2] environment")
    W = [generate(s) for s in range(4)]
    cfg = ConfigT()
    b = batch_from_levels([{"world": i, "legs_seed": 7 + i} for i in range(4)], W, 32, cfg, dev)
    B = len(b["legs"])
    env = TEnv(b, sample_pref3(B), domain=EVAL)
    o, oc, L = env.obs_pair()
    check(o.shape == (B, OBS_BASE + CTX) and oc.shape == (B, OBS_BASE + PRIV + CTX) and torch.isfinite(o).all() and torch.isfinite(oc).all(),
          f"obs [{B}, {o.shape[1]}], critic [{B}, {oc.shape[1]}], finite")
    check(bool(L[:, ADVANCE].all()) and not bool(L[:, BACK].any()), "at the start: advance legal, backtrack not")
    g = torch.Generator().manual_seed(0)
    a = torch.rand(B, 3, generator=g, dtype=torch.float64) * 2 - 1
    q, turn, r, snap = env._proposal(torch.full((B,), ADVANCE), a)
    free = ~snap
    p = env.cur
    hl = (q - p)[:, :2].norm(dim=-1)
    flown = torch.atan2(torch.sin(env._bearing(p, q) - env.heading), torch.cos(env._bearing(p, q) - env.heading))
    lim = torch.clamp(KAPPA * hl, max=TURN_MAX)
    check(bool((flown.abs() <= lim + 1e-9).all()) and bool(((r >= R_MIN - 1e-9) & (r <= R_MAX + 1e-9))[free].all()),
          f"turn bound |flown turn| <= min(45 deg, 45 deg / 0.5 m * horizontal step) and step in [{R_MIN}, {R_MAX}] m (random actions)")
    legs = [[0, 0, 1.8, 7.3, 0, 1.8], [0, 0, 1.8, 0, 9.1, 1.8], [-3, 2, 1.8, 4, -5, 1.8]]
    ob = build_batch(open_items(legs), dev)
    env = TEnv(ob, torch.full((3, 3), 1 / 3, dtype=torch.float64), domain=EVAL)
    for _ in range(60):
        r_, done = env.step(torch.full((3,), ADVANCE), torch.tensor([[1.0, 0.0, 0.0]] * 3))
        if done:
            break
    ratio = (env.time / env.t_st)
    check(bool(env.success.all()) and bool(torch.allclose(ratio, torch.ones(3, dtype=torch.float64, device=ratio.device), atol=1e-9)),
          f"straight flight in the open reaches the goal in exactly the straight-line time (ratio {ratio.tolist()})")
    P, n = env.executed()
    m = flight_metrics(P, n, ob["boxes"], env.obst)
    check(bool(torch.allclose(m["time_s"], env.time, atol=1e-9)), "env flight time == the judge's flight time (same path)")
    cb = build_batch(open_items([[0, 0, 1.8, 8, 0, 1.8]], [[3, 4, -1, 1, 3.0]]), dev)
    env = TEnv(cb, torch.full((1, 3), 1 / 3, dtype=torch.float64), domain=EVAL)
    tot = 0.0
    for _ in range(60):
        r_, done = env.step(torch.full((1,), ADVANCE), torch.tensor([[1.0, 0.0, 0.0]]))
        tot += r_.item()
        if done:
            break
    check(bool(env.crashed.all()) and not bool(env.success.any()) and tot < -5, f"flying into a box ends the episode (return {tot:.2f})")
    env = TEnv(ob, torch.full((3, 3), 1 / 3, dtype=torch.float64), domain=EVAL)
    env.step(torch.full((3,), ADVANCE), torch.tensor([[1.0, 0.0, 0.0]] * 3))
    p1, t1 = env.cur.clone(), env.time.clone()
    _, _, L = env.obs_pair()
    check(bool(L[:, BACK].all()), "after one waypoint: backtrack legal")
    env.step(torch.full((3,), BACK), torch.zeros(3, 3))
    seg = (p1 - env.cur).norm(dim=-1)
    check(bool(torch.allclose(env.cur, ob["legs"][:, :3])) and bool(torch.allclose(env.time - t1, seg / V_CRUISE + 4 * T_TURN)),
          "backtrack returns to the start and costs the reversal yaw (180 deg = 4 T_TURN) + the flight back")
    back_dir = env._bearing(p1, env.cur)
    check(bool(torch.allclose(env.heading, back_dir)), "after a backtrack the heading is the direction flown back")


def review_fix_checks(dev):
    print("[2b] review fixes (flown turn bound, executed objectives, walls)")
    W = [generate(s) for s in range(4)]
    b = batch_from_levels([{"world": i, "legs_seed": 30 + i} for i in range(4)], W, 32, ConfigT(), dev)
    B = len(b["legs"])
    env = TEnv(b, sample_pref3(B), domain=EVAL, generator=torch.Generator(device=dev).manual_seed(1))
    g = torch.Generator().manual_seed(3)
    for _ in range(T_MAX):
        _, _, L = env.obs_pair(False)
        mode = torch.multinomial(L.float().cpu(), 1, generator=g)[:, 0]
        _, done = env.step(mode, torch.rand(B, 3, generator=g, dtype=torch.float64) * 2 - 1)
        if done:
            break
    Pe, ne = env.executed()
    m = flight_metrics(Pe, ne, b["boxes"], env.obst)
    check(int(env.backs.sum()) > 0 and bool(torch.allclose(m["time_s"], env.time, atol=1e-9)),
          f"random episodes ({int(env.backs.sum())} backtracks): env time == the judge's time on the executed polyline")
    check(bool((m["turn_per_m"] <= KAPPA * (1 + 1e-9)).all()) and bool((m["max_turn"] <= TURN_MAX + 1e-9).all()),
          f"flown turn per horizontal metre <= {math.degrees(KAPPA):.0f} deg/m and turns <= 45 deg on every executed path "
          f"(max {math.degrees(m['turn_per_m'].max().item()):.0f} deg/m; backtrack reversals excluded)")
    s = env.success
    if s.any():
        Pr, nr = env.path()
        mr = flight_metrics(Pr, nr, b["boxes"], env.obst)
        check(bool(torch.allclose(env.F_exec[s], m["F"][s])) and bool(torch.allclose(env.F[s], mr["F"][s])) and
              bool((env.F_exec[s, 0] >= env.F[s, 0] - 1e-9).all()),
              f"F_exec = the executed flight's objectives (rewarded), F = the route's (reported), F_exec time >= F time "
              f"({int(s.sum())} successes)")
    # walls: a narrow hall; steering into the wall collides (it used to be clamped silently)
    hall = np.array([-30, 30, -1.4, 1.4])
    nb = build_batch([(hall, np.zeros((0, 5)), np.zeros((0, 4)), np.array([0, 0, 1.8, 8, 0, 1.8]), None)], dev)
    env = TEnv(nb, torch.full((1, 3), 1 / 3, dtype=torch.float64), domain=EVAL)
    for _ in range(4):
        _, done = env.step(torch.full((1,), ADVANCE), torch.tensor([[1.0, 1.0, 0.0]]))
        if done:
            break
    check(bool(env.crashed.all()) and float(env.cur[0, 1]) < 1.4 - 0.5 + 1e-9,
          "steering into a hall wall's 0.5 m margin is a collision (no clamping); the drone stays outside the margin")
    P = torch.tensor([[[0, 0, 1.8], [4, 1.2, 1.8], [8, 0, 1.8]]], dtype=torch.float64, device=dev)
    n3 = torch.tensor([3], device=dev)
    mw = flight_metrics(P, n3, nb["boxes"], env.obst, walls=nb["kmask"] & ~env.obst)
    mo = flight_metrics(P, n3, nb["boxes"], env.obst)
    check(bool(mw["collides"].all()) and not bool(mo["collides"].any()), "the judge: a path within 0.5 m of a wall collides")


def reward_checks(dev):
    print("[2c] reward")
    legs = [[0, 0, 1.8, 7.3, 0, 1.8], [0, 0, 1.8, 0, 9.1, 1.8], [-3, 2, 1.8, 4, -5, 1.8]]
    ob = build_batch(open_items(legs), dev)
    w = torch.tensor([[0.5, 0.25, 0.25], [1 / 3, 1 / 3, 1 / 3], [0.2, 0.4, 0.4]], dtype=torch.float64)
    env = TEnv(ob, w, domain=EVAL, c_prox=0.0, act_pen=0.0)
    tot = torch.zeros(3, dtype=torch.float64)
    for _ in range(60):
        r_, done = env.step(torch.full((3,), ADVANCE), torch.tensor([[1.0, 0.0, 0.0]] * 3))
        tot += r_.double().cpu()
        if done:
            break
    q = (env.r_term.cpu() - SUCC).neg()
    exp = SHAPE + SUCC - C_TIME * w[:, 0] - q
    check(bool(env.success.all()) and bool(torch.allclose(tot, exp, atol=1e-5)),
          f"straight open flight: return = SHAPE + SUCC - C_TIME w_time ({tot.tolist()})")
    W = [generate(s) for s in range(4)]
    b = batch_from_levels([{"world": i, "legs_seed": 60 + i} for i in range(4)], W, 32, ConfigT(), dev)
    B = len(b["legs"])
    pw = sample_pref3(B, torch.Generator().manual_seed(4))
    env = TEnv(b, pw, domain=EVAL)
    g = torch.Generator().manual_seed(5)
    tot = torch.zeros(B, dtype=torch.float64)
    for _ in range(T_MAX):
        _, _, L = env.obs_pair(False)
        mode = torch.multinomial(L.float().cpu(), 1, generator=g)[:, 0]
        r_, done = env.step(mode, torch.rand(B, 3, generator=g, dtype=torch.float64) * 2 - 1)
        tot += r_.double().cpu()
        if done:
            break
    s, f = env.success.cpu(), ~env.success.cpu()
    ratio = (env.time / env.t_st).cpu()
    charged = torch.where(s, C_TIME * pw[:, 0] * ratio.clamp(max=T_CAP), C_TIME * pw[:, 0] * T_CAP)
    fail_time = (-env.r_term.cpu() - FAIL)                                       # the top-up of failures
    check(int(env.backs.sum()) > 0 and bool(torch.allclose(env.r_time.cpu()[s], charged[s], atol=1e-9)) and
          bool(torch.allclose((env.r_time.cpu() + fail_time)[f], charged[f], atol=1e-9)),
          f"time charges = C_TIME w_time min(time / straight, T_CAP) on successes, the full cap on failures "
          f"({int(s.sum())} successes, {int(f.sum())} failures, {int(env.backs.sum())} backtracks)")
    worst_success = SHAPE + SUCC - C_TIME * T_CAP - Q_MAX
    best_failure = SHAPE - FAIL - C_TIME * T_CAP
    bound = SHAPE - FAIL - C_TIME * pw[:, 0] * T_CAP                             # per episode (its w_time)
    check(worst_success - best_failure >= 4.0 - 1e-9 and (not f.any() or bool((tot[f] <= bound[f] + 1e-5).all())),
          f"the worst success ({worst_success:+.1f} before step penalties) beats the best failure ({best_failure:+.1f}); "
          f"every random failure returned <= that")


def ppo_checks(dev, batch_n):
    print("[3] PPO")
    W = [generate(s) for s in range(6)]
    cfg = ConfigT()
    actor = make_actor_t(cfg).to(dev)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = batch_from_levels([{"world": i, "legs_seed": i} for i in range(4)], W, 8, cfg, dev)
    roll, _ = rollout_t(actor, b, sample_pref3(len(b["legs"])), cfg)
    keep = roll["ALIVE"]
    with torch.no_grad():
        dm, dc = actor.dist(roll["O"][keep], roll["LEGAL"][keep])
        lp = ActorG.logp(dm, dc, roll["MODE"][keep], roll["U"][keep])
    err = (lp - roll["LP"][keep]).abs().max().item()
    check(err < 1e-4, f"rollout log-probs == recomputation (max diff {err:.1e})")
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "vt.pt"
        c = ConfigT(iters=3, batch=batch_n, minibatch=batch_n * 8, levels_per_iter=8, device=str(dev), probe_every=1, ckpt_every=1)
        _, _, hist, _ = train_t(c, log=lambda s: None, ckpt=ck, worlds=W)
        check(len(hist) == 3 and all(math.isfinite(h["ret"]) and math.isfinite(h["value_loss"]) for h in hist),
              f"3 training iterations finite (success {hist[-1]['success'] * 100:.0f} %, crash {hist[-1]['crash'] * 100:.0f} %, "
              f"steps {hist[-1]['steps']:.1f})")
        c4 = ConfigT(**{**c.__dict__, "iters": 4})
        _, _, h2, _ = train_t(c4, log=lambda s: None, ckpt=ck, resume=load_checkpoint(ck), worlds=W)
        check(len(h2) == 4 and h2[-1]["it"] == 3, "resume continues at iteration 3")
        print("[4] eval_t.py (common judge, planning time)")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_t.py"), "--ckpt", str(ck),
                            "--vg", str(ROOT / "results" / "ckpt" / "vG3_box_seed1.pt"),
                            "--baselines", ",".join(str(ROOT / "results" / "ckpt" / f) for f in ("vS_seed1.pt", "vB_seed1.pt")),
                            "--sets", "heldout,factory2_real", "--worlds", "2", "--legs", "4", "--tag", "selftest"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and all(k in r.stdout for k in ("vT ", "vG3_box_seed1", "vS_seed1", "vB_seed1", "plan"))
        check(ok, "eval_t.py judges vT, vG, vS, vB with the same model and measures planning time"
              + ("" if ok else f": {r.stderr[-800:]}"))
        if ok:
            print("\n".join("      " + l for l in r.stdout.splitlines() if l.strip().startswith(("vT", "vG", "vS", "vB", "=="))))
        (ROOT / "results" / "eval_t_selftest.json").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    from rlpatrol.ppo import resolve_device
    dev = resolve_device(a.device)
    model_checks()
    env_checks(dev)
    review_fix_checks(dev)
    reward_checks(dev)
    ppo_checks(dev, a.batch)
    print(f"all {len(OK)} checks passed")


if __name__ == "__main__":
    main()
