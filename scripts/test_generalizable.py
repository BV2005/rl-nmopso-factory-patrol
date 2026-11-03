"""Self-test of the generalizable-policy pipeline (docs/generalizable_rl.md). Minutes on a CPU.
  1. worldgen: deterministic, valid (legs in the free component), the cylinder cover reproduces
     factory2's data/factory2_model.json cylinders from its rects (port of factory_cylinders.m)
  2. MyCost-3D == MyCost (nmopso_cost.Legs.cost) on factory2 paths with infinitely tall obstacles;
     the 3D rule: over a low box is feasible, through a tall one is not
  3. batched scan == sensors.sim_scan on factory2; height channel (low box known, tall rack unknown)
  4. env: shapes, legal modes, backtrack returns to the previous node, climb rises above the band,
     termination within T_DEC, finite rewards
  5. PPO: rollout log-probs == recomputation; PLR prefers high-score levels; 3 training iterations
     (P2 and P3 settings) finite and resumable; eval_worlds on 2 worlds
  6. round 2: defaults == round 1 (old checkpoints), proximity penalty near obstacles only, heavier
     quality weights never improve the terminal reward, exploration decay caps the std, the
     evaluation domain is noise-free, the candidate-set protocol (HV) runs
  7. round 3: no leg starts / ends in a collision zone (either model), unexpected boxes keep them clear;
     segment-to-box distance exact (vs brute force); box-model cost (beside / through / over a box) and
     collision check agree; the scan reports ranges to the zone (margin) and inflated_bins gives the same
     from a point cloud (flight); box-model training + NMOPSO re-scoring in eval_worlds
  python scripts/test_generalizable.py [--device cpu] [--batch 256]"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rlpatrol import sensors as S  # noqa: E402
from rlpatrol.cost3d import TOP_INF, cost3d  # noqa: E402
from rlpatrol.env_g import (ADVANCE, BACK, CLIMB, OBS_BASE, PRIV, T_DEC, ZBAND, GenEnv, build_batch,  # noqa: E402
                            factory2_level, sample_conditions)
from rlpatrol.nmopso_cost import N_NODES, Legs  # noqa: E402
from rlpatrol.ppo_g import (PLR, ActorG, ConfigG, batch_from_levels, load_checkpoint, make_actor_g,  # noqa: E402
                            rollout_g, train_g)
from rlpatrol.scan3d import H_MAX, sim_scan_b  # noqa: E402
from rlpatrol.world import load_world  # noqa: E402
from rlpatrol.worldgen import TRAIN, cylinders, generate  # noqa: E402

OK = []


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    OK.append(msg)
    print(f"  ok  {msg}", flush=True)


def worldgen_checks():
    print("[1] world generator")
    a, b = generate(7), generate(7)
    check(np.array_equal(a.boxes, b.boxes) and np.array_equal(a.cyl, b.cyl), "deterministic in the seed")
    W = [generate(s) for s in range(12)]
    check(all(len(w.boxes) >= 2 for w in W), f"12 worlds, {np.mean([len(w.boxes) for w in W]):.1f} boxes each")
    rng = np.random.default_rng(0)
    ok = True
    for w in W:
        L = w.sample_legs(20, rng)
        ok &= bool(w.point_free(L[:, 0], L[:, 1]).all() and w.point_free(L[:, 3], L[:, 4]).all())
        for x0, x1, y0, y1, _ in w.boxes:                          # leg ends outside every box + clearance
            for P in (L[:, :2], L[:, 3:5]):
                ok &= not bool(((P[:, 0] > x0 - 0.3) & (P[:, 0] < x1 + 0.3) & (P[:, 1] > y0 - 0.3) & (P[:, 1] < y1 + 0.3)).any())
    check(ok, "legs start and end in free space, clear of every box")
    J = json.loads((ROOT / "data" / "factory2_model.json").read_text())
    c = cylinders(np.array(J["rects"]), J["inflate"], 2.0, 0.7)
    ref = np.array(J["cylinders_m"])
    check(c.shape[0] == ref.shape[0] and np.allclose(c[:, :3], ref, atol=1e-9),
          f"cylinder cover == factory2's {len(ref)} cylinders (factory_cylinders.m port)")


def cost_checks(world):
    print("[2] MyCost-3D")
    rng = np.random.default_rng(1)
    from rlpatrol.legs import random_legs
    legs = random_legs(world, 64, rng)
    st, gl = world.to_units(legs[:, :3]), world.to_units(legs[:, 3:])
    L = Legs(world, st, gl)
    r = L.rmin[:, None] + (L.rmax - L.rmin)[:, None] * torch.rand(64, N_NODES, dtype=torch.float64)
    nodes = L.decode(r, (torch.rand(64, N_NODES) - 0.5).double() * 0.6, (torch.rand(64, N_NODES) - 0.5).double() * 0.6)
    F0 = L.cost(nodes)
    P = torch.cat([torch.tensor(legs[:, None, :3]), torch.tensor(world.to_metres(nodes.numpy())), torch.tensor(legs[:, None, 3:])], 1)
    hall, boxes, cyl = factory2_level(world)
    c = torch.tensor(cyl)[None].repeat(64, 1, 1)
    F1 = cost3d(P, torch.full((64,), N_NODES + 2), c, torch.ones(64, len(cyl), dtype=torch.bool), (1.6, 2.0))
    same = torch.isfinite(F0) == torch.isfinite(F1)
    fin = torch.isfinite(F0) & torch.isfinite(F1)
    err = (F0[fin] - F1[fin]).abs().max().item() if fin.any() else 0.0
    check(bool(same.all()) and err < 1e-9, f"MyCost-3D == MyCost on 64 factory2 paths, tall obstacles (max diff {err:.1e}, "
          f"{int(torch.isfinite(F0).all(-1).sum())} feasible)")
    box = np.array([[-0.5, 0.5, -0.5, 0.5, 1.0]])
    cl = torch.tensor(cylinders(box))[None]
    path = torch.tensor([[[-4.0, 0, 1.8], [0.0, 0, 2.4], [4.0, 0, 1.8]]])
    m = torch.ones(1, len(cl[0]), dtype=torch.bool)
    check(bool(torch.isfinite(cost3d(path, torch.tensor([3]), cl, m)).all()), "over a 1.0 m box at 1.8-2.4 m: feasible")
    cl2 = cl.clone(); cl2[..., 3] = 3.0
    check(bool(torch.isinf(cost3d(path, torch.tensor([3]), cl2, m)[0, 1])), "through a 3.0 m box: collision (J2 = inf)")


def scan_checks(world):
    print("[3] batched scan")
    rng = np.random.default_rng(2)
    from rlpatrol.legs import random_legs
    legs = random_legs(world, 32, rng)
    pos = torch.tensor(legs[:, :3]); yaw = torch.rand(32, dtype=torch.float64) * 6.28
    sb = S.scene_boxes(world)
    ref = S.sim_scan(sb, pos, yaw, S.SensorSpec(noise=False))
    km = torch.ones(32, len(sb), dtype=torch.bool)
    r, h = sim_scan_b(sb[None].repeat(32, 1, 1), km, pos, yaw, S.SensorSpec(noise=False))
    check(torch.allclose(r, ref, atol=1e-6), "batched range scan == sensors.sim_scan (32 factory2 poses)")
    far = [[-50.3, -50, -50, 50, 10], [50, 50.3, -50, 50, 10], [-50, 50, -50.3, -50, 10], [-50, 50, 50, 50.3, 10]]
    bx = torch.tensor([far + [[3.0, 3.6, -0.3, 0.3, 1.2]], far + [[1.5, 1.9, -3.0, 3.0, 3.5]]], dtype=torch.float64)
    p = torch.tensor([[0.0, 0, 1.8], [0.0, 0, 1.8]], dtype=torch.float64)
    r, h = sim_scan_b(bx, torch.ones(2, 5, dtype=torch.bool), p, torch.zeros(2, dtype=torch.float64), S.SensorSpec(noise=False))
    mid = S.SCAN_BINS // 2
    check(abs(h[0, mid].item() * H_MAX - 1.2) < 1e-6, f"a 1.2 m box ahead: height known ({h[0, mid].item() * H_MAX:.2f} m)")
    check(h[1, mid].item() == 1.0, "a 3.5 m rack 1.5 m ahead (top above the camera view): height unknown (1.0)")


def env_checks(dev, batch_n):
    print("[4] environment")
    W = [generate(s) for s in range(4)]
    cfg = ConfigG(p_extra=0.5, modes="all")
    b = batch_from_levels([{"world": i, "legs_seed": 10 + i} for i in range(4)], W, batch_n // 4, cfg, dev)
    B = len(b["legs"])
    w, u = sample_conditions(B, True)
    env = GenEnv(b, w, u)
    o, oc, L = env.obs_pair()
    check(o.shape == (B, OBS_BASE + 5) and oc.shape == (B, OBS_BASE + PRIV + 5) and torch.isfinite(o).all(),
          f"obs [{B}, {o.shape[1]}], critic [{B}, {oc.shape[1]}], finite")
    check(bool(L[:, ADVANCE].all()) and not bool(L[:, BACK].any()), "at node 0: advance legal, backtrack not")
    z0 = env.cur.clone()
    env.step(torch.full((B,), ADVANCE), torch.zeros(B, 3))
    adv = env.alive & (env.k == 1)
    p1 = env.cur.clone()
    _, _, L = env.obs_pair()
    check(bool(L[adv, BACK].all()), "after one node: backtrack legal")
    env.step(torch.full((B,), BACK), torch.zeros(B, 3))
    check(bool(torch.allclose(env.cur[adv], z0[adv])), "backtrack returns to the previous node")
    check(bool(env.tabu[adv, 0].any(-1).all()), "the direction tried from there is tabu")
    env2 = GenEnv(b, w, u)
    env2.step(torch.full((B,), CLIMB), torch.ones(B, 3))
    up = env2.alive & (env2.k == 1)
    check(bool((env2.cur[up, 2] > ZBAND[1] + 0.05).all()), "climb rises above the patrol band")
    env3 = GenEnv(b, w, u)
    tot = torch.zeros(B, device=dev)
    for t in range(T_DEC):
        _, _, L = env3.obs_pair(False)
        mode = torch.multinomial(L.float(), 1)[:, 0]
        r, done = env3.step(mode, torch.rand(B, 3) * 2 - 1)
        tot += r
        if done:
            break
    check(done and torch.isfinite(tot).all(), f"random episodes end within {T_DEC} decisions; rewards finite "
          f"(feasible {env3.feasible.float().mean() * 100:.0f} %, crash {env3.crashed.float().mean() * 100:.0f} %)")


def ppo_checks(dev, batch_n):
    print("[5] PPO, PLR, training, evaluation")
    W = [generate(s) for s in range(8)]
    cfg = ConfigG(modes="all")
    actor = make_actor_g(cfg).to(dev)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.05 * torch.randn_like(p))
    b = batch_from_levels([{"world": i, "legs_seed": i} for i in range(4)], W, 16, cfg, dev)
    w, u = sample_conditions(len(b["legs"]), True)
    roll, _ = rollout_g(actor, b, w, u, cfg)
    keep = roll["ALIVE"]
    with torch.no_grad():
        dm, dc = actor.dist(roll["O"][keep], roll["LEGAL"][keep])
        lp = ActorG.logp(dm, dc, roll["MODE"][keep], roll["U"][keep])
    err = (lp - roll["LP"][keep]).abs().max().item()
    check(err < 1e-4, f"rollout log-probs == recomputation (max diff {err:.1e})")
    plr = PLR(8, 100, 0.0, 0.1, 0.0, np.random.default_rng(0))
    lv = plr.sample(8, 0)
    plr.update(lv, np.arange(8, dtype=float), 0)
    picks = [plr.sample(1, 1)[0]["score"] for _ in range(200)]
    check(np.mean(picks) > 5.5, f"PLR replays high-score levels more often (mean score {np.mean(picks):.2f} of 0..7)")
    with tempfile.TemporaryDirectory() as d:
        for name, extra in (("p2", {}), ("p3", {"plr": True, "p_new": 0.5, "dr": 1})):
            ck = Path(d) / f"{name}.pt"
            c = ConfigG(iters=3, batch=batch_n, minibatch=batch_n * 4, levels_per_iter=8, device=str(dev), probe_every=1,
                        ckpt_every=1, **extra)
            _, _, hist, plr = train_g(c, log=lambda s: None, ckpt=ck, worlds=W)
            check(len(hist) == 3 and all(math.isfinite(h["ret"]) and math.isfinite(h["value_loss"]) for h in hist),
                  f"{name}: 3 iterations finite (feasible {hist[-1]['feasible'] * 100:.0f} %, modes "
                  f"{np.round(hist[-1]['mode_frac'], 2).tolist()}, plr {len(plr.levels)})")
        c = ConfigG(**{**c.__dict__, "iters": 4})
        _, _, hist2, _ = train_g(c, log=lambda s: None, ckpt=ck, resume=load_checkpoint(ck), worlds=W)
        check(len(hist2) == 4 and hist2[-1]["it"] == 3, "resume continues at iteration 3")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_worlds.py"), "--ckpt", str(ck), "--sets",
                            "heldout,factory2_real", "--worlds", "2", "--legs", "4", "--tag", "selftest"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and "== heldout" in r.stdout and "== factory2_real" in r.stdout
        check(ok, "eval_worlds.py runs on held-out worlds and factory2" + ("" if ok else f": {r.stderr[-600:]}"))
        (ROOT / "results" / "eval_g_selftest.json").unlink(missing_ok=True)


def round2_checks(dev, batch_n):
    print("[6] round 2")
    from rlpatrol.env_g import DR, EVAL
    from rlpatrol.ppo_g import env_options
    old = {"modes": "all", "budget": True}                          # a round-1 cfg dict: no round-2 keys
    o = env_options(old)
    check(o["wq"] == (1.0, 1.0, 1.0, 1.0) and o["c_prox"] == 0.0 and o["max_back"] == 5,
          "round-1 checkpoints keep round-1 rewards (defaults)")
    check(EVAL.noise is False and EVAL.exec_sigma == 0 and DR[2].exec_sigma == 0.03 and DR[1].exec_sigma == 0.1,
          "evaluation domain noise-free; dr 2 = moderate (0.03 m), dr 1 unchanged")
    far = [[-30.3, -30, -30, 30, 10], [30, 30.3, -30, 30, 10], [-30, 30, -30.3, -30, 10], [-30, 30, 30, 30.3, 10]]
    box = np.array([0.8, 1.2, 0.9, 1.3, 3.0])                       # beside the first step (cover ~0.2 m away), not on it
    items = [(np.array([-30, 30, -30, 30.0]), np.array([box]), cylinders(box[None]), np.array([0, 0, 1.8, 6, 0, 1.8]), None),
             (np.array([-30, 30, -30, 30.0]), np.zeros((0, 5)), np.zeros((0, 4)), np.array([0, 0, 1.8, 6, 0, 1.8]), None)]
    b = build_batch(items, dev)
    w = torch.tensor([[1 / 3, 0, 1 / 3, 1 / 3]] * 2, dtype=torch.float64)
    u = torch.tensor([0.6, 0.6], dtype=torch.float64)
    rs = []
    for cp in (0.0, 0.5):
        env = GenEnv(b, w, u, domain=EVAL, c_prox=cp, d_safe=1.0)
        r, _ = env.step(torch.full((2,), ADVANCE), torch.zeros(2, 3))
        rs.append(r)
    near = (rs[0] - rs[1])
    check(bool(near[0] > 1e-4) and abs(near[1].item()) < 1e-9 and not bool(env.crashed.any()),
          f"proximity penalty for the step beside a box ({near[0].item():.3f}), none in the open")
    W = [generate(s) for s in range(4)]
    cfg = ConfigG(modes="all")
    bb = batch_from_levels([{"world": i, "legs_seed": 30 + i} for i in range(4)], W, 16, cfg, dev)
    ww, uu = sample_conditions(len(bb["legs"]), True, gen=torch.Generator().manual_seed(1))
    term = []
    for wq in ((1.0, 1.0, 1.0, 1.0), (2.0, 1.0, 2.0, 2.0)):
        env = GenEnv(bb, ww, uu, domain=EVAL, wq=wq)
        tot = torch.zeros(len(bb["legs"]), device=dev)
        for t in range(T_DEC):
            r, done = env.step(torch.full((len(bb["legs"]),), ADVANCE), torch.zeros(len(bb["legs"]), 3))
            tot += r
            if done:
                break
        term.append(tot)
    check(bool((term[1] <= term[0] + 1e-9).all()), "heavier F1/F3/F4 weights never improve the terminal reward")
    with tempfile.TemporaryDirectory() as d:
        c = ConfigG(iters=3, batch=batch_n, minibatch=batch_n * 4, levels_per_iter=8, device=str(dev), probe_every=1,
                    noise_decay=True, log_std_end=-2.2, c_prox=0.5, w_len=2.0, w_alt=2.0, w_smooth=2.0, c_back=1.0, max_back=3)
        ck = Path(d) / "r2.pt"
        actor, _, hist, _ = train_g(c, log=lambda s: None, ckpt=ck, worlds=W)
        check(len(hist) == 3 and bool((actor.log_std <= -2.2 + 1e-6).all()),
              f"round-2 training finite; exploration decayed to log_std {actor.log_std.max().item():.2f} at the end")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_worlds.py"), "--ckpt", str(ck), "--sets", "heldout",
                            "--worlds", "1", "--legs", "2", "--candidates", "1", "--tag", "selftest_cand"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and "HV" in r.stdout
        check(ok, "candidate-set protocol (HV) runs" + ("" if ok else f": {r.stderr[-600:]}"))
        (ROOT / "results" / "eval_g_selftest_cand.json").unlink(missing_ok=True)


def round3_checks(dev, batch_n):
    print("[7] round 3: clean legs, box collision model")
    from rlpatrol.cost3d import cost3d_box, seg_box_dist, segment_check_box
    from rlpatrol.scan3d import inflated_bins
    from rlpatrol.worldgen import LEG_MARGIN, unexpected_box, zone_clearance
    rng = np.random.default_rng(3)
    bad = tot = 0
    clear_extra = True
    for s in range(20):
        w = generate(100 + s)
        L = w.sample_legs(40, rng)
        for P in (L[:, :2], L[:, 3:5]):
            bad += int((zone_clearance(P, w.cyl, w.boxes) < LEG_MARGIN).sum()); tot += len(P)
        for l in L[:8]:
            b = unexpected_box(l, rng)
            if b is not None:
                clear_extra &= bool((zone_clearance(np.array([l[:2], l[3:5]]), cylinders(b[None]), b[None]) >= LEG_MARGIN).all())
    check(bad == 0 and clear_extra, f"no leg end in a collision zone ({tot} ends, round 2: 6.8 %); unexpected boxes keep ends clear")
    g = torch.Generator().manual_seed(4)
    a = torch.rand(64, 1, 2, dtype=torch.float64, generator=g) * 6 - 3
    bb = torch.rand(64, 1, 2, dtype=torch.float64, generator=g) * 6 - 3
    c = torch.rand(64, 1, 2, dtype=torch.float64, generator=g) * 4 - 2
    box = torch.cat([c[..., :1] - 0.5, c[..., :1] + 0.5, c[..., 1:] - 0.3, c[..., 1:] + 0.3, torch.full((64, 1, 1), 2.0, dtype=torch.float64)], -1)
    d = seg_box_dist(a, bb, box)[:, 0, 0]
    t = torch.linspace(0, 1, 4001, dtype=torch.float64)[None, :, None]
    pts = a + t * (bb - a)
    dx = torch.clamp(torch.maximum(box[..., 0] - pts[..., 0], pts[..., 0] - box[..., 1]), min=0)
    dy = torch.clamp(torch.maximum(box[..., 2] - pts[..., 1], pts[..., 1] - box[..., 3]), min=0)
    brute = torch.hypot(dx, dy).min(1).values
    check(bool((d <= brute + 1e-9).all() and (brute - d).max() < 3e-3),
          f"segment-to-box distance exact (vs 4001-point brute force, max gap {(brute - d).max().item():.1e} m)")
    B1 = torch.tensor([[[-0.5, 0.5, -0.5, 0.5, 3.0]]], dtype=torch.float64)
    m1 = torch.ones(1, 1, dtype=torch.bool)
    n3 = torch.tensor([3])
    side = torch.tensor([[[-4.0, 1.1, 1.8], [0.0, 1.1, 1.8], [4.0, 1.1, 1.8]]], dtype=torch.float64)   # 0.6 m beside
    near = torch.tensor([[[-4.0, 0.9, 1.8], [0.0, 0.9, 1.8], [4.0, 0.9, 1.8]]], dtype=torch.float64)   # 0.4 m beside
    over = torch.tensor([[[-4.0, 0.0, 1.8], [0.0, 0.0, 2.4], [4.0, 0.0, 1.8]]], dtype=torch.float64)
    low = B1.clone(); low[..., 4] = 1.0
    ok = (torch.isfinite(cost3d_box(side, n3, B1, m1)).all() and torch.isinf(cost3d_box(near, n3, B1, m1)[0, 1])
          and torch.isfinite(cost3d_box(over, n3, low, m1)).all() and torch.isinf(cost3d_box(over, n3, B1, m1)[0, 1]))
    check(bool(ok), "box model: 0.6 m beside a box feasible, 0.4 m collides (margin 0.5), over a 1.0 m box feasible, over 3.0 m not")
    A = torch.rand(256, 3, dtype=torch.float64, generator=g) * 4 - 2; A[:, 2] = 1.8
    Bq = torch.rand(256, 3, dtype=torch.float64, generator=g) * 4 - 2; Bq[:, 2] = 1.8
    hit, gap = segment_check_box(A, Bq, B1.repeat(256, 1, 1), m1.repeat(256, 1))
    Fh = cost3d_box(torch.stack([A, Bq], 1), torch.full((256,), 2), B1.repeat(256, 1, 1), m1.repeat(256, 1))[:, 1]
    check(bool((hit == torch.isinf(Fh)).all()) and bool(((gap < 0) == hit).all()),
          f"collision check == cost-model collision on 256 random segments ({int(hit.sum())} hits)")
    far = [[-50.3, -50, -50, 50, 10], [50, 50.3, -50, 50, 10], [-50, 50, -50.3, -50, 10], [-50, 50, 50, 50.3, 10]]
    bx = torch.tensor([far + [[-6, 6, 5.0, 5.4, 3.5]]], dtype=torch.float64)          # a rack 5 m to the left
    p = torch.tensor([[0.0, 0, 1.8]], dtype=torch.float64)
    r0, _ = sim_scan_b(bx, torch.ones(1, 5, dtype=torch.bool), p, torch.zeros(1, dtype=torch.float64), S.SensorSpec(noise=False))
    r5, _ = sim_scan_b(bx, torch.ones(1, 5, dtype=torch.bool), p, torch.zeros(1, dtype=torch.float64), S.SensorSpec(noise=False), margin=0.5)
    side_bin = int(np.argmin(np.abs(S.bin_centres().numpy() - math.pi / 2)))
    check(abs((r0[0, side_bin] - r5[0, side_bin]).item() * S.SCAN_MAX - 0.5) < 1e-6,
          f"scan margin: the rack 5 m to the left reads {r5[0, side_bin].item() * S.SCAN_MAX:.2f} m (zone at 4.5 m)")
    xs = np.linspace(-6, 6, 2401)
    surf = np.c_[xs, np.full_like(xs, 5.0)]                                            # the rack's near face
    ib = inflated_bins(surf, 0.5)
    sim = r5[0].numpy() * S.SCAN_MAX
    lidar_only = np.abs(S.bin_centres().numpy()) > S.DEPTH_HFOV / 2 + S.BIN
    m = lidar_only & np.isfinite(ib) & (sim < S.SCAN_MAX - 1e-6)
    err = np.abs(ib[m] - sim[m]).max() if m.any() else 1.0
    check(m.any() and err < 0.03, f"inflated_bins (flight, point cloud) == simulated zone ranges ({m.sum()} bins, max {err:.3f} m)")
    W = [generate(s) for s in range(6)]
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "r3.pt"
        c3 = ConfigG(iters=3, batch=batch_n, minibatch=batch_n * 4, levels_per_iter=8, device=str(dev), probe_every=1,
                     collision="box", box_margin=0.5, c_prox=0.5)
        _, _, hist, _ = train_g(c3, log=lambda s: None, ckpt=ck, worlds=W)
        check(len(hist) == 3 and all(math.isfinite(h["ret"]) for h in hist),
              f"box-model training finite (feasible {hist[-1]['feasible'] * 100:.0f} %, crash {hist[-1]['crash'] * 100:.0f} %)")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_worlds.py"), "--ckpt", str(ck), "--sets",
                            "heldout,factory2_real", "--worlds", "2", "--legs", "4", "--tag", "selftest_r3"],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and "NMOPSO model" in r.stdout
        check(ok, "eval_worlds re-scores box-model paths with NMOPSO's cylinder model" + ("" if ok else f": {r.stderr[-600:]}"))
        (ROOT / "results" / "eval_g_selftest_r3.json").unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    from rlpatrol.ppo import resolve_device
    dev = resolve_device(a.device)
    world = load_world()
    worldgen_checks()
    cost_checks(world)
    scan_checks(world)
    env_checks(dev, a.batch)
    ppo_checks(dev, a.batch)
    round2_checks(dev, a.batch)
    round3_checks(dev, a.batch)
    print(f"all {len(OK)} checks passed")


if __name__ == "__main__":
    main()
