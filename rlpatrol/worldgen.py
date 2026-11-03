"""Procedural factory worlds for training a policy that generalizes beyond factory2.

A generated world is a rectangular hall with axis-aligned obstacle boxes [xmin xmax ymin ymax top]
(world metres, z = height above the floor): blocks of parallel rack rows (random count, length,
orientation, aisle width; optional dead-end connectors that close one end of an aisle), machines of
random footprint and height (some low enough to fly over), partitions with or without a gap, and in
the OOD variant L-shaped stacks. As NMOPSO models obstacles as cylinders, every box is also covered
with cylinders exactly as planner/factory_patrol/factory_cylinders.m does (inflate, cell, thinR),
each cylinder keeping the height of its box (the 3D cost, rlpatrol/cost3d.py).

Validity (else the seed is rejected and the next tried): enough free space (>= min_free of the hall
at the drone's clearance), one connected free component holding >= 80 % of it; legs are sampled in
that component only, so every leg is reachable on the free grid.

Everything is a deterministic function of (params, seed)."""
import math
from dataclasses import dataclass, field, replace

import numpy as np

Z_PATROL = 1.8
RACK_TOP, RACK_T = 3.5, 0.4


@dataclass
class GenParams:
    hall_x: tuple = (20.0, 40.0)            # hall size (m), uniform
    hall_y: tuple = (15.0, 25.0)
    n_rack_blocks: tuple = (0, 2)
    rows_per_block: tuple = (1, 4)
    rack_len: tuple = (5.0, 14.0)
    aisle: tuple = (1.6, 3.5)               # free width between rows (m)
    dead_end_p: float = 0.3                 # probability that a block closes one end of an aisle
    n_machines: tuple = (2, 8)
    machine_size: tuple = (0.6, 3.0)
    machine_top: tuple = (0.8, 3.0)         # some below the patrol band (overflyable), some taller
    n_partitions: tuple = (0, 1)
    partition_gap_p: float = 0.6
    l_shapes: tuple = (0, 0)                # OOD: L-shaped stacks (two joined boxes)
    wall_margin: float = 1.0                # nodes stay this far from the walls
    clearance: float = 0.35                 # drone radius for the free grid
    min_free: float = 0.40
    grid: float = 0.25
    # NMOPSO cylinder cover (planner/factory_patrol/factory_layout.m)
    inflate: float = 0.4
    cell: float = 2.0
    thin_r: float = 0.7


TRAIN = GenParams()
OOD = replace(GenParams(), hall_x=(36.0, 48.0), hall_y=(22.0, 30.0), aisle=(1.4, 1.8), n_rack_blocks=(1, 3),
              n_machines=(4, 10), l_shapes=(1, 4))


def cylinders(boxes, inflate=0.4, cell=2.0, thin_r=0.7):
    """factory_cylinders.m: cover every box, inflated by `inflate`, with cylinders [x y R top] (metres).
    Rectangles narrower than `cell` get one row of radius-thin_r circles along their long axis."""
    out = []
    for x0, x1, y0, y1, top in np.asarray(boxes, float).reshape(-1, 5):
        r = (x0 - inflate, x1 + inflate, y0 - inflate, y1 + inflate)
        w, h = r[1] - r[0], r[3] - r[2]
        if min(w, h) < cell and thin_r > min(w, h) / 2:
            long, half = max(w, h), min(w, h) / 2
            step = 2 * math.sqrt(thin_r ** 2 - half ** 2)
            k = int(math.ceil(long / step))
            s = (np.arange(1, k + 1) - 0.5) * long / k
            if w >= h:
                c = np.c_[r[0] + s, np.full(k, (r[2] + r[3]) / 2)]
            else:
                c = np.c_[np.full(k, (r[0] + r[1]) / 2), r[2] + s]
            out.append(np.c_[c, np.full(k, thin_r), np.full(k, top)])
            continue
        k1, k2 = max(1, int(math.ceil(w / cell))), max(1, int(math.ceil(h / cell)))
        dx, dy = w / k1, h / k2
        I, J = np.meshgrid(np.arange(k1), np.arange(k2))
        n = I.size
        out.append(np.c_[r[0] + (I.ravel("F") + 0.5) * dx, r[2] + (J.ravel("F") + 0.5) * dy,  # MATLAB I(:)
                         np.full(n, math.hypot(dx, dy) / 2), np.full(n, top)])
    return np.vstack(out) if out else np.zeros((0, 4))


def _overlap(a, b, gap=0.0):
    return not (a[1] + gap <= b[0] or b[1] + gap <= a[0] or a[3] + gap <= b[2] or b[3] + gap <= a[2])


@dataclass
class GenWorld:
    hall: np.ndarray                 # [xmin xmax ymin ymax]
    boxes: np.ndarray                # [K, 5]
    cyl: np.ndarray                  # [M, 4] x y R top (metres)
    seed: int
    params: GenParams = field(repr=False)
    free: np.ndarray = field(default=None, repr=False)        # [nx, ny] main free component
    gx: np.ndarray = field(default=None, repr=False)
    gy: np.ndarray = field(default=None, repr=False)

    def point_free(self, x, y):
        """in the main free component (grid lookup) and exactly clear of every box by the clearance."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ix = np.clip(np.round((x - self.gx[0]) / self.params.grid).astype(int), 0, len(self.gx) - 1)
        iy = np.clip(np.round((y - self.gy[0]) / self.params.grid).astype(int), 0, len(self.gy) - 1)
        ok = self.free[ix, iy].copy()
        c = self.params.clearance
        for x0, x1, y0, y1, _ in self.boxes:
            ok &= ~((x > x0 - c) & (x < x1 + c) & (y > y0 - c) & (y < y1 + c))
        return ok

    def sample_legs(self, n, rng, lmin=1.5, lmax=12.0, z=Z_PATROL):
        """n legs [ax ay az bx by bz] with both ends in the main free component, length in [lmin, lmax]."""
        out = np.zeros((0, 6))
        h, m = self.hall, self.params.wall_margin
        lo, hi = np.array([h[0] + m, h[2] + m]), np.array([h[1] - m, h[3] - m])
        for _ in range(200):
            if len(out) >= n:
                break
            k = 4 * (n - len(out)) + 32
            a = rng.uniform(lo, hi, (k, 2))
            ang = rng.uniform(0, 2 * np.pi, k)
            L = rng.uniform(lmin, lmax, k)
            b = a + L[:, None] * np.c_[np.cos(ang), np.sin(ang)]
            ok = (b >= lo).all(1) & (b <= hi).all(1)
            ok &= self.point_free(a[:, 0], a[:, 1]) & self.point_free(b[:, 0], b[:, 1])
            zz = np.full((ok.sum(), 1), z)
            out = np.vstack([out, np.hstack([a[ok], zz, b[ok], zz])])
        if len(out) < n:
            raise RuntimeError(f"world {self.seed}: could not sample {n} legs")
        return out[:n]

    def to_world(self):
        """an rlpatrol.world.World (NMOPSO units, 2D cylinders) for the map-based baselines (vB, vS)."""
        import torch
        from .world import World
        s, h = 10.0, self.hall
        mx, my = round((h[1] - h[0]) * s) + 1, round((h[3] - h[2]) * s) + 1
        wm = 1.5 * s
        bounds = np.array([1 + wm, mx - wm, 1 + wm, my - wm, 16.0, 20.0])
        c = self.cyl
        th = np.c_[(c[:, 0] - h[0]) * s + 1, (c[:, 1] - h[2]) * s + 1, np.zeros(len(c)), c[:, 2] * s]
        return World(hall=np.asarray(h, float), scale=s, bounds=bounds, threats=torch.tensor(th, dtype=torch.float64),
                     cyl_m=c[:, :3].copy(), cyl_margin=0.25, z_patrol=Z_PATROL, rects=self.boxes.copy(), direct_rays=True)

    def to_sdf(self, path, template="worlds/factory2.sdf"):
        """a Gazebo world: factory2.sdf's header (physics, PX4 systems, lights), a floor, walls and the boxes."""
        lines = open(template).read().splitlines()
        head = lines[:next(i for i, l in enumerate(lines) if '<model name="floor">' in l)]
        h = self.hall
        cx, cy, W, H = (h[0] + h[1]) / 2, (h[2] + h[3]) / 2, h[1] - h[0], h[3] - h[2]

        def box(name, x, y, z, sx, sy, sz, rgb):
            g = f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>"
            return (f'    <model name="{name}"><static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><link name="link">'
                    f'<collision name="c">{g}</collision><visual name="v">{g}<material><ambient>{rgb} 1</ambient>'
                    f'<diffuse>{rgb} 1</diffuse></material></visual></link></model>')
        body = [box("floor", cx, cy, -0.05, W + 0.6, H + 0.6, 0.1, "0.35 0.35 0.35")]
        for n, (x, y, sx, sy) in {"north_wall": (cx, h[3] + 0.15, W + 0.6, 0.3), "south_wall": (cx, h[2] - 0.15, W + 0.6, 0.3),
                                  "east_wall": (h[1] + 0.15, cy, 0.3, H), "west_wall": (h[0] - 0.15, cy, 0.3, H)}.items():
            body.append(box(n, x, y, 2.5, sx, sy, 5.0, "0.7 0.7 0.7"))
        for i, (x0, x1, y0, y1, top) in enumerate(self.boxes):
            body.append(box(f"obstacle_{i}", (x0 + x1) / 2, (y0 + y1) / 2, top / 2, x1 - x0, y1 - y0, top,
                            "0.55 0.45 0.3" if top >= 3 else "0.3 0.4 0.6"))
        open(path, "w").write("\n".join(head + body + ["  </world>", "</sdf>", ""]))


def _free_grid(hall, boxes, p):
    gx = np.arange(hall[0], hall[1] + 1e-9, p.grid)
    gy = np.arange(hall[2], hall[3] + 1e-9, p.grid)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    free = (X >= hall[0] + p.wall_margin) & (X <= hall[1] - p.wall_margin) & \
           (Y >= hall[2] + p.wall_margin) & (Y <= hall[3] - p.wall_margin)
    c = p.clearance
    for x0, x1, y0, y1, _ in boxes:
        free &= ~((X >= x0 - c) & (X <= x1 + c) & (Y >= y0 - c) & (Y <= y1 + c))
    return gx, gy, free


def _largest_component(free):
    lab = np.zeros(free.shape, int)
    best, best_n, cur = 0, 0, 0
    nx, ny = free.shape
    for i in range(nx):
        for j in range(ny):
            if free[i, j] and not lab[i, j]:
                cur += 1
                stack, n = [(i, j)], 0
                lab[i, j] = cur
                while stack:
                    a, b = stack.pop()
                    n += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        u, v = a + da, b + db
                        if 0 <= u < nx and 0 <= v < ny and free[u, v] and not lab[u, v]:
                            lab[u, v] = cur
                            stack.append((u, v))
                if n > best_n:
                    best, best_n = cur, n
    return lab == best, best_n


def _layout(rng, p):
    hw, hh = rng.uniform(*p.hall_x), rng.uniform(*p.hall_y)
    hall = np.array([-hw / 2, hw / 2, -hh / 2, hh / 2])
    m = p.wall_margin
    boxes = []

    def fits(b, gap=0.8):
        return (b[0] >= hall[0] + 0.3 and b[1] <= hall[1] - 0.3 and b[2] >= hall[2] + 0.3 and b[3] <= hall[3] - 0.3
                and not any(_overlap(b, o, gap) for o in boxes))

    for _ in range(rng.integers(p.n_rack_blocks[0], p.n_rack_blocks[1] + 1)):
        for _try in range(30):
            rows = int(rng.integers(p.rows_per_block[0], p.rows_per_block[1] + 1))
            L, aisle = rng.uniform(*p.rack_len), rng.uniform(*p.aisle)
            along_x = rng.random() < 0.5
            span = rows * RACK_T + (rows - 1) * aisle
            ext = (L, span) if along_x else (span, L)
            if ext[0] > hall[1] - hall[0] - 2 * m or ext[1] > hall[3] - hall[2] - 2 * m:
                continue
            x0 = rng.uniform(hall[0] + 0.3, hall[1] - 0.3 - ext[0])
            y0 = rng.uniform(hall[2] + 0.3, hall[3] - 0.3 - ext[1])
            block = []
            for r in range(rows):
                o = r * (RACK_T + aisle)
                block.append([x0, x0 + L, y0 + o, y0 + o + RACK_T, RACK_TOP] if along_x else
                             [x0 + o, x0 + o + RACK_T, y0, y0 + L, RACK_TOP])
            if rows >= 2 and rng.random() < p.dead_end_p:              # close one end of one aisle
                k = int(rng.integers(0, rows - 1))
                o = k * (RACK_T + aisle) + RACK_T
                if along_x:
                    xe = x0 if rng.random() < 0.5 else x0 + L - RACK_T
                    block.append([xe, xe + RACK_T, y0 + o, y0 + o + aisle, RACK_TOP])
                else:
                    ye = y0 if rng.random() < 0.5 else y0 + L - RACK_T
                    block.append([x0 + o, x0 + o + aisle, ye, ye + RACK_T, RACK_TOP])
            env = [min(b[0] for b in block), max(b[1] for b in block), min(b[2] for b in block), max(b[3] for b in block), 0]
            if fits(env, gap=1.2):
                boxes.extend(block)
                break
    for _ in range(rng.integers(p.n_partitions[0], p.n_partitions[1] + 1)):
        for _try in range(20):
            along_x = rng.random() < 0.5
            if along_x:
                y = rng.uniform(hall[2] + 3, hall[3] - 3)
                segs = [[hall[0] + 0.3, hall[1] - 0.3]]
            else:
                x = rng.uniform(hall[0] + 3, hall[1] - 3)
                segs = [[hall[2] + 0.3, hall[3] - 0.3]]
            if rng.random() < p.partition_gap_p:                       # a doorway
                a, b = segs[0]
                g = rng.uniform(a + 2, b - 4)
                segs = [[a, g], [g + rng.uniform(1.8, 3.0), b]]
            part = [[s0, s1, y - 0.15, y + 0.15, RACK_TOP] if along_x else [x - 0.15, x + 0.15, s0, s1, RACK_TOP]
                    for s0, s1 in segs if s1 - s0 > 0.5]
            if all(fits(b, gap=1.0) for b in part):
                boxes.extend(part)
                break
    for _ in range(rng.integers(p.n_machines[0], p.n_machines[1] + 1)):
        for _try in range(30):
            sx, sy = rng.uniform(*p.machine_size, 2)
            x0, y0 = rng.uniform(hall[0] + m, hall[1] - m - sx), rng.uniform(hall[2] + m, hall[3] - m - sy)
            b = [x0, x0 + sx, y0, y0 + sy, rng.uniform(*p.machine_top)]
            if fits(b):
                boxes.append(b)
                break
    for _ in range(rng.integers(p.l_shapes[0], p.l_shapes[1] + 1)):  # OOD: L-shaped stacks
        for _try in range(30):
            a, t, top = rng.uniform(2, 4), rng.uniform(0.6, 1.0), rng.uniform(1.0, 3.0)
            x0, y0 = rng.uniform(hall[0] + m, hall[1] - m - a), rng.uniform(hall[2] + m, hall[3] - m - a)
            L = [[x0, x0 + a, y0, y0 + t, top], [x0, x0 + t, y0 + t, y0 + a, top]]
            if all(fits(b) for b in L[:1]) and fits(L[1]):
                boxes.extend(L)
                break
    return hall, np.array(boxes, float).reshape(-1, 5)


def generate(seed, p=TRAIN, max_tries=50):
    """a valid GenWorld for this seed (tries seed, seed + 1e6, ... until valid)."""
    for k in range(max_tries):
        rng = np.random.default_rng(seed + k * 1_000_003)
        hall, boxes = _layout(rng, p)
        gx, gy, free = _free_grid(hall, boxes, p)
        comp, n = _largest_component(free)
        total = ((gx >= hall[0] + p.wall_margin) & (gx <= hall[1] - p.wall_margin)).sum() * \
                ((gy >= hall[2] + p.wall_margin) & (gy <= hall[3] - p.wall_margin)).sum()
        if free.sum() >= p.min_free * total and n >= 0.8 * free.sum():
            cyl = cylinders(boxes, p.inflate, p.cell, p.thin_r)
            return GenWorld(hall, boxes, cyl, seed, p, comp, gx, gy)
    raise RuntimeError(f"no valid world for seed {seed}")


def unexpected_box(leg, rng, size=(0.4, 1.0), top=(0.8, 2.6)):
    """an obstacle box on the middle part of a leg's straight line (the map does not have it)."""
    a, b = np.asarray(leg[:2]), np.asarray(leg[3:5])
    c = a + rng.uniform(0.35, 0.65) * (b - a)
    sx, sy = rng.uniform(*size, 2)
    return np.array([c[0] - sx / 2, c[0] + sx / 2, c[1] - sy / 2, c[1] + sy / 2, rng.uniform(*top)])
