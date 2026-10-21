"""The factory2 world as NMOPSO sees it, loaded from data/factory2_model.json (written by
matlab/export_world.m through factory_patrol's own CreateModel.m)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass
class World:
    hall: np.ndarray          # [xmin xmax ymin ymax] metres
    scale: float              # NMOPSO units per metre (10: 1 unit = 0.1 m)
    bounds: np.ndarray        # [xmin xmax ymin ymax zmin zmax] NMOPSO units (clamping of SphericalToCart2)
    threats: torch.Tensor     # [M, 4] cylinders [x y z R] in NMOPSO units
    cyl_m: np.ndarray         # [M, 3] cylinders [x y R] metres
    cyl_margin: float         # free-space margin outside the cylinders (m), as free_space.m
    z_patrol: float

    def to_units(self, p):
        """world metres [..., 3] -> NMOPSO units (CreateModel.m toU)."""
        p = np.asarray(p, dtype=np.float64)
        return np.stack([(p[..., 0] - self.hall[0]) * self.scale + 1,
                         (p[..., 1] - self.hall[2]) * self.scale + 1,
                         p[..., 2] * self.scale], -1)

    def to_metres(self, u):
        u = np.asarray(u, dtype=np.float64)
        return np.stack([(u[..., 0] - 1) / self.scale + self.hall[0],
                         (u[..., 1] - 1) / self.scale + self.hall[2],
                         u[..., 2] / self.scale], -1)

    def free(self, x, y):
        """free_space.m (auto layouts): at least cyl_margin outside every cylinder, inside the wall margin."""
        x = np.asarray(x); y = np.asarray(y)
        ok = np.ones(np.broadcast(x, y).shape, bool)
        for cx, cy, r in self.cyl_m:
            ok &= np.hypot(x - cx, y - cy) - r >= self.cyl_margin
        b = self.bounds
        xu = (x - self.hall[0]) * self.scale + 1
        yu = (y - self.hall[2]) * self.scale + 1
        return ok & (xu >= b[0]) & (xu <= b[1]) & (yu >= b[2]) & (yu <= b[3])


def load_world(name="factory2_model.json", device="cpu", dtype=torch.float64) -> World:
    J = json.loads((DATA / name).read_text())
    return World(hall=np.array(J["hall"], float), scale=float(J["scale"]),
                 bounds=np.array(J["bounds_units"], float),
                 threats=torch.tensor(J["threats_units"], dtype=dtype, device=device),
                 cyl_m=np.array(J["cylinders_m"], float), cyl_margin=float(J["cylMargin"]),
                 z_patrol=float(J["zPatrol"]))
