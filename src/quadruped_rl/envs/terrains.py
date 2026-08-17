"""Terrain generation from configs/terrain/*.yaml specs.

Produces heightfields / primitive layouts consumed by the simulator backends.
Each generator takes (terrain_cfg, level, rng) and returns a TerrainSpec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TerrainSpec:
    name: str
    level: str
    heightfield: np.ndarray | None = None  # [H, W] meters
    friction: float = 0.8
    extras: dict[str, Any] = field(default_factory=dict)  # dynamic bodies etc.


def make_terrain(
    terrain_cfg: dict[str, Any],
    level: str,
    rng: np.random.Generator,
    size_m: float = 8.0,
    resolution_m: float = 0.05,
) -> TerrainSpec:
    name = terrain_cfg["name"]
    params = terrain_cfg.get("levels", {}).get(level, {})
    friction = params.get("friction", terrain_cfg.get("friction", 0.8))
    n = int(size_m / resolution_m)
    hf = np.zeros((n, n))

    if name == "flat":
        pass
    elif name == "stairs":
        h, d = params["step_height_m"], params["step_depth_m"]
        steps_cells = max(int(d / resolution_m), 1)
        for i in range(n):
            hf[i, :] = (i // steps_cells) * h
    elif name in ("slope", "rough_slope"):
        incline = np.tan(np.radians(params["incline_deg"]))
        hf += np.linspace(0, size_m * incline, n)[:, None]
        if name == "rough_slope":
            hf += rng.uniform(-params["roughness_m"], params["roughness_m"], hf.shape)
    elif name == "gap":
        w = int(params["gap_width_m"] / resolution_m)
        hf[n // 2 : n // 2 + w, :] = -1.0
    elif name in ("gravel", "grass", "sand", "mud"):
        amp = (
            params.get("particle_size_m")
            or params.get("blade_height_m")
            or params.get("sink_depth_m", 0.02)
        )
        hf += rng.uniform(-amp, amp, hf.shape)
    elif name == "random_boxes":
        density, hmax = params["density_per_m2"], params["box_height_max_m"]
        for _ in range(int(density * size_m * size_m)):
            x, y = rng.integers(0, n - 10, 2)
            hf[x : x + 10, y : y + 10] = rng.uniform(0.02, hmax)
    elif name in ("moving_platform", "seesaw"):
        pass  # dynamic bodies configured by the sim backend via extras
    elif name == "slippery":
        pass  # flat geometry; friction carries the difficulty
    else:
        raise ValueError(f"Unknown terrain '{name}'")

    return TerrainSpec(
        name=name, level=level, heightfield=hf, friction=friction, extras=dict(params)
    )
