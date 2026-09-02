"""Terrain generation from configs."""

import numpy as np
import pytest

from quadruped_rl.envs.curriculum import Curriculum, gae_lambda_for_terrain
from quadruped_rl.envs.terrains import make_terrain
from quadruped_rl.harness.config import compose_config

ALL_TERRAINS = [
    "rough",
    "stairs",
    "slope",
    "gap",
    "gravel",
    "grass",
    "sand",
    "random_boxes",
    "moving_platform",
    "seesaw",
    "slippery",
    "mud",
    "rough_slope",
]


@pytest.mark.parametrize("name", ALL_TERRAINS)
@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_all_scenarios_generate(name, level):
    cfg = compose_config(terrain=name)
    spec = make_terrain(cfg["terrain"], level, np.random.default_rng(0))
    assert spec.heightfield is not None
    assert np.all(np.isfinite(spec.heightfield))
    assert 0.0 < spec.friction <= 1.0


def test_stairs_height_matches_config():
    cfg = compose_config(terrain="stairs")
    spec = make_terrain(cfg["terrain"], "hard", np.random.default_rng(0))
    steps = np.unique(np.round(spec.heightfield[:, 0], 6))
    diffs = np.diff(steps)
    assert np.allclose(diffs, 0.20)  # hard: 20 cm steps


def test_curriculum_promotion():
    cfg = compose_config()
    cfg["curriculum"]["window_episodes"] = 10
    cur = Curriculum(cfg)
    assert cur.level == "easy"
    promoted = False
    for _ in range(10):
        promoted = cur.report_episode(True) or promoted
    assert promoted and cur.level == "medium"


def test_gae_lambda_terrain_adjustment():
    assert gae_lambda_for_terrain(0.95, "extreme") < 0.95
    assert gae_lambda_for_terrain(0.95, "structured") == 0.95
