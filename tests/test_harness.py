"""End-to-end harness smoke: mock env + PPO, seeding, matrix expansion."""

import numpy as np

from quadruped_rl.harness.config import compose_config
from quadruped_rl.harness.matrix_runner import expand_matrix
from quadruped_rl.harness.seeding import set_global_seed
from quadruped_rl.registry import list_algorithms


def test_all_algorithms_registered():
    assert set(list_algorithms()) >= {"ppo", "trpo", "a3c", "sac", "td3", "ddpg"}


def test_seeding_reproducible():
    set_global_seed(7)
    a = np.random.rand(5)
    set_global_seed(7)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_matrix_expansion_phase2_size():
    cfg = compose_config(experiment="phase2_matrix")
    cells = expand_matrix(cfg)
    # 3 robots x 6 algorithms x 12 terrains x 10 seeds
    assert len(cells) == 3 * 6 * 12 * 10


def test_smoke_training_run(tmp_path):
    """Full Trainer loop on the mock env with PPO (~seconds, CPU)."""
    import torch  # noqa: F401  (skip via ImportError if torch missing)

    from quadruped_rl.harness.trainer import Trainer

    cfg = compose_config(
        algorithm="ppo",
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={"run": {"seed": 0, "smoke_test": True}, "logging": {"wandb": False}},
    )
    trainer = Trainer(cfg, run_dir=tmp_path / "run")
    result = trainer.train()
    assert "success_rate" in result["final"]
    assert (tmp_path / "run" / "config.yaml").exists()
    assert (tmp_path / "run" / "metrics.jsonl").exists()
    assert list((tmp_path / "run" / "checkpoints").glob("*.pt"))
