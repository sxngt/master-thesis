"""Config composition: merge order, inherit resolution, persistence."""

from quadruped_rl.harness.config import compose_config, deep_merge, save_resolved_config


def test_deep_merge_override_wins():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    out = deep_merge(base, {"a": {"y": 9}, "c": 4})
    assert out == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base["a"]["y"] == 2  # no mutation


def test_compose_full_stack():
    cfg = compose_config(
        algorithm="ppo",
        robot="a1",
        terrain="stairs",
        reward="traditional",
        experiment="phase1_baseline",
    )
    assert cfg["algorithm"]["name"] == "ppo"
    assert cfg["robot"]["name"] == "a1"
    assert cfg["terrain"]["name"] == "stairs"
    assert cfg["experiment"]["name"] == "phase1_baseline"
    assert "components" in cfg["reward"]


def test_hybrid_reward_inherit_resolution():
    cfg = compose_config(reward="hybrid_llm")
    # {inherit: reward/traditional.yaml} must be expanded
    assert (
        "components" in cfg["reward"]["traditional"]["reward"]
        or "components" in cfg["reward"]["traditional"]
    )


def test_cli_override_precedence():
    cfg = compose_config(algorithm="ppo", overrides={"run": {"seed": 42}})
    assert cfg["run"]["seed"] == 42


def test_save_resolved_config(tmp_path):
    cfg = compose_config(algorithm="ppo")
    out = save_resolved_config(cfg, tmp_path)
    assert out.exists() and out.name == "config.yaml"


def test_all_experiment_configs_compose():
    for exp in ["phase1_baseline", "phase2_matrix", "phase3_llm", "phase4_hardware"]:
        cfg = compose_config(experiment=exp)
        assert cfg["experiment"]["name"] == exp
