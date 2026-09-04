"""Reward coach: guardrail, schedulers, rollback loop, trainer wiring."""

import copy
import json

import pytest

from quadruped_rl.harness.config import compose_config
from quadruped_rl.llm_feedback.coach import (
    HillClimbScheduler,
    LLMScheduler,
    ParamSpace,
    RandomScheduler,
    RewardCoach,
)

torch = pytest.importorskip("torch")

SPACE = {
    "energy.weight": {"low": -5e-3, "high": -1e-6, "scale": "log"},
    "feet_air_time.weight": {"low": 0.0, "high": 3.0},
    "yaw_rate.weight": {"low": -2.0, "high": -0.05},
}
CURRENT = {"energy.weight": -2.5e-5, "feet_air_time.weight": 1.0, "yaw_rate.weight": -0.5}


class FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.last_usage = {"input_tokens": 1, "output_tokens": 1}
        self.calls = 0

    def complete(self, system, user, max_tokens=1024):
        self.calls += 1
        return self.response


# ------------------------------------------------------------- guardrail
def test_bounds_must_not_straddle_zero():
    with pytest.raises(ValueError):
        ParamSpace({"x.weight": {"low": -1.0, "high": 1.0}})


def test_clip_enforces_sign_lock_and_step_limits():
    space = ParamSpace(SPACE, max_rel_change=0.3, max_log_factor=3.0, max_params=3)
    accepted, notes = space.clip(
        {
            "energy.weight": +0.01,  # sign flip -> clipped back to a penalty
            "feet_air_time.weight": 5.0,  # too big a jump (+400%)
            "yaw_rate.weight": -0.6,  # within +-30%
            "bogus.weight": 1.0,
        },
        CURRENT,
    )
    assert "bogus.weight" not in accepted
    assert accepted["energy.weight"] < 0
    assert accepted["feet_air_time.weight"] == pytest.approx(1.3)
    assert accepted["yaw_rate.weight"] == pytest.approx(-0.6)
    assert any("bogus" in n for n in notes)


def test_log_scale_step_is_multiplicative():
    space = ParamSpace(SPACE, max_log_factor=3.0)
    accepted, _ = space.clip({"energy.weight": -2.5e-3}, CURRENT)  # 100x jump
    assert accepted["energy.weight"] == pytest.approx(-7.5e-5)  # capped at 3x


def test_zero_weight_can_be_enabled():
    space = ParamSpace(SPACE, max_rel_change=0.3)
    cur = dict(CURRENT, **{"feet_air_time.weight": 0.0})
    accepted, _ = space.clip({"feet_air_time.weight": 2.0}, cur)
    assert accepted["feet_air_time.weight"] == pytest.approx(0.9)  # 30% of range


def test_max_params_per_intervention():
    space = ParamSpace(SPACE, max_params=1)
    accepted, _ = space.clip({"yaw_rate.weight": -0.6, "feet_air_time.weight": 1.2}, CURRENT)
    assert list(accepted) == ["yaw_rate.weight"]


# ------------------------------------------------------------ schedulers
def test_random_scheduler_is_seeded_and_in_range():
    space = ParamSpace(SPACE)
    a = RandomScheduler(seed=3).propose({}, CURRENT, space).actions
    b = RandomScheduler(seed=3).propose({}, CURRENT, space).actions
    assert a == b
    accepted, _ = space.clip(a, CURRENT)
    for k, v in accepted.items():
        assert space.specs[k].low <= v <= space.specs[k].high


def test_hillclimb_flips_direction_after_rollback():
    space = ParamSpace(SPACE)
    hc = HillClimbScheduler()
    first = hc.propose({"history_objs": []}, CURRENT, space).actions
    (key, val), *_ = first.items()
    assert abs(val) > abs(CURRENT[key])  # first move: increase magnitude

    class H:  # minimal stand-in for Intervention
        status = "rolled_back"

    second = hc.propose({"history_objs": [H()]}, CURRENT, space).actions
    assert list(second)[0] != key  # moved to next parameter
    assert hc._dir == -1.0


VALID = json.dumps(
    {
        "diagnosis": "energy penalty dominates",
        "actions": [{"param": "energy.weight", "value": -1e-5, "rationale": "relax"}],
        "expected_effect": "walking emerges",
        "confidence": 0.8,
    }
)
REPORT = {
    "component_table": "-",
    "objective_text": "J",
    "rollback_tolerance": 0.05,
    "step": 1,
    "progress": 0.1,
    "k": 1,
    "kpi_table": "-",
    "stats_table": "-",
    "dynamics_table": "-",
    "window": 1,
    "history_text": "-",
}


def test_llm_scheduler_parses_valid_and_discards_garbage():
    space = ParamSpace(SPACE)
    ok = LLMScheduler({"model": "x"}, client=FakeClient(f"sure: {VALID}"))
    p = ok.propose(REPORT, CURRENT, space)
    assert p.actions == {"energy.weight": -1e-5}
    assert p.prompt and "energy.weight" in p.prompt["system"]

    bad = LLMScheduler({"model": "x"}, client=FakeClient("{not json"))
    p = bad.propose(REPORT, CURRENT, space)
    assert p.actions == {} and bad.discarded == 1


# --------------------------------------------------------- coach loop
class StubEnv:
    def __init__(self):
        self.params = dict(CURRENT)

    def reward_params(self):
        return dict(self.params)

    def set_reward_params(self, u):
        self.params.update(u)

    def training_stats(self):
        return {"gait/duty_factor": 0.6}


class StubAlgo:
    def __init__(self):
        self.saved = self.loaded = 0

    def save(self, path):
        self.saved += 1

    def load(self, path):
        self.loaded += 1


def _coach(tmp_path, client, **over):
    cfg = {
        "strategy": "llm",
        "interval_steps": 10,
        "rollback_tolerance": 0.05,
        "objective": {"success_rate": 1.0},
        "params": SPACE,
        "llm": {"model": "x"},
        **over,
    }
    env, algo = StubEnv(), StubAlgo()
    coach = RewardCoach(cfg, env, algo, tmp_path, client=client)
    return coach, env, algo


def test_coach_applies_then_rolls_back_on_regression(tmp_path):
    coach, env, algo = _coach(tmp_path, FakeClient(VALID))
    assert coach.on_eval(5, 100, {"success_rate": 0.5}, []) == {}  # before interval
    out = coach.on_eval(10, 100, {"success_rate": 0.5}, [{"loss": 1.0}])
    assert out["intervention"] == 1 and env.params["energy.weight"] == pytest.approx(-1e-5)
    assert algo.saved == 1
    # objective collapses -> rollback restores params and policy snapshot
    out = coach.on_eval(20, 100, {"success_rate": 0.2}, [])
    assert out.get("rolled_back") == 1.0
    assert env.params["energy.weight"] == pytest.approx(-2.5e-5)
    assert algo.loaded == 1
    assert coach.history[0].status == "rolled_back"
    assert out.get("cooldown")  # slot skipped after rollback
    lines = (tmp_path / "coach_log.jsonl").read_text().splitlines()
    assert len(lines) >= 2 and json.loads(lines[0])["prompt"]["user"]


def test_coach_keeps_improving_intervention(tmp_path):
    coach, env, algo = _coach(tmp_path, FakeClient(VALID))
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    coach.on_eval(20, 100, {"success_rate": 0.7}, [])
    assert coach.history[0].status == "kept"
    assert env.params["energy.weight"] == pytest.approx(-1e-5)


def test_coach_rejects_params_missing_from_env(tmp_path):
    cfg = {"strategy": "random", "params": {"nope.weight": {"low": 0.0, "high": 1.0}}}
    with pytest.raises(KeyError):
        RewardCoach(cfg, StubEnv(), StubAlgo(), tmp_path)


def test_trainer_runs_with_random_coach_on_mock_vec(tmp_path):
    from quadruped_rl.harness.trainer import Trainer

    cfg = compose_config(sim="mock", algorithm="ppo", robot="a1", terrain="flat", coach="random")
    cfg["sim"].update(backend="mock_vec", num_envs=4)
    cfg["run"].update(
        total_timesteps=2_000,
        eval_interval_steps=400,
        checkpoint_interval_steps=2_000,
        eval_episodes=2,
        device="cpu",
        seed=0,
    )
    cfg["logging"]["wandb"] = False
    cfg["algorithm"]["rollout_steps"] = 8
    cfg["coach"].update(interval_steps=400, warmup_steps=400)
    cfg["coach"]["params"] = {
        "drive.weight": {"low": 0.5, "high": 2.0},
        "action_cost.weight": {"low": 0.001, "high": 0.1, "scale": "log"},
    }
    result = Trainer(cfg, run_dir=tmp_path / "run").train()
    assert "coach-random" in result["run_id"]
    records = [
        json.loads(line) for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("coach/intervention" in r for r in records)
    assert (tmp_path / "run" / "coach_log.jsonl").exists()


def test_coach_analysis_on_trainer_output(tmp_path):
    """load_run / intervention_table / figures consume what the Trainer writes."""
    from quadruped_rl.analysis.coach import (
        intervention_table,
        load_coach_table,
        objective_curves,
        parameter_trajectories,
    )
    from quadruped_rl.harness.trainer import Trainer

    base = compose_config(sim="mock", algorithm="ppo", robot="a1", terrain="flat", coach="random")
    base["sim"].update(backend="mock_vec", num_envs=4)
    base["run"].update(
        total_timesteps=2_000,
        eval_interval_steps=400,
        checkpoint_interval_steps=2_000,
        eval_episodes=2,
        device="cpu",
    )
    base["logging"]["wandb"] = False
    base["algorithm"]["rollout_steps"] = 8
    base["coach"].update(interval_steps=400, warmup_steps=400)
    base["coach"]["params"] = {"drive.weight": {"low": 0.5, "high": 2.0}}
    results = tmp_path / "results"
    for seed in (0, 1):
        cfg = copy.deepcopy(base)
        cfg["run"]["seed"] = seed
        Trainer(cfg, run_dir=results / f"coached_s{seed}").train()

    table = load_coach_table(results)
    assert len(table) == 2 and set(table["condition"]) == {"random"}
    assert table["setting"].iloc[0] == "flat-easy-traditional"
    assert (table["n_interventions"] > 0).all()
    settled = table["n_kept"] + table["n_rolled_back"] + table["n_pending"]
    assert settled.eq(table["n_interventions"]).all()
    its = intervention_table(table)
    assert set(its["status"]) <= {"kept", "rolled_back", "pending"}
    objective_curves(table, "flat-easy-traditional", tmp_path / "fig" / "obj")
    parameter_trajectories(table.iloc[0], {"drive.weight": (0.5, 2.0)}, tmp_path / "fig" / "p")
    assert (tmp_path / "fig" / "obj.png").exists()
