"""Reward coach: guardrail, schedulers, rollback loop, trainer wiring."""

import copy
import json
import math
import threading

import numpy as np
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
    "objective_table": "-",
    "stats_table": "-",
    "dynamics_table": "-",
    "window": 1,
    "history_text": "-",
    "task_text": "-",
    "objective_trace": "-",
    "best_text": "-",
    "noise_text": "-",
}


def test_llm_scheduler_parses_valid_and_discards_garbage():
    space = ParamSpace(SPACE)
    ok = LLMScheduler({"model": "x"}, client=FakeClient(f"sure: {VALID}"))
    p = ok.propose(REPORT, CURRENT, space)
    assert p.actions == {"energy.weight": -1e-5}
    assert p.prompt and "energy.weight" in p.prompt["system"]

    bad = LLMScheduler({"model": "x", "retries": 1}, client=FakeClient("{not json"))
    p = bad.propose(REPORT, CURRENT, space)
    assert p.actions == {} and bad.discarded == 1
    assert bad.client.calls == 2  # one retry (reasoning can exhaust the token budget)


def test_hillclimb_flips_when_kept_move_did_not_improve():
    space = ParamSpace(SPACE)
    hc = HillClimbScheduler()
    first = hc.propose({"history_objs": []}, CURRENT, space).actions
    (key, _), *_ = first.items()

    class H:  # kept within tolerance, but J did not rise -> treated as failure
        status = "kept"
        objective_before = 0.0
        objective_after = 0.0
        applied = dict(first)
        params_before = dict(CURRENT)

    moved = {**CURRENT, **first}
    second = hc.propose({"history_objs": [H()]}, moved, space).actions
    assert hc._dir == -1.0
    assert second[key] == CURRENT[key]  # the non-improving move is undone
    new_key = [k for k in second if k != key]
    assert len(new_key) == 1 and new_key[0] != key  # ... and the search moves on


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
        self.hp = {"learning_rate": 3e-4}
        self.loaded_paths: list[str] = []

    def save(self, path):
        self.saved += 1

    def load(self, path):
        self.loaded += 1
        self.loaded_paths.append(str(path))

    def hyperparams(self):
        return dict(self.hp)

    def set_hyperparams(self, u):
        self.hp.update(u)


def _coach(tmp_path, client, env_params=None, **over):
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
    env.params.update(env_params or {})
    coach = RewardCoach(cfg, env, algo, tmp_path, client=client)
    return coach, env, algo


def test_coach_applies_then_rolls_back_on_regression(tmp_path):
    coach, env, algo = _coach(tmp_path, FakeClient(VALID))
    assert coach.on_eval(5, 100, {"success_rate": 0.5}, []) == {}  # before interval
    out = coach.on_eval(10, 100, {"success_rate": 0.5}, [{"loss": 1.0}])
    assert out["intervention"] == 1 and env.params["energy.weight"] == pytest.approx(-1e-5)
    assert algo.saved == 2  # best-so-far snapshot + rollback snapshot
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


def test_coach_tunes_algo_hyperparams_and_rolls_them_back(tmp_path):
    resp = json.dumps(
        {
            "diagnosis": "exploration collapsed",
            "actions": [{"param": "algo.learning_rate", "value": 1e-4, "rationale": "cool down"}],
            "confidence": 0.6,
        }
    )
    space = dict(SPACE, **{"algo.learning_rate": {"low": 3e-5, "high": 1e-3, "scale": "log"}})
    coach, env, algo = _coach(tmp_path, FakeClient(resp), params=space)
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    assert algo.hp["learning_rate"] == pytest.approx(1e-4)
    coach.on_eval(20, 100, {"success_rate": 0.1}, [])
    assert coach.history[0].status == "rolled_back"
    assert algo.hp["learning_rate"] == pytest.approx(3e-4)


def test_coach_restores_best_snapshot_on_request(tmp_path):
    restore = json.dumps({"diagnosis": "collapsed", "actions": [], "restore_best": True})
    coach, env, algo = _coach(tmp_path, FakeClient(restore))
    # evaluations between interventions also feed the best-snapshot tracker
    coach.on_eval(10, 100, {"success_rate": 0.3}, [])  # k=1: nothing better yet
    assert coach.history[0].restored_from_step is None
    (tmp_path / "coach_best.pt").touch()  # StubAlgo.save writes nothing
    coach.on_eval(15, 100, {"success_rate": 0.9}, [])  # off-cycle eval: new best
    # best = mean of best_confirm (2) consecutive evaluations, not the single spike
    assert coach.best_step == 15 and coach.best_obj == pytest.approx(0.6)
    out = coach.on_eval(20, 100, {"success_rate": 0.2}, [])  # k=2: collapsed -> restore
    assert out.get("restored") == 1.0
    assert coach.history[1].restored_from_step == 15
    assert algo.loaded_paths[-1].endswith("coach_best.pt")
    assert "restored best snapshot" in coach.history[1].summary()
    # the same snapshot cannot be restored again (max_restores_per_snapshot=1)
    out = coach.on_eval(30, 100, {"success_rate": 0.2}, [])
    assert "restored" not in out and coach.history[2].restored_from_step is None
    assert any("already restored" in n for n in coach.history[2].notes)


def test_coach_widens_tolerance_with_evaluation_noise(tmp_path):
    coach, env, algo = _coach(tmp_path, FakeClient(VALID), interval_steps=100)
    assert coach.effective_tolerance() == pytest.approx(0.05)  # no trace yet
    for i, j in enumerate([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]):  # quantised eval
        coach.on_eval(i + 1, 1000, {"success_rate": j}, [])
    sigma = coach.noise_sigma()
    assert sigma > 0.2
    assert coach.effective_tolerance() == pytest.approx(min(0.3, 2 * 2**0.5 * sigma))
    # a clean linear trend is not noise (second differences cancel it)
    quiet, _, _ = _coach(tmp_path / "q", FakeClient(VALID), interval_steps=100)
    for i in range(10):
        quiet.on_eval(i + 1, 1000, {"success_rate": 0.05 * i}, [])
    assert quiet.noise_sigma() == pytest.approx(0.0, abs=1e-9)
    assert quiet.effective_tolerance() == pytest.approx(0.05)


def test_coach_does_not_roll_back_within_noise(tmp_path):
    coach, env, algo = _coach(tmp_path, FakeClient(VALID), interval_steps=10)
    for i, j in enumerate([0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0]):  # steps 1..9
        coach.on_eval(i + 1, 1000, {"success_rate": j}, [])
    coach.on_eval(10, 1000, {"success_rate": 0.5}, [])  # k=1 applied
    assert coach.history[0].applied
    coach.on_eval(20, 1000, {"success_rate": 0.3}, [])  # -0.2 drop, but noise ~0.25
    assert coach.history[0].status == "kept"
    assert coach.history[0].tolerance_used > 0.2


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
    assert (table["n_restored"] == 0).all() and (table["n_algo_moves"] == 0).all()
    assert np.isfinite(table["objective_late"]).all()
    its = intervention_table(table)
    assert set(its["status"]) <= {"kept", "rolled_back", "pending"}
    assert "restored_from_step" in its.columns
    objective_curves(table, "flat-easy-traditional", tmp_path / "fig" / "obj")
    parameter_trajectories(table.iloc[0], {"drive.weight": (0.5, 2.0)}, tmp_path / "fig" / "p")
    assert (tmp_path / "fig" / "obj.png").exists()


def test_llm_scheduler_skips_report_on_api_failure():
    class Failing:
        last_usage: dict = {}
        calls = 0

        def complete(self, *a, **k):
            self.calls += 1
            raise RuntimeError("429 no credits")

    space = ParamSpace(SPACE)
    s = LLMScheduler({"model": "x", "api_retries": 2, "api_backoff_s": 0.0}, client=Failing())
    p = s.propose(REPORT, CURRENT, space)
    assert p.actions == {} and p.diagnosis.startswith("api error")
    assert s.api_errors == 1 and s.client.calls == 3  # 1 + 2 retries, then skipped


def test_objective_rows_state_command_ceiling(tmp_path):
    coach, _, _ = _coach(
        tmp_path,
        FakeClient(VALID),
        objective={"success_rate": 1.0, "mean_forward_velocity_ms": 0.5},
        kpi_command={"mean_forward_velocity_ms": "forward_velocity.target_ms"},
    )
    rows = coach.objective_rows(
        {"success_rate": 0.57, "mean_forward_velocity_ms": 0.56},
        {"forward_velocity.target_ms": 0.7},
    )
    assert rows["success_rate"] == "+1 * 0.570 = +0.570"
    vel = rows["mean_forward_velocity_ms"]
    assert vel.startswith("+0.5 * 0.560 = +0.280; commanded by forward_velocity.target_ms = 0.7")
    assert "(measured/commanded = 0.80)" in vel and "capped near +0.350" in vel
    slow = coach.objective_rows(
        {"success_rate": 0.0, "mean_forward_velocity_ms": 0.1}, {"forward_velocity.target_ms": 1.0}
    )["mean_forward_velocity_ms"]
    assert "not yet tracking" in slow and "would be +0.500" in slow
    assert rows["J"] == "0.8500"
    # no commanding parameter known -> plain contribution
    plain = coach.objective_rows({"success_rate": 0.57, "mean_forward_velocity_ms": 0.56}, {})
    assert "commanded" not in plain["mean_forward_velocity_ms"]


# ------------------------------------------------------------- coach v5
V5_RESP = json.dumps(
    {
        "diagnosis": "relax energy",
        "actions": [{"param": "energy.weight", "value": -1e-5, "rationale": "r"}],
        "predicted_delta_j": 0.1,
        "confidence": 0.7,
    }
)


def test_confirmatory_reevaluation_resolves_an_ambiguous_drop(tmp_path):
    """A drop inside (confirm_zone*tol, tol] is re-evaluated; the pooled mean decides."""
    fresh = iter([{"success_rate": 0.62, "n_episodes": 100}])
    calls = []

    def evaluate():
        m = next(fresh)
        calls.append(m)
        return m

    cfg = {"rollback_tolerance": 0.1, "confirm_evals": 1, "confirm_zone": 0.5, "noise_z": 0.0}
    coach, env, algo = _coach(tmp_path, FakeClient(V5_RESP), **cfg)
    coach.evaluate = evaluate
    coach.on_eval(10, 100, {"success_rate": 0.6, "n_episodes": 100}, [])  # k=1 applied
    # -0.07: beyond half the tolerance but not beyond it -> one more evaluation
    coach.on_eval(20, 100, {"success_rate": 0.53, "n_episodes": 100}, [])
    p = coach.history[0]
    assert len(calls) == 1 and p.confirm_evals == 1
    assert p.objective_after == pytest.approx(0.575)  # pooled mean of 0.53 and 0.62
    assert p.status == "kept"
    assert coach.eval_trace[-1] == (20, pytest.approx(0.575))
    assert p.effect is not None and p.predicted_delta_j == pytest.approx(0.1)
    assert p.objective_se == pytest.approx(math.sqrt(0.6 * 0.4 / 100))
    # a clear drop still rolls back without spending evaluations
    client = coach.scheduler.client
    client.response = V5_RESP.replace("-1e-05", "-5e-06")
    coach.on_eval(30, 100, {"success_rate": 0.58, "n_episodes": 100}, [])  # k=3 applied
    assert coach.history[2].applied
    coach.on_eval(40, 100, {"success_rate": 0.2, "n_episodes": 100}, [])
    assert coach.history[2].status == "rolled_back" and len(calls) == 1


def test_measurement_se_widens_tolerance_from_the_first_evaluation(tmp_path):
    coach, _, _ = _coach(tmp_path, FakeClient(V5_RESP), rollback_tolerance=0.01, noise_z=2.0)
    coach.on_eval(10, 100, {"success_rate": 0.5, "n_episodes": 64}, [])
    se = math.sqrt(0.25 / 64)
    assert coach.eval_se[-1] == pytest.approx(se)
    assert coach.effective_tolerance() == pytest.approx(2.0 * math.sqrt(2) * se)


def test_curriculum_release_steps_a_lowered_lever_back(tmp_path):
    space = dict(SPACE, **{"drive.target": {"low": 0.3, "high": 1.5}})
    lower = json.dumps(
        {"diagnosis": "curriculum", "actions": [{"param": "drive.target", "value": 0.7}]}
    )
    keep = json.dumps({"diagnosis": "fine", "actions": []})
    client = FakeClient(lower)
    coach, env, algo = _coach(
        tmp_path,
        client,
        params=space,
        curriculum_params=["drive.target"],
        release_from_progress=0.5,
        release_min_success=0.3,
        max_rel_change=0.3,
        env_params={"drive.target": 1.0},
    )
    coach.on_eval(10, 100, {"success_rate": 0.0}, [])  # k=1: LLM lowers to 0.7
    assert env.params["drive.target"] == pytest.approx(0.7)
    client.response = keep
    coach.on_eval(20, 100, {"success_rate": 0.0}, [])  # 20%: too early, no success
    assert env.params["drive.target"] == pytest.approx(0.7)
    coach.on_eval(60, 100, {"success_rate": 0.6}, [])  # 60% but previous success 0 -> wait
    assert env.params["drive.target"] == pytest.approx(0.7)
    out = coach.on_eval(70, 100, {"success_rate": 0.6}, [])  # two successful reports
    assert out.get("curriculum_release") == 1.0
    assert env.params["drive.target"] == pytest.approx(0.91)  # one +30 % guardrail step
    rec = coach.history[-1]
    assert rec.invariant == {"drive.target": pytest.approx(0.91)} and rec.applied == rec.invariant
    assert any("curriculum release" in n for n in rec.notes)
    assert "curriculum release" in rec.summary()
    coach.on_eval(80, 100, {"success_rate": 0.6}, [])
    assert env.params["drive.target"] == pytest.approx(1.0)  # capped at the baseline
    coach.on_eval(90, 100, {"success_rate": 0.6}, [])
    assert not coach.history[-1].invariant  # at baseline: nothing to release
    # a release that hurts is rolled back like any other move
    assert coach.history[-3].status == "kept"


def test_consolidate_phase_freezes_reward_shaping(tmp_path):
    space = dict(
        SPACE,
        **{
            "drive.target": {"low": 0.3, "high": 1.5},
            "algo.learning_rate": {"low": 3e-5, "high": 1e-3, "scale": "log"},
        },
    )
    resp = json.dumps(
        {
            "diagnosis": "late",
            "actions": [
                {"param": "energy.weight", "value": -1e-5},
                {"param": "drive.target", "value": 1.2},
                {"param": "algo.learning_rate", "value": 1e-4},
            ],
        }
    )
    coach, env, algo = _coach(
        tmp_path,
        FakeClient(resp),
        params=space,
        curriculum_params=["drive.target"],
        phases={"exploit": 0.5, "consolidate": 0.85},
        env_params={"drive.target": 1.0},
    )
    assert coach.phase(0.2) == "explore" and coach.phase(0.6) == "exploit"
    coach.on_eval(90, 100, {"success_rate": 0.5}, [])
    rec = coach.history[0]
    assert rec.phase == "consolidate"
    assert set(rec.applied) == {"drive.target", "algo.learning_rate"}
    assert env.params["energy.weight"] == pytest.approx(CURRENT["energy.weight"])
    assert any("frozen in the consolidate phase" in n for n in rec.notes)


def test_report_carries_evidence_ledger_and_calibration(tmp_path):
    coach, env, algo = _coach(
        tmp_path,
        FakeClient(V5_RESP),
        objective={"success_rate": 1.0, "mean_forward_velocity_ms": 0.5},
        kpi_command={"mean_forward_velocity_ms": "forward_velocity.target_ms"},
        params=dict(SPACE, **{"forward_velocity.target_ms": {"low": 0.3, "high": 1.5}}),
        interval_steps=10,
        env_params={"forward_velocity.target_ms": 1.0},
    )
    kpis = [
        {"success_rate": s, "mean_forward_velocity_ms": v, "n_episodes": 256}
        for s, v in [(0.0, 0.1), (0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]
    ]
    for i, k in enumerate(kpis):
        coach.on_eval(10 * (i + 1), 100, k, [])
    first = coach.history[0]  # the only applied move (later reports repeat it: noop)
    assert first.status == "kept" and first.effect is not None and first.expected_dj == 0.0
    assert all(h.status == "noop" for h in coach.history[1:-1])
    user = json.loads((tmp_path / "coach_log.jsonl").read_text().splitlines()[-1])["prompt"]["user"]
    assert "### Evidence (computed)" in user
    assert "Phase: exploit" in user
    assert "Learning curve (this policy lineage" in user
    assert "ceiling" in user and "each +0.1 of forward_velocity.target_ms is worth up to" in user
    assert "observed (commanded, measured, success)" in user
    assert "energy.weight up: effect on J" in user
    assert "Your dJ predictions so far" in user
    assert "n_episodes" not in user.split("### Objective decomposition")[0]  # not a KPI row
    assert "predicted_delta_j" in json.loads(
        (tmp_path / "coach_log.jsonl").read_text().splitlines()[0]
    )


# ------------------------------------------------------------ async coach
class BlockingClient(FakeClient):
    """Completes only once released (a slow local LLM)."""

    def __init__(self, response: str):
        super().__init__(response)
        self.release = threading.Event()

    def complete(self, system, user, max_tokens=1024):
        self.release.wait(timeout=10)
        return super().complete(system, user, max_tokens)


def test_async_coach_keeps_training_and_applies_when_the_call_lands(tmp_path):
    client = BlockingClient(VALID)
    coach, env, algo = _coach(tmp_path, client, async_llm=True)
    out = coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    assert out.get("llm_inflight") == 1.0 and "intervention" not in out
    assert env.params["energy.weight"] == pytest.approx(-2.5e-5)  # nothing applied yet
    assert coach.on_eval(15, 100, {"success_rate": 0.55}, []) == {}  # still thinking
    client.release.set()
    coach._inflight._thread.join(timeout=10)
    out = coach.on_eval(20, 100, {"success_rate": 0.6}, [])
    assert out["intervention"] == 1 and env.params["energy.weight"] == pytest.approx(-1e-5)
    rec = coach.history[0]
    assert rec.report_step == 10 and rec.step == 20 and rec.objective_before == 0.6
    assert any(n.startswith("async: proposed from the report at step 10") for n in rec.notes)
    assert coach.on_eval(25, 100, {"success_rate": 0.65}, []) == {}  # a full interval to settle
    out = coach.on_eval(30, 100, {"success_rate": 0.7}, [])
    assert rec.status == "kept" and out.get("llm_inflight") == 1.0  # next report submitted
    assert client.calls == 2
    first = json.loads((tmp_path / "coach_log.jsonl").read_text().splitlines()[0])
    assert first["report_step"] == 10 and first["llm_latency_s"] >= 0 and first["prompt"]["user"]


def test_async_coach_keeps_the_report_cadence_on_the_interval_grid(tmp_path):
    # evaluations fall a little past each multiple of the interval; a proposal
    # applied at such an off-grid step must not push the next report past the
    # next grid slot (that halved the intervention rate on the server)
    client = BlockingClient(VALID)
    coach, _, _ = _coach(tmp_path, client, async_llm=True)
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    client.release.set()
    coach._inflight._thread.join(timeout=10)
    out = coach.on_eval(12, 100, {"success_rate": 0.6}, [])  # applied off-grid
    assert out["intervention"] == 1 and coach._next == 20
    out = coach.on_eval(21, 100, {"success_rate": 0.7}, [])  # the next grid slot reports
    assert coach.history[0].status == "kept" and out.get("llm_inflight") == 1.0
    assert client.calls == 2


def test_async_coach_drops_a_call_still_in_flight_at_the_end(tmp_path):
    client = BlockingClient(VALID)
    coach, env, _ = _coach(tmp_path, client, async_llm=True)
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    assert coach._inflight is not None
    coach.finish()
    client.release.set()
    assert coach._inflight is None and coach.history == []
    assert env.params["energy.weight"] == pytest.approx(-2.5e-5)


# ------------------------------------------------------------ settled ledger
TARGET_UP = json.dumps(
    {
        "diagnosis": "release the command",
        "actions": [{"param": "forward_velocity.target_ms", "value": 1.1, "rationale": "r"}],
        "confidence": 0.6,
    }
)


def _curriculum_coach(tmp_path, client, **over):
    return _coach(
        tmp_path,
        client,
        objective={"success_rate": 1.0, "mean_forward_velocity_ms": 0.5},
        kpi_command={"mean_forward_velocity_ms": "forward_velocity.target_ms"},
        curriculum_params=["forward_velocity.target_ms"],
        params=dict(SPACE, **{"forward_velocity.target_ms": {"low": 0.3, "high": 1.5}}),
        interval_steps=10,
        env_params={"forward_velocity.target_ms": 1.0},
        **over,
    )


def _run(coach, curve):
    for i, (s, v) in enumerate(curve):
        coach.on_eval(10 * (i + 1), 100, {"success_rate": s, "mean_forward_velocity_ms": v}, [])


def _last_user(tmp_path) -> str:
    return json.loads((tmp_path / "coach_log.jsonl").read_text().splitlines()[-1])["prompt"]["user"]


def test_settled_ledger_credits_a_raised_command_that_dips_then_pays(tmp_path):
    # flat before the move; the raised target costs success at the next report
    # (immediate effect negative, inside the rollback tolerance) and pays after
    curve = [(0.5, 0.6), (0.46, 0.62), (0.6, 0.7), (0.65, 0.75), (0.66, 0.76)]
    coach, env, _ = _curriculum_coach(tmp_path, FakeClient(TARGET_UP), settled_reports=2)
    _run(coach, curve)
    first = coach.history[0]
    assert first.status == "kept" and first.effect < 0 and first.trend_slope == 0.0
    assert env.params["forward_velocity.target_ms"] == pytest.approx(1.1)
    user = _last_user(tmp_path)
    assert "forward_velocity.target_ms up: effect on J -0.030" in user
    assert "Settled effect of curriculum moves (J 2 reports after the move" in user
    # window ends at step 30: J 0.95 - 0.80 = +0.15 net of a flat trend
    assert "forward_velocity.target_ms up: settled effect on J +0.150" in user
    # off by default: the same run without settled_reports has no such row
    (tmp_path / "off").mkdir()
    coach0, _, _ = _curriculum_coach(tmp_path / "off", FakeClient(TARGET_UP))
    _run(coach0, curve)
    assert "settled effect" not in _last_user(tmp_path / "off")


def test_settled_ledger_skips_windows_that_are_not_clean(tmp_path):
    # too recent: the window (2 reports) has not closed one report after the move
    coach, _, _ = _curriculum_coach(tmp_path, FakeClient(TARGET_UP), settled_reports=2)
    _run(coach, [(0.5, 0.6), (0.46, 0.62)])
    assert "settled effect" not in _last_user(tmp_path)
    # rolled back inside the window: the move's lineage is gone, no settled row
    (tmp_path / "rb").mkdir()
    coach, _, _ = _curriculum_coach(tmp_path / "rb", FakeClient(TARGET_UP), settled_reports=2)
    _run(coach, [(0.5, 0.6), (0.1, 0.2), (0.6, 0.7), (0.65, 0.75)])
    assert coach.history[0].status == "rolled_back"
    assert "settled effect" not in _last_user(tmp_path / "rb")


# ---------------------------------------------------------------- ledger veto
def test_ledger_veto_drops_a_reward_move_the_run_already_scored_negative(tmp_path):
    coach, env, _ = _coach(tmp_path, FakeClient(VALID), ledger_veto_obs=3)
    for _ in range(3):
        coach.ledger.add("energy.weight", "up", -0.1, 0.01)
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    rec = coach.history[0]
    assert rec.applied == {} and env.params["energy.weight"] == pytest.approx(-2.5e-5)
    assert any("energy.weight up: vetoed by the ledger" in n for n in rec.notes)
    # the next report shows the veto in the history line so the model learns
    coach.on_eval(20, 100, {"success_rate": 0.5}, [])
    assert "[guardrail: energy.weight up: vetoed by the ledger" in _last_user(tmp_path)


def test_ledger_veto_needs_enough_evidence_and_spares_curriculum_levers(tmp_path):
    # two observations only: applied
    coach, env, _ = _coach(tmp_path, FakeClient(VALID), ledger_veto_obs=3)
    for _ in range(2):
        coach.ledger.add("energy.weight", "up", -0.1, 0.01)
    coach.on_eval(10, 100, {"success_rate": 0.5}, [])
    assert env.params["energy.weight"] == pytest.approx(-1e-5)
    # a curriculum lever is never vetoed by the (biased) immediate ledger
    (tmp_path / "cur").mkdir()
    coach, env, _ = _curriculum_coach(tmp_path / "cur", FakeClient(TARGET_UP), ledger_veto_obs=3)
    for _ in range(5):
        coach.ledger.add("forward_velocity.target_ms", "up", -0.1, 0.01)
    coach.on_eval(10, 100, {"success_rate": 0.5, "mean_forward_velocity_ms": 0.6}, [])
    assert env.params["forward_velocity.target_ms"] == pytest.approx(1.1)


# ------------------------------------------------------------ curriculum lock
TARGET_DOWN = json.dumps(
    {
        "diagnosis": "consolidate",
        "actions": [{"param": "forward_velocity.target_ms", "value": 0.8, "rationale": "r"}],
        "confidence": 0.6,
    }
)


class SeqClient(FakeClient):
    """One canned response per call, the last one repeated."""

    def __init__(self, *responses: str):
        super().__init__(responses[-1])
        self.responses = list(responses)

    def complete(self, system, user, max_tokens=1024):
        self.calls += 1
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def test_curriculum_lock_drops_a_lowering_in_the_release_phase(tmp_path):
    lower_again = TARGET_DOWN.replace('"value": 0.8', '"value": 0.6')
    coach, env, _ = _curriculum_coach(
        tmp_path,
        SeqClient(TARGET_DOWN, lower_again),
        curriculum_lock=True,
        release_from_progress=0.5,
        release_min_success=0.3,
    )
    walking = {"success_rate": 0.55, "mean_forward_velocity_ms": 0.7}
    # before the release phase the lowering goes through
    coach.on_eval(10, 100, walking, [])
    assert env.params["forward_velocity.target_ms"] == pytest.approx(0.8)
    # in the release phase (two successful reports) it is dropped and the
    # release invariant steps the lever back up instead
    coach.on_eval(60, 100, walking, [])
    rec = coach.history[-1]
    assert rec.proposed["forward_velocity.target_ms"] == 0.6
    assert any("locked in the release phase" in n for n in rec.notes)
    assert rec.invariant and env.params["forward_velocity.target_ms"] > 0.8
    # the next report shows the dropped move
    coach.on_eval(70, 100, walking, [])
    assert "[guardrail: forward_velocity.target_ms: lowering" in _last_user(tmp_path)


def test_curriculum_lock_is_a_tunable_override():
    from quadruped_rl.llm_feedback import evolve as ev

    clean, errors = ev.validate_overrides({"curriculum_lock": True, "ledger_veto_obs": 3}, [])
    assert not errors and clean == {"curriculum_lock": True, "ledger_veto_obs": 3}
    _, errors = ev.validate_overrides({"ledger_veto_obs": 9}, [])
    assert errors
