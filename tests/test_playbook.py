"""Cross-run playbook: case extraction, evidence lines, JSON round trip, coach hook."""

import json
from pathlib import Path

import yaml

from quadruped_rl.llm_feedback import evolve as ev
from quadruped_rl.llm_feedback.playbook import Playbook, case_of

BASE = {"forward_velocity.target_ms": 1.0, "energy.weight": -0.002, "action_rate.weight": -0.01}


def write_run(
    root: Path,
    run_id: str,
    seed: int,
    strategy: str,
    evals: list[tuple[int, float, float]],
    moves: list[tuple[int, dict[str, float], str, float | None]] = (),
    reward: str = "naive",
) -> Path:
    """A minimal run dir: config.yaml, metrics.jsonl (eval/final), coach_log.jsonl."""
    d = root / run_id
    d.mkdir(parents=True)
    cfg = {
        "run": {"seed": seed, "total_timesteps": evals[-1][0]},
        "algorithm": {"name": "ppo"},
        "sim": {"terrain_level": "medium"},
        "terrain": {"name": "rough"},
        "reward": {"name": reward},
        "coach": {"strategy": strategy},
    }
    (d / "config.yaml").write_text(yaml.safe_dump(cfg))
    recs = [
        {"step": s, "eval/success_rate": succ, "eval/mean_forward_velocity_ms": v}
        for s, succ, v in evals
    ]
    s, succ, v = evals[-1]
    recs.append({"step": s, "final/success_rate": succ, "final/mean_forward_velocity_ms": v})
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    params = dict(BASE)
    log = []

    def succ_at(step: int) -> float:
        return max([e[1] for e in evals if e[0] <= step] or [0.0])

    for k, (step, applied, status, effect) in enumerate(moves):
        rec = {
            "k": k,
            "step": step,
            "phase": "exploit",
            "status": "pending",
            "params_before": dict(params),
            "applied": applied,
            "objective_before": 0.1,
            "kpi_before": {"success_rate": succ_at(step)},
        }
        log.append(dict(rec))
        settled = dict(rec, status=status, effect=effect)
        log.append(settled)
        params.update(applied)
    (d / "coach_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    return d


def build_batch(root: Path) -> Playbook:
    floor = [(2_000_000, 0.0, 0.05), (4_000_000, 0.0, 0.05)]
    # two coached runs that left the floor after the same first move
    for seed in (0, 1):
        write_run(
            root,
            f"esc{seed}",
            seed,
            "llm",
            floor + [(6_000_000, 0.5, 0.8), (8_000_000, 0.9, 1.2)],
            [
                (
                    2_000_000,
                    {"forward_velocity.target_ms": 0.7, "energy.weight": -0.0007},
                    "kept",
                    None,
                ),
                (6_000_000, {"forward_velocity.target_ms": 1.4}, "kept", -0.05),
            ],
        )
    # one coached run stuck on the floor with a different move
    write_run(
        root,
        "stuck",
        2,
        "llm",
        floor + [(6_000_000, 0.0, 0.05), (8_000_000, 0.0, 0.05)],
        [(2_000_000, {"action_rate.weight": -0.03}, "kept", None)],
    )
    # a random coach (not experience) and a control
    write_run(
        root,
        "rnd",
        3,
        "random",
        floor + [(8_000_000, 0.3, 0.5)],
        [(2_000_000, {"energy.weight": -0.004}, "kept", None)],
    )
    write_run(root, "ctl", 4, "none", floor + [(8_000_000, 0.0, 0.05)])
    return Playbook.build(Playbook.run_dirs_under(root))


def test_case_of_extracts_breakout_moves_and_final_params(tmp_path):
    build_batch(tmp_path)
    c = case_of(tmp_path / "esc0")
    assert c.setting == "rough-medium-naive" and c.condition == "llm"
    assert c.breakout_step == 6_000_000
    assert [m.step for m in c.moves] == [2_000_000, 6_000_000]
    assert c.moves[0].directions() == {
        ("forward_velocity.target_ms", "down"),
        ("energy.weight", "up"),
    }
    assert c.final_params["forward_velocity.target_ms"] == 1.4
    assert abs(c.final_j - (0.9 + 0.5 * 1.2)) < 1e-9
    assert case_of(tmp_path / "missing") is None


def test_evidence_on_the_floor_names_the_recipe_and_the_stuck_moves(tmp_path):
    pb = build_batch(tmp_path)
    assert sum(pb.coached(c) for c in pb.cases) == 3  # the random coach is not experience
    text = pb.evidence("rough-medium-naive", success_now=0.0)
    assert "2/3 left the floor" in text
    assert "without a coach 0/1" in text
    assert "forward_velocity.target_ms down 2/2 (2 at the first report)" in text
    assert "energy.weight up 2/2" in text
    assert "1 run(s) never left the floor" in text
    assert "moves seen only in stuck runs: action_rate.weight down" in text
    assert "first success came 4.0M steps" in text
    # the current run is never its own evidence
    assert "1/2 left the floor" in pb.evidence("rough-medium-naive", 0.0, exclude_run="esc0")


def test_evidence_while_walking_shows_ledger_and_final_outcome(tmp_path):
    pb = build_batch(tmp_path)
    text = pb.evidence("rough-medium-naive", success_now=0.6)
    assert "forward_velocity.target_ms up: 2 moves, kept 100%, effect -0.050" in text
    assert "final J by where forward_velocity.target_ms ended" in text
    assert ">= 1.4: J 1.50 (n=2)" in text


def test_unknown_setting_falls_back_to_any_task(tmp_path):
    pb = build_batch(tmp_path)
    text = pb.evidence("stairs-hard-traditional", 0.0)
    assert "other tasks" in text or "any task" in text
    assert "left the floor" in text
    assert Playbook([]).evidence("rough-medium-naive", 0.0) == ""


def test_json_round_trip_and_stalled_text(tmp_path):
    pb = build_batch(tmp_path)
    out = tmp_path / "pb" / "playbook.json"
    pb.save(out)
    pb2 = Playbook.load(out)
    assert [c.run_id for c in pb2.cases] == [c.run_id for c in pb.cases]
    assert pb2.evidence("rough-medium-naive", 0.0) == pb.evidence("rough-medium-naive", 0.0)
    stalled = pb2.stalled_text()
    assert "(stuck)" in stalled and "action_rate.weight -0.01->-0.03" in stalled
    assert "esc0" not in stalled


def test_playbook_enabled_is_a_tunable_override():
    clean, errors = ev.validate_overrides({"playbook.enabled": True}, [])
    assert not errors and clean == {"playbook": {"enabled": True}}
    _, errors = ev.validate_overrides({"playbook.enabled": "yes"}, [])
    assert errors


def test_diagnostics_lists_stuck_runs_from_run_dirs(tmp_path):
    build_batch(tmp_path)
    from quadruped_rl.analysis.coach import load_run

    rows = []
    for name in ("esc0", "esc1", "stuck"):
        r = load_run(tmp_path / name)
        r["setting"] = "rough-medium-naive"
        rows.append(r)
    text = ev.diagnostics(rows)
    assert "never left the floor" in text and "(stuck)" in text


def test_coach_report_carries_playbook_evidence(tmp_path):
    from test_coach import VALID, FakeClient, _coach

    pb = build_batch(tmp_path / "runs")
    path = tmp_path / "playbook.json"
    pb.save(path)
    (tmp_path / "run").mkdir()
    coach, _env, _algo = _coach(
        tmp_path / "run", FakeClient(VALID), playbook={"enabled": True, "path": str(path)}
    )
    coach.task.update(terrain="rough", level="medium", reward="naive")
    coach.on_eval(10, 100, {"success_rate": 0.0}, [])
    rec = json.loads((tmp_path / "run" / "coach_log.jsonl").read_text().splitlines()[-1])
    user = rec["prompt"]["user"]
    assert "Earlier coached runs of this task" in user
    assert "forward_velocity.target_ms down 2/2 (2 at the first report)" in user
    # a missing file is a warning, not a failure
    (tmp_path / "run2").mkdir()
    coach2, _, _ = _coach(
        tmp_path / "run2", FakeClient(VALID), playbook={"enabled": True, "path": "nope.json"}
    )
    assert coach2.playbook is None


def test_run_dirs_resolve_through_job_logs(tmp_path, monkeypatch):
    build_batch(tmp_path / "data" / "results")
    batch = tmp_path / "data" / "results" / "coach_x"
    (batch / "jobs_logs").mkdir(parents=True)
    run = "ppo_a1_rough_coach-llm_s0_20260905-112658_21e045"
    (tmp_path / "data" / "results" / run).mkdir()
    (tmp_path / "data" / "results" / run / "config.yaml").write_text("{}")
    (batch / "jobs_logs" / "000.log").write_text(f'run dir "data/results/{run}" created\n')
    monkeypatch.chdir(tmp_path)
    assert [d.name for d in Playbook.run_dirs_under(batch)] == [run]
    # a root with runs of its own never falls back to the logs
    assert len(Playbook.run_dirs_under(tmp_path / "data" / "results")) == 6


def test_run_dirs_under_lists_overlapping_roots_once(tmp_path):
    build_batch(tmp_path)
    once = Playbook.run_dirs_under(tmp_path)
    twice = Playbook.run_dirs_under(tmp_path, tmp_path / "esc0", tmp_path)
    assert len(once) == 5 and twice == once
