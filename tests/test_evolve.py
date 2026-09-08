"""Coach self-evolution: version dirs, proposal validation, paired acceptance, state."""

import json
import math

import pytest
from pydantic import ValidationError

from quadruped_rl.harness.config import compose_config
from quadruped_rl.harness.trainer import make_run_id
from quadruped_rl.llm_feedback import evolve as ev
from quadruped_rl.llm_feedback.coach import load_prompts, resolve_version_dir
from quadruped_rl.llm_feedback.prompts import COACH_SYSTEM, COACH_USER
from quadruped_rl.llm_feedback.translator import LLMClient, strip_thinking

PARAMS = ["forward_velocity.target_ms", "forward_velocity.weight", "action_rate.weight"]


def incumbent() -> ev.Version:
    return ev.Version(name="v5", dir=None, system=COACH_SYSTEM, user=COACH_USER, overrides={})


def proposal(**kw) -> ev.Proposal:
    base = dict(hypothesis="h", system_md=COACH_SYSTEM, user_md=COACH_USER, coach_overrides={})
    base.update(kw)
    return ev.Proposal(**base)


def row(setting, seed, j, ints=None, **kw):
    r = {
        "setting": setting,
        "seed": seed,
        "objective_final": j,
        "success_rate": min(1.0, j),
        "mean_forward_velocity_ms": 0.5,
        "n_kept": 3,
        "n_rolled_back": 1,
        "_interventions": ints or [],
        "_final_params": {"forward_velocity.target_ms": 0.7},
    }
    r.update(kw)
    return r


# ------------------------------------------------------------ versions
def test_v5_version_dir_matches_builtin_prompts():
    v = ev.Version.load(resolve_version_dir("configs/coach/versions/v5"))
    assert v.name == "v5" and v.system == COACH_SYSTEM and v.overrides == {}
    assert (
        ev.validate_proposal(proposal(coach_overrides={"noise_z": 2.5}), incumbent(), PARAMS) == []
    )
    assert load_prompts("configs/coach/versions/v5") == (COACH_SYSTEM, COACH_USER)
    assert load_prompts(None) == (COACH_SYSTEM, COACH_USER)


def test_version_dir_merges_overrides_and_tags_run_id():
    cfg = compose_config(
        algorithm="ppo",
        robot="a1",
        terrain="stairs",
        coach="llm",
        overrides={"coach": {"version_dir": "configs/coach/versions/v5"}},
    )
    assert cfg["coach"]["strategy"] == "llm" and cfg["coach"]["version_dir"]
    assert "coach-llm-v5" in make_run_id(cfg)


def test_version_dir_yaml_overrides_apply_but_cli_wins(tmp_path):
    d = tmp_path / "v9"
    d.mkdir()
    (d / "system.md").write_text(COACH_SYSTEM)
    (d / "user.md").write_text(COACH_USER)
    (d / "coach.yaml").write_text("coach:\n  noise_z: 2.5\n  max_params: 1\n")
    cfg = compose_config(coach="llm", overrides={"coach": {"version_dir": str(d), "max_params": 3}})
    assert cfg["coach"]["noise_z"] == 2.5
    assert cfg["coach"]["max_params"] == 3  # CLI override re-applied after the version merge


# ----------------------------------------------------------- proposals
def test_validate_proposal_rejects_broken_templates():
    inc = incumbent()
    errs = ev.validate_proposal(
        proposal(system_md="{task_text} {nope}", user_md="{kpi_table} only"), inc, PARAMS
    )
    assert any("unknown placeholders ['nope']" in e for e in errs)
    assert any("system_md: required placeholders missing" in e for e in errs)
    assert any("user_md: required placeholders missing" in e for e in errs)
    errs = ev.validate_proposal(proposal(user_md=COACH_USER + " {"), inc, PARAMS)
    assert any("bad braces" in e for e in errs)
    errs = ev.validate_proposal(proposal(system_md=COACH_SYSTEM * 3), inc, PARAMS)
    assert any("longer than" in e for e in errs)
    assert "hypothesis is empty" in ev.validate_proposal(proposal(hypothesis=" "), inc, PARAMS)
    # literal JSON braces doubled in the template are fine
    ok = proposal(system_md=COACH_SYSTEM + '\nReturn {{"x": 1}}.')
    assert ev.validate_proposal(ok, inc, PARAMS) == []


def test_validate_overrides_ranges_and_types():
    clean, errs = ev.validate_overrides(
        {"noise_z": 2.0, "phases": {"exploit": 0.5}, "llm.reasoning_effort": "low"}, PARAMS
    )
    assert errs == []
    assert clean == {"noise_z": 2.0, "phases": {"exploit": 0.5}, "llm": {"reasoning_effort": "low"}}
    clean, errs = ev.validate_overrides({"best_confirm": 2.4, "interval_steps": 2e6}, PARAMS)
    assert clean == {"best_confirm": 2, "interval_steps": 2_000_000}
    _, errs = ev.validate_overrides({"noise_z": 9, "bogus": 1, "interval_steps": 1.5e6}, PARAMS)
    assert any("outside" in e for e in errs)
    assert any("not a tunable key" in e for e in errs)
    assert any("multiple of 1,000,000" in e for e in errs)
    _, errs = ev.validate_overrides({"llm": {"reasoning_effort": "max"}}, PARAMS)
    assert any("must be one of" in e for e in errs)


def test_validate_effect_prior():
    good = {"effect_prior": {"forward_velocity.target_ms": {"up": [0.05, 0.05]}}}
    clean, errs = ev.validate_overrides(good, PARAMS)
    assert errs == [] and clean == good
    bad = {"effect_prior": {"unknown.param": {"up": [0.0, 0.1]}}}
    assert ev.validate_overrides(bad, PARAMS)[1]
    bad = {"effect_prior": {"action_rate.weight": {"sideways": [0.0, 0.1]}}}
    assert any("expected [mean, sd]" in e for e in ev.validate_overrides(bad, PARAMS)[1])
    bad = {"effect_prior": {"action_rate.weight": {"down": [0.9, 0.1]}}}
    assert any("|mean| <= 0.3" in e for e in ev.validate_overrides(bad, PARAMS)[1])


def test_write_version_roundtrip(tmp_path):
    prop = proposal(
        hypothesis="calmer",
        coach_overrides={"noise_z": 2.5, "bogus": 3},
        code_ideas=["expose gait phase"],
        expected_delta_j=0.03,
    )
    d = ev.write_version(tmp_path, "v6", prop, "v5", PARAMS, 1)
    v = ev.Version.load(d)
    assert v.name == "v6" and v.system == COACH_SYSTEM
    assert v.overrides == {"noise_z": 2.5}  # invalid keys are dropped on write
    assert "parent: v5" in v.changelog and "expose gait phase" in v.changelog


def test_proposal_from_json_tolerates_prose_around_the_object():
    text = 'Sure:\n{"hypothesis": "x", "system_md": "a", "user_md": "b"}\nDone.'
    p = ev.Proposal.from_json(text)
    assert p.hypothesis == "x" and p.coach_overrides == {} and p.code_ideas == []
    with pytest.raises(ValueError):
        ev.Proposal.from_json("no json here")
    with pytest.raises(ValidationError):
        ev.Proposal.from_json('{"hypothesis": 3.5, "system_md": []}')  # wrong types


def test_omitted_templates_inherit_from_incumbent(tmp_path):
    inc = incumbent()
    # settings-only proposal: templates null -> incumbent's
    p = ev.Proposal.from_json('{"hypothesis": "x", "coach_overrides": {"max_rel_change": 0.4}}')
    assert p.system_md is None and p.user_md is None
    assert ev.validate_proposal(p, inc, PARAMS) == []
    r = p.resolve(inc)
    assert (
        r.system_md == COACH_SYSTEM
        and r.user_md == COACH_USER
        and r.coach_overrides == {"max_rel_change": 0.4}
    )
    # the incumbent's own overrides carry over (deltas are written against the
    # effective settings); the proposal's keys win, nested tables merge
    inc2 = ev.Version(
        name="v7p",
        dir=None,
        system=COACH_SYSTEM,
        user=COACH_USER,
        overrides={"playbook": {"enabled": True}, "curriculum_lock": True, "noise_z": 2.0},
    )
    p2 = ev.Proposal.from_json(
        '{"hypothesis": "x", "coach_overrides": {"noise_z": 2.5, "playbook": {"enabled": false}}}'
    )
    r2 = p2.resolve(inc2)
    assert r2.coach_overrides == {
        "playbook": {"enabled": False},
        "curriculum_lock": True,
        "noise_z": 2.5,
    }
    assert ev.validate_proposal(p2, inc2, PARAMS) == []
    # same templates, only inherited overrides -> still "changes nothing"
    errs = ev.validate_proposal(ev.Proposal.from_json('{"hypothesis": "x"}'), inc2, PARAMS)
    assert any("changes nothing" in e for e in errs)
    # an unresolved proposal cannot be written; a resolved one can
    with pytest.raises(ValueError):
        ev.write_version(tmp_path, "v6", p, "v5", PARAMS, 1)
    ev.write_version(tmp_path, "v6", r, "v5", PARAMS, 1)
    assert ev.Version.load(tmp_path / "v6").system == COACH_SYSTEM
    # nothing changed at all -> rejected
    errs = ev.validate_proposal(ev.Proposal.from_json('{"hypothesis": "x"}'), inc, PARAMS)
    assert any("changes nothing" in e for e in errs)


# ------------------------------------------------------------ analysis
def test_summarize_and_paired_compare():
    inc = [row("a", s, 0.5 + 0.01 * s) for s in range(3)] + [row("b", s, 0.8) for s in range(3)]
    cand = [row("a", s, 0.6 + 0.01 * s) for s in range(3)] + [row("b", s, 0.82) for s in range(3)]
    s = ev.summarize(inc)
    assert s["a"]["n"] == 3 and s["a"]["mean"] == pytest.approx(0.51)
    assert s["a"]["ci95"] == pytest.approx(1.96 * 0.01 / math.sqrt(3))
    assert s["b"]["sd"] == pytest.approx(0.0) and s["a"]["kept"] == 3
    cmp = ev.paired_compare(cand, inc)
    assert cmp["n"] == 6 and cmp["mean"] == pytest.approx((0.1 * 3 + 0.02 * 3) / 6)
    assert cmp["per_setting"]["a"] == pytest.approx(0.1)
    assert cmp["per_setting"]["b"] == pytest.approx(0.02)
    assert 0 < cmp["p_one_sided"] < 0.05 and cmp["cohen_d"] > 1
    assert ev.paired_compare(cand, [])["n"] == 0
    assert "n=6" in ev.compare_text(cmp)
    assert "| a |" in ev.results_table(s)


def test_accept_rule():
    rule = ev.AcceptRule(min_pairs=6, p_max=0.2, worst_setting=-0.05)
    ok, reason = rule.decide({"n": 3, "mean": 0.1, "p_one_sided": 0.01, "per_setting": {}})
    assert not ok and "pairs" in reason
    ok, _ = rule.decide(
        {"n": 6, "mean": 0.05, "p_one_sided": 0.1, "per_setting": {"a": 0.1, "b": -0.1}}
    )
    assert not ok  # one setting regressed beyond the tolerance
    ok, _ = rule.decide(
        {"n": 6, "mean": 0.05, "p_one_sided": 0.5, "per_setting": {"a": 0.05, "b": 0.05}}
    )
    assert not ok  # not significant
    ok, reason = rule.decide(
        {
            "n": 6,
            "mean": 0.05,
            "ci95": 0.02,
            "cohen_d": 1.2,
            "p_one_sided": 0.05,
            "per_setting": {"a": 0.06, "b": 0.04},
        }
    )
    assert ok and "pooled dJ +0.050" in reason


def test_diagnostics_reports_calibration_and_worst_moves():
    ints = [
        {
            "step": 4_000_000,
            "phase": "explore",
            "status": "rolled_back",
            "effect": -0.12,
            "predicted_delta_j": 0.05,
            "applied": {"forward_velocity.target_ms": 1.0},
            "diagnosis": "pushing speed",
        },
        {
            "step": 6_000_000,
            "phase": "exploit",
            "status": "kept",
            "effect": 0.03,
            "predicted_delta_j": 0.02,
            "applied": {"action_rate.weight": -0.02},
            "diagnosis": "smoother",
        },
    ]
    text = ev.diagnostics([row("a", 0, 0.7, ints), row("a", 1, 0.6)])
    assert "a seed 0: J=0.700" in text and "final target_ms 0.70" in text
    assert "sign accuracy 0.50" in text
    assert "step 4M [rolled_back] effect -0.120" in text and "pushing speed" in text
    assert ev.diagnostics([]) == "(no runs)"


# --------------------------------------------------------------- state
def test_evolve_state_names_and_history(tmp_path):
    st = ev.EvolveState(tmp_path / "state.json")
    assert st.incumbent is None and st.next_name() == "v6"
    st.data["incumbent"] = "v5"
    st.version("v5").update(generation=0, status="incumbent")
    st.version("v6").update(parent="v5", generation=1, status="rejected", hypothesis="h6")
    st.log("hello")
    st.save()
    assert st.next_name() == "v7"
    # a fresh root never overwrites version dirs an earlier root wrote
    assert st.next_name(["v5p", "v9", "notes"]) == "v10"
    st2 = ev.EvolveState(tmp_path / "state.json")
    assert st2.incumbent == "v5" and st2.data["log"][-1]["msg"] == "hello"
    hist = st2.history_text()
    assert "v6" in hist and "h6" in hist


def test_meta_prompt_formats_with_and_without_errors():
    inc = incumbent()
    summary = ev.summarize([row("a", 0, 0.5), row("a", 1, 0.6)])
    last = {
        "name": "v6",
        "status": "rejected",
        "reason": "p=0.4",
        "hypothesis": "h6",
        "compare": {
            "n": 6,
            "mean": -0.01,
            "sd": 0.02,
            "ci95": 0.016,
            "t": -1,
            "p_one_sided": 0.4,
            "cohen_d": -0.5,
            "per_setting": {"a": -0.01},
        },
        "summary": summary,
        "diagnostics": "(diag)",
    }
    system, user = ev.meta_prompt(inc, summary, "(diag)", "(hist)", last, PARAMS, ["bad braces"])
    assert "task_text" in system and "forward_velocity.target_ms" in system
    assert "v6" in user and "bad braces" in user and COACH_USER[:40] in user
    _, user = ev.meta_prompt(inc, {}, "(no runs)", "", None, PARAMS)
    assert "(none yet)" in user


def test_meta_prompt_shows_effective_settings_and_flags_noop_overrides():
    base = {"max_rel_change": 0.3, "phases": {"exploit": 0.5, "consolidate": 0.85}}
    inc = ev.Version(
        name="v5",
        dir=None,
        system=COACH_SYSTEM,
        user=COACH_USER,
        overrides={"phases": {"exploit": 0.4}},
    )
    text = ev.effective_settings(base, inc.overrides)
    assert "- max_rel_change: 0.3\n" in text
    assert "- phases.exploit: 0.4  (set by this version)" in text
    assert "- phases.consolidate: 0.85\n" in text
    _, user = ev.meta_prompt(inc, {}, "(no runs)", "", None, PARAMS, base_coach=base)
    assert "phases.exploit: 0.4" in user
    # an override equal to the effective value is a no-op only when the preset is known
    noop = proposal(coach_overrides={"max_rel_change": 0.3, "phases": {"exploit": 0.4}})
    assert ev.validate_proposal(noop, inc, PARAMS) == []
    errs = ev.validate_proposal(noop, inc, PARAMS, base)
    assert any("equal the current effective values" in e for e in errs)
    real = proposal(coach_overrides={"max_rel_change": 0.3, "noise_z": 2.5})
    assert ev.validate_proposal(real, inc, PARAMS, base) == []


# --------------------------------------------------------- local client
def test_llm_client_local_base_url_uses_max_tokens(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kw):
            calls.update(kw)

            class R:
                choices = [
                    type("C", (), {"message": type("M", (), {"content": "<think>x</think>{}"})()})()
                ]
                usage = None

            return R()

    class FakeOpenAI:
        def __init__(self, **kw):
            calls["init"] = kw
            self.chat = type("Chat", (), {"completions": FakeCompletions()})()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    c = LLMClient(
        "openai",
        "gpt-oss:20b",
        reasoning_effort="low",
        json_mode=True,
        base_url="http://127.0.0.1:11435/v1",
        api_key_env="LOCAL_LLM_API_KEY",
        timeout_s=30,
    )
    assert c.complete("s", "u", max_tokens=123) == "{}"
    assert calls["init"]["base_url"] == "http://127.0.0.1:11435/v1"
    assert calls["init"]["api_key"] == "local" and calls["init"]["timeout"] == 30
    assert calls["max_tokens"] == 123 and "max_completion_tokens" not in calls
    assert calls["response_format"] == {"type": "json_object"}


def test_strip_thinking():
    assert strip_thinking('<think>\nplan\n</think>\n{"a": 1}') == '{"a": 1}'
    assert strip_thinking("plain") == "plain"
    assert json.loads(strip_thinking("<think>a</think>{}")) == {}
