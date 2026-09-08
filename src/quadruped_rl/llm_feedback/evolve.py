"""Coach evolution: derive coach version N+1 from the measured batch of version N.

The reward coach (coach.py) is the outer loop around PPO. This module is the
loop around the coach: a *version* is a directory ``configs/coach/versions/<v>/``
holding the prompt templates (``system.md``, ``user.md``), decision-layer
overrides (``coach.yaml``) and a ``CHANGELOG.md``. One generation is

    run incumbent (once) -> run candidate on the same settings x seeds
    -> paired comparison of the final objective J -> accept / reject
    -> meta-LLM reads the evidence and writes the next candidate.

Everything measurable is computed here (paired differences, CI, one-sided
paired t, per-setting means, coach-log diagnostics, prediction calibration);
the meta-LLM only proposes text and bounded numbers, and every proposal is
validated (placeholders, key names, ranges) before it becomes a version.
Code changes are never applied automatically: the meta-LLM's ``code_ideas``
are appended to ``proposals.md`` for a human.

Pure logic — no simulator, no network — so it is unit-tested (tests/test_evolve.py).
The driver that runs jobs and talks to the LLM server is scripts/evolve_coach.py.
"""

from __future__ import annotations

import json
import math
import string
import time
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quadruped_rl.llm_feedback.prompts import EVOLVE_SYSTEM, EVOLVE_USER
from quadruped_rl.llm_feedback.schemas import MetaOutput

# Fields the coach fills into the templates (coach.py::LLMScheduler.propose).
SYSTEM_FIELDS = {
    "task_text",
    "component_table",
    "param_table",
    "max_params",
    "max_rel_change",
    "max_log_factor",
    "rollback_tolerance",
    "objective_text",
}
USER_FIELDS = {
    "step",
    "progress",
    "k",
    "kpi_table",
    "objective_table",
    "stats_table",
    "dynamics_table",
    "window",
    "trace",
    "best",
    "noise",
    "history",
    "evidence",
    "phase",
}
# Placeholders a version must keep: without these the coach reasons blind.
SYSTEM_REQUIRED = {"task_text", "component_table", "param_table", "objective_text"}
USER_REQUIRED = {"kpi_table", "objective_table", "trace", "history", "evidence"}

# Decision-layer keys the meta-LLM may set, with their admissible range
# (inclusive). Dotted keys address nested config. Anything else is rejected.
TUNABLE: dict[str, tuple[float, float] | set[str]] = {
    "rollback_tolerance": (0.02, 0.15),
    "noise_z": (1.0, 3.0),
    "tolerance_max": (0.1, 0.4),
    "best_confirm": (1, 3),
    "cooldown_after_rollback": (0, 2),
    "max_restores_per_snapshot": (0, 2),
    "confirm_evals": (0, 2),
    "confirm_zone": (0.0, 1.0),
    "trend_points": (2, 6),
    "release_from_progress": (0.3, 0.8),
    "release_min_success": (0.0, 0.8),
    "phases.exploit": (0.2, 0.7),
    "phases.consolidate": (0.7, 0.97),
    "effect_prior_sd": (0.02, 0.3),
    "tracking_ratio_prior": (0.6, 1.0),
    "forecast_min_points": (3, 8),
    "max_params": (1, 3),
    "max_rel_change": (0.1, 0.5),
    "max_log_factor": (1.5, 5.0),
    "interval_steps": (1_000_000, 4_000_000),
    "warmup_steps": (1_000_000, 6_000_000),
    "llm.reasoning_effort": {"none", "low", "medium", "high"},
    "llm.max_tokens": (2000, 32000),
    "playbook.enabled": {True, False},
    "settled_reports": (0, 3),
    "ledger_veto_obs": (0, 6),
    "curriculum_lock": {True, False},
}
INT_KEYS = {
    "settled_reports",
    "ledger_veto_obs",
    "best_confirm",
    "cooldown_after_rollback",
    "max_restores_per_snapshot",
    "confirm_evals",
    "trend_points",
    "forecast_min_points",
    "max_params",
    "interval_steps",
    "warmup_steps",
    "llm.max_tokens",
}
EFFECT_PRIOR_MEAN_MAX = 0.3
EFFECT_PRIOR_SD = (0.01, 0.3)
MAX_TEMPLATE_GROWTH = 2.5  # candidate template length <= this x the incumbent's


# ----------------------------------------------------------------- versions
@dataclass
class Version:
    name: str
    dir: Path
    system: str
    user: str
    overrides: dict[str, Any]
    changelog: str = ""

    @classmethod
    def load(cls, d: str | Path) -> Version:
        d = Path(d)
        raw = yaml.safe_load((d / "coach.yaml").read_text()) if (d / "coach.yaml").exists() else {}
        raw = raw or {}
        overrides = raw.get("coach", raw) or {}
        return cls(
            name=d.name,
            dir=d,
            system=(d / "system.md").read_text(),
            user=(d / "user.md").read_text(),
            overrides=dict(overrides),
            changelog=(d / "CHANGELOG.md").read_text() if (d / "CHANGELOG.md").exists() else "",
        )


@dataclass
class Proposal:
    hypothesis: str
    system_md: str | None  # None = inherit the incumbent's template (see resolve)
    user_md: str | None
    coach_overrides: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_delta_j: float | None = None
    code_ideas: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, text: str) -> Proposal:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object in meta response")
        m = MetaOutput.model_validate(json.loads(text[start : end + 1]))
        return cls(
            hypothesis=m.hypothesis,
            system_md=m.system_md,
            user_md=m.user_md,
            coach_overrides=dict(m.coach_overrides),
            rationale=m.rationale,
            expected_delta_j=m.expected_delta_j,
            code_ideas=list(m.code_ideas),
        )

    def resolve(self, incumbent: Version) -> Proposal:
        """Fill omitted templates from the incumbent (a proposal may change only
        the settings, or only one of the two templates)."""
        return replace(
            self,
            system_md=incumbent.system if self.system_md is None else self.system_md,
            user_md=incumbent.user if self.user_md is None else self.user_md,
        )


def placeholders(template: str) -> set[str]:
    """Named fields of a str.format template (raises ValueError on bad braces)."""
    return {f for _, f, _, _ in string.Formatter().parse(template) if f}


def _sample_fields(template: str, names: set[str]) -> dict[str, Any]:
    """Dummy values that satisfy each placeholder's format spec (numbers where
    a spec such as ``:,`` or ``:.2f`` is used, text otherwise)."""
    out: dict[str, Any] = {n: "x" for n in names}
    for _, f, spec, _ in string.Formatter().parse(template):
        if f and spec:
            out[f] = 1 if ("," in spec or "d" in spec) and "f" not in spec else 1.0
    return out


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k != "effect_prior":
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _nest(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in flat.items():
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def validate_overrides(
    overrides: dict[str, Any], params: list[str]
) -> tuple[dict[str, Any], list[str]]:
    """Check dotted/nested decision-layer overrides against TUNABLE; returns
    (clean nested dict, errors). Unknown keys and out-of-range values are errors."""
    errors: list[str] = []
    clean: dict[str, Any] = {}
    for key, val in _flatten(dict(overrides)).items():
        if key == "effect_prior":
            ok = True
            if not isinstance(val, dict):
                errors.append("effect_prior must be a mapping param -> {up|down: [mean, sd]}")
                continue
            for p, dirs in val.items():
                if p not in params or not isinstance(dirs, dict):
                    errors.append(f"effect_prior: unknown param or bad shape '{p}'")
                    ok = False
                    continue
                for d, ms in dirs.items():
                    if d not in ("up", "down") or not isinstance(ms, (list, tuple)) or len(ms) != 2:
                        errors.append(f"effect_prior.{p}.{d}: expected [mean, sd]")
                        ok = False
                        continue
                    mean, sd = float(ms[0]), float(ms[1])
                    if abs(mean) > EFFECT_PRIOR_MEAN_MAX or not (
                        EFFECT_PRIOR_SD[0] <= sd <= EFFECT_PRIOR_SD[1]
                    ):
                        errors.append(f"effect_prior.{p}.{d}: |mean| <= 0.3 and sd in [0.01, 0.3]")
                        ok = False
            if ok:
                clean["effect_prior"] = val
            continue
        if key not in TUNABLE:
            errors.append(f"'{key}' is not a tunable key (allowed: {sorted(TUNABLE)})")
            continue
        rng = TUNABLE[key]
        if isinstance(rng, set):
            if val not in rng:
                errors.append(f"'{key}' must be one of {sorted(rng)}, got {val!r}")
                continue
            clean[key] = val
            continue
        try:
            x = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{key}' must be a number, got {val!r}")
            continue
        lo, hi = rng
        if not (lo <= x <= hi):
            errors.append(f"'{key}'={x} outside [{lo}, {hi}]")
            continue
        clean[key] = int(round(x)) if key in INT_KEYS else x
    if "interval_steps" in clean and clean["interval_steps"] % 1_000_000:
        errors.append("interval_steps must be a multiple of 1,000,000 (eval interval)")
        clean.pop("interval_steps")
    return _nest(clean), errors


def validate_proposal(
    prop: Proposal,
    incumbent: Version,
    params: list[str],
    base_coach: dict[str, Any] | None = None,
) -> list[str]:
    """All reasons a proposal cannot become a version (empty = valid).
    With ``base_coach`` (the coach preset) overrides equal to the effective
    value are reported as no-ops."""
    prop = prop.resolve(incumbent)
    errors: list[str] = []
    same_templates = (
        prop.system_md.strip() == incumbent.system.strip()
        and prop.user_md.strip() == incumbent.user.strip()
    )
    if same_templates and not prop.coach_overrides:
        errors.append("proposal changes nothing (templates and overrides equal the incumbent)")
    elif same_templates and base_coach is not None:
        current = _merge_coach(base_coach, incumbent.overrides or {})
        clean, _ = validate_overrides(prop.coach_overrides, params)
        noop = [
            k
            for k, v in _flatten(clean).items()
            if k in TUNABLE and _lookup(current, k) is not None and _lookup(current, k) == v
        ]
        if noop and len(noop) == len(_flatten(clean)):
            errors.append(
                f"proposal changes nothing: overrides {noop} equal the current effective values"
            )
    for label, text, allowed, required, base in (
        ("system_md", prop.system_md, SYSTEM_FIELDS, SYSTEM_REQUIRED, incumbent.system),
        ("user_md", prop.user_md, USER_FIELDS, USER_REQUIRED, incumbent.user),
    ):
        if not text.strip():
            errors.append(f"{label} is empty")
            continue
        try:
            used = placeholders(text)
        except ValueError as e:
            errors.append(f"{label}: bad braces ({e}); literal braces must be doubled {{{{ }}}}")
            continue
        unknown = used - allowed
        if unknown:
            errors.append(f"{label}: unknown placeholders {sorted(unknown)}")
        missing = required - used
        if missing:
            errors.append(f"{label}: required placeholders missing {sorted(missing)}")
        if len(text) > MAX_TEMPLATE_GROWTH * max(len(base), 1):
            errors.append(f"{label}: longer than {MAX_TEMPLATE_GROWTH}x the incumbent")
        if not unknown:
            try:
                text.format(**_sample_fields(text, allowed))
            except (KeyError, IndexError, ValueError) as e:
                errors.append(f"{label}: does not format ({type(e).__name__}: {e})")
    _, errs = validate_overrides(prop.coach_overrides, params)
    errors.extend(errs)
    if not prop.hypothesis.strip():
        errors.append("hypothesis is empty")
    return errors


def write_version(
    root: Path, name: str, prop: Proposal, parent: str, params: list[str], gen: int
) -> Path:
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    if prop.system_md is None or prop.user_md is None:
        raise ValueError("write_version needs a resolved proposal (Proposal.resolve)")
    (d / "system.md").write_text(prop.system_md)
    (d / "user.md").write_text(prop.user_md)
    clean, _ = validate_overrides(prop.coach_overrides, params)
    (d / "coach.yaml").write_text(
        f"# Decision-layer overrides of coach version {name} (evolution generation {gen}).\n"
        + yaml.safe_dump({"coach": clean}, sort_keys=False)
    )
    ideas = "".join(f"- {i}\n" for i in prop.code_ideas) or "- (none)\n"
    (d / "CHANGELOG.md").write_text(
        f"# {name} — generation {gen} ({time.strftime('%Y-%m-%d')})\n\n"
        f"parent: {parent}\n"
        f"hypothesis: {prop.hypothesis}\n"
        f"expected_delta_j: {prop.expected_delta_j}\n\n"
        f"## Rationale\n{prop.rationale}\n\n## Code ideas (not applied)\n{ideas}"
    )
    return d


# ---------------------------------------------------------------- analysis
def summarize(
    rows: list[dict[str, Any]], metric: str = "objective_final"
) -> dict[str, dict[str, float]]:
    """Per setting: n, mean, sd, ci95 of ``metric`` plus coach-outcome means."""
    out: dict[str, dict[str, float]] = {}
    by: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["setting"], []).append(r)
    for setting, rs in sorted(by.items()):
        x = np.array([float(r[metric]) for r in rs])
        n = len(x)
        sd = float(x.std(ddof=1)) if n > 1 else 0.0
        out[setting] = {
            "n": n,
            "mean": float(x.mean()),
            "sd": sd,
            "ci95": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0,
            "success_rate": float(np.mean([r.get("success_rate", float("nan")) for r in rs])),
            "velocity": float(
                np.mean([r.get("mean_forward_velocity_ms", float("nan")) for r in rs])
            ),
            "kept": float(np.mean([r.get("n_kept", 0) for r in rs])),
            "rolled_back": float(np.mean([r.get("n_rolled_back", 0) for r in rs])),
            "restored": float(np.mean([r.get("n_restored", 0) for r in rs])),
            "api_errors": float(np.mean([r.get("n_api_errors", 0) for r in rs])),
            "discarded": float(np.mean([r.get("n_discarded", 0) for r in rs])),
            "latency_s": float(np.mean([r.get("llm_latency_s", 0) for r in rs])),
            "tokens_out": float(np.mean([r.get("tokens_out_per_call", 0) for r in rs])),
        }
    return out


def _t_sf(t: float, df: int) -> float:
    """One-sided p of Student t (scipy if available, else normal approximation)."""
    try:
        from scipy import stats

        return float(stats.t.sf(t, df))
    except Exception:  # pragma: no cover - scipy is a project dependency
        return 0.5 * math.erfc(t / math.sqrt(2))


def paired_compare(
    cand: list[dict[str, Any]], inc: list[dict[str, Any]], metric: str = "objective_final"
) -> dict[str, Any]:
    """Candidate minus incumbent on matching (setting, seed) pairs.

    Returns pooled mean/sd/ci95, one-sided paired-t p (H1: candidate > incumbent),
    Cohen's d of the differences, and per-setting mean differences."""
    ref = {(r["setting"], int(r["seed"])): float(r[metric]) for r in inc}
    pairs = [
        (r["setting"], int(r["seed"]), float(r[metric]) - ref[(r["setting"], int(r["seed"]))])
        for r in cand
        if (r["setting"], int(r["seed"])) in ref
    ]
    if not pairs:
        return {"n": 0, "pairs": []}
    d = np.array([x for _, _, x in pairs])
    n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = (
        float(d.mean() / (sd / math.sqrt(n)))
        if n > 1 and sd > 0
        else (math.inf if d.mean() > 0 else -math.inf)
    )
    p = _t_sf(t, n - 1) if n > 1 and math.isfinite(t) else (0.0 if t > 0 else 1.0)
    per: dict[str, float] = {}
    for s in sorted({s for s, _, _ in pairs}):
        per[s] = float(np.mean([x for ss, _, x in pairs if ss == s]))
    return {
        "n": n,
        "mean": float(d.mean()),
        "sd": sd,
        "ci95": 1.96 * sd / math.sqrt(n) if n > 1 else 0.0,
        "t": t,
        "p_one_sided": p,
        "cohen_d": float(d.mean() / sd) if sd > 0 else float("inf"),
        "per_setting": per,
        "pairs": pairs,
    }


@dataclass
class AcceptRule:
    """Pre-registered acceptance of a candidate over the incumbent."""

    min_pairs: int = 6
    p_max: float = 0.2  # one-sided paired t
    worst_setting: float = -0.05  # no setting may lose more than this (mean dJ)

    def decide(self, cmp: dict[str, Any]) -> tuple[bool, str]:
        if cmp.get("n", 0) < self.min_pairs:
            return False, f"only {cmp.get('n', 0)} pairs (< {self.min_pairs})"
        if cmp["mean"] <= 0:
            return False, f"pooled dJ {cmp['mean']:+.3f} <= 0"
        worst = min(cmp["per_setting"].items(), key=lambda kv: kv[1])
        if worst[1] < self.worst_setting:
            return False, f"{worst[0]} loses {worst[1]:+.3f} (< {self.worst_setting})"
        if cmp["p_one_sided"] > self.p_max:
            return (
                False,
                f"pooled dJ {cmp['mean']:+.3f} but p={cmp['p_one_sided']:.2f} > {self.p_max}",
            )
        return True, (
            f"pooled dJ {cmp['mean']:+.3f} +- {cmp['ci95']:.3f} (n={cmp['n']}, "
            f"p={cmp['p_one_sided']:.3f}, d={cmp['cohen_d']:.2f}), "
            f"worst setting {worst[0]} {worst[1]:+.3f}"
        )


# ------------------------------------------------------------- diagnostics
def diagnostics(rows: list[dict[str, Any]], max_worst: int = 6) -> str:
    """What the coach did in these runs, from the settled coach_log records
    (``_interventions`` of analysis.coach.load_run): outcome counts, final
    curriculum value, calibration of the coach's dJ predictions, and the
    interventions with the worst detrended effect (with the coach's own
    diagnosis, so the meta-LLM sees *why* it moved)."""
    lines: list[str] = []
    worst: list[tuple[float, str]] = []
    preds: list[tuple[float, float]] = []
    for r in sorted(rows, key=lambda r: (r["setting"], int(r["seed"]))):
        ints = r.get("_interventions") or []
        final_params = r.get("_final_params") or {}
        target = final_params.get("forward_velocity.target_ms")
        phases = {}
        for it in ints:
            phases[it.get("phase") or "?"] = phases.get(it.get("phase") or "?", 0) + 1
            eff, pred = it.get("effect"), it.get("predicted_delta_j")
            if eff is not None and pred is not None:
                preds.append((float(pred), float(eff)))
            if eff is not None and it.get("applied"):
                moved = ", ".join(f"{k}={v:.3g}" for k, v in it["applied"].items())
                worst.append(
                    (
                        float(eff),
                        f"{r['setting']} s{r['seed']} step {it['step'] / 1e6:.0f}M "
                        f"[{it['status']}] effect {eff:+.3f}: {moved} — "
                        f'"{str(it.get("diagnosis") or "")[:160]}"',
                    )
                )
        succ = r.get("success_rate", float("nan"))
        vel = r.get("mean_forward_velocity_ms", float("nan"))
        lines.append(
            f"- {r['setting']} seed {r['seed']}: J={r['objective_final']:.3f} "
            f"(success {succ:.2f}, v {vel:.2f}); "
            f"kept {r.get('n_kept', 0)}, rolled back {r.get('n_rolled_back', 0)}, "
            f"restored {r.get('n_restored', 0)}, api errors {r.get('n_api_errors', 0)}, "
            f"malformed {r.get('n_discarded', 0)}; "
            f"final target_ms {target if target is None else f'{target:.2f}'}; phases {phases}"
        )
    if preds:
        pr = np.array(preds)
        sign = float(np.mean(np.sign(pr[:, 0]) == np.sign(pr[:, 1])))
        mae = float(np.mean(np.abs(pr[:, 0] - pr[:, 1])))
        bias = float(np.mean(pr[:, 0] - pr[:, 1]))
        lines.append(
            f"- coach dJ predictions: n={len(preds)}, sign accuracy {sign:.2f}, "
            f"MAE {mae:.3f}, bias {bias:+.3f}"
        )
    if worst:
        lines.append("- worst interventions (detrended effect on J):")
        for _eff, txt in sorted(worst)[:max_worst]:
            lines.append(f"    {txt}")
    stalled = _stalled_text(rows)
    if stalled:
        lines.append(stalled)
    return "\n".join(lines) if lines else "(no runs)"


def _stalled_text(rows: list[dict[str, Any]]) -> str:
    """Runs that never left the floor and every move they made there — the
    failure the meta-LLM most needs to see (2026-09-08: two naive-reward seeds
    stuck for 40M steps while the coach wandered)."""
    dirs = [r["run_dir"] for r in rows if r.get("run_dir")]
    if not dirs:
        return ""
    from quadruped_rl.llm_feedback.playbook import Playbook

    return Playbook.build(dirs, conditions=None).stalled_text()


def results_table(summary: dict[str, dict[str, float]]) -> str:
    hdr = (
        "| setting | n | J mean +- sd | success | velocity | kept | rolled back "
        "| restored | api err | malformed | LLM s/call | out tok/call |"
    )
    out = [hdr, "|" + "---|" * 12]
    for s, v in summary.items():
        out.append(
            f"| {s} | {v['n']} | {v['mean']:.3f} +- {v['sd']:.3f} | {v['success_rate']:.2f} "
            f"| {v['velocity']:.2f} | {v['kept']:.1f} | {v['rolled_back']:.1f} "
            f"| {v['restored']:.1f} | {v['api_errors']:.1f} | {v['discarded']:.1f} "
            f"| {v.get('latency_s', 0):.0f} | {v.get('tokens_out', 0):.0f} |"
        )
    return "\n".join(out)


def compare_text(cmp: dict[str, Any]) -> str:
    if cmp.get("n", 0) == 0:
        return "(no paired runs)"
    per = ", ".join(f"{s} {d:+.3f}" for s, d in cmp["per_setting"].items())
    return (
        f"pooled dJ = {cmp['mean']:+.3f} +- {cmp['ci95']:.3f} (95% CI, n={cmp['n']} pairs), "
        f"one-sided p = {cmp['p_one_sided']:.3f}, Cohen's d = {cmp['cohen_d']:.2f}; "
        f"per setting: {per}"
    )


# ------------------------------------------------------------------- state
class EvolveState:
    """Resumable record of the evolution (``<root>/state.json``)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"generation": 0, "incumbent": None, "versions": {}, "log": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    @property
    def incumbent(self) -> str | None:
        return self.data["incumbent"]

    def version(self, name: str) -> dict[str, Any]:
        return self.data["versions"].setdefault(
            name, {"parent": None, "generation": 0, "status": "new"}
        )

    def next_name(self, taken: Iterable[str] = ()) -> str:
        """v6, v7, ... after the highest v<N> known to the state or in ``taken``
        (the version dirs on disk: a fresh root numbers past every dir an
        earlier root wrote instead of overwriting it)."""
        names = set(self.data["versions"]) | set(taken)
        nums = [int(n[1:]) for n in names if n[1:].isdigit()]
        return f"v{max(nums, default=5) + 1}"

    def log(self, msg: str) -> None:
        self.data["log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg})
        self.save()

    def history_text(self, limit: int = 8) -> str:
        """Accepted/rejected versions with their numbers, newest last."""
        items = sorted(self.data["versions"].items(), key=lambda kv: kv[1].get("generation", 0))
        lines = []
        for name, v in items[-limit:]:
            res = v.get("compare") or {}
            per = ", ".join(f"{s} {d:+.3f}" for s, d in (res.get("per_setting") or {}).items())
            lines.append(
                f"- {name} (gen {v.get('generation')}, parent {v.get('parent')}): {v.get('status')}"
                + (
                    f"; dJ vs parent {res['mean']:+.3f} (p={res['p_one_sided']:.2f}); {per}"
                    if res.get("n")
                    else ""
                )
                + (f"; hypothesis: {v['hypothesis'][:200]}" if v.get("hypothesis") else "")
                + (f"; verdict: {v['reason']}" if v.get("reason") else "")
            )
        return "\n".join(lines) if lines else "(first generation)"


# ------------------------------------------------------------- meta prompt
def _lookup(cfg: dict[str, Any], dotted: str) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def effective_settings(base_coach: dict[str, Any] | None, overrides: dict[str, Any]) -> str:
    """Current value of every tunable key: the coach preset merged with the
    version's overrides. Shown to the meta-LLM so a "change" is measured
    against the real value, not against a guess of the default."""
    merged = _merge_coach(base_coach, overrides)
    lines = []
    for key in TUNABLE:
        val = _lookup(merged, key)
        changed = _lookup(overrides, key) is not None
        lines.append(f"- {key}: {val}" + ("  (set by this version)" if changed else ""))
    return "\n".join(lines)


def _merge_coach(base_coach: dict[str, Any] | None, overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base_coach or {})
    for k, v in (overrides or {}).items():
        merged[k] = (
            {**merged[k], **v} if isinstance(v, dict) and isinstance(merged.get(k), dict) else v
        )
    return merged


def meta_prompt(
    incumbent: Version,
    inc_summary: dict[str, dict[str, float]],
    inc_diag: str,
    history: str,
    last_candidate: dict[str, Any] | None,
    params: list[str],
    validation_errors: list[str] | None = None,
    base_coach: dict[str, Any] | None = None,
) -> tuple[str, str]:
    tunable = "\n".join(
        f"- {k}: {'one of ' + str(sorted(v)) if isinstance(v, set) else f'[{v[0]}, {v[1]}]'}"
        for k, v in TUNABLE.items()
    )
    system = EVOLVE_SYSTEM.format(
        tunable=tunable,
        system_fields=", ".join(sorted(SYSTEM_FIELDS)),
        user_fields=", ".join(sorted(USER_FIELDS)),
        params=", ".join(params),
    )
    cand_text = "(none yet)"
    if last_candidate:
        cmp = last_candidate.get("compare") or {}
        cand_text = (
            f"{last_candidate.get('name')} — {last_candidate.get('status')}: "
            f"{last_candidate.get('reason', '')}\n"
            f"hypothesis: {last_candidate.get('hypothesis', '')}\n"
            f"{compare_text(cmp)}\n"
            f"{results_table(last_candidate.get('summary') or {})}\n"
            f"coach behaviour:\n{last_candidate.get('diagnostics', '')}"
        )
    user = EVOLVE_USER.format(
        incumbent=incumbent.name,
        incumbent_settings=effective_settings(base_coach, incumbent.overrides or {}),
        incumbent_results=results_table(inc_summary),
        incumbent_diag=inc_diag,
        history=history,
        last_candidate=cand_text,
        system_md=incumbent.system,
        user_md=incumbent.user,
        errors=("\n".join(f"- {e}" for e in validation_errors) if validation_errors else "(none)"),
    )
    return system, user
