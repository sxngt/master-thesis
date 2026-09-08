"""Reward coach: LLM-guided (bi-level) reward shaping during PPO training.

Outer loop (this module), every `interval_steps` env-steps, right after the
periodic deterministic evaluation:

    report  = Observer(KPIs, per-component reward contributions, gait
              descriptors, learning dynamics, intervention history)
    actions = Scheduler.propose(report)            # llm | random | hillclimb | none
    actions = ParamSpace.clip(actions)             # bounds, sign lock, step size
    env.set_reward_params(actions); snapshot(policy)
    ... next report: objective dropped > tolerance -> rollback policy + params

Parameters are flat keys of the vectorised reward (``energy.weight``) plus,
optionally, live algorithm hyperparameters addressed as ``algo.<key>``
(PPO: learning_rate, entropy_coef, clip_range). The coach also keeps the
best-objective policy snapshot; the LLM may ask to restore it
(``restore_best``) when it judges the policy to have collapsed.

Decision layer (coach v5, shared by every scheduler so the LLM/random/
hillclimb comparison stays fair): evaluation uncertainty (binomial SE of the
success rate, sd/sqrt(n) of the means) combined with the process noise of the
J trace; a confirmatory re-evaluation when a drop is ambiguous; detrended
effect attribution of every intervention into a per-parameter ledger with
shrinkage; a saturating learning-curve forecast; a training-phase schedule
(explore / exploit / consolidate); and a curriculum-release invariant that
steps a lowered curriculum lever back to its baseline once the policy
succeeds, so a curriculum reduction can never be forfeited by omission
(coach_stats.py; docs/coach_versions.md).

Asynchronous mode (``coach.async_llm``): the scheduler call runs on a
background thread and training carries on; the proposal is applied at the
first evaluation after it lands (the record keeps ``report_step`` and
``llm_latency_s``). Worth it only when the simulation GPU would otherwise
idle (one run per GPU behind a queued LLM: 1-8 min per call, ~40 % of a run
in 2026-09-07's single-daemon stack). With one LLM daemon per concurrent run
a call takes 40-90 s while an early 1M-step eval interval takes ~20 s, so an
async proposal lands 2-4 evaluations after the report it was reasoned from
(rough-hard 2026-09-08: first breakout 15-16M steps vs 10-11M synchronous);
and with 3 runs sharing the GPU a synchronous call costs no throughput, the
other runs absorb the idle time. Prefer synchronous there.

Inner loop (PPO) is untouched: it only sees R = sum_i w_i r_i with the
current parameters. Thesis mapping: R_LLM == sum_i dw_i(LLM) * r_i, i.e. the
LLM reward is a re-weighting term over the traditional components.

The LLM itself is reached only through translator.LLMClient (CLAUDE.md rule).
Every prompt/response is written verbatim to <run_dir>/coach_log.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quadruped_rl.llm_feedback.coach_stats import (
    EffectLedger,
    calibration,
    detrended_effect,
    fit_saturating,
    linear_trend,
    move_direction,
    objective_se,
    tracking_ratio,
)
from quadruped_rl.llm_feedback.playbook import Playbook
from quadruped_rl.llm_feedback.prompts import COACH_SYSTEM, COACH_USER
from quadruped_rl.llm_feedback.schemas import CoachOutput
from quadruped_rl.llm_feedback.translator import LLMClient, _extract_json

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- space
@dataclass(frozen=True)
class ParamSpec:
    key: str
    low: float
    high: float
    scale: str = "linear"  # linear | log (log: step limit is a multiplicative factor)
    description: str = ""


class ParamSpace:
    """Guardrail: the only parameters a scheduler may touch, with bounds,
    sign lock (bounds never straddle zero) and per-intervention step limits."""

    def __init__(
        self,
        params: dict[str, dict[str, Any]],
        max_rel_change: float = 0.3,
        max_log_factor: float = 3.0,
        max_params: int = 3,
    ):
        self.specs: dict[str, ParamSpec] = {}
        for key, spec in params.items():
            low, high = float(spec["low"]), float(spec["high"])
            if low > high:
                raise ValueError(f"{key}: low > high")
            if low < 0.0 < high:
                raise ValueError(f"{key}: bounds must not straddle zero (sign lock)")
            self.specs[key] = ParamSpec(
                key, low, high, spec.get("scale", "linear"), spec.get("description", "")
            )
        self.max_rel_change = float(max_rel_change)
        self.max_log_factor = float(max_log_factor)
        self.max_params = int(max_params)

    def keys(self) -> list[str]:
        return list(self.specs)

    def clip(
        self, proposals: dict[str, float], current: dict[str, float]
    ) -> tuple[dict[str, float], list[str]]:
        """Return (accepted updates, notes). Unknown keys are dropped; values
        are clipped to bounds and to the per-step change limit; at most
        max_params entries survive (proposal order)."""
        accepted: dict[str, float] = {}
        notes: list[str] = []
        for key, value in proposals.items():
            if key not in self.specs:
                notes.append(f"{key}: not tunable, dropped")
                continue
            if len(accepted) >= self.max_params:
                notes.append(f"{key}: exceeds max {self.max_params} params, dropped")
                continue
            spec = self.specs[key]
            cur = float(current.get(key, value))
            if not math.isfinite(value):
                notes.append(f"{key}: non-finite, dropped")
                continue
            v = self._limit_step(spec, cur, float(value))
            v = min(max(v, spec.low), spec.high)
            if v != value:
                notes.append(f"{key}: {value:.4g} clipped to {v:.4g}")
            if v == cur:
                notes.append(f"{key}: no change")
                continue
            accepted[key] = v
        return accepted, notes

    def _limit_step(self, spec: ParamSpec, cur: float, value: float) -> float:
        if spec.scale == "log" and cur != 0.0 and (value == 0.0 or (value > 0) == (cur > 0)):
            f = self.max_log_factor
            lo, hi = sorted((cur / f, cur * f))
            return min(max(value, lo), hi)
        span = abs(cur) if cur != 0.0 else (spec.high - spec.low)
        step = self.max_rel_change * span
        return min(max(value, cur - step), cur + step)

    def table(self, current: dict[str, float]) -> str:
        rows = ["| key | allowed | current | scale | meaning |", "|---|---|---|---|---|"]
        for k, s in self.specs.items():
            rows.append(
                f"| {k} | [{s.low:g}, {s.high:g}] | {current.get(k, float('nan')):.4g} "
                f"| {s.scale} | {s.description} |"
            )
        return "\n".join(rows)


# ------------------------------------------------------------------ records
@dataclass
class Intervention:
    k: int
    step: int
    scheduler: str
    diagnosis: str
    proposed: dict[str, float]
    applied: dict[str, float]
    params_before: dict[str, float]
    objective_before: float
    kpi_before: dict[str, float]
    notes: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)  # training stats in the report
    expected_effect: str = ""
    confidence: float = 0.0
    objective_after: float | None = None
    kpi_after: dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending | kept | rolled_back | noop
    raw_response_sha: str = ""
    restored_from_step: int | None = None  # best snapshot reloaded before applying
    tolerance_used: float = 0.0  # effective (noise-aware) rollback tolerance
    # coach v5 evidence
    phase: str = ""
    predicted_delta_j: float | None = None  # the scheduler's own forecast of dJ
    effect: float | None = None  # detrended dJ attributed to this intervention
    expected_dj: float | None = None  # dJ the pre-intervention trend predicted
    trend_slope: float | None = None  # pre-intervention dJ per step (settled effect)
    objective_se: float = 0.0  # measurement SE of J at the report
    confirm_evals: int = 0  # extra evaluations spent on an ambiguous decision
    invariant: dict[str, float] = field(default_factory=dict)  # curriculum release
    report_step: int | None = None  # async: the evaluation the report was built from
    llm_latency_s: float = 0.0  # wall-clock time of the scheduler call

    def summary(self) -> str:
        change = ", ".join(
            f"{k}: {self.params_before[k]:.3g}->{v:.3g}" for k, v in self.applied.items()
        )
        if self.restored_from_step is not None:
            change = f"restored best snapshot (step {self.restored_from_step:,}); " + change
        if self.invariant:
            change += "; curriculum release: " + ", ".join(
                f"{k}->{v:.3g}" for k, v in self.invariant.items()
            )
        if not change:
            change = "no change"
        after = "" if self.objective_after is None else f" -> {self.objective_after:.3f}"
        eff = ""
        if self.effect is not None:
            eff = f", effect net of trend {self.effect:+.3f}"
            if self.predicted_delta_j is not None:
                eff += f" (you predicted {self.predicted_delta_j:+.3f})"
        dropped = [n for n in self.notes if "dropped" in n]
        guard = f" [guardrail: {'; '.join(dropped)}]" if dropped else ""
        return (
            f"#{self.k} step {self.step:,} [{self.status}] {change} "
            f"(objective {self.objective_before:.3f}{after}{eff}); {self.diagnosis[:160]}{guard}"
        )


@dataclass
class Proposal:
    actions: dict[str, float]
    diagnosis: str = ""
    restore_best: bool = False
    expected_effect: str = ""
    confidence: float = 0.0
    raw: str = ""
    prompt: dict[str, str] | None = None
    usage: dict[str, int] | None = None
    predicted_delta_j: float | None = None


class ProposalCall:
    """One ``Scheduler.propose`` call, run inline or on a daemon thread so
    training continues while a slow (local) LLM thinks."""

    def __init__(
        self,
        scheduler: Scheduler,
        report: dict[str, Any],
        current: dict[str, float],
        space: ParamSpace,
        step: int,
        k: int,
        stats: dict[str, float],
    ):
        self.scheduler, self.report, self.current, self.space = scheduler, report, current, space
        self.step, self.k, self.stats = step, k, dict(stats)
        self.started_at = 0.0
        self.latency_s = 0.0
        self._result: Proposal | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def run(self) -> None:
        self.started_at = time.time()
        try:
            self._result = self.scheduler.propose(self.report, self.current, self.space)
        except BaseException as e:  # re-raised in the training thread by proposal()
            self._error = e
        self.latency_s = time.time() - self.started_at

    def start(self) -> None:
        # daemon: an unfinished call must not keep the process alive at the end of training
        self._thread = threading.Thread(target=self.run, name="coach-llm", daemon=True)
        self._thread.start()

    def done(self) -> bool:
        return self._thread is None or not self._thread.is_alive()

    def proposal(self) -> Proposal:
        if self._error is not None:
            raise self._error
        assert self._result is not None, "proposal() before run()"
        return self._result


# --------------------------------------------------------------- schedulers
class Scheduler:
    name = "none"

    def propose(
        self, report: dict[str, Any], current: dict[str, float], space: ParamSpace
    ) -> Proposal:
        return Proposal(actions={}, diagnosis="fixed reward")


class RandomScheduler(Scheduler):
    """Control condition: same cadence and step limits, no information."""

    name = "random"

    def __init__(self, seed: int = 0, p_change: float = 1.0):
        self.rng = random.Random(seed)
        self.p_change = p_change

    def propose(self, report, current, space) -> Proposal:
        if self.rng.random() > self.p_change:
            return Proposal(actions={}, diagnosis="random: skip")
        keys = self.rng.sample(space.keys(), k=min(space.max_params, len(space.keys())))
        actions = {}
        for key in keys:
            spec, cur = space.specs[key], current[key]
            if spec.scale == "log" and cur != 0.0:
                actions[key] = cur * math.exp(
                    self.rng.uniform(-1, 1) * math.log(space.max_log_factor)
                )
            else:
                span = abs(cur) if cur != 0.0 else (spec.high - spec.low)
                actions[key] = cur + self.rng.uniform(-1, 1) * space.max_rel_change * span
        return Proposal(actions=actions, diagnosis="random perturbation")


class HillClimbScheduler(Scheduler):
    """LLM-free adaptive control: (1+1)-ES over one parameter at a time.

    Cycles through the tunable keys; an intervention that improved the
    objective repeats the same direction on the same key, one that did not
    (rolled back, or kept within tolerance but not better) is undone — the
    parameter is put back to its pre-move value in the next proposal, the
    direction flips and the search moves on. Without the undo, a move that
    was kept only because the objective did not drop by more than the
    tolerance stayed in place for the rest of training (coach_v2: energy
    weight ratcheted x28 in 6 M steps, standing became optimal). Uses
    exactly the same objective feedback the LLM receives."""

    name = "hillclimb"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._i = 0
        self._dir = 1.0

    def propose(self, report, current, space) -> Proposal:
        keys = space.keys()
        history: list[Intervention] = report.get("history_objs", [])
        undo: dict[str, float] = {}
        if history:
            last = history[-1]
            improved = (
                last.status == "kept"
                and last.objective_after is not None
                and last.objective_after > last.objective_before
            )
            if last.status == "kept" and not improved:
                # the policy stays (it kept learning), only the move is undone
                undo = {k: last.params_before[k] for k in last.applied if k in last.params_before}
            if last.status == "rolled_back" or (last.status == "kept" and not improved):
                self._dir *= -1.0
                self._i = (self._i + 1) % len(keys)
            elif last.status == "noop":
                self._i = (self._i + 1) % len(keys)
        key = keys[self._i]
        spec, cur = space.specs[key], current[key]
        if spec.scale == "log" and cur != 0.0:
            value = cur * (space.max_log_factor if self._dir > 0 else 1.0 / space.max_log_factor)
        else:
            span = abs(cur) if cur != 0.0 else (spec.high - spec.low)
            value = cur + self._dir * space.max_rel_change * span
        actions = {**{k: v for k, v in undo.items() if k != key}, key: value}
        return Proposal(actions=actions, diagnosis=f"hillclimb on {key} dir {self._dir:+.0f}")


class LLMScheduler(Scheduler):
    name = "llm"

    def __init__(
        self,
        llm_cfg: dict[str, Any],
        client: LLMClient | None = None,
        prompt_dir: str | Path | None = None,
    ):
        self.cfg = llm_cfg
        self.client = client or LLMClient(
            llm_cfg.get("provider", "openai"),
            llm_cfg.get("model", "gpt-5.4-2026-03-05"),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            json_mode=True,
            base_url=llm_cfg.get("base_url"),
            api_key_env=llm_cfg.get("api_key_env"),
            timeout_s=llm_cfg.get("timeout_s"),
        )
        # prompt templates of a coach *version* (configs/coach/versions/<v>/):
        # the evolution loop rewrites these files, never this module
        self.system_template, self.user_template = load_prompts(prompt_dir)
        self.max_tokens = int(llm_cfg.get("max_tokens", 16000))
        self.retries = int(llm_cfg.get("retries", 1))
        # transport/API failures (rate limit, exhausted credits, network): retried
        # with backoff, then the report is skipped (no change) rather than
        # crashing a multi-hour training run; the coach log records the error
        # so the analysis can flag the run (coach_v4 lost 15 runs to a 429).
        self.api_retries = int(llm_cfg.get("api_retries", 3))
        self.api_backoff_s = float(llm_cfg.get("api_backoff_s", 30.0))
        self.discarded = 0
        self.api_errors = 0

    def _complete(self, system: str, user: str) -> str:
        for attempt in range(self.api_retries + 1):
            try:
                return self.client.complete(system, user, max_tokens=self.max_tokens)
            except (ValueError, KeyError, json.JSONDecodeError):
                raise
            except Exception as e:  # API/transport errors of any provider
                if attempt >= self.api_retries:
                    raise
                wait = self.api_backoff_s * 2**attempt
                log.warning("LLM API error (%s: %s), retry in %.0fs", type(e).__name__, e, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def propose(self, report, current, space) -> Proposal:
        system = self.system_template.format(
            task_text=report["task_text"],
            component_table=report["component_table"],
            param_table=space.table(current),
            max_params=space.max_params,
            max_rel_change=space.max_rel_change,
            max_log_factor=space.max_log_factor,
            rollback_tolerance=report["rollback_tolerance"],
            objective_text=report["objective_text"],
        )
        user = self.user_template.format(
            step=report["step"],
            progress=report["progress"],
            k=report["k"],
            kpi_table=report["kpi_table"],
            objective_table=report["objective_table"],
            stats_table=report["stats_table"],
            dynamics_table=report["dynamics_table"],
            window=report["window"],
            trace=report["objective_trace"],
            best=report["best_text"],
            noise=report["noise_text"],
            history=report["history_text"],
            evidence=report.get("evidence_text", "(none)"),
            phase=report.get("phase_text", ""),
        )
        raw, parsed, err = "", None, None
        for _ in range(self.retries + 1):
            try:
                raw = self._complete(system, user)
                parsed = CoachOutput.model_validate(_extract_json(raw))
                break
            except (ValidationError, ValueError, json.JSONDecodeError, KeyError) as e:
                err = e  # e.g. reasoning consumed the whole token budget -> empty text
                log.warning("Malformed coach response (%s), retrying", e)
            except Exception as e:  # API failure after retries: skip this report
                self.api_errors += 1
                log.error("LLM API failure, coach report skipped: %s: %s", type(e).__name__, e)
                return Proposal(
                    actions={},
                    diagnosis=f"api error: {type(e).__name__}: {str(e)[:200]}",
                    prompt={"system": system, "user": user},
                )
        if parsed is None:
            self.discarded += 1
            log.warning("Discarded malformed coach response: %s", err)
            return Proposal(
                actions={},
                diagnosis=f"discarded malformed response ({type(err).__name__})",
                raw=raw,
                prompt={"system": system, "user": user},
                usage=dict(self.client.last_usage),
            )
        actions = {a.param: float(a.value) for a in parsed.actions}
        return Proposal(
            actions=actions,
            diagnosis=parsed.diagnosis,
            restore_best=bool(parsed.restore_best),
            expected_effect=parsed.expected_effect,
            confidence=parsed.confidence,
            raw=raw,
            prompt={"system": system, "user": user},
            usage=dict(self.client.last_usage),
            predicted_delta_j=parsed.predicted_delta_j,
        )


def make_scheduler(
    coach_cfg: dict[str, Any], seed: int, client: LLMClient | None = None
) -> Scheduler:
    strategy = coach_cfg.get("strategy", "none")
    if strategy == "none":
        return Scheduler()
    if strategy == "random":
        return RandomScheduler(seed=seed)
    if strategy == "hillclimb":
        return HillClimbScheduler(seed=seed)
    if strategy == "llm":
        return LLMScheduler(
            coach_cfg["llm"], client=client, prompt_dir=coach_cfg.get("version_dir")
        )
    raise ValueError(f"Unknown coach strategy '{strategy}'")


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_version_dir(version_dir: str | Path | None) -> Path | None:
    """`coach.version_dir` is relative to the repo root unless absolute."""
    if not version_dir:
        return None
    d = Path(version_dir)
    return d if d.is_absolute() else REPO_ROOT / d


def load_prompts(prompt_dir: str | Path | None) -> tuple[str, str]:
    """(system, user) templates: ``<dir>/system.md`` / ``<dir>/user.md`` when
    present, else the module defaults (prompts.py)."""
    d = resolve_version_dir(prompt_dir)
    system, user = COACH_SYSTEM, COACH_USER
    if d is not None:
        if (d / "system.md").exists():
            system = (d / "system.md").read_text()
        if (d / "user.md").exists():
            user = (d / "user.md").read_text()
    return system, user


# --------------------------------------------------------------------- coach
class RewardCoach:
    """Outer-loop controller wired into harness.Trainer.

    Call `on_eval(step, total, eval_metrics, train_window)` right after each
    periodic evaluation; it decides whether an intervention is due, resolves
    the pending one (keep / rollback) and applies the next.
    """

    def __init__(
        self,
        coach_cfg: dict[str, Any],
        env,
        algorithm,
        run_dir: str | Path,
        seed: int = 0,
        client: LLMClient | None = None,
        task: dict[str, Any] | None = None,
        evaluate: Callable[[], dict[str, float]] | None = None,
    ):
        self.cfg = coach_cfg
        self.env = env
        self.algorithm = algorithm
        self.evaluate = evaluate  # fresh deterministic evaluation (confirmatory re-eval)
        self.run_dir = Path(run_dir)
        self.task = dict(task or {})
        self.enabled = coach_cfg.get("strategy", "none") != "none"
        self.interval = int(coach_cfg.get("interval_steps", 2_000_000))
        self.warmup = int(coach_cfg.get("warmup_steps", 0))
        self.tolerance = float(coach_cfg.get("rollback_tolerance", 0.05))
        # Noise-aware decisions (coach v4): the fixed tolerance is widened to
        # noise_z * sigma of the evaluation noise estimated from the trace, a
        # "best" must be confirmed by best_confirm consecutive evaluations, and
        # a best snapshot can be restored at most max_restores_per_snapshot
        # times. Batch 2 (deterministic resets, J quantised in 0.25 steps) had
        # the LLM roll back and restore on pure noise: one run restored the
        # same 3M-step snapshot 7 times, discarding 14M steps of training.
        self.noise_window = int(coach_cfg.get("noise_window", 10))
        self.noise_z = float(coach_cfg.get("noise_z", 2.0))
        self.tolerance_max = float(coach_cfg.get("tolerance_max", 0.3))
        self.best_confirm = max(1, int(coach_cfg.get("best_confirm", 2)))
        self.max_restores = int(coach_cfg.get("max_restores_per_snapshot", 1))
        self._restores: dict[int, int] = {}  # best_step -> times restored
        self._trace_mark = 0  # eval_trace index where the current policy lineage starts
        self.objective_w: dict[str, float] = dict(coach_cfg.get("objective", {"success_rate": 1.0}))
        # KPI -> tunable parameter that commands it (e.g. mean velocity -> target
        # speed). The report then states the ceiling that command puts on the
        # objective term: batch 4 (2026-09-05) stairs runs lowered the target
        # to 0.7 m/s as a curriculum and never raised it, forfeiting 0.16 of J.
        self.kpi_command: dict[str, str] = dict(coach_cfg.get("kpi_command", {}))
        self.tracking_ratio = float(coach_cfg.get("tracking_ratio", 0.75))
        # ---- coach v5 decision layer (docs/coach_versions.md)
        self.n_key = str(coach_cfg.get("n_episodes_key", "n_episodes"))
        self.sd_suffix = str(coach_cfg.get("sd_suffix", "_sd"))
        self.trend_points = int(coach_cfg.get("trend_points", 4))
        # Ledger veto: a proposed move on a reward weight whose in-run ledger
        # already says "this direction hurts" (posterior mean + z * sd < 0 over
        # at least ledger_veto_obs observations) is dropped and the model is
        # told so. 0 = off. evolve2 v6 stairs s1 (2026-09-08): the coach kept
        # strengthening stability/termination penalties through five
        # observations of "stability.weight down: -0.085 +- 0.006" and the
        # run never left the floor. Curriculum levers are exempt (their
        # immediate effect is structurally biased; see settled_reports).
        self.ledger_veto_obs = int(coach_cfg.get("ledger_veto_obs", 0))
        self.ledger_veto_z = float(coach_cfg.get("ledger_veto_z", 2.0))
        self.confirm_evals = int(coach_cfg.get("confirm_evals", 1))
        self.confirm_zone = float(coach_cfg.get("confirm_zone", 0.5))
        self.curriculum_params: list[str] = list(coach_cfg.get("curriculum_params", []))
        self.release_from_progress = float(coach_cfg.get("release_from_progress", 0.5))
        self.release_min_success = float(coach_cfg.get("release_min_success", 0.3))
        # Curriculum lock: in the release phase (same gate as the release
        # invariant) a proposal to lower a curriculum lever is dropped. v7's
        # prompt rule ("never lower target_ms on a rising trend") enforced as a
        # guardrail: evolve2 v6 stairs s0 (2026-09-08) lowered target_ms
        # 0.98 -> 0.8 at 34M with success 0.55 and the invariant had to undo it.
        self.curriculum_lock = bool(coach_cfg.get("curriculum_lock", False))
        self.success_key = str(coach_cfg.get("success_key", "success_rate"))
        phases = dict(coach_cfg.get("phases", {}))
        self.phase_exploit = float(phases.get("exploit", 0.5))
        self.phase_consolidate = float(phases.get("consolidate", 0.85))
        self.tracking_prior = float(coach_cfg.get("tracking_ratio_prior", 0.89))
        priors = {
            (p, d): (float(v[0]), float(v[1]))
            for p, dirs in dict(coach_cfg.get("effect_prior", {})).items()
            for d, v in dict(dirs).items()
        }
        self.ledger = EffectLedger(
            prior_sd=float(coach_cfg.get("effect_prior_sd", 0.1)), priors=priors
        )
        self.forecast_min_points = int(coach_cfg.get("forecast_min_points", 4))
        # settled effect of curriculum moves: J this many reports after the
        # move, net of trend (0 = off). The immediate ledger is systematically
        # negative for a raised command (success dips at the next report and
        # recovers at the following one), which made the coach stop releasing
        # the curriculum (rough-hard v6 s1, 2026-09-08: target_ms up "-0.104
        # +- 0.021" while J rose 1.41 -> 1.48 after 1.0 -> 1.1)
        self.settled_reports = int(coach_cfg.get("settled_reports", 0))
        self.playbook = self._load_playbook(coach_cfg.get("playbook") or {})
        self.eval_se: list[float] = []  # measurement SE of J at every eval
        self.space = ParamSpace(
            coach_cfg.get("params", {}),
            max_rel_change=coach_cfg.get("max_rel_change", 0.3),
            max_log_factor=coach_cfg.get("max_log_factor", 3.0),
            max_params=coach_cfg.get("max_params", 3),
        )
        self.scheduler = make_scheduler(coach_cfg, seed, client)
        self.component_docs: dict[str, str] = dict(coach_cfg.get("component_docs", {}))
        self.history: list[Intervention] = []
        self.pending: Intervention | None = None
        self.async_llm = bool(coach_cfg.get("async_llm", False))
        self._inflight: ProposalCall | None = None
        self.cooldown = 0
        self._next = max(self.interval, self.warmup)
        self._prev_kpi: dict[str, float] = {}
        self._snapshot = self.run_dir / "coach_snapshot.pt"
        self._best_snapshot = self.run_dir / "coach_best.pt"
        self._log_path = self.run_dir / "coach_log.jsonl"
        self.best_obj, self.best_step = -math.inf, 0
        self.best_params: dict[str, float] = {}
        self.eval_trace: list[tuple[int, float]] = []  # (step, J) at every eval
        self.baseline: dict[str, float] = {}
        if self.enabled:
            self.baseline = dict(self.current_params())
            unknown = [k for k in self.space.keys() if k not in self.baseline]
            if unknown:
                raise KeyError(f"coach params not tunable (reward or algo.*): {unknown}")
            missing = [k for k in self.curriculum_params if k not in self.space.specs]
            if missing:  # e.g. a config composed for another reward set
                log.warning("curriculum_params not tunable here, ignored: %s", missing)
                self.curriculum_params = [k for k in self.curriculum_params if k not in missing]

    def _load_playbook(self, cfg: dict[str, Any]) -> Playbook | None:
        """Cross-run experience (``coach.playbook: {enabled, path}``); absent or
        unreadable -> the report carries in-run evidence only."""
        if not self.enabled or not cfg.get("enabled"):
            return None
        path = Path(str(cfg.get("path") or "data/results/playbook.json"))
        if not path.exists():
            log.warning("coach.playbook enabled but %s is missing; no cross-run evidence", path)
            return None
        try:
            pb = Playbook.load(path)
        except (OSError, ValueError, KeyError) as e:
            log.warning("coach.playbook %s unreadable (%s); ignored", path, e)
            return None
        log.info("coach playbook: %d runs (%s)", len(pb.cases), pb.built)
        return pb

    def _setting(self) -> str:
        return (
            f"{self.task.get('terrain', '?')}-{self.task.get('level', 'easy')}-"
            f"{self.task.get('reward', 'traditional')}"
        )

    # --------------------------------------------------------- parameters
    def current_params(self) -> dict[str, float]:
        """Reward parameters plus live algorithm hyperparameters (``algo.*``)."""
        out = dict(self.env.reward_params())
        out.update({f"algo.{k}": v for k, v in self.algorithm.hyperparams().items()})
        return out

    def apply_params(self, updates: dict[str, float]) -> None:
        reward = {k: v for k, v in updates.items() if not k.startswith("algo.")}
        algo = {k[len("algo.") :]: v for k, v in updates.items() if k.startswith("algo.")}
        if reward:
            self.env.set_reward_params(reward)
        if algo:
            self.algorithm.set_hyperparams(algo)

    # ---------------------------------------------------------- objective
    def objective(self, kpi: dict[str, float]) -> float:
        return float(sum(w * float(kpi.get(m, 0.0)) for m, w in self.objective_w.items()))

    def noise_sigma(self) -> float:
        """Robust per-evaluation noise of J from second differences of the trace.

        Second differences cancel a linear learning trend; for iid noise of
        std s they have std sqrt(6) s. MAD-based so a single jump (e.g. the
        effect of an intervention, which contaminates two second differences)
        does not dominate; 0 until five second differences are available so
        the median is not carried by such a jump.
        """
        vals = [j for _, j in self.eval_trace[-(self.noise_window + 2) :]]
        d2 = [vals[i] - 2 * vals[i + 1] + vals[i + 2] for i in range(len(vals) - 2)]
        if len(d2) < 5:
            return 0.0
        mad = sorted(abs(x) for x in d2)[len(d2) // 2]
        return 1.4826 * mad / math.sqrt(6)

    def measurement_se(self, kpi: dict[str, float]) -> float:
        """SE of J from one evaluation (binomial success rate, sd/sqrt(n) means)."""
        return objective_se(
            kpi, self.objective_w, int(kpi.get(self.n_key, 0)), sd_suffix=self.sd_suffix
        )

    def sigma(self) -> float:
        """Per-evaluation sd of J: the larger of the process-noise estimate from
        the trace and the measurement SE of the latest evaluation (the latter is
        available from the first evaluation, the former needs seven)."""
        se = self.eval_se[-1] if self.eval_se else 0.0
        return max(self.noise_sigma(), se)

    def effective_tolerance(self) -> float:
        """Rollback/restore threshold: max(fixed tolerance, noise_z sigma of a J difference)."""
        tol = max(self.tolerance, self.noise_z * math.sqrt(2) * self.sigma())
        return min(tol, self.tolerance_max)

    def phase(self, progress: float) -> str:
        if progress >= self.phase_consolidate:
            return "consolidate"
        return "exploit" if progress >= self.phase_exploit else "explore"

    def objective_rows(self, kpi: dict[str, float], current: dict[str, float]) -> dict[str, str]:
        """Per-term contribution to J, with the ceiling implied by a commanding parameter."""
        rows: dict[str, str] = {}
        for m, w in self.objective_w.items():
            v = float(kpi.get(m, 0.0))
            text = f"{w:+g} * {v:.3f} = {w * v:+.3f}"
            cmd = self.kpi_command.get(m)
            if cmd is not None and cmd in current:
                target = float(current[cmd])
                ratio = v / target if target else float("nan")
                text += f"; commanded by {cmd} = {target:g} (measured/commanded = {ratio:.2f})"
                if ratio >= self.tracking_ratio:
                    text += (
                        f" - the policy tracks its command, so this term is capped near "
                        f"{w * target:+.3f} until {cmd} is raised"
                    )
                else:
                    text += (
                        f" - not yet tracking; the ceiling at this command would be "
                        f"{w * target:+.3f}"
                    )
            rows[m] = text
        rows["J"] = f"{self.objective(kpi):.4f}"
        return rows

    def objective_text(self) -> str:
        terms = " + ".join(f"{w:+g} * {m}" for m, w in self.objective_w.items())
        return (
            f"maximise J = {terms} (deterministic policy, {self.cfg.get('eval_note', '')})".strip()
        )

    # ---------------------------------------------------------------- hook
    def on_eval(
        self,
        step: int,
        total: int,
        eval_metrics: dict[str, float],
        train_window: list[dict[str, float]],
    ) -> dict[str, float]:
        """Returns scalar metrics to log under coach/ (empty if nothing ran)."""
        if not self.enabled:
            return {}
        obj = self.objective(eval_metrics)
        self.eval_trace.append((step, obj))
        self.eval_se.append(self.measurement_se(eval_metrics))
        if self._inflight is not None:
            # async: training went on while the scheduler was thinking; apply
            # the proposal at the first evaluation after it lands
            if not self._inflight.done():
                self._track_best(step, obj)
                return {}
            call, self._inflight = self._inflight, None
            self._track_best(step, obj)
            out: dict[str, float] = {"objective": obj}
            out["tolerance"] = self.effective_tolerance()
            out["objective_se"] = self.eval_se[-1]
            out.update(self._apply(step, total, obj, eval_metrics, call))
            # next report on the interval grid, not ``step + interval``:
            # evaluations fall a few 10k steps past each multiple of the
            # eval interval, so an off-grid anchor missed every other slot
            # (2026-09-08: 8 reports in 32M steps instead of 16)
            self._next = (step // self.interval + 1) * self.interval
            return out
        if step < self._next:
            self._track_best(step, obj)
            return {}
        self._next += self.interval
        out = {"objective": obj}

        # 1) resolve the pending intervention with this fresh evaluation
        rolled_back = False
        tol = self.effective_tolerance()
        out["tolerance"] = tol
        out["objective_se"] = self.eval_se[-1]
        if self.pending is not None:
            p = self.pending
            acted = bool(p.applied) or p.restored_from_step is not None
            obj, eval_metrics = self._resolve(p, step, obj, eval_metrics, tol, acted)
            if p.status == "rolled_back":
                self.apply_params(p.params_before)
                self.algorithm.load(self._snapshot)
                rolled_back = True
                self._trace_mark = len(self.eval_trace)
                self.cooldown = int(self.cfg.get("cooldown_after_rollback", 1))
                out["rolled_back"] = 1.0
            if p.effect is not None:
                out["effect"] = p.effect
            if p.confirm_evals:
                out["confirm_evals"] = float(p.confirm_evals)
            self._write(p)
            self.pending = None
        if not rolled_back:
            self._track_best(step, obj)

        stats = self.env.training_stats()  # harvested every interval regardless
        if self.cooldown > 0:
            self.cooldown -= 1
            self._prev_kpi = dict(eval_metrics)
            out["cooldown"] = float(self.cooldown + 1)
            return out

        # 2) report -> propose (inline or in the background) -> apply
        current = self.current_params()
        k = len(self.history) + 1
        report = self._report(step, total, k, eval_metrics, stats, train_window, current)
        call = ProposalCall(self.scheduler, report, current, self.space, step, k, stats)
        if self.async_llm:
            call.start()
            self._inflight = call
            self._next = math.inf  # no new report until this proposal has been applied
            self._prev_kpi = dict(eval_metrics)
            out["llm_inflight"] = 1.0
            return out
        call.run()
        out.update(self._apply(step, total, obj, eval_metrics, call))
        return out

    def _apply(
        self,
        step: int,
        total: int,
        obj: float,
        eval_metrics: dict[str, float],
        call: ProposalCall,
    ) -> dict[str, float]:
        """Guardrail -> phase gate -> curriculum invariant -> apply the
        proposal of ``call`` at the current evaluation (the report's own
        evaluation when synchronous, a later one when asynchronous)."""
        out: dict[str, float] = {}
        proposal = call.proposal()
        current = self.current_params()
        params_before = {key: current[key] for key in self.space.keys()}
        k, stats = call.k, call.stats
        progress = step / max(total, 1)
        phase = self.phase(progress)
        tol = self.effective_tolerance()
        restored: int | None = None
        notes: list[str] = []
        if step != call.step:
            notes.append(
                f"async: proposed from the report at step {call.step:,} "
                f"(LLM latency {call.latency_s:.0f} s), applied at step {step:,}"
            )
            if self.phase(call.step / max(total, 1)) != phase:
                notes.append(f"phase moved to {phase} since the report")
        if proposal.restore_best:
            # honoured only for a real collapse (same tolerance as rollback),
            # so an over-eager "restore" cannot undo ordinary learning progress,
            # and at most max_restores times per snapshot so the run cannot be
            # pinned to one early policy
            if not self._best_snapshot.exists() or self.best_obj - obj <= tol:
                notes.append(f"restore_best: J not below best by more than {tol:.3f}, ignored")
            elif self._restores.get(self.best_step, 0) >= self.max_restores:
                notes.append(
                    f"restore_best: snapshot at step {self.best_step:,} already restored "
                    f"{self.max_restores}x, ignored (train through instead)"
                )
            else:
                self.algorithm.save(self._snapshot)  # rollback target = pre-restore state
                self.algorithm.load(self._best_snapshot)
                self.apply_params(self.best_params)
                current = self.current_params()
                restored = self.best_step
                self._restores[self.best_step] = self._restores.get(self.best_step, 0) + 1
                self._trace_mark = len(self.eval_trace)
                out["restored"] = 1.0
        proposed = dict(proposal.actions)
        if phase == "consolidate":
            # the last part of the budget consolidates the policy: reward-shaping
            # changes no longer have time to pay off, only curriculum release
            # and optimiser settings (e.g. a learning-rate decay) go through
            blocked = [
                key
                for key in proposed
                if key not in self.curriculum_params and not key.startswith("algo.")
            ]
            for key in blocked:
                proposed.pop(key)
                notes.append(f"{key}: reward changes are frozen in the consolidate phase, dropped")
        if self.curriculum_lock and self._release_ready(progress, eval_metrics):
            for key in self.curriculum_params:
                try:
                    lowered = key in proposed and float(proposed[key]) < float(current[key])
                except (TypeError, ValueError):
                    lowered = False
                if lowered:
                    proposed.pop(key)
                    notes.append(
                        f"{key}: lowering a curriculum lever is locked in the release phase "
                        f"(success >= {self.release_min_success:g} for 2 reports), dropped"
                    )
        for key in list(proposed):
            veto = self._ledger_veto(key, current.get(key), proposed[key])
            if veto:
                proposed.pop(key)
                notes.append(veto)
        applied, clip_notes = self.space.clip(proposed, current)
        notes += clip_notes
        invariant = self._curriculum_release(progress, current, applied, eval_metrics, notes)
        applied.update(invariant)
        rec = Intervention(
            k=k,
            step=step,
            scheduler=self.scheduler.name,
            diagnosis=proposal.diagnosis,
            proposed=dict(proposal.actions),
            applied=applied,
            params_before=params_before,
            objective_before=obj,
            kpi_before=dict(eval_metrics),
            notes=notes,
            stats=dict(stats),
            expected_effect=proposal.expected_effect,
            confidence=proposal.confidence,
            raw_response_sha=hashlib.sha256(proposal.raw.encode()).hexdigest()[:16]
            if proposal.raw
            else "",
            restored_from_step=restored,
            phase=phase,
            predicted_delta_j=proposal.predicted_delta_j,
            objective_se=self.eval_se[-1],
            invariant=invariant,
            report_step=call.step,
            llm_latency_s=call.latency_s,
        )
        if applied:
            if restored is None:
                self.algorithm.save(self._snapshot)
            self.apply_params(applied)
        self.history.append(rec)
        self.pending = rec
        self._prev_kpi = dict(eval_metrics)
        self._write(rec, proposal)
        out["intervention"] = float(k)
        out["n_applied"] = float(len(applied))
        out["confidence"] = float(proposal.confidence)
        if invariant:
            out["curriculum_release"] = 1.0
        if proposal.usage:
            out.update({f"tokens_{k_}": float(v) for k_, v in proposal.usage.items()})
        out.update({f"param/{key}": v for key, v in self.current_params().items()})
        return out

    def finish(self) -> None:
        """End of training: a proposal still in flight can no longer be applied."""
        if self._inflight is not None:
            log.warning(
                "coach: proposal from step %s still in flight at the end of training, dropped",
                f"{self._inflight.step:,}",
            )
            self._inflight = None

    # ------------------------------------------------------------ decisions
    def _resolve(
        self,
        p: Intervention,
        step: int,
        obj: float,
        kpi: dict[str, float],
        tol: float,
        acted: bool,
    ) -> tuple[float, dict[str, float]]:
        """Keep / roll back the pending intervention; returns the (possibly
        pooled) objective and KPIs used for the decision.

        A drop deeper than the tolerance rolls back. A drop inside the
        ambiguous zone (more than confirm_zone * tolerance but not beyond it)
        triggers up to confirm_evals fresh evaluations whose mean replaces the
        single noisy measurement (sequential test: the sd of the mean of m
        evaluations is sigma / sqrt(m)). The effect of the intervention is
        recorded net of the pre-intervention learning trend and enters the
        parameter ledger.
        """
        dj = obj - p.objective_before
        if acted and self.evaluate is not None and self.confirm_evals > 0:
            if -tol <= dj < -self.confirm_zone * tol:
                pooled = [kpi]
                for _ in range(self.confirm_evals):
                    pooled.append(dict(self.evaluate()))
                kpi = _mean_dicts(pooled)
                obj = self.objective(kpi)
                p.confirm_evals = len(pooled) - 1
                self.eval_trace[-1] = (step, obj)
                self.eval_se[-1] = self.measurement_se(kpi) / math.sqrt(len(pooled))
                dj = obj - p.objective_before
        p.objective_after, p.kpi_after, p.tolerance_used = obj, dict(kpi), tol
        if acted:
            before = [t for t in self.eval_trace[self._trace_mark : -1] if t[0] <= p.step]
            if before:
                p.effect, p.expected_dj = detrended_effect(before, step, obj, self.trend_points)
                p.trend_slope = linear_trend(before, self.trend_points)[0]
                obs_sd = math.sqrt(2.0) * max(self.sigma(), 1e-6)
                for key, v in p.applied.items():
                    self.ledger.add(key, move_direction(p.params_before[key], v), p.effect, obs_sd)
            p.status = "rolled_back" if dj < -tol else "kept"
        else:
            p.status = "noop"
        return obj, kpi

    def _curriculum_release(
        self,
        progress: float,
        current: dict[str, float],
        applied: dict[str, float],
        kpi: dict[str, float],
        notes: list[str],
    ) -> dict[str, float]:
        """Invariant: a curriculum lever lowered below its baseline is stepped
        back up (one guardrail step per report, still subject to rollback) once
        the policy succeeds and the budget is past release_from_progress,
        unless the scheduler itself moved it this report. Batch 4 (2026-09-05):
        every stairs run lowered target_ms to 0.7 and never raised it,
        forfeiting the capped velocity term for good.
        """
        release: dict[str, float] = {}
        if not self._release_ready(progress, kpi):
            return release
        success = float(kpi.get(self.success_key, 0.0))
        for key in self.curriculum_params:
            base, cur = self.baseline[key], float(current[key])
            if key in applied or cur >= base - 1e-9:
                continue
            spec = self.space.specs[key]
            step_up = self.space._limit_step(spec, cur, base)
            value = min(max(step_up, spec.low), spec.high)
            if value > cur:
                release[key] = value
                notes.append(
                    f"{key}: curriculum release {cur:.3g}->{value:.3g} towards baseline "
                    f"{base:.3g} (success {success:.2f} >= {self.release_min_success})"
                )
        return release

    def _release_ready(self, progress: float, kpi: dict[str, float]) -> bool:
        """Release-phase gate: past release_from_progress and success at or
        above release_min_success at this and the previous report."""
        if progress < self.release_from_progress:
            return False
        recent = [h.kpi_before.get(self.success_key, 0.0) for h in self.history[-1:]]
        recent.append(float(kpi.get(self.success_key, 0.0)))
        return min(recent) >= self.release_min_success

    def _track_best(self, step: int, obj: float) -> None:
        # a best must be confirmed by best_confirm consecutive evaluations of the
        # same policy lineage (mean), so a single noisy spike is not a best
        lineage = [j for _, j in self.eval_trace[self._trace_mark :]]
        if len(lineage) < self.best_confirm:
            return
        recent = lineage[-self.best_confirm :]
        conf = sum(recent) / len(recent)
        if conf > self.best_obj:
            self.best_obj, self.best_step = conf, step
            self.best_params = {key: v for key, v in self.current_params().items()}
            self.algorithm.save(self._best_snapshot)

    # -------------------------------------------------------------- report
    def _report(self, step, total, k, kpi, stats, train_window, current) -> dict[str, Any]:
        def table(rows: dict[str, Any]) -> str:
            return "\n".join(f"- {key}: {_fmt(v)}" for key, v in rows.items()) or "- (none)"

        kpi_rows = {}
        for key, v in kpi.items():
            if key == self.n_key or key.endswith(self.sd_suffix):
                continue  # used for the uncertainty estimate, not a KPI
            prev = self._prev_kpi.get(key)
            kpi_rows[key] = f"{_fmt(v)}" + ("" if prev is None else f" (prev {_fmt(prev)})")
        kpi_rows["objective J"] = f"{self.objective(kpi):.4f}" + (
            f" (prev {self.objective(self._prev_kpi):.4f})" if self._prev_kpi else ""
        )
        dyn: dict[str, float] = {}
        if train_window:
            keys = {key for m in train_window for key in m}
            for key in sorted(keys):
                vals = [m[key] for m in train_window if key in m]
                dyn[key] = sum(vals) / len(vals)
        comp_rows = (
            "\n".join(f"- {name}: {doc}" for name, doc in self.component_docs.items())
            or "- (see parameter table)"
        )
        history_text = "\n".join(h.summary() for h in self.history[-12:]) or "(none yet)"
        trace = ", ".join(f"{s / 1e6:.1f}M: {j:.3f}" for s, j in self.eval_trace[-16:])
        tol = self.effective_tolerance()
        if self._best_snapshot.exists():
            left = self.max_restores - self._restores.get(self.best_step, 0)
            best_text = (
                f"J = {self.best_obj:.3f} (mean of {self.best_confirm} consecutive "
                f"evaluations) at step {self.best_step:,}; snapshot restorable "
                f"{max(left, 0)} more time(s)"
            )
        else:
            best_text = "(none yet)"
        noise_text = (
            f"sd of J per evaluation ~ {self.sigma():.3f} (measurement SE of this "
            f"evaluation {self.eval_se[-1]:.3f} from {int(kpi.get(self.n_key, 0))} "
            f"episodes; process noise from the last {self.noise_window} evaluations "
            f"{self.noise_sigma():.3f}); a difference between two evaluations is "
            f"significant beyond ~{tol:.3f}. Effective rollback/restore threshold "
            f"now: {tol:.3f}"
        )
        progress = step / max(total, 1)
        phase = self.phase(progress)
        reports_left = max(int((total - step) // self.interval), 0)
        phase_text = (
            f"Phase: {phase} ({progress:.0%} of budget, ~{reports_left} reports left; "
            f"exploit from {self.phase_exploit:.0%}, consolidate from "
            f"{self.phase_consolidate:.0%} - in consolidate only "
            f"{', '.join(self.curriculum_params) or 'no curriculum lever'} and algo.* "
            f"may change)"
        )
        task = self.task
        task_text = (
            f"terrain: {task.get('terrain', '?')} (level {task.get('level', '?')}); "
            f"success = travel {task.get('course_length_m', 5.0):g} m from the spawn point "
            f"within {task.get('episode_length_s', 20.0):g} s (reaching it ends the episode); "
            f"an episode also ends on a fall (base contact or tipping over). "
            f"Algorithm: {task.get('algorithm', 'PPO')}, {task.get('num_envs', '?')} "
            f"parallel environments."
        )
        return {
            "step": step,
            "progress": step / max(total, 1),
            "k": k,
            "window": len(train_window),
            "kpi_table": table(kpi_rows),
            "objective_table": table(self.objective_rows(kpi, current)),
            "stats_table": table(stats),
            "dynamics_table": table(dyn),
            "history_text": history_text,
            "objective_trace": trace or "(none yet)",
            "best_text": best_text,
            "noise_text": noise_text,
            "task_text": task_text,
            "evidence_text": self._evidence(step, total, kpi, current),
            "phase_text": phase_text,
            "history_objs": self.history,
            "component_table": comp_rows,
            "objective_text": self.objective_text(),
            "rollback_tolerance": tol,
            "current": current,
        }

    def _command_pairs(self, kpi_key: str, cmd: str) -> list[tuple[float, float, float]]:
        """(commanded, measured, success) at every past report plus the current one."""
        pairs = []
        for h in self.history:
            if cmd in h.params_before and kpi_key in h.kpi_before:
                pairs.append(
                    (
                        float(h.params_before[cmd]),
                        float(h.kpi_before[kpi_key]),
                        float(h.kpi_before.get(self.success_key, 0.0)),
                    )
                )
        return pairs

    def _evidence(
        self, step: int, total: int, kpi: dict[str, float], current: dict[str, float]
    ) -> str:
        """Computed evidence for the report: forecast, headroom, ledger, calibration."""
        lines: list[str] = []
        # learning-curve forecast of the current lineage
        lineage = self.eval_trace[self._trace_mark :]
        fit = fit_saturating(lineage, self.forecast_min_points)
        if fit is not None and fit.tau_steps > 0:
            lines.append(
                f"- Learning curve (this policy lineage, {fit.n} evaluations): J -> "
                f"{fit.asymptote:.3f} asymptotically (time constant {fit.tau_steps / 1e6:.1f}M "
                f"steps, R^2 {fit.r2:.2f}); projected J at the end of the budget with no "
                f"further change: {fit.predict(total):.3f}"
            )
        else:
            lines.append("- Learning curve: too few evaluations of this lineage to fit yet")
        # headroom per objective term
        ceiling = 0.0
        for m, w in self.objective_w.items():
            v = float(kpi.get(m, 0.0))
            cmd = self.kpi_command.get(m)
            if cmd is not None and cmd in current and cmd in self.space.specs:
                pairs = self._command_pairs(m, cmd) + [
                    (float(current[cmd]), v, float(kpi.get(self.success_key, 0.0)))
                ]
                fitted = tracking_ratio(pairs)
                ratio, n = fitted if fitted else (self.tracking_prior, 0)
                src = f"fitted on {n} reports" if n else "prior from earlier batches"
                hi = self.space.specs[cmd].high
                cap_now, cap_max = w * ratio * float(current[cmd]), w * ratio * hi
                ceiling += cap_max
                lines.append(
                    f"- {m}: {w * v:+.3f} now; ceiling {cap_now:+.3f} at {cmd} = "
                    f"{current[cmd]:g}, {cap_max:+.3f} at its maximum {hi:g} (tracking "
                    f"ratio {ratio:.2f}, {src}); each +0.1 of {cmd} is worth up to "
                    f"{w * ratio * 0.1:+.3f} J if success holds"
                )
                seen = sorted({(round(c, 2), round(vv, 2), round(sr, 2)) for c, vv, sr in pairs})
                if len(seen) > 1:
                    lines.append(
                        "  observed (commanded, measured, success): "
                        + ", ".join(f"({c:g}, {vv:.2f}, {sr:.2f})" for c, vv, sr in seen[-8:])
                    )
            elif m == self.success_key:
                ceiling += w
                lines.append(
                    f"- {m}: {w * v:+.3f} now; ceiling {w:+.3f} (headroom {w * (1 - v):+.3f})"
                )
            else:
                lines.append(f"- {m}: {w * v:+.3f} now")
        lines.append(f"- Objective ceiling under the parameter bounds: J <= {ceiling:.3f}")
        # ledger of what past moves did in this run
        rows = self.ledger.rows()
        if rows:
            lines.append("- Effect of past moves in this run (posterior mean +- sd, net of trend):")
            lines += [f"  {r}" for r in rows]
        rows = self._settled_rows()
        if rows:
            lines.append(
                f"- Settled effect of curriculum moves (J {self.settled_reports} reports after "
                "the move, net of trend; a raised command costs success at the next report "
                "and pays at the following ones, so weigh this row, not the immediate one, "
                "for a curriculum lever):"
            )
            lines += [f"  {r}" for r in rows]
        # calibration of the scheduler's own predictions
        pairs = [
            (h.predicted_delta_j, h.effect)
            for h in self.history
            if h.predicted_delta_j is not None and h.effect is not None
        ]
        cal = calibration(pairs)
        if cal["n"]:
            lines.append(
                f"- Your dJ predictions so far: sign correct {cal['sign_accuracy']:.0%} of "
                f"{cal['n']}, mean abs error {cal['mae']:.3f}, bias {cal['bias']:+.3f} "
                f"(positive = you over-predict)"
            )
        if self.playbook is not None:
            lines.append(
                self.playbook.evidence(
                    self._setting(),
                    float(kpi.get(self.success_key, 0.0)),
                    exclude_run=self.run_dir.name,
                )
            )
        return "\n".join(lines)

    def _ledger_veto(self, key: str, current: float | None, value: float) -> str | None:
        """Reason to drop a proposed move because this run's ledger already
        shows the direction hurting, or None."""
        if self.ledger_veto_obs <= 0 or current is None or key in self.curriculum_params:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        if v == current:
            return None
        direction = move_direction(current, v)
        mean, sd, n = self.ledger.posterior(key, direction)
        if n < self.ledger_veto_obs or mean + self.ledger_veto_z * sd >= 0:
            return None
        return (
            f"{key} {direction}: vetoed by the ledger (effect on J {mean:+.3f} +- {sd:.3f} "
            f"over {n} obs of this run), dropped"
        )

    def _settled_rows(self) -> list[str]:
        """Ledger of curriculum moves read ``settled_reports`` reports after each
        kept move (first evaluation at or past move step + k * interval), net of
        the pre-move trend. A move whose lever was touched again, or whose
        lineage was rolled back or restored, before that evaluation is skipped
        (its window is not clean); a move too recent to have settled is
        skipped as well."""
        k = self.settled_reports
        if k <= 0 or not self.curriculum_params:
            return []
        ledger = EffectLedger(prior_sd=self.ledger.prior_sd)
        obs_sd = math.sqrt(2.0) * max(self.sigma(), 1e-6)
        for i, h in enumerate(self.history):
            if h.status != "kept" or h.trend_slope is None:
                continue
            keys = [p for p in h.applied if p in self.curriculum_params]
            if not keys:
                continue
            end = h.step + k * self.interval
            clean = True
            for later in self.history[i + 1 :]:
                if later.step >= end:
                    break
                if (
                    later.status == "rolled_back"
                    or later.restored_from_step is not None
                    or any(p in later.applied for p in keys)
                ):
                    clean = False
                    break
            settled = [t for t in self.eval_trace if t[0] >= end]
            if not clean or not settled:
                continue
            s_end, j_end = settled[0]
            effect = j_end - h.objective_before - h.trend_slope * (s_end - h.step)
            for p in keys:
                ledger.add(p, move_direction(h.params_before[p], h.applied[p]), effect, obs_sd)
        return [r.replace("effect on J", "settled effect on J") for r in ledger.rows()]

    def _write(self, rec: Intervention, proposal: Proposal | None = None) -> None:
        entry: dict[str, Any] = {"time": time.time(), **asdict(rec)}
        if proposal is not None:
            entry["prompt"] = proposal.prompt
            entry["raw_response"] = proposal.raw
            entry["usage"] = proposal.usage
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _mean_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = {k for d in dicts for k in d}
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(d[k]) for d in dicts if k in d]
        out[k] = sum(vals) / len(vals)
    return out
