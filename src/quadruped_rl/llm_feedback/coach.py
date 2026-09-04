"""Reward coach: LLM-guided (bi-level) reward shaping during PPO training.

Outer loop (this module), every `interval_steps` env-steps, right after the
periodic deterministic evaluation:

    report  = Observer(KPIs, per-component reward contributions, gait
              descriptors, learning dynamics, intervention history)
    actions = Scheduler.propose(report)            # llm | random | hillclimb | none
    actions = ParamSpace.clip(actions)             # bounds, sign lock, step size
    env.set_reward_params(actions); snapshot(policy)
    ... next report: objective dropped > tolerance -> rollback policy + params

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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

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

    def summary(self) -> str:
        change = ", ".join(
            f"{k}: {self.params_before[k]:.3g}->{v:.3g}" for k, v in self.applied.items()
        )
        if not change:
            change = "no change"
        after = "" if self.objective_after is None else f" -> {self.objective_after:.3f}"
        return (
            f"#{self.k} step {self.step:,} [{self.status}] {change} "
            f"(objective {self.objective_before:.3f}{after}); {self.diagnosis[:160]}"
        )


@dataclass
class Proposal:
    actions: dict[str, float]
    diagnosis: str = ""
    expected_effect: str = ""
    confidence: float = 0.0
    raw: str = ""
    prompt: dict[str, str] | None = None
    usage: dict[str, int] | None = None


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

    Cycles through the tunable keys; a kept intervention repeats the same
    direction on the same key, a rollback flips direction and moves on.
    Uses exactly the same objective/rollback feedback the LLM receives."""

    name = "hillclimb"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self._i = 0
        self._dir = 1.0

    def propose(self, report, current, space) -> Proposal:
        keys = space.keys()
        history: list[Intervention] = report.get("history_objs", [])
        if history:
            last = history[-1]
            if last.status == "rolled_back":
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
        return Proposal(actions={key: value}, diagnosis=f"hillclimb on {key} dir {self._dir:+.0f}")


class LLMScheduler(Scheduler):
    name = "llm"

    def __init__(self, llm_cfg: dict[str, Any], client: LLMClient | None = None):
        self.cfg = llm_cfg
        self.client = client or LLMClient(
            llm_cfg.get("provider", "openai"),
            llm_cfg.get("model", "gpt-5.4-2026-03-05"),
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            json_mode=True,
        )
        self.max_tokens = int(llm_cfg.get("max_tokens", 4000))
        self.discarded = 0

    def propose(self, report, current, space) -> Proposal:
        system = COACH_SYSTEM.format(
            component_table=report["component_table"],
            param_table=space.table(current),
            max_params=space.max_params,
            max_rel_change=space.max_rel_change,
            max_log_factor=space.max_log_factor,
            rollback_tolerance=report["rollback_tolerance"],
            objective_text=report["objective_text"],
        )
        user = COACH_USER.format(
            step=report["step"],
            progress=report["progress"],
            k=report["k"],
            kpi_table=report["kpi_table"],
            stats_table=report["stats_table"],
            dynamics_table=report["dynamics_table"],
            window=report["window"],
            history=report["history_text"],
        )
        raw = ""
        try:
            raw = self.client.complete(system, user, max_tokens=self.max_tokens)
            parsed = CoachOutput.model_validate(_extract_json(raw))
        except (ValidationError, ValueError, json.JSONDecodeError, KeyError) as e:
            self.discarded += 1
            log.warning("Discarded malformed coach response: %s", e)
            return Proposal(
                actions={},
                diagnosis=f"discarded malformed response ({type(e).__name__})",
                raw=raw,
                prompt={"system": system, "user": user},
                usage=dict(self.client.last_usage),
            )
        actions = {a.param: float(a.value) for a in parsed.actions}
        return Proposal(
            actions=actions,
            diagnosis=parsed.diagnosis,
            expected_effect=parsed.expected_effect,
            confidence=parsed.confidence,
            raw=raw,
            prompt={"system": system, "user": user},
            usage=dict(self.client.last_usage),
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
        return LLMScheduler(coach_cfg["llm"], client=client)
    raise ValueError(f"Unknown coach strategy '{strategy}'")


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
    ):
        self.cfg = coach_cfg
        self.env = env
        self.algorithm = algorithm
        self.run_dir = Path(run_dir)
        self.enabled = coach_cfg.get("strategy", "none") != "none"
        self.interval = int(coach_cfg.get("interval_steps", 2_000_000))
        self.warmup = int(coach_cfg.get("warmup_steps", 0))
        self.tolerance = float(coach_cfg.get("rollback_tolerance", 0.05))
        self.objective_w: dict[str, float] = dict(coach_cfg.get("objective", {"success_rate": 1.0}))
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
        self.cooldown = 0
        self._next = max(self.interval, self.warmup)
        self._prev_kpi: dict[str, float] = {}
        self._snapshot = self.run_dir / "coach_snapshot.pt"
        self._log_path = self.run_dir / "coach_log.jsonl"
        if self.enabled:
            unknown = [k for k in self.space.keys() if k not in env.reward_params()]
            if unknown:
                raise KeyError(f"coach params not in env reward: {unknown}")

    # ---------------------------------------------------------- objective
    def objective(self, kpi: dict[str, float]) -> float:
        return float(sum(w * float(kpi.get(m, 0.0)) for m, w in self.objective_w.items()))

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
        if not self.enabled or step < self._next:
            return {}
        self._next += self.interval
        obj = self.objective(eval_metrics)
        out: dict[str, float] = {"objective": obj}

        # 1) resolve the pending intervention with this fresh evaluation
        if self.pending is not None:
            p = self.pending
            p.objective_after, p.kpi_after = obj, dict(eval_metrics)
            if p.applied and obj < p.objective_before - self.tolerance:
                self.env.set_reward_params(p.params_before)
                self.algorithm.load(self._snapshot)
                p.status = "rolled_back"
                self.cooldown = int(self.cfg.get("cooldown_after_rollback", 1))
                out["rolled_back"] = 1.0
            else:
                p.status = "kept" if p.applied else "noop"
            self._write(p)
            self.pending = None

        stats = self.env.training_stats()  # harvested every interval regardless
        if self.cooldown > 0:
            self.cooldown -= 1
            self._prev_kpi = dict(eval_metrics)
            out["cooldown"] = float(self.cooldown + 1)
            return out

        # 2) propose -> guardrail -> apply
        current = self.env.reward_params()
        k = len(self.history) + 1
        report = self._report(step, total, k, eval_metrics, stats, train_window, current)
        proposal = self.scheduler.propose(report, current, self.space)
        applied, notes = self.space.clip(proposal.actions, current)
        rec = Intervention(
            k=k,
            step=step,
            scheduler=self.scheduler.name,
            diagnosis=proposal.diagnosis,
            proposed=dict(proposal.actions),
            applied=applied,
            params_before={key: current[key] for key in self.space.keys()},
            objective_before=obj,
            kpi_before=dict(eval_metrics),
            notes=notes,
            stats=dict(stats),
            expected_effect=proposal.expected_effect,
            confidence=proposal.confidence,
            raw_response_sha=hashlib.sha256(proposal.raw.encode()).hexdigest()[:16]
            if proposal.raw
            else "",
        )
        if applied:
            self.algorithm.save(self._snapshot)
            self.env.set_reward_params(applied)
        self.history.append(rec)
        self.pending = rec
        self._prev_kpi = dict(eval_metrics)
        self._write(rec, proposal)
        out["intervention"] = float(k)
        out["n_applied"] = float(len(applied))
        out["confidence"] = float(proposal.confidence)
        if proposal.usage:
            out.update({f"tokens_{k_}": float(v) for k_, v in proposal.usage.items()})
        out.update({f"param/{key}": v for key, v in self.env.reward_params().items()})
        return out

    # -------------------------------------------------------------- report
    def _report(self, step, total, k, kpi, stats, train_window, current) -> dict[str, Any]:
        def table(rows: dict[str, Any]) -> str:
            return "\n".join(f"- {key}: {_fmt(v)}" for key, v in rows.items()) or "- (none)"

        kpi_rows = {}
        for key, v in kpi.items():
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
        return {
            "step": step,
            "progress": step / max(total, 1),
            "k": k,
            "window": len(train_window),
            "kpi_table": table(kpi_rows),
            "stats_table": table(stats),
            "dynamics_table": table(dyn),
            "history_text": history_text,
            "history_objs": self.history,
            "component_table": comp_rows,
            "objective_text": self.objective_text(),
            "rollback_tolerance": self.tolerance,
            "current": current,
        }

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
