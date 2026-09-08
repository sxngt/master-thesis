"""Cross-run experience for the reward coach (the "playbook").

The coach's evidence (``coach_stats``) is computed from the current run only,
so a coach starting a run knows nothing about what worked in earlier runs of
the same task: 2026-09-08, on the naive-reward rough terrain, the local coach
left two of three seeds on the floor (success 0 for 40M steps) although every
earlier run that escaped it had applied the same three moves at the first
report. The playbook is a compact index of every finished coached run:

- when the policy left the floor (first evaluation with a success rate),
- which (parameter, direction) moves were applied on the floor before that,
  and which moves the runs that never left it applied,
- how each move fared once the policy was walking (kept / rolled back, mean
  detrended effect), i.e. a ledger across runs.

``Playbook.evidence`` turns this into a few computed lines for the coach's
report (same task first, any task as a fallback); ``stalled_text`` summarises
stuck runs for the meta-LLM. Built from run directories (config.yaml,
metrics.jsonl, coach_log.jsonl) by ``scripts/build_playbook.py`` or by the
evolution driver after every batch; the coach loads the JSON when
``coach.playbook.enabled`` is true. Pure logic, no simulator or pandas.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from quadruped_rl.llm_feedback.coach_stats import move_direction

DEFAULT_WEIGHTS = {"success_rate": 1.0, "mean_forward_velocity_ms": 0.5}
FLOOR_SUCCESS = 0.1  # success rate at which a policy has "left the floor"
# runs whose moves count as experience: random/hillclimb coaches move blindly
COACHED_CONDITIONS = frozenset({"llm"})


@dataclass
class Move:
    step: int
    phase: str
    status: str
    j_before: float
    success_before: float
    applied: dict[str, list[float]]  # param -> [before, after]
    effect: float | None = None
    restored: bool = False

    def directions(self) -> set[tuple[str, str]]:
        return {(p, move_direction(b, a)) for p, (b, a) in self.applied.items() if a != b}


@dataclass
class Case:
    run_id: str
    setting: str
    seed: int
    condition: str  # coach strategy or "none"
    total_steps: int
    final_j: float
    breakout_step: int | None  # first eval with success >= FLOOR_SUCCESS
    evals: list[list[float]] = field(default_factory=list)  # [step, J, success]
    moves: list[Move] = field(default_factory=list)
    final_params: dict[str, float] = field(default_factory=dict)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def case_of(
    run_dir: str | Path,
    weights: dict[str, float] | None = None,
    floor_success: float = FLOOR_SUCCESS,
) -> Case | None:
    """Extract one Case from a finished run directory (None if incomplete)."""
    run_dir = Path(run_dir)
    cfg_path, metrics_path = run_dir / "config.yaml", run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and metrics_path.exists()):
        return None
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    weights = weights or DEFAULT_WEIGHTS
    recs = _read_jsonl(metrics_path)
    finals = [r for r in recs if "final/success_rate" in r]
    if not finals:
        return None
    final = finals[-1]
    evals = []
    breakout = None
    for r in recs:
        if "eval/success_rate" not in r:
            continue
        j = sum(w * float(r.get(f"eval/{m}", 0.0)) for m, w in weights.items())
        s = float(r["eval/success_rate"])
        evals.append([int(r["step"]), round(j, 4), round(s, 4)])
        if breakout is None and s >= floor_success:
            breakout = int(r["step"])
    settled: dict[int, dict[str, Any]] = {}
    for rec in _read_jsonl(run_dir / "coach_log.jsonl"):
        if rec.get("status") != "pending" or rec.get("k") not in settled:
            settled[rec["k"]] = rec
    moves = []
    for rec in settled.values():
        applied = rec.get("applied") or {}
        if not applied or rec.get("status") == "noop":
            continue
        before = rec.get("params_before") or {}
        moves.append(
            Move(
                step=int(rec.get("step", 0)),
                phase=str(rec.get("phase") or "?"),
                status=str(rec.get("status")),
                j_before=float(rec.get("objective_before") or 0.0),
                success_before=float((rec.get("kpi_before") or {}).get("success_rate", 0.0)),
                applied={
                    p: [float(before.get(p, v)), float(v)]
                    for p, v in applied.items()
                    if isinstance(v, (int, float))
                },
                effect=None if rec.get("effect") is None else float(rec["effect"]),
                restored=rec.get("restored_from_step") is not None,
            )
        )
    moves.sort(key=lambda m: m.step)
    final_params: dict[str, float] = {}
    if settled:
        last = settled[max(settled)]
        final_params = {
            k: float(v)
            for k, v in {**(last.get("params_before") or {}), **(last.get("applied") or {})}.items()
            if isinstance(v, (int, float))
        }
    sim = cfg.get("sim") or {}
    return Case(
        run_id=run_dir.name,
        setting=(
            f"{(cfg.get('terrain') or {}).get('name', '?')}-"
            f"{sim.get('terrain_level', 'easy')}-"
            f"{(cfg.get('reward') or {}).get('name', 'traditional')}"
        ),
        seed=int((cfg.get("run") or {}).get("seed", 0)),
        condition=str((cfg.get("coach") or {}).get("strategy") or "none"),
        total_steps=int((cfg.get("run") or {}).get("total_timesteps", 0)),
        final_j=sum(w * float(final.get(f"final/{m}", 0.0)) for m, w in weights.items()),
        breakout_step=breakout,
        evals=evals,
        moves=moves,
        final_params=final_params,
    )


_RUN_DIR_RE = re.compile(r"data/results/[\w.-]+_s\d+_\d{8}-\d{6}_[0-9a-f]{6}")


def _run_dirs_from_job_logs(logs: Path) -> list[Path]:
    """Run directories named in run_jobs logs, resolved against the cwd and the
    ancestors of ``logs`` (the repo root when the batch lives under it)."""
    bases = [Path.cwd(), *logs.parents]
    out: dict[str, Path] = {}
    for log in sorted(logs.glob("*.log")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for rel in set(_RUN_DIR_RE.findall(text)):
            for base in bases:
                d = base / rel
                if (d / "config.yaml").exists():
                    out.setdefault(d.name, d)
                    break
    return [out[k] for k in sorted(out)]


def _fmt_dir(param: str, direction: str) -> str:
    return f"{param} {direction}"


class Playbook:
    """Index of finished runs and the computed evidence drawn from it."""

    def __init__(
        self,
        cases: list[Case],
        built: str = "",
        floor_success: float = FLOOR_SUCCESS,
        conditions: frozenset[str] | set[str] = COACHED_CONDITIONS,
    ):
        self.cases = list(cases)
        self.built = built or time.strftime("%Y-%m-%d %H:%M")
        self.floor_success = floor_success
        self.conditions = frozenset(conditions)

    def coached(self, c: Case) -> bool:
        return c.condition in self.conditions and bool(c.moves)

    # ------------------------------------------------------------ build / io
    @classmethod
    def build(
        cls,
        run_dirs: list[str | Path],
        weights: dict[str, float] | None = None,
        floor_success: float = FLOOR_SUCCESS,
        conditions: frozenset[str] | set[str] | None = COACHED_CONDITIONS,
    ) -> Playbook:
        """``conditions=None`` counts every coached strategy as experience."""
        cases: dict[str, Case] = {}
        for d in run_dirs:
            c = case_of(d, weights, floor_success)
            if c is not None:
                cases[c.run_id] = c  # the same run seen twice (symlinked roots) counts once
        if conditions is None:
            conditions = {c.condition for c in cases.values() if c.condition != "none"}
        return cls(list(cases.values()), floor_success=floor_success, conditions=conditions)

    @staticmethod
    def run_dirs_under(*roots: str | Path) -> list[Path]:
        """Every ``<root>/**/config.yaml`` directory (roots may be run parents or run
        dirs). A batch root that only holds ``jobs_logs/`` (older run_jobs batches
        whose runs live flat under data/results/) is resolved through the run-dir
        paths printed in those logs. Overlapping roots (a batch root listed
        next to a parent of it) yield every run dir once."""
        out: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            if (root / "config.yaml").exists():
                found = [root]
            else:
                found = sorted(p.parent for p in root.glob("**/config.yaml"))
                if not found and (root / "jobs_logs").is_dir():
                    found = _run_dirs_from_job_logs(root / "jobs_logs")
            for d in found:
                key = d.resolve()
                if key not in seen:
                    seen.add(key)
                    out.append(d)
        return out

    def to_json(self) -> str:
        return json.dumps(
            {
                "built": self.built,
                "floor_success": self.floor_success,
                "conditions": sorted(self.conditions),
                "cases": [asdict(c) for c in self.cases],
            }
        )

    @classmethod
    def from_json(cls, text: str) -> Playbook:
        raw = json.loads(text)
        cases = []
        for c in raw.get("cases", []):
            moves = [Move(**m) for m in c.pop("moves", [])]
            cases.append(Case(**c, moves=moves))
        return cls(
            cases,
            raw.get("built", ""),
            raw.get("floor_success", FLOOR_SUCCESS),
            frozenset(raw.get("conditions", COACHED_CONDITIONS)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Playbook:
        return cls.from_json(Path(path).read_text())

    # ------------------------------------------------------------- evidence
    def _select(self, setting: str, exclude: str | None) -> tuple[list[Case], bool]:
        """Coached cases of ``setting`` (fallback: any setting); flag = fallback used."""
        pool = [c for c in self.cases if self.coached(c) and c.run_id != exclude]
        same = [c for c in pool if c.setting == setting]
        return (same, False) if same else (pool, True)

    def evidence(
        self,
        setting: str,
        success_now: float,
        exclude_run: str | None = None,
        max_moves: int = 8,
    ) -> str:
        """Report lines for a coach at ``setting`` whose current success rate is
        ``success_now``: on the floor -> what preceded the breakout in earlier
        runs (and what the stuck runs did); walking -> the cross-run ledger."""
        cases, fallback = self._select(setting, exclude_run)
        if not cases:
            return ""
        where = "other tasks (none of this task yet)" if fallback else "this task"
        out_cases = [c for c in cases if c.breakout_step is not None]
        stuck = [c for c in cases if c.breakout_step is None]
        finals = np.array([c.final_j for c in cases])
        head = (
            f"- Earlier coached runs of {where} ({len(cases)} runs, playbook {self.built}): "
            f"{len(out_cases)}/{len(cases)} left the floor"
        )
        if out_cases:
            med = float(np.median([c.breakout_step for c in out_cases]))
            head += f" (median at {med / 1e6:.1f}M steps)"
        head += (
            f"; final J {finals.mean():.2f} +- {finals.std(ddof=1) if len(finals) > 1 else 0:.2f}"
        )
        controls = [
            c
            for c in self.cases
            if c.condition == "none" and c.setting == setting and c.run_id != exclude_run
        ]
        if controls:
            n_out = sum(c.breakout_step is not None for c in controls)
            head += (
                f"; without a coach {n_out}/{len(controls)} runs left the floor, final J "
                f"{np.mean([c.final_j for c in controls]):.2f}"
            )
        lines = [head]
        if success_now < self.floor_success:
            lines += self._floor_lines(out_cases, stuck, max_moves)
        else:
            lines += self._ledger_lines(cases, max_moves)
        return "\n".join(lines)

    def _floor_lines(self, out_cases: list[Case], stuck: list[Case], max_moves: int) -> list[str]:
        lines = []
        if out_cases:
            cnt: Counter[tuple[str, str]] = Counter()
            first: Counter[tuple[str, str]] = Counter()
            gaps = []
            for c in out_cases:
                on_floor = [m for m in c.moves if m.step <= c.breakout_step]
                seen: set[tuple[str, str]] = set()
                for i, m in enumerate(on_floor):
                    for d in m.directions():
                        if d not in seen:
                            seen.add(d)
                            cnt[d] += 1
                            if i == 0:
                                first[d] += 1
                if on_floor:
                    gaps.append((c.breakout_step - on_floor[0].step) / 1e6)
            if cnt:
                parts = [
                    f"{_fmt_dir(*d)} {n}/{len(out_cases)}"
                    + (f" ({first[d]} at the first report)" if first[d] else "")
                    for d, n in cnt.most_common(max_moves)
                ]
                lines.append(
                    "  moves applied on the floor before the breakout in the runs that left it: "
                    + ", ".join(parts)
                )
                if gaps:
                    lines.append(
                        f"  the first success came {np.median(gaps):.1f}M steps (median) after "
                        f"the first of those moves"
                    )
            else:
                lines.append("  those runs left the floor without any parameter change")
        if stuck:
            cnt = Counter()
            for c in stuck:
                seen = set()
                for m in c.moves:
                    for d in m.directions():
                        if d not in seen:
                            seen.add(d)
                            cnt[d] += 1
            out_dirs: set[tuple[str, str]] = set()
            for c in out_cases:
                for m in c.moves:
                    if m.step <= c.breakout_step:
                        out_dirs |= m.directions()
            only = [d for d, _ in cnt.most_common() if d not in out_dirs]
            lines.append(
                f"  {len(stuck)} run(s) never left the floor; they applied: "
                + ", ".join(
                    f"{_fmt_dir(*d)} {n}/{len(stuck)}" for d, n in cnt.most_common(max_moves)
                )
                + (
                    " — moves seen only in stuck runs: " + ", ".join(_fmt_dir(*d) for d in only[:5])
                    if only and out_cases
                    else ""
                )
            )
        return lines

    def _ledger_lines(self, cases: list[Case], max_moves: int) -> list[str]:
        stats: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"n": [], "kept": [], "effect": []}
        )
        for c in cases:
            for m in c.moves:
                if m.success_before < self.floor_success:
                    continue  # floor moves are summarised by _floor_lines
                for d in m.directions():
                    st = stats[d]
                    st["n"].append(1.0)
                    st["kept"].append(1.0 if m.status == "kept" else 0.0)
                    if m.effect is not None:
                        st["effect"].append(m.effect)
        lines = self._final_command_lines(cases)
        if not stats:
            return lines + ["  (no moves recorded while walking)"]
        rows = sorted(stats.items(), key=lambda kv: -len(kv[1]["n"]))[:max_moves]
        lines.append(
            "  ledger of moves made while walking in those runs (kept share, mean effect "
            "net of trend at the next report - the immediate effect, not the final outcome):"
        )
        for d, st in rows:
            n = len(st["n"])
            kept = sum(st["kept"]) / n
            eff = f"{np.mean(st['effect']):+.3f} (n={len(st['effect'])})" if st["effect"] else "n/a"
            lines.append(f"    {_fmt_dir(*d)}: {n} moves, kept {kept:.0%}, effect {eff}")
        return lines

    def _final_command_lines(
        self, cases: list[Case], param: str = "forward_velocity.target_ms"
    ) -> list[str]:
        """Final outcome by where the curriculum lever ended: the immediate
        ledger effect of raising it is negative (tracking lags), the runs that
        raised it finish highest — both are shown."""
        bins: dict[str, list[float]] = defaultdict(list)
        for c in cases:
            v = c.final_params.get(param)
            if v is None:
                continue
            key = ">= 1.4" if v >= 1.4 else ("1.0 - 1.39" if v >= 1.0 else "< 1.0")
            bins[key].append(c.final_j)
        if len(bins) < 2:
            return []
        parts = [
            f"{k}: J {np.mean(v):.2f} (n={len(v)})"
            for k, v in sorted(bins.items(), key=lambda kv: kv[0], reverse=True)
        ]
        return [f"  final J by where {param} ended in those runs: " + ", ".join(parts)]

    # --------------------------------------------------------------- meta
    def stalled_text(self, run_ids: set[str] | None = None) -> str:
        """Runs that never left the floor and the moves they made — for the
        meta-LLM's diagnostics (``run_ids`` restricts to one batch)."""
        stuck = [
            c
            for c in self.cases
            if self.coached(c)
            and c.breakout_step is None
            and (run_ids is None or c.run_id in run_ids)
        ]
        if not stuck:
            return ""
        lines = [
            f"- runs that never left the floor (success < {self.floor_success:g} at every "
            "evaluation):"
        ]
        for c in stuck:
            seq = "; ".join(
                f"{m.step / 1e6:.0f}M "
                + ", ".join(f"{p} {b:.3g}->{a:.3g}" for p, (b, a) in m.applied.items())
                for m in c.moves[:8]
            )
            lines.append(
                f"    {c.setting} s{c.seed} ({c.run_id}): J={c.final_j:.3f}; "
                f"moves: {seq or '(none)'}"
            )
        return "\n".join(lines)
