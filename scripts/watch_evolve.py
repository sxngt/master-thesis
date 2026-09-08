#!/usr/bin/env python3
"""Live view of coach batches and evolve roots: training numbers, LLM responses,
progress — refreshed from the run directories on disk (no W&B, stdlib only, so it
also runs with the server's system python3).

Two modes:

  dashboard (default)  one screen, redrawn every --interval seconds: driver state,
                       GPU / LLM-pool health, one row per run (step, speed, ETA,
                       PPO loss/KL/beta, last eval success/velocity/J, coach
                       phase and last move), the latest LLM responses and the
                       driver log tail.
  --follow             event stream (tail -f across every run): every new
                       evaluation, every LLM proposal (diagnosis, actions,
                       forecast) and its later verdict (kept / rolled back,
                       effect), crashes, driver log lines. --full adds the raw
                       LLM response, --prompt the user prompt it answered.

Usage:
  python3 scripts/watch_evolve.py data/results/evolve3                 # dashboard
  python3 scripts/watch_evolve.py data/results/evolve3 --follow --full # stream
  python3 scripts/watch_evolve.py data/results/evolve3 --run <run_dir> # one run, full history
  python3 scripts/watch_evolve.py data/results/evolve3 --remote        # same, on dongbeen via ssh

Roots may be evolve roots (<root>/<version>/runs/<run>), batch dirs
(<batch>/runs/<run> or <batch>/<run>) or single run dirs; several may be given.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_OBJECTIVE = {"success_rate": 1.0, "mean_forward_velocity_ms": 0.5}
STALE_S = 900.0  # no new metrics for this long while unfinished = "stalled?"

# ----------------------------------------------------------------- terminal
USE_COLOR = sys.stdout.isatty()


def c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if USE_COLOR else s


def red(s: str) -> str:
    return c("31", s)


def green(s: str) -> str:
    return c("32", s)


def yellow(s: str) -> str:
    return c("33", s)


def cyan(s: str) -> str:
    return c("36", s)


def dim(s: str) -> str:
    return c("2", s)


def bold(s: str) -> str:
    return c("1", s)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def clip(s: str, width: int) -> str:
    """Cut a (possibly coloured) line to the terminal width."""
    if visible_len(s) <= width:
        return s
    out, n = [], 0
    for tok in re.split(r"(\x1b\[[0-9;]*m)", s):
        if tok.startswith("\x1b["):
            out.append(tok)
            continue
        room = width - n
        if room <= 0:
            break
        out.append(tok[:room])
        n += min(len(tok), room)
    return "".join(out) + ("\x1b[0m" if USE_COLOR else "")


def fmt_steps(n: float) -> str:
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.0f}k"
    return f"{n:.0f}"


def fmt_dur(s: float) -> str:
    if s != s or s < 0:
        return "-"
    s = int(s)
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    if s >= 60:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s}s"


def fmt_num(v: Any, spec: str = ".3f", empty: str = "-") -> str:
    if v is None:
        return empty
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:
        return "nan"
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.1e}"
    return f"{v:{spec}}"


def fmt_params(d: dict[str, Any]) -> str:
    return ", ".join(
        f"{k.split('.')[-1] if k.count('.') > 1 else k}={fmt_num(v, '.3g')}" for k, v in d.items()
    )


def wrap(text: str, width: int, indent: str) -> list[str]:
    text = " ".join(str(text).split())
    return textwrap.wrap(
        text, width=max(width - len(indent), 20), initial_indent=indent, subsequent_indent=indent
    ) or [indent]


# ----------------------------------------------------------------- tiny yaml
def _scalar(v: str) -> Any:
    v = v.strip()
    if v in ("", "null", "~"):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    if v.startswith(("'", '"')) and v.endswith(("'", '"')) and len(v) >= 2:
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def load_config(path: Path) -> dict[str, Any]:
    """config.yaml -> nested dict. PyYAML when available, else an indentation
    parser that is enough for the resolved config the trainer writes
    (mappings + scalars; lists are ignored)."""
    text = path.read_text()
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        pass
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #")[0].rstrip()
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, val = line.strip().partition(":")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if val.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _scalar(val)
    return root


def cfg_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ----------------------------------------------------------------- jsonl tail
class Tail:
    """Incremental JSONL reader: only the bytes appended since the last poll."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.buf = b""

    def poll(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.offset:  # truncated / rewritten
            self.offset, self.buf = 0, b""
        if size == self.offset:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        self.offset += len(chunk)
        self.buf += chunk
        lines = self.buf.split(b"\n")
        self.buf = lines.pop()  # partial last line (or empty)
        out = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


# ----------------------------------------------------------------- run state
class Run:
    def __init__(self, run_dir: Path, group: str):
        self.dir = run_dir
        self.group = group  # version / batch this run belongs to
        self.id = run_dir.name
        self.cfg = (
            load_config(run_dir / "config.yaml") if (run_dir / "config.yaml").exists() else {}
        )
        self.total = float(cfg_get(self.cfg, "run.total_timesteps", 0) or 0)
        obj = cfg_get(self.cfg, "coach.objective")
        self.objective = (
            {k: float(v) for k, v in obj.items()}
            if isinstance(obj, dict)
            else dict(DEFAULT_OBJECTIVE)
        )
        self.terrain = cfg_get(self.cfg, "terrain.name", "?")
        self.level = cfg_get(self.cfg, "sim.terrain_level", "?")
        self.reward = cfg_get(self.cfg, "reward.name", "traditional")
        self.cond = cfg_get(self.cfg, "coach.strategy", "none") or "none"
        self.seed = cfg_get(self.cfg, "run.seed", "?")
        self.version = Path(str(cfg_get(self.cfg, "coach.version_dir", ""))).name
        self.llm_url = cfg_get(self.cfg, "coach.llm.base_url")
        self.metrics = Tail(run_dir / "metrics.jsonl")
        self.coach = Tail(run_dir / "coach_log.jsonl")
        self.step = 0
        self.train: dict[str, Any] = {}
        self.train_hist: list[tuple[float, float]] = []  # (time, step)
        self.evals: list[dict[str, Any]] = []
        self.final: dict[str, Any] | None = None
        self.interventions: dict[int, dict[str, Any]] = {}  # k -> latest record
        self.n_events = 0
        self.last_mtime = 0.0
        self.crash: str | None = None
        self.nonfinite = 0.0
        self.kl_stops = 0

    # -- identity
    @property
    def setting(self) -> str:
        rw = "trad" if self.reward == "traditional" else self.reward
        return f"{self.terrain}-{self.level}-{rw}"

    @property
    def label(self) -> str:
        tag = self.version or self.cond
        tag = "" if tag == self.group else f" {tag}"
        return f"{self.group}/{self.setting}{tag} s{self.seed}"

    def objective_of(self, rec: dict[str, Any], prefix: str) -> float | None:
        vals = [rec.get(prefix + k) for k in self.objective]
        if any(v is None for v in vals):
            return None
        # zip(strict=) needs 3.10; the server's system python is 3.8
        return sum(w * float(v) for w, v in zip(self.objective.values(), vals))  # noqa: B905

    # -- polling
    def poll(self) -> list[tuple[str, dict[str, Any]]]:
        """Absorb new records; returns (kind, record) events in file order."""
        events: list[tuple[str, dict[str, Any]]] = []
        for rec in self.metrics.poll():
            step = rec.get("step")
            if step is not None:
                self.step = max(self.step, int(step))
            if "train/loss" in rec:
                self.train = rec
                if "time" in rec:
                    self.train_hist.append((float(rec["time"]), float(step)))
                    self.train_hist = self.train_hist[-200:]
                self.nonfinite += float(rec.get("train/nonfinite_minibatches", 0) or 0)
                self.kl_stops += int(bool(rec.get("train/kl_early_stop")))
            if "eval/success_rate" in rec:
                rec["_J"] = self.objective_of(rec, "eval/")
                self.evals.append(rec)
                events.append(("eval", rec))
            if "final/success_rate" in rec:
                rec["_J"] = self.objective_of(rec, "final/")
                self.final = rec
                events.append(("final", rec))
        for rec in self.coach.poll():
            k = int(rec.get("k", -1))
            prev = self.interventions.get(k)
            self.interventions[k] = rec
            kind = "proposal" if rec.get("status") == "pending" or prev is None else "verdict"
            if rec.get("status") == "pending" and prev is not None:
                kind = "proposal"  # re-issued (confirm eval) — show again
            events.append((kind, rec))
        try:
            self.last_mtime = max(
                p.stat().st_mtime
                for p in (self.dir / "metrics.jsonl", self.dir / "coach_log.jsonl")
                if p.exists()
            )
        except ValueError:
            self.last_mtime = self.dir.stat().st_mtime
        self.n_events += len(events)
        return events

    # -- derived
    def rate(self) -> float | None:
        h = self.train_hist
        if len(h) < 2 or h[-1][0] - h[0][0] <= 0:
            return None
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])

    def progress(self) -> float:
        return self.step / self.total if self.total else 0.0

    def status(self, now: float) -> str:
        if self.final is not None:
            return "done"
        if self.crash:
            return "crash"
        if self.step == 0 and self.last_mtime == 0:
            return "queued"
        if now - self.last_mtime > STALE_S:
            return "stalled?"
        return "running"

    def last_eval(self) -> dict[str, Any] | None:
        return self.evals[-1] if self.evals else None

    def best_j(self) -> float | None:
        js = [e["_J"] for e in self.evals if e.get("_J") is not None]
        return max(js) if js else None

    def last_intervention(self) -> dict[str, Any] | None:
        if not self.interventions:
            return None
        return self.interventions[max(self.interventions)]

    def coach_cell(self) -> str:
        if self.cond == "none":
            return dim("-")
        rec = self.last_intervention()
        if rec is None:
            return dim("warmup")
        st = rec.get("status", "?")
        col = {"kept": green, "rolled_back": red, "pending": yellow}.get(st, str)
        change = fmt_params(rec.get("applied") or {}) or "noop"
        if rec.get("restored_from_step") is not None:
            change = "restore; " + change
        eff = rec.get("effect")
        tail = f" ({float(eff):+.2f})" if eff is not None else ""
        diag = str(rec.get("diagnosis", ""))
        api = " " + red("API-ERR") if diag.startswith("api error") else ""
        return f"k{rec.get('k')} {rec.get('phase', '')[:4]} {col(st)} {change}{tail}{api}"


# ----------------------------------------------------------------- discovery
def discover(roots: list[Path]) -> list[Run]:
    """Run dirs under evolve roots / batch dirs / plain run dirs (config.yaml
    + metrics.jsonl mark a run)."""
    found: dict[Path, Run] = {}

    def is_run(d: Path) -> bool:
        return (d / "config.yaml").exists() and (d / "metrics.jsonl").exists()

    for root in roots:
        if is_run(root):
            found[root] = Run(
                root, root.parent.parent.name if root.parent.name == "runs" else root.parent.name
            )
            continue
        for depth in ((root,), root.glob("*"), root.glob("*/runs/*"), root.glob("runs/*")):
            for d in depth:
                if d.is_dir() and is_run(d) and d not in found:
                    group = d.parent.parent.name if d.parent.name == "runs" else d.parent.name
                    found[d] = Run(d, group)
    return list(found.values())


def scan_crashes(runs: list[Run]) -> None:
    """Job logs (<batch>/jobs_logs/*.log) that mention a run and a Traceback."""
    seen: set[Path] = set()
    by_id = {r.id: r for r in runs}
    for r in runs:
        for logs_dir in (r.dir.parent.parent / "jobs_logs", r.dir.parent / "jobs_logs"):
            if logs_dir in seen or not logs_dir.is_dir():
                continue
            seen.add(logs_dir)
            for log in logs_dir.glob("*.log"):
                try:
                    text = log.read_text(errors="replace")
                except OSError:
                    continue
                if "Traceback" not in text:
                    continue
                for rid, run in by_id.items():
                    if rid in text and run.final is None:
                        lines = [ln for ln in text.splitlines() if ln.strip()]
                        run.crash = lines[-1][:120] if lines else "Traceback"


# ----------------------------------------------------------------- driver
class Driver:
    """evolve state.json (generation, incumbent, verdicts, log) + driver.log."""

    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "state.json"
        self.log_tail = Tail(root / "driver.log")  # not JSONL — read raw below
        self.log_lines: list[str] = []
        self.state: dict[str, Any] = {}
        self.n_log_seen = 0

    def poll(self) -> list[str]:
        new: list[str] = []
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
            log = self.state.get("log") or []
            for rec in log[self.n_log_seen :]:
                new.append(f"{rec.get('time', '')} {rec.get('msg', '')}")
            self.n_log_seen = len(log)
        p = self.root / "driver.log"
        if p.exists():
            size = p.stat().st_size
            if size > self.log_tail.offset:
                with open(p, "rb") as f:
                    f.seek(self.log_tail.offset)
                    chunk = f.read()
                self.log_tail.offset = size
                for ln in chunk.decode(errors="replace").splitlines():
                    if ln.strip() and not ln.startswith("[jobs] "):
                        new.append(ln.rstrip())
        self.log_lines = (self.log_lines + new)[-200:]
        return new

    def header(self) -> str:
        if not self.state:
            return ""
        s = self.state
        parts = [f"gen {s.get('generation')}", f"incumbent {bold(str(s.get('incumbent')))}"]
        for name, v in (s.get("versions") or {}).items():
            st = v.get("status", "?")
            col = {"accepted": green, "incumbent": green, "rejected": red, "running": yellow}.get(
                st, str
            )
            cmp_ = v.get("compare") or {}
            dj = (
                f" dJ {cmp_['mean']:+.3f} p={cmp_.get('p_one_sided', float('nan')):.2f}"
                if "mean" in cmp_
                else ""
            )
            parts.append(f"{name}:{col(st)}{dj}")
        return "  ".join(parts)


# ----------------------------------------------------------------- probes
def gpu_line() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    cells = []
    for row in out.strip().splitlines():
        try:
            i, util, used, total = [x.strip() for x in row.split(",")]
            cells.append(f"{i}:{int(util):3d}% {int(used) / 1024:4.1f}/{int(total) / 1024:.0f}G")
        except ValueError:
            continue
    return "GPU " + " | ".join(cells) if cells else ""


def llm_line(urls: set[str]) -> str:
    cells = []
    for url in sorted(urls):
        base = url.rsplit("/v1", 1)[0]
        port = base.rsplit(":", 1)[-1]
        try:
            with urllib.request.urlopen(base + "/api/ps", timeout=1.5) as resp:
                models = json.loads(resp.read().decode()).get("models") or []
            names = ",".join(m.get("name", "?") for m in models) or "idle"
            cells.append(f"{port} {green('up')} {names}")
        except Exception:  # noqa: BLE001 — any failure = the daemon is not answering
            cells.append(f"{port} {red('down')}")
    return "LLM " + " | ".join(cells) if cells else ""


# ----------------------------------------------------------------- rendering
def event_lines(
    run: Run, kind: str, rec: dict[str, Any], width: int, full: bool, prompt: bool
) -> list[str]:
    ts = time.strftime("%H:%M:%S", time.localtime(float(rec.get("time", time.time()))))
    who = cyan(run.label)
    step = fmt_steps(float(rec.get("step", run.step)))
    if kind == "eval":
        j = rec.get("_J")
        return [
            f"{dim(ts)} {who} @{step} eval  succ {fmt_num(rec.get('eval/success_rate'), '.2f')}  "
            f"v {fmt_num(rec.get('eval/mean_forward_velocity_ms'), '.2f')}  J {bold(fmt_num(j))}  "
            f"CoT {fmt_num(rec.get('eval/cost_of_transport'), '.2f')}  "
            f"falls/min {fmt_num(rec.get('eval/fall_frequency_per_min'), '.1f')}"
        ]
    if kind == "final":
        j = rec.get("_J")
        return [
            f"{dim(ts)} {who} @{step} {green('FINAL')} "
            f"succ {fmt_num(rec.get('final/success_rate'), '.2f')}  "
            f"v {fmt_num(rec.get('final/mean_forward_velocity_ms'), '.2f')}  J {bold(fmt_num(j))}"
        ]
    if kind == "crash":
        return [f"{dim(ts)} {who} {red('CRASH')} {rec.get('msg', '')}"]
    if kind == "driver":
        return [f"{dim(ts)} {yellow('driver')} {rec.get('msg', '')}"]
    # coach proposal / verdict
    k = rec.get("k")
    st = rec.get("status", "?")
    col = {"kept": green, "rolled_back": red, "pending": yellow}.get(st, str)
    head = (
        f"{dim(ts)} {who} @{step} coach #{k} {rec.get('phase', '')} {col(st)}  "
        f"J {fmt_num(rec.get('objective_before'))}"
    )
    lines = [head]
    ind = "      "
    if kind == "proposal":
        diag = str(rec.get("diagnosis", ""))
        lines += wrap(f"diagnosis: {diag}", width, ind)
        proposed = rec.get("proposed") or {}
        applied = rec.get("applied") or {}
        before = rec.get("params_before") or {}
        if proposed:
            lines.append(ind + "proposed: " + fmt_params(proposed))
        if applied:
            lines.append(
                ind
                + "applied:  "
                + ", ".join(
                    f"{p} {fmt_num(before.get(p), '.3g')}->{fmt_num(v, '.3g')}"
                    for p, v in applied.items()
                )
            )
        elif proposed:
            lines.append(ind + "applied:  " + yellow("nothing (all dropped)"))
        if rec.get("restored_from_step") is not None:
            lines.append(
                ind
                + yellow(
                    "restored best snapshot from step "
                    + fmt_steps(float(rec["restored_from_step"]))
                )
            )
        notes = rec.get("notes") or []
        for n in notes:
            lines += wrap(f"note: {n}", width, ind)
        extra = []
        if rec.get("predicted_delta_j") is not None:
            extra.append(f"predicted dJ {float(rec['predicted_delta_j']):+.3f}")
        if rec.get("confidence") is not None:
            extra.append(f"confidence {float(rec['confidence']):.2f}")
        if rec.get("llm_latency_s"):
            extra.append(f"llm {float(rec['llm_latency_s']):.0f}s")
        u = rec.get("usage") or {}
        if u:
            extra.append(f"tokens in/out {u.get('input_tokens', 0)}/{u.get('output_tokens', 0)}")
        if rec.get("expected_effect"):
            lines += wrap(f"expected: {rec['expected_effect']}", width, ind)
        if extra:
            lines.append(ind + dim("  ".join(extra)))
        if prompt and isinstance(rec.get("prompt"), dict):
            lines.append(ind + bold("--- user prompt ---"))
            for raw in str(rec["prompt"].get("user", "")).splitlines():
                lines += wrap(raw, width, ind) if raw.strip() else [ind]
        if full and rec.get("raw_response"):
            lines.append(ind + bold("--- raw response ---"))
            for raw in str(rec["raw_response"]).splitlines():
                lines += wrap(raw, width, ind) if raw.strip() else [ind]
    else:  # verdict
        after = rec.get("objective_after")
        eff = rec.get("effect")
        exp = rec.get("expected_dj")
        pred = rec.get("predicted_delta_j")
        bits = [f"J {fmt_num(rec.get('objective_before'))} -> {fmt_num(after)}"]
        if eff is not None:
            bits.append(f"effect net of trend {float(eff):+.3f}")
        if exp is not None:
            bits.append(f"trend expected {float(exp):+.3f}")
        if pred is not None:
            bits.append(f"LLM predicted {float(pred):+.3f}")
        if rec.get("tolerance_used"):
            bits.append(f"tol {float(rec['tolerance_used']):.3f}")
        if rec.get("confirm_evals"):
            bits.append(f"confirm evals {rec['confirm_evals']}")
        lines.append(ind + "  ".join(bits))
        if rec.get("invariant"):
            lines.append(ind + "curriculum release: " + fmt_params(rec["invariant"]))
    return [clip(ln, width) for ln in lines]


def render_dashboard(
    runs: list[Run],
    drivers: list[Driver],
    recent: list[tuple[float, Run, str, dict[str, Any]]],
    interval: float,
    n_llm: int,
    llm_urls: set[str],
    probe: bool,
) -> str:
    width = shutil.get_terminal_size((160, 50)).columns
    now = time.time()
    out: list[str] = []
    roots = ", ".join(str(d.root) for d in drivers) or ", ".join(
        sorted({str(r.dir.parent) for r in runs})
    )
    out.append(
        bold(f"{roots}") + dim(f"   {time.strftime('%Y-%m-%d %H:%M:%S')}   every {interval:.0f}s")
    )
    for d in drivers:
        h = d.header()
        if h:
            out.append(h)
    if probe:
        g = gpu_line()
        if g:
            out.append(g)
        alive = {r.llm_url for r in runs if r.llm_url and r.status(now) == "running"}
        alive = alive or {r.llm_url for r in runs if r.llm_url} or llm_urls
        ll = llm_line(alive)
        if ll:
            out.append(ll)
    out.append("")

    groups: dict[str, list[Run]] = {}
    for r in runs:
        groups.setdefault(r.group, []).append(r)
    hdr = (
        f"{'setting':<22} {'cond':<9} {'s':>2} {'step':>12} {'%':>4} {'sps':>5} {'ETA':>6} "
        f"{'loss':>7} {'KL':>7} {'beta':>6} {'succ':>5} {'vel':>5} {'J':>6} {'bestJ':>6}  coach"
    )
    for name, rs in groups.items():
        counts: dict[str, int] = {}
        for r in rs:
            counts[r.status(now)] = counts.get(r.status(now), 0) + 1
        summary = "  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        js = [r.final["_J"] for r in rs if r.final and r.final.get("_J") is not None]
        if js:
            mean = sum(js) / len(js)
            sd = (sum((j - mean) ** 2 for j in js) / max(len(js) - 1, 1)) ** 0.5
            summary += f"   final J {mean:.3f} ± {sd:.3f} (n={len(js)})"
        out.append(bold(f"── {name} ") + dim(summary))
        out.append(dim(hdr))
        for r in sorted(rs, key=lambda x: (x.setting, x.cond, str(x.seed))):
            st = r.status(now)
            rate = r.rate()
            eta = (r.total - r.step) / rate if rate and st == "running" else float("nan")
            ev = r.last_eval()
            fin = r.final
            if fin is not None:
                succ, vel, j = (
                    fin.get("final/success_rate"),
                    fin.get("final/mean_forward_velocity_ms"),
                    fin.get("_J"),
                )
            elif ev is not None:
                succ, vel, j = (
                    ev.get("eval/success_rate"),
                    ev.get("eval/mean_forward_velocity_ms"),
                    ev.get("_J"),
                )
            else:
                succ = vel = j = None
            tr = r.train
            beta = tr.get("train/kl_penalty")
            beta_s = fmt_num(beta, ".2f")
            if beta is not None and float(beta) >= 8:
                beta_s = yellow(beta_s)
            kl = tr.get("train/approx_kl")
            kl_s = fmt_num(kl, ".4f")
            if kl is not None and abs(float(kl)) > 0.1:
                kl_s = red(kl_s)
            step_s = f"{fmt_steps(r.step)}/{fmt_steps(r.total)}" if r.total else fmt_steps(r.step)
            if st == "done":
                step_s = green(step_s)
            elif st == "crash":
                step_s = red("CRASH")
            elif st == "stalled?":
                step_s = yellow(step_s)
            flags = ""
            if r.nonfinite:
                flags += red(f" nonfinite×{int(r.nonfinite)}")
            if r.kl_stops:
                flags += yellow(f" klstop×{r.kl_stops}")
            sps = fmt_steps(rate) if rate else "-"
            row = (
                f"{r.setting:<22} {(r.version or r.cond):<9} {str(r.seed):>2} {step_s:>12} "
                f"{100 * r.progress():>4.0f} {sps:>5} {fmt_dur(eta):>6} "
                f"{fmt_num(tr.get('train/loss'), '.3f'):>7} {kl_s:>7} {beta_s:>6} "
                f"{fmt_num(succ, '.2f'):>5} {fmt_num(vel, '.2f'):>5} {bold(fmt_num(j)):>6} "
                f"{fmt_num(r.best_j()):>6}  "
                f"{r.coach_cell()}{flags}"
            )
            if st == "crash" and r.crash:
                row += dim("  " + r.crash)
            out.append(clip(row, width))
        out.append("")

    if n_llm and recent:
        out.append(bold("── latest LLM responses"))
        for _t, r, kind, rec in recent[-n_llm:]:
            out.extend(event_lines(r, kind, rec, width, full=False, prompt=False)[:6])
        out.append("")
    for d in drivers:
        if d.log_lines:
            out.append(bold(f"── driver log {d.root}"))
            out.extend(clip(dim(ln), width) for ln in d.log_lines[-6:])
    return "\n".join(out)


# ----------------------------------------------------------------- main loops
def remote_exec(argv: list[str]) -> None:
    host = os.environ.get("REMOTE_HOST", "sxngt@100.104.103.77")
    port = os.environ.get("REMOTE_PORT", "12888")
    root = os.environ.get("REMOTE_ROOT", "/mnt/sdb1/sxngt/workspace/master-thesis")
    args = [a for a in argv if a != "--remote"]
    cmd = f"cd {shlex.quote(root)} && python3 scripts/watch_evolve.py {shlex.join(args)}"
    os.execvp("ssh", ["ssh", "-t", "-p", port, host, cmd])


def main() -> None:
    global USE_COLOR
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "roots",
        nargs="*",
        default=["data/results/evolve3"],
        help="evolve roots / batch dirs / run dirs",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help="only this run dir (repeatable); implies full history",
    )
    p.add_argument(
        "--follow", "-f", action="store_true", help="event stream instead of the dashboard"
    )
    p.add_argument(
        "--full", action="store_true", help="stream: print the raw LLM response of each proposal"
    )
    p.add_argument(
        "--prompt", action="store_true", help="stream: print the user prompt of each proposal"
    )
    p.add_argument(
        "--history",
        action="store_true",
        help="stream: replay every past event first (default: last --last)",
    )
    p.add_argument(
        "--last",
        type=int,
        default=8,
        help="stream: past events to replay; dashboard: LLM responses shown",
    )
    p.add_argument("--interval", type=float, default=15.0, help="refresh / poll seconds")
    p.add_argument(
        "--no-probe", action="store_true", help="dashboard: skip nvidia-smi and LLM-pool probes"
    )
    p.add_argument(
        "--llm-url", action="append", default=[], help="extra OpenAI-compatible base URL to probe"
    )
    p.add_argument("--once", action="store_true", help="render once and exit")
    p.add_argument("--no-color", action="store_true")
    p.add_argument(
        "--remote",
        action="store_true",
        help="run this command on the server over ssh (REMOTE_HOST/PORT/ROOT)",
    )
    args = p.parse_args()
    if args.remote:
        remote_exec(sys.argv[1:])
    if args.no_color or not sys.stdout.isatty():
        USE_COLOR = False

    roots = [Path(r) for r in (args.run or args.roots)]
    runs = discover(roots)
    drivers = [
        Driver(r) for r in roots if (r / "state.json").exists() or (r / "driver.log").exists()
    ]
    follow = args.follow or bool(args.run)
    replay_all = args.history or bool(args.run)
    width = shutil.get_terminal_size((160, 50)).columns

    # first pass: absorb everything that already exists
    backlog: list[tuple[float, Run, str, dict[str, Any]]] = []
    for r in runs:
        for kind, rec in r.poll():
            backlog.append((float(rec.get("time", 0.0)), r, kind, rec))
    for d in drivers:
        for ln in d.poll():
            backlog.append((0.0, runs[0] if runs else None, "driver", {"msg": ln}))  # type: ignore[arg-type]
    scan_crashes(runs)
    backlog.sort(key=lambda e: e[0])
    llm_events = [e for e in backlog if e[2] in ("proposal", "verdict")]

    if follow:
        shown = backlog if replay_all else [e for e in backlog if e[2] != "driver"][-args.last :]
        for _t, r, kind, rec in shown:
            if kind == "driver":
                print(f"{yellow('driver')} {rec['msg']}")
                continue
            print("\n".join(event_lines(r, kind, rec, width, args.full, args.prompt)))
        if not runs:
            print(dim(f"no runs under {', '.join(map(str, roots))} yet — waiting"))
        print(
            dim(f"--- following {len(runs)} runs, poll {args.interval:.0f}s (Ctrl-C to stop) ---")
        )
        sys.stdout.flush()
        if args.once:
            return
        known = {r.dir for r in runs}
        while True:
            time.sleep(args.interval)
            for r in discover(roots):  # new runs appear as the driver launches them
                if r.dir not in known:
                    known.add(r.dir)
                    runs.append(r)
                    print(
                        f"{dim(time.strftime('%H:%M:%S'))} {cyan(r.label)} "
                        f"{green('NEW RUN')} {r.dir}"
                    )
            for r in runs:
                was_crashed = r.crash
                for kind, rec in r.poll():
                    print("\n".join(event_lines(r, kind, rec, width, args.full, args.prompt)))
                if was_crashed is None:
                    scan_crashes([r])
                    if r.crash:
                        print(
                            "\n".join(
                                event_lines(r, "crash", {"msg": r.crash}, width, False, False)
                            )
                        )
            for d in drivers:
                for ln in d.poll():
                    print(f"{yellow('driver')} {ln}")
            sys.stdout.flush()
        return

    known = {r.dir for r in runs}
    while True:
        screen = render_dashboard(
            runs,
            drivers,
            llm_events,
            args.interval,
            args.last,
            set(args.llm_url),
            not args.no_probe,
        )
        if not args.once:
            sys.stdout.write("\x1b[2J\x1b[H" if USE_COLOR else "\n" * 3)
        sys.stdout.write(screen + "\n")
        sys.stdout.flush()
        if args.once:
            return
        time.sleep(args.interval)
        for r in discover(roots):
            if r.dir not in known:
                known.add(r.dir)
                runs.append(r)
        for r in runs:
            for kind, rec in r.poll():
                if kind in ("proposal", "verdict"):
                    llm_events.append((float(rec.get("time", time.time())), r, kind, rec))
        llm_events = llm_events[-200:]
        scan_crashes(runs)
        for d in drivers:
            d.poll()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
