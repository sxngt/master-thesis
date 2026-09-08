#!/usr/bin/env python
"""Self-improving coach loop: v5 -> v6 -> v7 ... with a local LLM (zero API cost).

Three GPUs serve the coach model (scripts/llm_server.sh), the fourth trains.
Each generation (llm_feedback/evolve.py has the logic, this file the plumbing):

  1. gate      wait until the training GPUs are free and the LLM server answers
  2. incumbent run it once (settings x seeds) if it has no results yet
  3. candidate meta-LLM writes configs/coach/versions/v<N+1>/ (validated), run it;
               a hand-written version dir whose CHANGELOG says `parent: <incumbent>`
               and that state.json does not know yet is tried first (queued candidate)
  4. compare   paired dJ on (setting, seed); accept -> new incumbent, else keep
  5. report    <root>/report.md, state.json, proposals.md (code ideas for a human)

Resumable: every run is skipped when its jobs file is already done (run_jobs
status), every version keeps its results in <root>/<version>/runs, and
state.json is rewritten after each step. Never kills anything.

On dongbeen (docs/evolve.md):
    scripts/llm_server.sh start && scripts/llm_server.sh warm
    export ISAAC_PY=/mnt/sdb1/sxngt/isaac-sim-4.5.0/python.sh
    nohup $ISAAC_PY scripts/evolve_coach.py --root data/results/evolve --sim-gpus 3 \
        --parallel 2 >> data/results/evolve/driver.log 2>&1 &
Dry run on a laptop (vectorised mock simulator + its 2-parameter coach space,
real local LLM, ~10 minutes; --versions-dir keeps scratch versions out of configs/):
    uv run python scripts/evolve_coach.py --root data/results/evolve_dry --sim mock_vec \
        --coach llm_local_mock --python .venv/bin/python --sim-gpus 0 --min-free-mib 2000 \
        --llm-model gpt-oss:20b --meta-reasoning medium --steps 60000 --eval-interval 10000 \
        --seeds 0 1 --settings flat-easy-naive --parallel 2 --generations 1 --min-pairs 2 \
        --versions-dir /tmp/evolve_versions \
        --extra-override coach.interval_steps=20000 --extra-override coach.warmup_steps=20000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from quadruped_rl.analysis.coach import load_run  # noqa: E402
from quadruped_rl.harness.config import load_yaml  # noqa: E402
from quadruped_rl.llm_feedback.evolve import (  # noqa: E402
    AcceptRule,
    EvolveState,
    Proposal,
    Version,
    compare_text,
    diagnostics,
    meta_prompt,
    paired_compare,
    results_table,
    summarize,
    validate_proposal,
    write_version,
)
from quadruped_rl.llm_feedback.playbook import Playbook  # noqa: E402
from quadruped_rl.llm_feedback.translator import LLMClient  # noqa: E402

DEFAULT_VERSIONS = "configs/coach/versions"
DEFAULT_PY = "${ISAAC_PY:-$HOME/anaconda3/envs/env_isaaclab/bin/python}"
DEFAULT_SETTINGS = ["rough-hard-traditional", "stairs-medium-traditional", "rough-medium-naive"]


# ------------------------------------------------------------------- jobs
def job_line(
    args, setting: str, seed: int, coach: str, version: str | None, root: Path, index: int = 0
) -> str:
    terrain, level, reward = setting.split("-")
    parts = [
        f"PYTHONPATH=src {args.python} scripts/train.py --sim {args.sim}",
        "--algorithm ppo --robot a1",
        f"--terrain {terrain} --reward {reward} --seed {seed}",
        f"--override sim.terrain_level={level}",
        f"--override run.total_timesteps={args.steps}",
        f"--override run.eval_interval_steps={args.eval_interval}",
        f"--override run.checkpoint_interval_steps={max(args.steps, 10_000_000)}",
        "--override logging.wandb=false",
        f"--override run.results_root={root.relative_to(REPO)}",
    ]
    if coach != "none":
        parts.append(f"--coach {coach}")
        # cross-run evidence, used only by versions that set coach.playbook.enabled
        parts.append(
            f"--override coach.playbook.path={Path(args.root).relative_to(REPO)}/playbook.json"
        )
    if version:
        parts.append(f"--override coach.version_dir={args.versions_dir}/{version}")
    if args.llm_model and coach == args.coach:
        parts.append(f"--override coach.llm.model={args.llm_model}")
    if args.llm_url and coach == args.coach:
        # several URLs = one daemon per GPU (llm_server.sh pool-*): jobs are
        # dealt round-robin so concurrent runs talk to different daemons
        url = args.llm_url[index % len(args.llm_url)]
        parts.append(f"--override coach.llm.base_url={url}")
    parts += [f"--override {o}" for o in args.extra_override]
    return " ".join(parts)


def run_jobs(args, jobs: Path, lines: list[str]) -> None:
    """Write the jobs file and run it to completion with scripts/run_jobs.py
    (resumable through its .status.json; a failed job never stops the batch)."""
    jobs.parent.mkdir(parents=True, exist_ok=True)
    if not jobs.exists():
        jobs.write_text("\n".join(lines) + "\n")
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_jobs.py"),
        "--jobs",
        str(jobs),
        "--parallel",
        str(args.parallel),
        "--gpus",
        args.sim_gpus,
        "--success-pattern",
        '"final"',
        "--reconcile",
    ]
    with open(str(jobs) + ".driver.log", "a") as f:
        subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, check=False)


def rows_of(root: Path) -> list[dict]:
    rows = []
    if root.exists():
        for d in sorted(root.iterdir()):
            r = load_run(d)
            if r is not None:
                rows.append(r)
    return rows


# ------------------------------------------------------------------ gates
def gpu_free_mib(ids: list[int]) -> dict[int, int]:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return dict.fromkeys(ids, 10**6)  # no nvidia-smi (mock dry run): never block
    free = {}
    for line in out.strip().splitlines():
        i, used, total = (int(x) for x in line.split(","))
        if i in ids:
            free[i] = total - used
    return free


def llm_alive(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_for_resources(args, state: EvolveState) -> None:
    """Block until the training GPUs have room and the LLM answers. Polls
    forever (the other research on the box decides when we may start)."""
    ids = [int(x) for x in args.sim_gpus.split(",")]
    notified = False
    while True:
        free = gpu_free_mib(ids)
        busy = [i for i, f in free.items() if f < args.min_free_mib]
        llm_ok = args.sim == "mock" or all(llm_alive(u) for u in args.llm_url)
        if not busy and llm_ok:
            if notified:
                state.log(f"resources available (free MiB {free}), continuing")
            return
        if not notified:
            state.log(f"waiting: busy GPUs {busy} (free MiB {free}), llm alive={llm_ok}")
            notified = True
        time.sleep(args.poll_s)


# --------------------------------------------------------------- meta step
def propose(
    args, state: EvolveState, incumbent: Version, params: list[str]
) -> tuple[str, Proposal]:
    inc = state.version(incumbent.name)
    last = None
    for name, v in state.data["versions"].items():
        if v.get("parent") == incumbent.name and v.get("status") in ("rejected", "accepted"):
            if last is None or v.get("generation", 0) > last.get("generation", 0):
                last = dict(v, name=name)
    client = LLMClient(
        "openai",
        args.meta_model,
        reasoning_effort=args.meta_reasoning,
        json_mode=True,
        base_url=args.llm_url[0],
        timeout_s=3600,
    )
    errors: list[str] = []
    name = state.next_name(p.name for p in Path(args.versions_dir).iterdir() if p.is_dir())
    for attempt in range(args.meta_retries + 1):
        system, user = meta_prompt(
            incumbent,
            inc.get("summary") or {},
            inc.get("diagnostics") or "(no runs)",
            state.history_text(),
            last,
            params,
            errors,
            base_coach=args.base_coach,
        )
        t0, effort = time.time(), client.reasoning_effort
        raw = client.complete(system, user, max_tokens=args.meta_max_tokens)
        if not raw.strip() and client.reasoning_effort in ("high", "medium"):
            # thinking consumed the whole budget: one step down for the next attempt
            client.reasoning_effort = {"high": "medium", "medium": "low"}[client.reasoning_effort]
            state.log(f"meta response empty; reasoning effort -> {client.reasoning_effort}")
        rec = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "candidate": name,
            "attempt": attempt,
            "seconds": round(time.time() - t0),
            "reasoning_effort": effort,
            "usage": client.last_usage,
            "prompt": {"system": system, "user": user},
            "raw": raw,
        }
        with open(Path(args.root) / "meta_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        try:
            prop = Proposal.from_json(raw)
        except Exception as e:  # malformed JSON / schema
            errors = [f"response was not the required JSON object: {type(e).__name__}: {e}"]
            state.log(f"meta proposal attempt {attempt} malformed: {errors[0][:200]}")
            continue
        errors = validate_proposal(prop, incumbent, params, args.base_coach)
        if not errors:
            return name, prop.resolve(incumbent)
        state.log(f"meta proposal attempt {attempt} invalid: {errors}")
    raise RuntimeError(f"meta-LLM produced no valid proposal in {args.meta_retries + 1} attempts")


def queued_candidates(
    versions: Path, parent: str, state: EvolveState, skip: Iterable[str] = ()
) -> list[tuple[str, str]]:
    """Hand-written version dirs (name, hypothesis) that declare ``parent:
    <incumbent>`` in their CHANGELOG and are not in state.json (or ``skip``:
    versions already judged in an earlier root) yet, oldest CHANGELOG first
    (queue order = the order the candidates were written)."""
    out = []
    dirs = [p for p in versions.iterdir() if p.is_dir() and (p / "CHANGELOG.md").exists()]
    for d in sorted(dirs, key=lambda p: ((p / "CHANGELOG.md").stat().st_mtime, p.name)):
        if d.name in state.data["versions"] or d.name in skip:
            continue
        meta: dict[str, str] = {}
        key = None
        for line in (d / "CHANGELOG.md").read_text().splitlines():
            head = line.split(":", 1)[0].strip()
            if ":" in line and head in ("parent", "hypothesis", "expected_delta_j"):
                key = head
                meta[key] = line.split(":", 1)[1].strip()
            elif key and line.strip() and not line.startswith("#"):
                meta[key] += " " + line.strip()  # wrapped continuation line
            else:
                key = None
        if meta.get("parent") == parent:
            out.append((d.name, meta.get("hypothesis", "")))
    return out


# ------------------------------------------------------------------ report
def write_report(args, state: EvolveState) -> None:
    root = Path(args.root)
    lines = [
        f"# Coach evolution — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"incumbent: **{state.incumbent}**, generation {state.data['generation']}, "
        f"settings {args.settings}, seeds {args.seeds}, {args.steps:,} steps, "
        f"coach model `{args.llm_model}`, meta model `{args.meta_model}`",
        "",
        "Acceptance (pre-registered): paired (setting, seed) dJ vs incumbent, one-sided paired "
        f"t p <= {args.p_max}, no setting below {args.worst_setting:+.2f}, "
        f">= {args.min_pairs} pairs.",
        "",
    ]
    ctrl = state.data.get("controls")
    if ctrl:
        lines += ["## Controls (no coach, same seeds)", results_table(ctrl), ""]
    items = sorted(state.data["versions"].items(), key=lambda kv: kv[1].get("generation", 0))
    for name, v in items:
        lines.append(
            f"## {name} — {v.get('status')} (gen {v.get('generation')}, parent {v.get('parent')})"
        )
        if v.get("hypothesis"):
            lines.append(f"hypothesis: {v['hypothesis']}")
        if v.get("summary"):
            lines += ["", results_table(v["summary"]), ""]
        if v.get("compare", {}).get("n"):
            lines.append(f"vs parent: {compare_text(v['compare'])}")
        if v.get("reason"):
            lines.append(f"verdict: {v['reason']}")
        if v.get("diagnostics"):
            lines += ["", "<details><summary>coach behaviour</summary>", ""]
            lines += [v["diagnostics"], "", "</details>"]
        lines.append("")
    (root / "report.md").write_text("\n".join(lines))


# ------------------------------------------------------------------- main
def build_playbook(args, state: EvolveState) -> None:
    """Rebuild <root>/playbook.json from every finished run under the root and
    --playbook-roots (older batches), so the next batch's coaches see all of it."""
    root = Path(args.root)
    dirs = Playbook.run_dirs_under(root, *[REPO / r for r in args.playbook_roots])
    pb = Playbook.build(dirs)
    pb.save(root / "playbook.json")
    n_coached = sum(pb.coached(c) for c in pb.cases)
    state.log(f"playbook: {len(pb.cases)} runs ({n_coached} coached) from {len(dirs)} dirs")


def measure(args, state: EvolveState, name: str, coach: str, version: str | None) -> list[dict]:
    """Run (or resume) the batch of one version and return its rows."""
    root = Path(args.root) / name
    runs = root / "runs"
    if coach != "none":
        build_playbook(args, state)
    lines = [
        job_line(args, s, seed, coach, version, runs, i)
        for i, (s, seed) in enumerate((s, seed) for s in args.settings for seed in args.seeds)
    ]
    wait_for_resources(args, state)
    state.log(f"running {name}: {len(lines)} jobs on GPUs {args.sim_gpus}")
    run_jobs(args, root / "jobs.txt", lines)
    rows = rows_of(runs)
    state.log(f"{name}: {len(rows)}/{len(lines)} runs finished")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="data/results/evolve")
    p.add_argument("--incumbent", default="v5", help="starting version (configs/coach/versions/)")
    p.add_argument("--coach", default="llm_local", help="coach preset for LLM runs")
    p.add_argument("--versions-dir", default=DEFAULT_VERSIONS, help="where v<N> dirs live")
    p.add_argument("--python", default=DEFAULT_PY, help="interpreter for train.py (Isaac python)")
    p.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help="terrain-level-reward")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=40_000_000)
    p.add_argument("--eval-interval", type=int, default=1_000_000)
    p.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="key=value added to every train.py call (e.g. coach.interval_steps=20000)",
    )
    p.add_argument("--sim", default="isaaclab")
    p.add_argument("--sim-gpus", default="3", help="training GPUs (the LLM has the others)")
    p.add_argument("--parallel", type=int, default=2, help="concurrent runs (2 per 4090)")
    p.add_argument("--min-free-mib", type=int, default=16000, help="per training GPU to launch")
    p.add_argument("--poll-s", type=int, default=300)
    p.add_argument(
        "--llm-url",
        nargs="+",
        default=["http://127.0.0.1:11435/v1"],
        help="local LLM endpoint(s); several = round-robin over per-GPU daemons",
    )
    p.add_argument("--llm-model", default=None, help="override coach.llm.model of the preset")
    p.add_argument("--meta-model", default=None, help="model writing versions (default: coach)")
    p.add_argument("--meta-reasoning", default=None, help="gpt-oss only: low|medium|high")
    p.add_argument("--meta-max-tokens", type=int, default=32000)
    p.add_argument("--meta-retries", type=int, default=3)
    p.add_argument("--controls", nargs="*", default=["none"], help="no-coach references run once")
    p.add_argument(
        "--playbook-roots",
        nargs="*",
        default=[],
        help="extra result roots (relative to the repo) indexed into <root>/playbook.json",
    )
    p.add_argument(
        "--skip-versions",
        nargs="*",
        default=[],
        help="version dirs never taken as queued candidates (judged in an earlier root)",
    )
    p.add_argument("--generations", type=int, default=1000)
    p.add_argument("--min-pairs", type=int, default=6)
    p.add_argument("--p-max", type=float, default=0.2)
    p.add_argument("--worst-setting", type=float, default=-0.05)
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_absolute():
        root = REPO / root
    args.root = str(root)
    root.mkdir(parents=True, exist_ok=True)
    versions = Path(args.versions_dir)
    versions = versions if versions.is_absolute() else REPO / versions
    state = EvolveState(root / "state.json")
    rule = AcceptRule(args.min_pairs, args.p_max, args.worst_setting)
    base_coach = load_yaml(REPO / "configs" / "coach" / f"{args.coach}.yaml")["coach"]
    params = list(base_coach["params"])
    args.base_coach = base_coach
    # model names: CLI > coach preset (the meta-LLM defaults to the coach's model)
    args.llm_model = args.llm_model or base_coach["llm"]["model"]
    args.meta_model = args.meta_model or args.llm_model
    # local-only policy (2026-09-07): no paid API may be reached from this loop
    coach_url = base_coach.get("llm", {}).get("base_url")
    for label, url in [("coach", coach_url)] + [("meta", u) for u in args.llm_url]:
        if not str(url).startswith(("http://127.0.0.1", "http://localhost")):
            sys.exit(f"{label} LLM url {url!r} is not local; this loop runs on local models only")

    if state.incumbent is None:
        state.data["incumbent"] = args.incumbent
        v = state.version(args.incumbent)
        v.update(parent=None, generation=0, status="incumbent")
        state.log(f"start: incumbent {args.incumbent}, {args.settings}, seeds {args.seeds}")

    # controls: the same seeds without a coach (reference lines in the report)
    for c in args.controls:
        if c not in state.data.get("controls_done", []):
            rows = measure(args, state, f"control-{c}", c, None)
            state.data.setdefault("controls", {}).update(
                {f"{c}:{k}": v for k, v in summarize(rows).items()}
            )
            state.data.setdefault("controls_done", []).append(c)
            state.save()
            write_report(args, state)

    for _ in range(args.generations):
        inc_name = state.incumbent
        inc = state.version(inc_name)
        incumbent = Version.load(versions / inc_name)
        if not inc.get("summary"):
            rows = measure(args, state, inc_name, args.coach, inc_name)
            inc.update(summary=summarize(rows), diagnostics=diagnostics(rows))
            state.save()
            write_report(args, state)
        inc_rows = rows_of(root / inc_name / "runs")

        gen = state.data["generation"] + 1
        pending = [
            n
            for n, v in state.data["versions"].items()
            if v.get("status") == "candidate" and v.get("parent") == inc_name
        ]
        queued = queued_candidates(versions, inc_name, state, args.skip_versions)
        if pending:
            name = pending[0]
            state.log(f"resuming candidate {name}")
        elif queued:
            name, hypothesis = queued[0]
            state.version(name).update(
                parent=inc_name, generation=gen, status="candidate", hypothesis=hypothesis
            )
            state.data["generation"] = gen
            state.log(f"adopting queued candidate {name}: {hypothesis}")
            state.save()
        else:
            wait_for_resources(args, state)
            state.log(f"generation {gen}: asking the meta-LLM for the successor of {inc_name}")
            name, prop = propose(args, state, incumbent, params)
            write_version(versions, name, prop, inc_name, params, gen)
            state.version(name).update(
                parent=inc_name,
                generation=gen,
                status="candidate",
                hypothesis=prop.hypothesis,
                expected_delta_j=prop.expected_delta_j,
            )
            if prop.code_ideas:
                with open(root / "proposals.md", "a") as f:
                    f.write(f"\n## {name} (gen {gen})\n")
                    f.write("".join(f"- {i}\n" for i in prop.code_ideas))
            state.data["generation"] = gen
            state.log(f"{name} written: {prop.hypothesis}")
            state.save()

        rows = measure(args, state, name, args.coach, name)
        cmp = paired_compare(rows, inc_rows)
        ok, reason = rule.decide(cmp)
        cand = state.version(name)
        cand.update(
            summary=summarize(rows),
            diagnostics=diagnostics(rows),
            compare={k: v for k, v in cmp.items() if k != "pairs"},
            status="accepted" if ok else "rejected",
            reason=reason,
        )
        if ok:
            state.data["incumbent"] = name
            state.version(inc_name)["status"] = "superseded"
        state.log(f"{name} {'ACCEPTED' if ok else 'rejected'}: {reason}")
        state.save()
        write_report(args, state)


if __name__ == "__main__":
    main()
