#!/usr/bin/env python3
"""Pretty CLI status board for run_jobs.py batches.

Usage:
    python3 scripts/batch_status.py [--jobs data/results/night_batch/jobs.txt]
    watch -n 30 python3 scripts/batch_status.py        # live board
"""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

# ---------- ANSI ----------
R, G, Y, B, C, DIM, BOLD, END = ("\033[31m", "\033[32m", "\033[33m", "\033[34m",
                                 "\033[36m", "\033[2m", "\033[1m", "\033[0m")


def key(cmd: str) -> str:
    return hashlib.sha1(cmd.encode()).hexdigest()[:12]


def label(cmd: str) -> str:
    def pick(pat, default="?"):
        m = re.search(pat, cmd)
        return m.group(1) if m else default
    robot = pick(r"--robot (\S+)")
    algo = pick(r"--algorithm (\S+)")
    terrain = pick(r"--terrain (\S+)")
    level = pick(r"terrain_level=(\S+?)( |$)", "easy").split()[0]
    seed = pick(r"--seed (\S+)")
    return f"{robot:9s} {terrain}-{level:<7s} {algo:4s} s{seed}"


def bar(frac: float, width: int = 40) -> str:
    full = int(frac * width)
    return G + "█" * full + DIM + "░" * (width - full) + END


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", default="data/results/night_batch/jobs.txt")
    p.add_argument("--tail", type=int, default=8, help="recent completions to show")
    args = p.parse_args()

    jobs_path = Path(args.jobs)
    jobs = [ln.strip() for ln in jobs_path.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
    status_path = jobs_path.with_suffix(jobs_path.suffix + ".status.json")
    status = json.loads(status_path.read_text()) if status_path.exists() else {}

    running_cmds = subprocess.run(["pgrep", "-af", "scripts/train.py"],
                                  capture_output=True, text=True).stdout
    done, failed, running, pending = [], [], [], []
    for i, cmd in enumerate(jobs):
        st = status.get(key(cmd), {})
        if st.get("state") == "done":
            done.append((i, cmd, st))
        elif st.get("state") == "failed":
            failed.append((i, cmd, st))
        else:
            toks = re.findall(r"--robot \S+|--terrain \S+|--seed \S+|--algorithm \S+", cmd)
            lvl = re.search(r"--override sim.terrain_level=\S+", cmd)
            if lvl:
                toks.append(lvl.group(0))
            is_run = any(all(t in line for t in toks)
                         and (lvl or "terrain_level" not in line)
                         for line in running_cmds.splitlines())
            (running if is_run and toks else pending).append((i, cmd, st))

    n = len(jobs)
    print(f"\n{BOLD}{C}◢ BATCH STATUS ◣{END}  {DIM}{jobs_path}{END}")
    print(f"  {bar(len(done)/n)}  {BOLD}{len(done)}/{n}{END} "
          f"({G}done {len(done)}{END} · {Y}running {len(running)}{END} · "
          f"{DIM}pending {len(pending)}{END} · {R}failed {len(failed)}{END})")

    if done:
        secs = [s["seconds"] for _, _, s in done]
        avg = sum(secs) / len(secs)
        eta_s = (len(pending) + len(running)) * avg / max(len(running), 1)
        eta = time.strftime("%H:%M", time.localtime(time.time() + eta_s))
        print(f"  {DIM}avg {avg/60:.0f} min/run · rough ETA ~{eta}{END}")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"  {DIM}GPU {gpu}{END}\n")

    if running:
        print(f"{BOLD}{Y}▶ RUNNING{END}")
        for i, cmd, _ in running:
            print(f"  {Y}▶{END} [{i:02d}] {label(cmd)}")
    if failed:
        print(f"{BOLD}{R}✗ FAILED{END}")
        for i, cmd, st in failed:
            print(f"  {R}✗{END} [{i:02d}] {label(cmd)}  rc={st['rc']}  {DIM}{st['log']}{END}")

    if done:
        print(f"{BOLD}{G}✓ RECENT{END} {DIM}(last {min(args.tail, len(done))}){END}")
        for i, cmd, st in done[-args.tail:]:
            print(f"  {G}✓{END} [{i:02d}] {label(cmd)}  {DIM}{st['seconds']/60:.0f}min{END}")

    if pending:
        nxt = ", ".join(f"[{i:02d}]" for i, _, _ in pending[:6])
        print(f"{BOLD}⏳ NEXT{END}  {DIM}{nxt}{' …' if len(pending) > 6 else ''}{END}")
    print()


if __name__ == "__main__":
    main()
