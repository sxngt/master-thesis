#!/usr/bin/env python
"""Parallel training-job scheduler for single- or multi-GPU machines.

Runs shell-command jobs (one per line in a jobs file) with N concurrent
slots. Measured on the RTX 4080: one run uses ~3 GB VRAM at ~70 % util, so
two concurrent runs fit comfortably (~1.4-1.6x batch throughput); scales to
N GPUs via least-loaded CUDA_VISIBLE_DEVICES (4x GPUs ~= /4 wall time).

- Resume: finished jobs are recorded in <jobs>.status.json and skipped.
- Failure isolation: a failed job never stops the batch.
- matrix_runner's export_jobs() output is directly consumable here.

Example:
    python scripts/run_jobs.py --jobs jobs.txt --parallel 2 --gpus 0
    python scripts/run_jobs.py --jobs jobs.txt --parallel 4 --gpus 0,1,2,3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def job_key(cmd: str) -> str:
    return hashlib.sha1(cmd.encode()).hexdigest()[:12]


def job_slug(cmd: str, idx: int) -> str:
    words = re.findall(r"--algorithm (\S+)|--terrain (\S+)|--seed (\S+)", cmd)
    parts = [w for group in words for w in group if w]
    return f"{idx:03d}_" + ("_".join(parts) if parts else job_key(cmd))


class JobRunner:
    def __init__(
        self,
        jobs_path: Path,
        parallel: int,
        gpus: list[str],
        log_dir: Path,
        success_pattern: str | None = None,
        skip_running: bool = False,
    ):
        self.success_pattern = success_pattern
        self.skip_running = skip_running
        self.jobs = [
            ln.strip()
            for ln in jobs_path.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        self.parallel = parallel
        self.gpus = gpus
        self.log_dir = log_dir
        self.status_path = jobs_path.with_suffix(jobs_path.suffix + ".status.json")
        self.status: dict[str, dict] = (
            json.loads(self.status_path.read_text()) if self.status_path.exists() else {}
        )
        self._lock = threading.Lock()
        self._load: dict[str, int] = {g: 0 for g in gpus}  # running jobs per GPU
        self._assigned: dict[str, int] = {g: 0 for g in gpus}  # tie-break: round-robin

    def _save(self) -> None:
        with self._lock:
            self.status_path.write_text(json.dumps(self.status, indent=2))

    def _acquire_gpu(self) -> str:
        # least-loaded GPU (a plain round-robin counter piles jobs onto one
        # GPU as soon as run times diverge)
        with self._lock:
            gpu = min(self.gpus, key=lambda g: (self._load[g], self._assigned[g]))
            self._load[gpu] += 1
            self._assigned[gpu] += 1
            return gpu

    def _release_gpu(self, gpu: str) -> None:
        with self._lock:
            self._load[gpu] -= 1

    def _run_one(self, idx: int, cmd: str) -> None:
        key = job_key(cmd)
        gpu = self._acquire_gpu()
        slug = job_slug(cmd, idx)
        log = self.log_dir / f"{slug}.log"
        print(f"[jobs] START {slug} gpu={gpu}", flush=True)
        t0 = time.time()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        with open(log, "w") as f:
            rc = subprocess.run(
                cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT
            ).returncode
        self._release_gpu(gpu)
        state = "done" if rc == 0 else "failed"
        # some frameworks (e.g. Isaac Sim) swallow exceptions and exit 0 —
        # optionally require a success marker in the job log
        if state == "done" and self.success_pattern:
            import re as _re

            if not _re.search(self.success_pattern, log.read_text(errors="ignore")):
                state, rc = "failed", -2
        self.status[key] = {
            "state": state,
            "cmd": cmd,
            "rc": rc,
            "gpu": gpu,
            "seconds": round(time.time() - t0),
            "log": str(log),
        }
        self._save()
        print(f"[jobs] END   {slug} rc={rc} ({round(time.time() - t0)}s)", flush=True)

    def reconcile(self) -> int:
        """Mark jobs as done whose log already carries the success marker.

        Recovers runs that finished after the driver itself was killed (the
        children keep running and writing their logs). Returns the count.
        """
        if not self.success_pattern:
            raise ValueError("--reconcile needs --success-pattern")
        n = 0
        for idx, cmd in enumerate(self.jobs):
            key = job_key(cmd)
            if self.status.get(key, {}).get("state") == "done":
                continue
            log = self.log_dir / f"{job_slug(cmd, idx)}.log"
            if log.exists() and re.search(self.success_pattern, log.read_text(errors="ignore")):
                self.status[key] = {
                    "state": "done",
                    "cmd": cmd,
                    "rc": 0,
                    "gpu": "?",
                    "seconds": -1,
                    "log": str(log),
                }
                n += 1
        self._save()
        print(f"[jobs] reconciled {n} job(s) from logs", flush=True)
        return n

    def _is_running(self, cmd: str) -> bool:
        """True if some process was started with exactly this command line
        (``sh -c <cmd>``), i.e. a child of an earlier driver is still on it.
        Linux only (/proc); the log's mtime is useless here because Isaac
        block-buffers stdout for minutes."""
        if not self.skip_running:
            return False
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                argv = (proc / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if cmd.encode() in argv:
                return True
        return False

    def run(self) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        pending = []
        for i, c in enumerate(self.jobs):
            if self.status.get(job_key(c), {}).get("state") == "done":
                continue
            if self._is_running(c):
                print(f"[jobs] SKIP  {job_slug(c, i)} (already running)", flush=True)
                continue
            pending.append((i, c))
        print(
            f"[jobs] {len(pending)} to run / {len(self.jobs)} total, "
            f"parallel={self.parallel}, gpus={','.join(self.gpus)}"
        )
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            list(pool.map(lambda ic: self._run_one(*ic), pending))
        failed = [s for s in self.status.values() if s["state"] == "failed"]
        print(f"[jobs] ALL DONE — failed: {len(failed)}")
        for s in failed:
            print(f"[jobs]   FAILED rc={s['rc']}: {s['cmd'][:100]}  ({s['log']})")
        return 1 if failed else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True, help="file with one shell command per line")
    p.add_argument(
        "--parallel", type=int, default=2, help="concurrent slots (4080: 2 is the sweet spot)"
    )
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids, least-loaded first")
    p.add_argument("--log-dir", default=None)
    p.add_argument(
        "--success-pattern",
        default=None,
        help="regex that must appear in the job log for success "
        "(guards against frameworks that swallow errors)",
    )
    p.add_argument(
        "--reconcile",
        action="store_true",
        help="before running, mark jobs done whose log has the success marker",
    )
    p.add_argument(
        "--skip-running",
        action="store_true",
        help="skip jobs whose command is already running (children of a killed "
        "driver keep going); mark them with --reconcile on a later run",
    )
    args = p.parse_args()

    jobs_path = Path(args.jobs)
    log_dir = Path(args.log_dir or jobs_path.parent / (jobs_path.stem + "_logs"))
    runner = JobRunner(
        jobs_path,
        args.parallel,
        args.gpus.split(","),
        log_dir,
        success_pattern=args.success_pattern,
        skip_running=args.skip_running,
    )
    if args.reconcile:
        runner.reconcile()
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
