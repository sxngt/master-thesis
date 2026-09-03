#!/usr/bin/env python
"""Parallel training-job scheduler for single- or multi-GPU machines.

Runs shell-command jobs (one per line in a jobs file) with N concurrent
slots. Measured on the RTX 4080: one run uses ~3 GB VRAM at ~70 % util, so
two concurrent runs fit comfortably (~1.4-1.6x batch throughput); scales to
N GPUs via round-robin CUDA_VISIBLE_DEVICES (4x GPUs ~= /4 wall time).

- Resume: finished jobs are recorded in <jobs>.status.json and skipped.
- Failure isolation: a failed job never stops the batch.
- matrix_runner's export_jobs() output is directly consumable here.

Example:
    python scripts/run_jobs.py --jobs jobs.txt --parallel 2 --gpus 0
    python scripts/run_jobs.py --jobs jobs.txt --parallel 4 --gpus 0,1,2,3
"""

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
    ):
        self.success_pattern = success_pattern
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
        self._slot_gpu: dict[int, str] = {}
        self._next_slot = 0

    def _save(self) -> None:
        with self._lock:
            self.status_path.write_text(json.dumps(self.status, indent=2))

    def _acquire_gpu(self) -> str:
        with self._lock:
            gpu = self.gpus[self._next_slot % len(self.gpus)]
            self._next_slot += 1
            return gpu

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

    def run(self) -> int:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        pending = [
            (i, c)
            for i, c in enumerate(self.jobs)
            if self.status.get(job_key(c), {}).get("state") != "done"
        ]
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
    p.add_argument("--gpus", default="0", help="comma-separated GPU ids, round-robin")
    p.add_argument("--log-dir", default=None)
    p.add_argument(
        "--success-pattern",
        default=None,
        help="regex that must appear in the job log for success "
        "(guards against frameworks that swallow errors)",
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
    )
    raise SystemExit(runner.run())


if __name__ == "__main__":
    main()
