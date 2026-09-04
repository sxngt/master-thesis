"""Parallel job scheduler: concurrency, resume, failure isolation."""

import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "run_jobs.py"


def _run(jobs_file, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--jobs", str(jobs_file), *extra],
        capture_output=True,
        text=True,
    )


def test_parallel_runs_concurrently(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("sleep 1\nsleep 1\n")
    t0 = time.time()
    r = _run(jobs, "--parallel", "2")
    elapsed = time.time() - t0
    assert r.returncode == 0
    assert elapsed < 1.9, f"not concurrent: {elapsed:.1f}s"  # serial would be >2s


def test_resume_skips_done_and_isolates_failure(tmp_path):
    jobs = tmp_path / "jobs.txt"
    marker = tmp_path / "ran.txt"
    jobs.write_text(f"echo once >> {marker}\nfalse\n")
    r1 = _run(jobs, "--parallel", "1")
    assert r1.returncode == 1  # `false` job failed, batch still completed
    status = json.loads((tmp_path / "jobs.txt.status.json").read_text())
    states = sorted(s["state"] for s in status.values())
    assert states == ["done", "failed"]
    r2 = _run(jobs, "--parallel", "1")  # rerun: done skipped, failed retried
    assert "1 to run / 2 total" in r2.stdout
    assert marker.read_text().count("once") == 1


def test_gpu_round_robin_env(tmp_path):
    jobs = tmp_path / "jobs.txt"
    out = tmp_path / "gpus.txt"
    jobs.write_text(f'echo "$CUDA_VISIBLE_DEVICES" >> {out}\n' * 4)
    # 4 identical lines dedupe to one key — make them unique
    jobs.write_text("".join(f'echo "job{i} $CUDA_VISIBLE_DEVICES" >> {out}\n' for i in range(4)))
    r = _run(jobs, "--parallel", "1", "--gpus", "0,1")
    assert r.returncode == 0
    seen = sorted(line.split()[-1] for line in out.read_text().splitlines())
    assert seen == ["0", "0", "1", "1"]


def test_reconcile_marks_finished_logs_done(tmp_path):
    jobs = tmp_path / "jobs.txt"
    jobs.write_text("echo one\necho two\n")
    log_dir = tmp_path / "jobs_logs"
    log_dir.mkdir()
    # job "one" finished after its driver died: log carries the marker, no status
    (log_dir / "000_4d4d7281bbfd.log").write_text('{"final": {}}\n')
    r = _run(jobs, "--reconcile", "--success-pattern", '"final"')
    assert "reconciled 1 job" in r.stdout
    status = json.loads((jobs.parent / "jobs.txt.status.json").read_text())
    by_cmd = {v["cmd"]: v for v in status.values()}
    assert by_cmd["echo one"]["seconds"] == -1  # recovered, not re-run
    assert by_cmd["echo two"]["seconds"] >= 0  # actually executed


def test_skip_running_leaves_active_jobs_alone(tmp_path):
    jobs = tmp_path / "jobs.txt"
    marker = tmp_path / "ran.txt"
    one, two = f"sleep 3; echo one >> {marker}", f"echo two >> {marker}"
    jobs.write_text(f"{one}\n{two}\n")
    # job "one" is still running under an earlier (killed) driver
    orphan = subprocess.Popen(["sh", "-c", one])
    try:
        r = _run(jobs, "--skip-running")
        assert "SKIP  000_" in r.stdout and "1 to run / 2 total" in r.stdout
        assert marker.read_text() == "two\n"
        status = json.loads((jobs.parent / "jobs.txt.status.json").read_text())
        assert [v["cmd"] for v in status.values()] == [two]  # "one" left for reconcile
    finally:
        orphan.kill()
