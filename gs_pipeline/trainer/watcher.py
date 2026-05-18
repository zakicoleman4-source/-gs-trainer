"""Inbox watcher: filesystem listener that turns new .zip drops into jobs.

Runs as a long-lived process inside the Docker container alongside the
Streamlit UI. Watches ``inbox/`` for new ``*.zip`` files; for each:

1. Generates a job id.
2. Renames the zip to ``inbox/<job_id>__<original>.zip`` (single-writer claim
   — if the watcher crashes mid-job, the rename is the recovery hint).
3. Calls ``pipeline.run_job`` as a subprocess so a trainer crash never kills
   the watcher itself. Stdout/stderr go to ``logs/<job_id>/runner.{out,err}``.
4. On exit, leaves the renamed zip in ``inbox/`` so the user can resubmit
   manually if needed (we never auto-delete inputs).

Only one job runs at a time on the single GPU. New zips arriving during a
job sit in the queue (the watcher rescans inbox/ between jobs).

Two entry points:

- ``run_forever(...)``: blocks, watches indefinitely. The Docker process.
- ``process_one(...)``: drain the queue once and return. Used by the
  integration test.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from gs_pipeline.trainer.job_state import JobState, safe_read_state, state_path_for
from gs_pipeline.trainer.pipeline import (
    job_log_dir,
    job_outbox_dir,
    job_work_dir,
    run_job,
)

_log = logging.getLogger(__name__)


# Subdirectory under inbox/ for zips currently claimed by a worker.
CLAIM_PREFIX = "claim__"
# How often the watcher rescans inbox/ when idle.
DEFAULT_POLL_INTERVAL = 2.0


@dataclass
class WatcherPaths:
    inbox: Path
    work: Path
    outbox: Path
    logs: Path
    config_yaml: Optional[Path] = None

    def __post_init__(self) -> None:
        self.inbox = Path(self.inbox)
        self.work = Path(self.work)
        self.outbox = Path(self.outbox)
        self.logs = Path(self.logs)
        for d in (self.inbox, self.work, self.outbox, self.logs):
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# In-process (used by tests)
# ---------------------------------------------------------------------------

def process_one(
    paths: WatcherPaths,
    *,
    quality_preset: str = "Auto",
    train_fn: Optional[Callable] = None,
) -> Optional[JobState]:
    """Pick the oldest unclaimed .zip in inbox/ and run one job.

    Returns the JobState (DONE or FAILED), or None if the inbox is empty.
    Runs entirely in-process — no subprocess. This is what the integration
    test exercises.
    """
    zip_path = _claim_next(paths.inbox)
    if zip_path is None:
        return None
    job_id = _job_id_from_claim(zip_path)
    return run_job(
        job_id=job_id,
        bundle_zip=zip_path,
        work_root=paths.work,
        outbox_root=paths.outbox,
        log_root=paths.logs,
        config_yaml=paths.config_yaml,
        quality_preset=quality_preset,
        train_fn=train_fn,
        bundle_filename=_original_name(zip_path),
    )


# ---------------------------------------------------------------------------
# Subprocess-based (used in the Docker container)
# ---------------------------------------------------------------------------

def process_one_subprocess(
    paths: WatcherPaths,
    *,
    quality_preset: str = "Auto",
    python: str = sys.executable,
) -> Optional[Path]:
    """Like ``process_one`` but spawns ``run_job`` in a subprocess.

    Returns the claimed zip path, or None if the inbox was empty. The caller
    can read ``state.json`` to observe the outcome.
    """
    zip_path = _claim_next(paths.inbox)
    if zip_path is None:
        return None
    job_id = _job_id_from_claim(zip_path)
    log_dir = job_log_dir(paths.logs, job_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = (log_dir / "runner.out").open("ab")
    err_log = (log_dir / "runner.err").open("ab")
    env = os.environ.copy()
    cmd = [
        python, "-m", "gs_pipeline.trainer.pipeline",
        "--job-id", job_id,
        "--bundle-zip", str(zip_path),
        "--work-root", str(paths.work),
        "--outbox-root", str(paths.outbox),
        "--log-root", str(paths.logs),
        "--quality-preset", quality_preset,
    ]
    if paths.config_yaml is not None:
        cmd += ["--config-yaml", str(paths.config_yaml)]
    _log.info("spawning trainer for job %s: %s", job_id, " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, env=env)
    proc.wait()  # block until done; only one job at a time on this GPU
    out_log.close(); err_log.close()
    _log.info("trainer for job %s exited with code %s", job_id, proc.returncode)
    return zip_path


def run_forever(
    paths: WatcherPaths,
    *,
    quality_preset: str = "Auto",
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    stop_event=None,
) -> None:
    """Block forever, processing zips as they arrive.

    Pass a ``threading.Event`` as ``stop_event`` to break the loop cleanly
    (the integration test does this).
    """
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        claimed = process_one_subprocess(paths, quality_preset=quality_preset)
        if claimed is None:
            # Idle: sleep and rescan.
            if stop_event is not None:
                if stop_event.wait(poll_interval):
                    return
            else:
                time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Inbox queue mechanics
# ---------------------------------------------------------------------------

def _claim_next(inbox: Path) -> Optional[Path]:
    """Atomically rename the oldest unclaimed .zip to a claim__<job>__name path.

    Returns the renamed path (now safe to operate on), or None if no
    unclaimed zip exists.
    """
    inbox = Path(inbox)
    candidates = sorted(
        (p for p in inbox.glob("*.zip") if not p.name.startswith(CLAIM_PREFIX)),
        key=lambda p: p.stat().st_mtime,
    )
    for candidate in candidates:
        job_id = uuid.uuid4().hex[:12]
        claimed_path = inbox / f"{CLAIM_PREFIX}{job_id}__{candidate.name}"
        try:
            candidate.rename(claimed_path)
        except FileNotFoundError:
            continue  # someone else grabbed it (shouldn't happen with one worker, but safe)
        return claimed_path
    return None


def _job_id_from_claim(claim_path: Path) -> str:
    """Extract the 12-hex job id from a claim__<id>__<name>.zip path."""
    name = claim_path.name
    if not name.startswith(CLAIM_PREFIX):
        raise ValueError(f"not a claim path: {claim_path}")
    rest = name[len(CLAIM_PREFIX):]
    return rest.split("__", 1)[0]


def _original_name(claim_path: Path) -> str:
    """Recover the original .zip name from a claim path."""
    name = claim_path.name
    if not name.startswith(CLAIM_PREFIX):
        return name
    rest = name[len(CLAIM_PREFIX):]
    return rest.split("__", 1)[1] if "__" in rest else rest


# ---------------------------------------------------------------------------
# CLI (subprocess target)
# ---------------------------------------------------------------------------

def watcher_main(argv: Optional[list] = None) -> int:
    """``python -m gs_pipeline.trainer.watcher`` entry point — starts ``run_forever``."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the gs_pipeline inbox watcher.")
    parser.add_argument("--inbox", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--outbox", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--config-yaml", type=Path, default=None)
    parser.add_argument("--quality-preset", default="Auto", choices=["Auto", "Maximum"])
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL)
    args = parser.parse_args(argv)
    paths = WatcherPaths(
        inbox=args.inbox, work=args.work, outbox=args.outbox,
        logs=args.logs, config_yaml=args.config_yaml,
    )
    run_forever(paths, quality_preset=args.quality_preset, poll_interval=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(watcher_main())
