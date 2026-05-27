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

from gs_pipeline.trainer.job_state import JobState, State, safe_read_state, state_path_for

# Lazy-import pipeline to avoid pulling in numpy/plyfile at watcher startup.
# pipeline.py imports init_from_pcd.py (numpy + plyfile) and parse_metashape.py
# (numpy) at module level. The watcher daemon only needs the lightweight path
# helpers and run_job, which are imported on first use inside function bodies.
# This prevents the watcher from crashing in Docker if heavy deps aren't yet
# importable (e.g. pip/Python version mismatch or missing native libraries).

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
    # In-process mode (tests): no settle delay — files are created and
    # consumed in the same process with no upload-in-progress risk.
    zip_path = _claim_next(paths.inbox, settle_seconds=0.0)
    if zip_path is None:
        return None
    job_id = _job_id_from_claim(zip_path)
    from gs_pipeline.trainer.pipeline import run_job  # lazy import (heavy deps)
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
    log_dir = Path(paths.logs) / job_id  # inline; avoids importing pipeline at module level
    log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()

    # Per-job quality preset: read from opts sidecar written by the UI,
    # fall back to the watcher-level default (from supervisord / CLI arg).
    job_quality = _read_opts_quality(zip_path, default=quality_preset)

    cmd = [
        python, "-m", "gs_pipeline.trainer.pipeline",
        "--job-id", job_id,
        "--bundle-zip", str(zip_path),
        "--work-root", str(paths.work),
        "--outbox-root", str(paths.outbox),
        "--log-root", str(paths.logs),
        "--quality-preset", job_quality,
    ]
    if paths.config_yaml is not None:
        cmd += ["--config-yaml", str(paths.config_yaml)]
    _log.info("spawning trainer for job %s: %s", job_id, " ".join(cmd))
    with (log_dir / "runner.out").open("ab") as out_log, \
         (log_dir / "runner.err").open("ab") as err_log:
        proc = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, env=env)
        proc.wait()
    _log.info("trainer for job %s exited with code %s", job_id, proc.returncode)
    return zip_path


def _recover_stale_claims(paths: WatcherPaths) -> int:
    """Mark jobs left in non-terminal state as FAILED on startup.

    When the container restarts mid-training, claimed zips stay in inbox/ and
    their ``state.json`` is stuck in TRAINING/PREFLIGHT. This function:
    1. Scans inbox/ for ``claim__*`` files.
    2. For each, reads ``state.json`` from the work directory.
    3. If the state is non-terminal (QUEUED/PREFLIGHT/TRAINING/RESUMING),
       marks it FAILED so the UI shows the correct status.
    4. Unclaims the zip (renames back to original name) so the user can
       resubmit if desired.

    Returns the number of stale jobs recovered.
    """
    inbox = Path(paths.inbox)
    recovered = 0
    for claim_path in list(inbox.glob(f"{CLAIM_PREFIX}*.zip")):
        try:
            job_id = _job_id_from_claim(claim_path)
        except ValueError:
            continue
        state_path = state_path_for(paths.work, job_id)
        js = safe_read_state(state_path)
        if js is None:
            # No state.json yet — job never got past claiming. Unclaim so it
            # can be retried.
            _unclaim(claim_path, inbox)
            recovered += 1
            _log.warning(
                "Recovered stale claim %s (no state.json) — unclaimed for resubmission",
                claim_path.name,
            )
            continue
        if js.state in (State.DONE, State.FAILED):
            # Terminal state — nothing to recover. Leave the claim in place.
            continue
        # Non-terminal: mark as failed.
        try:
            js.mark_failed(
                "Job interrupted by container restart or worker crash. "
                "The original bundle has been unclaimed for resubmission."
            )
        except Exception:
            js.error_msg = "Job interrupted by container restart or worker crash."
            js.state = State.FAILED
            js.status_msg = "failed"
        from gs_pipeline.trainer.job_state import write_state
        write_state(js, state_path)
        _unclaim(claim_path, inbox)
        recovered += 1
        _log.warning(
            "Recovered stale job %s (was %s) — marked FAILED, unclaimed",
            job_id, js.state.value if hasattr(js.state, 'value') else js.state,
        )
    return recovered


def _unclaim(claim_path: Path, inbox: Path) -> None:
    """Rename a claim__<id>__<name>.zip back to <name>.zip for resubmission."""
    original_name = _original_name(claim_path)
    target = inbox / original_name
    if target.exists():
        # Avoid clobbering; add a suffix.
        i = 1
        stem = Path(original_name).stem
        while (inbox / f"{stem}_{i}.zip").exists():
            i += 1
        target = inbox / f"{stem}_{i}.zip"
    try:
        claim_path.rename(target)
    except OSError as e:
        _log.warning("Failed to unclaim %s: %s", claim_path, e)


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
    # On startup, recover any jobs left in non-terminal state from a prior crash.
    n_recovered = _recover_stale_claims(paths)
    if n_recovered:
        _log.info("Recovered %d stale job(s) on startup", n_recovered)

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

def _claim_next(inbox: Path, *, settle_seconds: float = 2.0) -> Optional[Path]:
    """Atomically rename the oldest unclaimed .zip to a claim__<job>__name path.

    Returns the renamed path (now safe to operate on), or None if no
    unclaimed zip exists. Also renames a companion ``.opts.json`` sidecar
    (written by the UI with per-job quality settings) so it stays alongside.

    Files whose mtime is less than ``settle_seconds`` ago are skipped — they
    may still be in the process of being written (upload in progress). They
    will be picked up on the next scan.
    """
    inbox = Path(inbox)
    now = time.time()

    def _safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return float("inf")
    candidates = sorted(
        (p for p in inbox.glob("*.zip")
         if not p.name.startswith(CLAIM_PREFIX)
         and now - _safe_mtime(p) >= settle_seconds),
        key=_safe_mtime,
    )
    for candidate in candidates:
        job_id = uuid.uuid4().hex[:12]
        claimed_path = inbox / f"{CLAIM_PREFIX}{job_id}__{candidate.name}"
        try:
            candidate.rename(claimed_path)
        except FileNotFoundError:
            continue  # someone else grabbed it (shouldn't happen with one worker, but safe)
        # Rename companion opts sidecar so it tracks the claim.
        opts_src = candidate.with_suffix(".opts.json")
        if opts_src.exists():
            try:
                opts_src.rename(claimed_path.with_suffix(".opts.json"))
            except OSError:
                pass
        return claimed_path
    return None


def _read_opts_quality(claim_path: Path, *, default: str = "Auto") -> str:
    """Read the quality preset from a per-job ``.opts.json`` sidecar if present."""
    opts_path = claim_path.with_suffix(".opts.json")
    if not opts_path.is_file():
        return default
    try:
        import json as _json
        opts = _json.loads(opts_path.read_text(encoding="utf-8"))
        q = opts.get("quality", default)
        if q in ("Auto", "Maximum"):
            return q
    except Exception:
        pass
    return default


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )
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
    _log.info("watcher starting (inbox=%s)", args.inbox)
    paths = WatcherPaths(
        inbox=args.inbox, work=args.work, outbox=args.outbox,
        logs=args.logs, config_yaml=args.config_yaml,
    )
    run_forever(paths, quality_preset=args.quality_preset, poll_interval=args.poll_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(watcher_main())
