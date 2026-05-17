"""Tests for ``gs_pipeline.trainer.job_state``."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from gs_pipeline.trainer.job_state import (
    InvalidStateTransition,
    JobState,
    OutputsSnapshot,
    PreflightSnapshot,
    SCHEMA_VERSION,
    STATE_FILENAME,
    State,
    new_job_state,
    read_state,
    safe_read_state,
    state_path_for,
    write_state,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preflight() -> PreflightSnapshot:
    return PreflightSnapshot(
        n_cameras=80,
        total_megapixels=240.0,
        dense_pts=500_000,
        target_splats=5_800_000,
        hard_cap_splats=14_700_000,
        iterations=30_000,
        downscale_factor=0.5,
        image_max_side=2000,
        quality_preset="Auto",
        gpu_name="RTX A5000",
        gpu_total_vram_bytes=24 * 1_000_000_000,
        notes=["heads up"],
    )


# ---------------------------------------------------------------------------
# Construction / transitions
# ---------------------------------------------------------------------------

def test_new_job_state_initial_fields():
    js = new_job_state("abc-123", bundle_filename="my_scene.zip")
    assert js.job_id == "abc-123"
    assert js.state is State.QUEUED
    assert js.status_msg == "queued"
    assert js.bundle_filename == "my_scene.zip"
    assert js.last_update_at is not None
    assert js.started_at is None
    assert js.error_msg is None


def test_happy_path_transitions():
    js = new_job_state("j1")
    js.start_preflight()
    assert js.state is State.PREFLIGHT
    assert js.started_at is not None
    preflight = _make_preflight()
    js.start_training(preflight)
    assert js.state is State.TRAINING
    assert js.preflight == preflight
    assert "0/30000" in js.status_msg
    js.tick(current_step=5000, current_splats=2_100_000, psnr=22.1, loss=0.04)
    assert js.progress.current_step == 5000
    assert js.progress.current_splats == 2_100_000
    assert js.progress.psnr_history == [[5000, 22.1]]
    assert js.progress.loss_history == [[5000, 0.04]]
    assert "5000/30000" in js.status_msg
    outputs = OutputsSnapshot(final_ply="/outbox/j1/scene.ply")
    js.finish(outputs)
    assert js.state is State.DONE
    assert js.outputs.final_ply == "/outbox/j1/scene.ply"


def test_resume_edge():
    js = new_job_state("j1")
    js.start_preflight()
    js.start_training(_make_preflight())
    js.mark_resuming(msg="resuming from ckpt_5000")
    assert js.state is State.RESUMING
    assert js.status_msg == "resuming from ckpt_5000"
    # First tick after resume flips back to TRAINING.
    js.tick(current_step=5001, current_splats=2_000_000)
    assert js.state is State.TRAINING


def test_mark_failed_allowed_from_any_non_terminal():
    for setup in (
        lambda j: None,                       # from QUEUED
        lambda j: j.start_preflight(),        # from PREFLIGHT
    ):
        js = new_job_state("j")
        setup(js)
        js.mark_failed("bang")
        assert js.state is State.FAILED
        assert js.error_msg == "bang"

    # From TRAINING.
    js = new_job_state("j")
    js.start_preflight()
    js.start_training(_make_preflight())
    js.mark_failed("crashed mid-train")
    assert js.state is State.FAILED


def test_mark_failed_rejects_terminal_states():
    js = new_job_state("j")
    js.start_preflight()
    js.start_training(_make_preflight())
    js.finish(OutputsSnapshot())
    with pytest.raises(InvalidStateTransition):
        js.mark_failed("too late")


def test_invalid_transitions_raise():
    # tick without TRAINING.
    js = new_job_state("j")
    with pytest.raises(InvalidStateTransition):
        js.tick(current_step=1, current_splats=1)
    # Skipping preflight.
    with pytest.raises(InvalidStateTransition):
        js.start_training(_make_preflight())
    # Resuming from queued.
    with pytest.raises(InvalidStateTransition):
        js.mark_resuming()


def test_done_is_terminal_no_further_transitions():
    js = new_job_state("j")
    js.start_preflight()
    js.start_training(_make_preflight())
    js.finish(OutputsSnapshot())
    with pytest.raises(InvalidStateTransition):
        js.start_preflight()
    with pytest.raises(InvalidStateTransition):
        js.tick(current_step=1, current_splats=1)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_round_trip():
    js = new_job_state("j", bundle_filename="x.zip")
    js.start_preflight()
    preflight = _make_preflight()
    js.start_training(preflight)
    js.tick(current_step=1000, current_splats=900_000, psnr=18.2, ssim=0.62, loss=0.07)
    js.finish(OutputsSnapshot(final_ply="/o/scene.ply", checkpoints=["/w/c.pt"]))
    d = js.to_dict()
    s = json.dumps(d)
    js2 = JobState.from_dict(json.loads(s))
    assert js2.job_id == "j"
    assert js2.state is State.DONE
    assert js2.preflight == preflight
    assert js2.progress.current_step == 1000
    assert js2.progress.psnr_history == [[1000, 18.2]]
    assert js2.outputs.final_ply == "/o/scene.ply"
    assert js2.outputs.checkpoints == ["/w/c.pt"]
    assert js2.bundle_filename == "x.zip"


def test_from_dict_rejects_wrong_schema_version():
    d = new_job_state("j").to_dict()
    d["schema_version"] = SCHEMA_VERSION + 99
    with pytest.raises(ValueError, match="schema_version"):
        JobState.from_dict(d)


# ---------------------------------------------------------------------------
# Atomic file IO
# ---------------------------------------------------------------------------

def test_write_state_creates_parent_dir(tmp_path: Path):
    js = new_job_state("abc")
    path = tmp_path / "work" / "abc" / STATE_FILENAME
    assert not path.parent.exists()
    write_state(js, path)
    assert path.is_file()
    loaded = read_state(path)
    assert loaded.job_id == "abc"


def test_state_path_for(tmp_path: Path):
    p = state_path_for(tmp_path, "abc")
    assert p == tmp_path / "abc" / STATE_FILENAME


def test_write_is_atomic_no_tempfile_left_on_success(tmp_path: Path):
    js = new_job_state("abc")
    js.start_preflight()
    path = tmp_path / "abc" / STATE_FILENAME
    write_state(js, path)
    assert path.is_file()
    # No leftover .tmp files.
    siblings = list(path.parent.iterdir())
    assert all(not p.name.startswith(".state.") for p in siblings)


def test_safe_read_state_missing_returns_none(tmp_path: Path):
    assert safe_read_state(tmp_path / "missing.json") is None


def test_safe_read_state_malformed_returns_none(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{ not json", encoding="utf-8")
    assert safe_read_state(p) is None


def test_concurrent_writes_never_corrupt_reads(tmp_path: Path):
    """Hammer the file from one writer + one reader thread; reader never sees partial data."""
    js = new_job_state("j")
    js.start_preflight()
    js.start_training(_make_preflight())
    path = tmp_path / "j" / STATE_FILENAME
    write_state(js, path)
    stop = threading.Event()
    errors: list[str] = []

    def writer() -> None:
        step = 0
        while not stop.is_set():
            step += 1
            try:
                js.tick(current_step=step, current_splats=1000 * step)
            except InvalidStateTransition:
                # If we transitioned to RESUMING somehow, fix back to TRAINING.
                js.state = State.TRAINING
            write_state(js, path)
            time.sleep(0.0005)

    def reader() -> None:
        while not stop.is_set():
            loaded = safe_read_state(path)
            if loaded is None:
                errors.append("got None from safe_read_state")
                return
            if loaded.state not in (State.TRAINING, State.RESUMING):
                errors.append(f"unexpected state {loaded.state}")
                return
            time.sleep(0.0005)

    t_w = threading.Thread(target=writer)
    t_r = threading.Thread(target=reader)
    t_w.start(); t_r.start()
    time.sleep(0.5)
    stop.set()
    t_w.join(); t_r.join()
    assert not errors, errors


def test_write_does_not_leave_tempfile_on_failure(tmp_path: Path, monkeypatch):
    js = new_job_state("j")
    path = tmp_path / "j" / STATE_FILENAME

    # Make os.replace raise to simulate filesystem failure.
    real_replace = os.replace
    def boom(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="boom"):
        write_state(js, path)
    monkeypatch.setattr(os, "replace", real_replace)

    # No .state.*.tmp left behind.
    leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".state.")]
    assert leftovers == [], leftovers


def test_started_at_set_on_first_non_queued_transition():
    js = new_job_state("j")
    assert js.started_at is None
    js.start_preflight()
    assert js.started_at is not None
    first = js.started_at
    js.start_training(_make_preflight())
    # started_at is sticky from the first transition.
    assert js.started_at == first
