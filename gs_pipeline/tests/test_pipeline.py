"""Integration tests for ``pipeline.run_job`` and ``watcher.process_one``.

The actual GPU trainer is stubbed via the ``train_fn`` injection point, so
the orchestration is fully exercised on CPU CI:

  unzip -> parse cameras.xml -> downsample dense.ply -> compute budget ->
  state.json -> stub train -> write final .ply -> mark done.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.tests.fixtures.make_synthetic import build_bundle, zip_bundle
from gs_pipeline.trainer.budget import Budget, GPUInfo
from gs_pipeline.trainer.export_ply import (
    num_sh_rest_coeffs,
    read_inria_ply,
    write_inria_ply,
)
from gs_pipeline.trainer.init_from_pcd import InitCloud
from gs_pipeline.trainer.job_state import (
    JobState,
    OutputsSnapshot,
    State,
    safe_read_state,
    state_path_for,
)
from gs_pipeline.trainer.parse_metashape import ParsedScene
from gs_pipeline.trainer.pipeline import job_outbox_dir, run_job
from gs_pipeline.trainer.watcher import (
    CLAIM_PREFIX,
    WatcherPaths,
    _job_id_from_claim,
    _original_name,
    process_one,
)


GPU_24 = GPUInfo(name="(test) 24GB", total_vram_bytes=24_000_000_000)


# ---------------------------------------------------------------------------
# Stub trainer
# ---------------------------------------------------------------------------

def _stub_train(
    *, scene: ParsedScene, init_cloud: InitCloud, budget: Budget,
    config, job_state: JobState, job_state_path: Path,
    work_dir: Path, outbox_dir: Path,
) -> OutputsSnapshot:
    """Fake trainer: tick a few times, write a 5-splat scene.ply, return outputs.

    Lets us drive the full orchestration without torch/gsplat. Mirrors the
    contract of the real ``train_mcmc.train`` exactly.
    """
    from gs_pipeline.trainer.job_state import write_state
    n = 5
    sh_degree = 0
    means = np.zeros((n, 3), dtype=np.float32)
    scales = np.full((n, 3), -2.0, dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32); quats[:, 0] = 1.0
    opacities = np.zeros((n,), dtype=np.float32)
    sh_dc = np.zeros((n, 3), dtype=np.float32)
    sh_rest = np.zeros((n, num_sh_rest_coeffs(sh_degree), 3), dtype=np.float32)

    # Two progress ticks so the state file shows real progress.
    for step in (250, 500):
        job_state.tick(current_step=step, current_splats=n, psnr=15.0, ssim=0.5, loss=0.3)
        write_state(job_state, job_state_path)

    final_ply = outbox_dir / "scene.ply"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    write_inria_ply(
        out_path=final_ply, means=means, scales=scales, quats=quats,
        opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest,
    )

    metrics_csv = work_dir / "metrics.csv"
    metrics_csv.write_text("step,loss,holdout_psnr,holdout_ssim\n500,0.30,15.00,0.50\n",
                           encoding="utf-8")
    report = {"job_id": job_state.job_id, "final_step": 500, "final_splat_count": n}
    report_json = work_dir / "report.json"
    report_json.write_text(json.dumps(report), encoding="utf-8")

    return OutputsSnapshot(
        checkpoints=[],
        preview_png=None,
        final_ply=str(final_ply),
        metrics_csv=str(metrics_csv),
        report_json=str(report_json),
    )


def _failing_train(**kwargs) -> OutputsSnapshot:
    raise RuntimeError("trainer exploded mid-run for test")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle_zip(tmp_path: Path) -> Path:
    bdir = tmp_path / "bundle_src"
    build_bundle(bdir, n_cameras=4, image_size=64, n_points=120)
    out = tmp_path / "my_scene.zip"
    zip_bundle(bdir, out)
    return out


# ---------------------------------------------------------------------------
# run_job: happy path
# ---------------------------------------------------------------------------

def test_run_job_happy_path(bundle_zip: Path, tmp_job_root: Path):
    state = run_job(
        job_id="job-001", bundle_zip=bundle_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
        bundle_filename="my_scene.zip",
    )
    assert state.state is State.DONE
    assert state.bundle_filename == "my_scene.zip"
    assert state.outputs.final_ply is not None
    assert Path(state.outputs.final_ply).is_file()
    assert state.preflight is not None
    assert state.preflight.n_cameras == 4
    assert state.preflight.dense_pts > 0
    assert state.preflight.gpu_name == GPU_24.name


def test_run_job_writes_state_json_at_each_stage(bundle_zip: Path, tmp_job_root: Path):
    sp = state_path_for(tmp_job_root / "work", "job-002")
    # Before run: no state file.
    assert not sp.exists()
    state = run_job(
        job_id="job-002", bundle_zip=bundle_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
    )
    assert state.state is State.DONE
    loaded = safe_read_state(sp)
    assert loaded is not None
    assert loaded.state is State.DONE
    assert loaded.preflight is not None
    assert loaded.outputs.final_ply is not None
    # Progress ticks from the stub should show up.
    assert any(step == 500 for step, _ in loaded.progress.psnr_history)


def test_final_ply_is_readable_inria_layout(bundle_zip: Path, tmp_job_root: Path):
    state = run_job(
        job_id="job-003", bundle_zip=bundle_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
    )
    assert state.outputs.final_ply
    loaded = read_inria_ply(Path(state.outputs.final_ply))
    assert loaded.means.shape == (5, 3)
    assert loaded.sh_degree == 0


# ---------------------------------------------------------------------------
# run_job: failure paths
# ---------------------------------------------------------------------------

def test_run_job_marks_failed_on_trainer_exception(bundle_zip: Path, tmp_job_root: Path):
    state = run_job(
        job_id="job-fail", bundle_zip=bundle_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_failing_train,
    )
    assert state.state is State.FAILED
    assert "trainer exploded" in (state.error_msg or "")
    # Error traceback file exists.
    err_file = tmp_job_root / "logs" / "job-fail" / "pipeline_error.txt"
    assert err_file.is_file()
    assert "trainer exploded" in err_file.read_text()


def test_run_job_marks_failed_on_missing_bundle(tmp_job_root: Path):
    state = run_job(
        job_id="job-noinput",
        bundle_zip=tmp_job_root / "nope.zip",  # doesn't exist
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
    )
    assert state.state is State.FAILED


def test_run_job_marks_failed_on_path_traversal_in_zip(tmp_job_root: Path):
    """Defense-in-depth: a malicious zip with '..' entries must be rejected."""
    bad_zip = tmp_job_root / "evil.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("../../escape.txt", "boom")
    state = run_job(
        job_id="job-evil", bundle_zip=bad_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
    )
    assert state.state is State.FAILED
    assert "unsafe path" in (state.error_msg or "")


def test_run_job_marks_failed_on_bundle_missing_cameras_xml(tmp_job_root: Path):
    """A zip without cameras.xml fails preflight cleanly."""
    bad_zip = tmp_job_root / "incomplete.zip"
    with zipfile.ZipFile(bad_zip, "w") as zf:
        zf.writestr("README.txt", "no cameras here")
    state = run_job(
        job_id="job-nocams", bundle_zip=bad_zip,
        work_root=tmp_job_root / "work",
        outbox_root=tmp_job_root / "outbox",
        log_root=tmp_job_root / "logs",
        gpu=GPU_24, train_fn=_stub_train,
    )
    assert state.state is State.FAILED
    assert "cameras.xml" in (state.error_msg or "")


# ---------------------------------------------------------------------------
# watcher.process_one + claim mechanics
# ---------------------------------------------------------------------------

def test_process_one_returns_none_when_inbox_empty(tmp_job_root: Path):
    paths = WatcherPaths(
        inbox=tmp_job_root / "inbox", work=tmp_job_root / "work",
        outbox=tmp_job_root / "outbox", logs=tmp_job_root / "logs",
    )
    assert process_one(paths, train_fn=_stub_train) is None


def test_process_one_drains_a_single_bundle(bundle_zip: Path, tmp_job_root: Path):
    paths = WatcherPaths(
        inbox=tmp_job_root / "inbox", work=tmp_job_root / "work",
        outbox=tmp_job_root / "outbox", logs=tmp_job_root / "logs",
    )
    inbox_zip = paths.inbox / "my_scene.zip"
    bundle_zip.rename(inbox_zip)

    state = process_one(paths, train_fn=_stub_train)
    assert state is not None
    assert state.state is State.DONE

    # Job's outbox has a scene.ply we can read.
    outbox_for_job = job_outbox_dir(paths.outbox, state.job_id)
    assert (outbox_for_job / "scene.ply").is_file()
    loaded = read_inria_ply(outbox_for_job / "scene.ply")
    assert loaded.means.shape[0] > 0

    # The claimed zip stayed in inbox/ under the claim__ prefix (not deleted).
    leftovers = list(paths.inbox.glob("*.zip"))
    assert len(leftovers) == 1
    assert leftovers[0].name.startswith(CLAIM_PREFIX)
    assert leftovers[0].name.endswith("__my_scene.zip")


def test_process_one_assigns_unique_ids(tmp_job_root: Path, bundle_zip: Path):
    """Two bundles in the inbox yield two distinct job ids."""
    paths = WatcherPaths(
        inbox=tmp_job_root / "inbox", work=tmp_job_root / "work",
        outbox=tmp_job_root / "outbox", logs=tmp_job_root / "logs",
    )
    import shutil
    a = paths.inbox / "a.zip"
    b = paths.inbox / "b.zip"
    shutil.copy(bundle_zip, a); shutil.copy(bundle_zip, b)

    s1 = process_one(paths, train_fn=_stub_train)
    s2 = process_one(paths, train_fn=_stub_train)
    assert s1 is not None and s2 is not None
    assert s1.job_id != s2.job_id


def test_claim_helpers_extract_id_and_name():
    claim = Path("/inbox/claim__abcdef012345__my_scene.zip")
    assert _job_id_from_claim(claim) == "abcdef012345"
    assert _original_name(claim) == "my_scene.zip"
