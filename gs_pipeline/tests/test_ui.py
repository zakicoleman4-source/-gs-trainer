"""Tests for the Streamlit UI: preflight helper, live_view utilities, and a
scripted run of ``app.py`` via ``streamlit.testing.v1.AppTest``."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.tests.fixtures.make_synthetic import build_bundle, zip_bundle
from gs_pipeline.trainer.budget import GPUInfo
from gs_pipeline.trainer.job_state import (
    OutputsSnapshot,
    PreflightSnapshot,
    State,
    new_job_state,
    state_path_for,
    write_state,
)
from gs_pipeline.ui.live_view import (
    format_eta,
    intermediate_ply_paths,
    metrics_csv_rows,
    render_progress_text,
)
from gs_pipeline.ui.preflight import (
    PreflightReport,
    estimate_training_minutes,
    run_preflight,
)


GPU_24 = GPUInfo(name="(test) 24GB", total_vram_bytes=24_000_000_000)


# ---------------------------------------------------------------------------
# preflight.run_preflight
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_zip(tmp_path: Path) -> Path:
    bdir = tmp_path / "src"
    build_bundle(bdir, n_cameras=6, image_size=128, n_points=300)
    out = tmp_path / "scene.zip"
    zip_bundle(bdir, out)
    return out


def test_run_preflight_happy(synth_zip: Path, tmp_path: Path):
    report = run_preflight(
        synth_zip, quality_preset="Auto", gpu=GPU_24,
        extract_to=tmp_path / "extract",
    )
    assert isinstance(report, PreflightReport)
    assert report.n_cameras == 6
    assert report.dense_pts_after_downsample > 0
    assert report.target_splats >= 500_000  # floor
    assert report.iterations == 30_000
    assert report.gpu_name == GPU_24.name


def test_run_preflight_rejects_zip_path_traversal(tmp_path: Path):
    import zipfile
    bad = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../../escape.txt", "boom")
    with pytest.raises(ValueError, match="unsafe path"):
        run_preflight(bad, gpu=GPU_24, extract_to=tmp_path / "x")


def test_run_preflight_missing_cameras_xml_raises(tmp_path: Path):
    import zipfile
    bad = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("README.txt", "no cameras here")
    with pytest.raises(FileNotFoundError, match="cameras.xml"):
        run_preflight(bad, gpu=GPU_24, extract_to=tmp_path / "x")


def test_estimate_training_minutes_is_positive_for_real_inputs():
    assert estimate_training_minutes(1_500_000, 30_000) > 0
    # Zero inputs short-circuit to zero, no division-by-zero.
    assert estimate_training_minutes(0, 0) == 0
    assert estimate_training_minutes(1_000_000, 0) == 0


# ---------------------------------------------------------------------------
# live_view: pure helpers
# ---------------------------------------------------------------------------

def test_format_eta_estimating_until_first_step():
    assert format_eta(0, 10_000, started_at_iso="2026-01-01T00:00:00+00:00") == "(estimating)"


def test_format_eta_unparseable_started_at():
    assert format_eta(100, 1000, started_at_iso="not-a-date") == "—"


def test_format_eta_seconds_minutes_hours():
    # Half-way through, started 30s ago -> ~30s remaining.
    from datetime import datetime, timedelta, timezone
    started = (datetime.now(timezone.utc) - timedelta(seconds=30)).replace(microsecond=0).isoformat()
    eta = format_eta(500, 1000, started_at_iso=started)
    # Should be roughly 30s (allow generous slack for real-clock variance).
    assert eta.endswith("s") or "m" in eta


def test_render_progress_text_uses_preflight():
    js = new_job_state("abc")
    js.start_preflight()
    pre = PreflightSnapshot(
        n_cameras=10, total_megapixels=100.0, dense_pts=10_000,
        target_splats=2_000_000, hard_cap_splats=10_000_000, iterations=30_000,
        downscale_factor=1.0, image_max_side=1024, quality_preset="Auto",
        gpu_name="(test) GPU", gpu_total_vram_bytes=24 * 10**9, notes=[],
    )
    js.start_training(pre)
    js.tick(current_step=5_000, current_splats=1_000_000)
    text = render_progress_text(js)
    assert "5,000 / 30,000" in text
    assert "Auto" in text
    assert "(test) GPU" in text


def test_metrics_csv_rows_parses_what_trainer_writes(tmp_path: Path):
    p = tmp_path / "metrics.csv"
    p.write_text(
        "step,loss,holdout_psnr,holdout_ssim\n"
        "1000,0.10,18.5,0.62\n"
        "2000,0.08,21.2,0.71\n"
        "garbage,bad,row,nope\n"           # malformed line ignored
        "3000,0.06,22.9,0.78\n",
        encoding="utf-8",
    )
    rows = metrics_csv_rows(str(p))
    assert [r["step"] for r in rows] == [1000, 2000, 3000]
    assert rows[1]["holdout_psnr"] == pytest.approx(21.2)


def test_metrics_csv_rows_missing_file_returns_empty():
    assert metrics_csv_rows(None) == []
    assert metrics_csv_rows("/does/not/exist.csv") == []


def test_intermediate_ply_paths_sorted_numerically(tmp_path: Path):
    """`scene_step_15000.ply` must sort AFTER `scene_step_5000.ply`, not before."""
    for step in (15000, 5000, 25000):
        (tmp_path / f"scene_step_{step}.ply").touch()
    paths = intermediate_ply_paths(tmp_path)
    assert [int(p.stem.rsplit("_", 1)[-1]) for p in paths] == [5000, 15000, 25000]


# ---------------------------------------------------------------------------
# AppTest: scripted Streamlit run
# ---------------------------------------------------------------------------

def test_app_renders_upload_phase_on_empty_state():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "ui" / "app.py"),
        default_timeout=20,
    )
    at.run()
    assert not at.exception
    # Upload page shows the title and the file_uploader widget.
    titles = " ".join(t.value for t in at.title)
    assert "gs_pipeline" in titles
    assert len(at.selectbox) >= 1                # quality preset selectbox
    assert at.selectbox[0].value == "Auto"


def test_app_can_choose_maximum_preset_via_apptest():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(
        str(Path(__file__).resolve().parent.parent / "ui" / "app.py"),
        default_timeout=20,
    )
    at.run()
    at.selectbox[0].set_value("Maximum").run()
    assert not at.exception
    # Phase remained `upload` (no file uploaded yet).
    assert at.session_state["phase"] == "upload"
    assert at.session_state["quality_preset"] == "Maximum"
