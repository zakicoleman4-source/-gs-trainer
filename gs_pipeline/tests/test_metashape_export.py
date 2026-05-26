"""Tests for ``gs_pipeline.metashape.export_for_splat`` using the stub chunk.

We can't run real Metashape in CI (no license), so we drive the exporter
with the ``metashape_stub`` chunk. The end-to-end check is that the bundle
the exporter writes is consumable by the trainer's own parser — round-trip
through the real ``parse_metashape.parse_cameras_xml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.metashape.export_for_splat import (
    BUNDLE_FORMAT,
    ChunkValidation,
    export_chunk_to_bundle_dir,
    export_chunk_to_zip,
    select_chunk,
    validate_chunk,
    zip_bundle,
)
from gs_pipeline.tests.stubs.metashape_stub import (
    StubChunk,
    build_stub_chunk,
)
from gs_pipeline.trainer.init_from_pcd import load_and_downsample
from gs_pipeline.trainer.parse_metashape import parse_cameras_xml


# ---------------------------------------------------------------------------
# validate_chunk
# ---------------------------------------------------------------------------

def test_validate_chunk_ok():
    chunk = build_stub_chunk(n_cameras=8)
    v = validate_chunk(chunk)
    assert isinstance(v, ChunkValidation)
    assert v.ok
    assert v.n_cameras_total == 8
    assert v.n_cameras_aligned == 8
    assert v.has_dense_cloud
    assert v.chunk_transform_is_identity
    assert "frame" in v.sensor_types


def test_validate_chunk_too_few_aligned_errors():
    chunk = build_stub_chunk(n_cameras=8, unaligned_count=5)
    v = validate_chunk(chunk)
    assert not v.ok
    assert any("aligned" in e for e in v.errors)


def test_validate_chunk_missing_dense_errors():
    chunk = build_stub_chunk(n_cameras=8, no_dense=True)
    v = validate_chunk(chunk)
    assert not v.ok
    assert any("dense cloud" in e for e in v.errors)


def test_validate_chunk_metashape_2x_point_cloud():
    """Metashape 2.x renamed dense_cloud to point_cloud and sparse to tie_points."""
    chunk = build_stub_chunk(n_cameras=8, no_dense=True)
    chunk.dense_cloud = None
    chunk.point_cloud = object()
    chunk.tie_points = object()
    v = validate_chunk(chunk)
    assert v.ok
    assert v.has_dense_cloud


def test_validate_chunk_no_cameras_errors():
    chunk = build_stub_chunk(n_cameras=0)
    v = validate_chunk(chunk)
    assert not v.ok
    assert any("no cameras" in e for e in v.errors)


def test_validate_chunk_warns_on_non_identity_transform():
    t = np.eye(4); t[0, 0] = -1.0
    chunk = build_stub_chunk(n_cameras=8, chunk_transform=t)
    v = validate_chunk(chunk)
    assert v.ok  # warning only, not an error
    assert not v.chunk_transform_is_identity
    assert any("geo-transform" in w for w in v.warnings)


def test_validate_chunk_identity_transform_from_matrix_like_object():
    """If chunk.transform is a Matrix-shaped object, it should still validate."""
    class _Mat:
        def __iter__(self):
            return iter([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
    chunk = build_stub_chunk(n_cameras=8, chunk_transform=_Mat())
    v = validate_chunk(chunk)
    assert v.chunk_transform_is_identity


# ---------------------------------------------------------------------------
# select_chunk
# ---------------------------------------------------------------------------

def test_select_chunk_prefers_current_chunk():
    class FakeDoc:
        chunk = build_stub_chunk(n_cameras=4, label="current")
        chunks = [build_stub_chunk(n_cameras=4, label="other")]
    assert select_chunk(FakeDoc()).label == "current"


def test_select_chunk_falls_back_to_first_enabled_when_no_current():
    a = build_stub_chunk(n_cameras=4, label="A"); a.enabled = False
    b = build_stub_chunk(n_cameras=4, label="B"); b.enabled = True
    class FakeDoc:
        chunk = None
        chunks = [a, b]
    assert select_chunk(FakeDoc()).label == "B"


def test_select_chunk_raises_when_empty():
    class FakeDoc:
        chunk = None
        chunks: list = []
    with pytest.raises(ValueError, match="no chunks"):
        select_chunk(FakeDoc())


# ---------------------------------------------------------------------------
# export_chunk_to_bundle_dir
# ---------------------------------------------------------------------------

def test_export_writes_all_required_files(tmp_path: Path):
    chunk = build_stub_chunk(n_cameras=6, image_size=64)
    bundle_dir = tmp_path / "bundle"
    export_chunk_to_bundle_dir(chunk, bundle_dir, metashape_module=_stub_module())
    assert (bundle_dir / "cameras.xml").is_file()
    assert (bundle_dir / "dense.ply").is_file()
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "images").is_dir()
    images = list((bundle_dir / "images").glob("*.png"))
    assert len(images) == 6


def test_exported_bundle_round_trips_through_trainer_parser(tmp_path: Path):
    """The acid test: a bundle written by the exporter must be loadable by the
    trainer's parser. Confirms cameras.xml schema parity and dense.ply layout."""
    chunk = build_stub_chunk(n_cameras=5, image_size=128)
    bundle_dir = tmp_path / "bundle"
    export_chunk_to_bundle_dir(chunk, bundle_dir, metashape_module=_stub_module())

    scene = parse_cameras_xml(bundle_dir / "cameras.xml",
                               image_dir=bundle_dir / "images")
    assert len(scene) == 5
    cloud = load_and_downsample(bundle_dir / "dense.ply")
    assert cloud.xyz.shape[0] > 0


def test_export_manifest_records_camera_count_and_format(tmp_path: Path):
    chunk = build_stub_chunk(n_cameras=7, image_size=64)
    export_chunk_to_bundle_dir(chunk, tmp_path / "bundle", metashape_module=_stub_module())
    import json
    m = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert m["format"] == BUNDLE_FORMAT
    assert m["n_cameras"] == 7
    assert m["image_size"] == [64, 64]


def test_export_progress_callback_is_called(tmp_path: Path):
    chunk = build_stub_chunk(n_cameras=4)
    msgs: list[tuple[str, float]] = []
    def progress(msg: str, frac: float) -> None:
        msgs.append((msg, frac))
    export_chunk_to_bundle_dir(chunk, tmp_path / "bundle",
                                progress=progress, metashape_module=_stub_module())
    # Should have at least the "done" callback.
    assert any(m[0] == "done" and m[1] == 1.0 for m in msgs)


# ---------------------------------------------------------------------------
# export_chunk_to_zip + validation refusal
# ---------------------------------------------------------------------------

def test_export_chunk_to_zip_writes_zip_when_valid(tmp_path: Path):
    chunk = build_stub_chunk(n_cameras=5)
    out_zip = tmp_path / "out.zip"
    zp, v = export_chunk_to_zip(
        chunk, out_zip=out_zip, work_dir=tmp_path / "work",
        metashape_module=_stub_module(),
    )
    assert v.ok
    assert zp == out_zip and out_zip.is_file()


def test_export_chunk_to_zip_refuses_invalid_chunk(tmp_path: Path):
    chunk = build_stub_chunk(n_cameras=8, no_dense=True)
    with pytest.raises(ValueError, match="dense cloud"):
        export_chunk_to_zip(
            chunk, out_zip=tmp_path / "out.zip",
            work_dir=tmp_path / "work",
            metashape_module=_stub_module(),
        )
    assert not (tmp_path / "out.zip").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stub_module():
    """Return a tiny module-like object with the constants the exporter looks up."""
    import gs_pipeline.tests.stubs.metashape_stub as m
    return m


# ---------------------------------------------------------------------------
# Undistorted flag in manifest
# ---------------------------------------------------------------------------

def test_manifest_undistorted_true_when_undistortphotos_succeeds(tmp_path: Path):
    """Normal export via stub (undistortPhotos works) → manifest.undistorted = True."""
    import json
    chunk = build_stub_chunk(n_cameras=4, image_size=64)
    export_chunk_to_bundle_dir(chunk, tmp_path / "bundle", metashape_module=_stub_module())
    m = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert m["undistorted"] is True


def test_manifest_undistorted_false_when_undistortphotos_raises(tmp_path: Path):
    """If undistortPhotos raises, fallback copies raw photos → undistorted = False."""
    import json
    from gs_pipeline.metashape.export_for_splat import export_chunk_to_bundle_dir

    chunk = build_stub_chunk(n_cameras=4, image_size=64)

    # Monkey-patch undistortPhotos to fail so the fallback path is exercised.
    original = chunk.undistortPhotos
    chunk.undistortPhotos = lambda path: (_ for _ in ()).throw(RuntimeError("simulated failure"))

    export_chunk_to_bundle_dir(chunk, tmp_path / "bundle", metashape_module=_stub_module())
    m = json.loads((tmp_path / "bundle" / "manifest.json").read_text())
    assert m["undistorted"] is False


# ---------------------------------------------------------------------------
# Watcher quality sidecar
# ---------------------------------------------------------------------------

def test_read_opts_quality_returns_default_when_no_file(tmp_path: Path):
    from gs_pipeline.trainer.watcher import _read_opts_quality
    fake_zip = tmp_path / "claim__abc123__scene.zip"
    assert _read_opts_quality(fake_zip, default="Auto") == "Auto"


def test_read_opts_quality_reads_sidecar(tmp_path: Path):
    import json
    from gs_pipeline.trainer.watcher import _read_opts_quality
    fake_zip = tmp_path / "claim__abc123__scene.zip"
    fake_zip.with_suffix(".opts.json").write_text(json.dumps({"quality": "Maximum"}))
    assert _read_opts_quality(fake_zip, default="Auto") == "Maximum"


def test_read_opts_quality_ignores_invalid_value(tmp_path: Path):
    import json
    from gs_pipeline.trainer.watcher import _read_opts_quality
    fake_zip = tmp_path / "claim__abc123__scene.zip"
    fake_zip.with_suffix(".opts.json").write_text(json.dumps({"quality": "UltraHD"}))
    assert _read_opts_quality(fake_zip, default="Auto") == "Auto"  # unknown → fall back to default


def test_claim_next_also_renames_opts_sidecar(tmp_path: Path):
    """When _claim_next renames a zip, the companion .opts.json renames too."""
    import json
    from gs_pipeline.trainer.watcher import _claim_next, CLAIM_PREFIX
    # Create a fake inbox zip + opts sidecar
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    zip_file = inbox / "scene.zip"
    zip_file.write_bytes(b"PK")   # minimal content
    opts_file = zip_file.with_suffix(".opts.json")
    opts_file.write_text(json.dumps({"quality": "Maximum"}))

    claimed = _claim_next(inbox)
    assert claimed is not None
    assert claimed.name.startswith(CLAIM_PREFIX)
    assert not opts_file.exists()
    opts_claimed = claimed.with_suffix(".opts.json")
    assert opts_claimed.is_file()
    assert json.loads(opts_claimed.read_text())["quality"] == "Maximum"
