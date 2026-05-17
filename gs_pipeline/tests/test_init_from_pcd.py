"""Tests for ``gs_pipeline.trainer.init_from_pcd``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from gs_pipeline.tests.fixtures.make_synthetic import build_bundle
from gs_pipeline.trainer.init_from_pcd import (
    DEFAULT_TARGET_MAX_POINTS,
    InitCloud,
    load_and_downsample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_ply(
    out_path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    binary: bool = True,
    color_dtype: str = "u1",
) -> Path:
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if rgb is not None:
        fields += [("red", color_dtype), ("green", color_dtype), ("blue", color_dtype)]
    data = np.empty(xyz.shape[0], dtype=fields)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    if rgb is not None:
        data["red"] = rgb[:, 0]
        data["green"] = rgb[:, 1]
        data["blue"] = rgb[:, 2]
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=not binary).write(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_synthetic_dense(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_points=3000)
    cloud = load_and_downsample(bundle.root / "dense.ply")
    assert isinstance(cloud, InitCloud)
    assert cloud.xyz.dtype == np.float32
    assert cloud.rgb.dtype == np.float32
    assert cloud.xyz.shape[1] == 3
    assert cloud.rgb.shape == cloud.xyz.shape
    # Colors must be normalized to [0,1].
    assert cloud.rgb.min() >= 0.0
    assert cloud.rgb.max() <= 1.0
    # Cube half-extent 0.5 => AABB ~ [-0.5, 0.5]; diagonal ~ sqrt(3).
    assert cloud.scene_extent == pytest.approx(np.sqrt(3.0), rel=0.05)


def test_returned_count_respects_target_cap(tmp_path: Path):
    rng = np.random.default_rng(0)
    # 50k random points in a [-1,1]^3 cube.
    xyz = rng.uniform(-1.0, 1.0, size=(50_000, 3)).astype(np.float32)
    rgb = (rng.uniform(0.0, 255.0, size=(50_000, 3))).astype(np.uint8)
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb)
    cloud = load_and_downsample(ply, target_max_points=2000)
    assert cloud.n_loaded == 50_000
    assert cloud.xyz.shape[0] <= 2000


def test_no_color_falls_back_to_gray(tmp_path: Path):
    rng = np.random.default_rng(1)
    xyz = rng.uniform(-1.0, 1.0, size=(500, 3)).astype(np.float32)
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb=None)
    cloud = load_and_downsample(ply, target_max_points=10_000)
    assert cloud.rgb.shape == cloud.xyz.shape
    np.testing.assert_allclose(cloud.rgb, 0.5, atol=1e-6)


def test_float_colors_are_clipped_not_rescaled(tmp_path: Path):
    """Some pipelines write float colors already in [0,1]. They must not be /255."""
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-1.0, 1.0, size=(200, 3)).astype(np.float32)
    rgb = rng.uniform(0.0, 1.0, size=(200, 3)).astype(np.float32)
    # Write with float color dtype.
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("red", "f4"), ("green", "f4"), ("blue", "f4")]
    data = np.empty(200, dtype=fields)
    data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]
    data["red"] = rgb[:, 0]; data["green"] = rgb[:, 1]; data["blue"] = rgb[:, 2]
    el = PlyElement.describe(data, "vertex")
    out = tmp_path / "p.ply"
    PlyData([el], text=False).write(str(out))
    cloud = load_and_downsample(out, target_max_points=10_000)
    assert cloud.rgb.max() <= 1.0
    # If the loader incorrectly divided by 255, max would be near 1/255 = 0.004.
    assert cloud.rgb.max() > 0.05


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_and_downsample(tmp_path / "nope.ply")


def test_empty_ply_raises(tmp_path: Path):
    ply = _write_ply(tmp_path / "p.ply", np.zeros((0, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        load_and_downsample(ply)


def test_degenerate_aabb_raises(tmp_path: Path):
    # All points at the origin => zero extent.
    xyz = np.zeros((50, 3), dtype=np.float32)
    rgb = np.zeros((50, 3), dtype=np.uint8)
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb)
    with pytest.raises(ValueError, match="degenerate"):
        load_and_downsample(ply)


def test_downsample_preserves_average_color(tmp_path: Path):
    """Each voxel's color is the mean of its source points."""
    # Two clusters of 100 red points each in two voxels.
    xyz = np.concatenate([
        np.full((100, 3), -0.4, dtype=np.float32) + np.random.default_rng(0).normal(0, 1e-3, (100, 3)).astype(np.float32),
        np.full((100, 3), +0.4, dtype=np.float32) + np.random.default_rng(0).normal(0, 1e-3, (100, 3)).astype(np.float32),
    ], axis=0)
    rgb = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (200, 1))
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb)
    cloud = load_and_downsample(ply, target_max_points=10_000, fixed_voxel_size=0.2)
    # All input was pure red; all output voxels must be ~red.
    np.testing.assert_allclose(cloud.rgb[:, 0], 1.0, atol=1e-3)
    np.testing.assert_allclose(cloud.rgb[:, 1], 0.0, atol=1e-3)
    np.testing.assert_allclose(cloud.rgb[:, 2], 0.0, atol=1e-3)


def test_aabb_and_extent_match_input(tmp_path: Path):
    xyz = np.array([
        [-1.0, -2.0, -3.0],
        [4.0, 5.0, 6.0],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)
    rgb = np.full_like(xyz, 0.5, dtype=np.float32) * 255
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb.astype(np.uint8))
    cloud = load_and_downsample(ply, target_max_points=100, fixed_voxel_size=10.0)
    np.testing.assert_allclose(cloud.aabb_min, [-1.0, -2.0, -3.0])
    np.testing.assert_allclose(cloud.aabb_max, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(cloud.scene_extent, np.linalg.norm([5.0, 7.0, 9.0]), rtol=1e-6)


def test_default_voxel_size_scales_with_scene_extent(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_points=3000)
    cloud = load_and_downsample(bundle.root / "dense.ply")
    # voxel_size_factor default is 1024 -> scene_extent / 1024.
    assert cloud.voxel_size == pytest.approx(cloud.scene_extent / 1024.0, rel=0.01)


def test_large_cloud_handled_under_cap(tmp_path: Path):
    """200k input -> default 1M cap is comfortably satisfied; no error."""
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-5.0, 5.0, size=(200_000, 3)).astype(np.float32)
    rgb = rng.uniform(0, 255, size=(200_000, 3)).astype(np.uint8)
    ply = _write_ply(tmp_path / "p.ply", xyz, rgb)
    cloud = load_and_downsample(ply)
    assert cloud.n_loaded == 200_000
    assert cloud.xyz.shape[0] <= DEFAULT_TARGET_MAX_POINTS
    assert cloud.xyz.shape[0] > 0
