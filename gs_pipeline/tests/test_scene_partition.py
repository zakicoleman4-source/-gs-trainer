"""Tests for ``gs_pipeline.trainer.scene_partition``.

All tests are pure CPU — no torch, no gsplat.  Synthetic scenes are built
using SimpleNamespace to mock ParsedScene / InitCloud.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gs_pipeline.trainer.scene_partition import (
    SceneBlock,
    _auto_grid_size,
    _camera_block_visibility,
    _extract_camera_positions,
    partition_scene,
    should_partition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_w2c_translation(tx: float, ty: float, tz: float) -> np.ndarray:
    """Return a w2c matrix that is a pure translation (camera at world position
    (-R^T t) = (tx, ty, tz) since R=I means c2w[:3,3] = -t... we want the
    camera *centre* at (tx, ty, tz), so set w2c as the inverse of a c2w
    whose translation is (tx, ty, tz)."""
    # c2w: identity rotation, translation (tx, ty, tz)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [tx, ty, tz]
    return np.linalg.inv(c2w)


def _identity_K(width: float = 1000.0, height: float = 1000.0, f: float = 800.0) -> np.ndarray:
    """Return a simple pinhole K with principal point at image centre."""
    return np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _make_synthetic_scene(
    cam_positions: np.ndarray,
    *,
    width: float = 1000.0,
    height: float = 1000.0,
    f: float = 800.0,
) -> SimpleNamespace:
    """Build a minimal ParsedScene mock from an (N, 3) array of camera centres."""
    n = cam_positions.shape[0]
    w2c_list = []
    for i in range(n):
        tx, ty, tz = cam_positions[i]
        w2c_list.append(_make_w2c_translation(tx, ty, tz))
    w2c_arr = np.stack(w2c_list, axis=0)          # (N, 4, 4)
    K_arr = np.tile(_identity_K(width, height, f)[None], (n, 1, 1))  # (N, 3, 3)
    return SimpleNamespace(
        w2c_per_camera=w2c_arr,
        K_per_camera=K_arr,
    )


def _make_synthetic_init_cloud(
    n_points: int = 10_000,
    xyz_min: float = 0.0,
    xyz_max: float = 100.0,
    *,
    seed: int = 42,
) -> SimpleNamespace:
    """Build a minimal InitCloud mock with random points in [xyz_min, xyz_max]^3."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(xyz_min, xyz_max, size=(n_points, 3)).astype(np.float32)
    return SimpleNamespace(xyz=xyz)


# ---------------------------------------------------------------------------
# 1. test_extract_camera_positions_returns_correct_shape
# ---------------------------------------------------------------------------

def test_extract_camera_positions_returns_correct_shape():
    """Camera positions extracted from w2c matrices must have shape (N, 3)
    and reproduce the known world-space centres."""
    known_positions = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)

    scene = _make_synthetic_scene(known_positions)
    extracted = _extract_camera_positions(scene)

    assert extracted.shape == (3, 3), f"Expected (3, 3), got {extracted.shape}"
    np.testing.assert_allclose(extracted, known_positions, atol=1e-9,
                                err_msg="Extracted positions do not match known camera centres.")


# ---------------------------------------------------------------------------
# 2. test_auto_grid_size_for_various_counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cameras,expected_rows,expected_cols_min,expected_cols_max", [
    # 300 / 150 = 2 blocks  → (1, 2)
    (300, 1, 2, 2),
    # 750 / 150 = 5 blocks  → (1, 5)
    (750, 1, 5, 5),
    # 1500 / 150 = 10 blocks → rows*cols ≈ 10, rows ≥ 2
    # ceil(sqrt(10)) = 4; rows = ceil(10/4) = 3 → (3, 4) [12 cells] or (2, 5) acceptable
    (1500, 2, 4, 6),
])
def test_auto_grid_size_for_various_counts(
    n_cameras, expected_rows, expected_cols_min, expected_cols_max
):
    rows, cols = _auto_grid_size(n_cameras)
    assert rows >= expected_rows, (
        f"n_cameras={n_cameras}: expected rows >= {expected_rows}, got {rows}"
    )
    assert expected_cols_min <= cols <= expected_cols_max, (
        f"n_cameras={n_cameras}: expected {expected_cols_min} <= cols <= {expected_cols_max}, got {cols}"
    )
    # The total blocks must be at least 2.
    assert rows * cols >= 2


def test_auto_grid_size_never_returns_single_block():
    """Even for very low camera counts, we should get at least 2 blocks."""
    for n in (150, 200, 299):
        rows, cols = _auto_grid_size(n)
        assert rows * cols >= 2, f"n_cameras={n}: got only {rows*cols} block(s)"


# ---------------------------------------------------------------------------
# 3. test_camera_block_visibility_center_camera_returns_high
# ---------------------------------------------------------------------------

def test_camera_block_visibility_center_camera_returns_high():
    """A camera looking directly at a large nearby block should have high visibility.

    Block spans ±8 in X and Y at depth z=2..4 (very close, large coverage).
    With f=800, W=H=1000 and at distance z=2 the projected half-width is
    800 * 8 / 2 = 3200 px, which saturates the image → clamped area ~ 100 %.
    """
    aabb_min = np.array([-8.0, -8.0, 2.0])
    aabb_max = np.array([8.0, 8.0, 4.0])

    # Camera at origin looking down +Z (identity w2c).
    w2c = np.eye(4, dtype=np.float64)
    K = _identity_K(1000.0, 1000.0, 800.0)

    vis = _camera_block_visibility(K, w2c, aabb_min, aabb_max)
    assert vis > 0.5, f"Expected visibility > 0.5 for centre-facing camera, got {vis:.4f}"


# ---------------------------------------------------------------------------
# 4. test_camera_block_visibility_distant_camera_returns_low
# ---------------------------------------------------------------------------

def test_camera_block_visibility_distant_camera_returns_low():
    """A camera far away and looking in the opposite direction returns low visibility."""
    # Block at (0, 0, 10) with ±1 extent.
    aabb_min = np.array([-1.0, -1.0, 9.0])
    aabb_max = np.array([1.0, 1.0, 11.0])

    # Camera at (0, 0, -1000) pointing DOWN −Z (reversed): we make the block
    # appear *behind* the camera by rotating 180° around Y (negates Z in cam
    # space).  In practice we just put the block behind the camera.
    # Rotate camera 180° around Y so +Z world becomes −Z cam (behind camera).
    R_flip = np.array([
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = R_flip
    # Camera position far along +X axis (1000 m away from block).
    w2c[:3, 3] = [-1000.0, 0.0, 0.0]

    K = _identity_K(1000.0, 1000.0, 800.0)
    vis = _camera_block_visibility(K, w2c, aabb_min, aabb_max)
    assert vis < 0.1, f"Expected visibility < 0.1 for distant/away camera, got {vis:.4f}"


# ---------------------------------------------------------------------------
# 5. test_partition_scene_basic
# ---------------------------------------------------------------------------

def test_partition_scene_basic():
    """Partition a 600-camera scene laid out in a 30×20 grid over [0,100]^2.

    Checks:
    - At least 2 blocks are returned.
    - Each block has a non-empty camera_indices list.
    - Every camera appears in at least 1 block.
    - Every block's point_mask has at least 1 True entry.
    """
    n_cams = 600
    n_rows_cam = 30
    n_cols_cam = 20
    assert n_rows_cam * n_cols_cam == n_cams

    # Lay cameras out on a regular grid over [0, 100] × [0, 100] at height 10.
    xs = np.linspace(0.0, 100.0, n_cols_cam)
    ys = np.linspace(0.0, 100.0, n_rows_cam)
    xx, yy = np.meshgrid(xs, ys)
    cam_x = xx.ravel()
    cam_y = yy.ravel()
    cam_z = np.full(n_cams, 10.0)  # uniform altitude (up axis = Z)
    cam_positions = np.column_stack([cam_x, cam_y, cam_z])  # (600, 3)

    scene = _make_synthetic_scene(cam_positions, width=4000.0, height=3000.0, f=3000.0)
    init_cloud = _make_synthetic_init_cloud(n_points=10_000, xyz_min=0.0, xyz_max=100.0)

    blocks = partition_scene(
        scene,
        init_cloud,
        target_cameras_per_block=150,
        overlap_factor=0.20,
        visibility_threshold=0.25,
        min_cameras_per_block=20,
    )

    assert blocks is not None, "partition_scene returned None for a 600-camera scene"
    assert len(blocks) >= 2, f"Expected >= 2 blocks, got {len(blocks)}"

    # Every block must have at least one camera.
    for b in blocks:
        assert isinstance(b, SceneBlock)
        assert len(b.camera_indices) > 0, f"Block {b.block_id} has no cameras"
        assert b.point_mask.dtype == bool, "point_mask must be bool"
        assert b.point_mask.shape == (10_000,), (
            f"point_mask shape {b.point_mask.shape} != (10000,)"
        )
        assert b.n_points > 0, f"Block {b.block_id} has no points"

    # Every camera must appear in at least one block.
    covered = set()
    for b in blocks:
        covered.update(b.camera_indices)
    assert len(covered) == n_cams, (
        f"Only {len(covered)}/{n_cams} cameras were assigned to a block"
    )


# ---------------------------------------------------------------------------
# 6. test_should_partition_threshold
# ---------------------------------------------------------------------------

def test_should_partition_threshold():
    """Below threshold → False; at or above → True."""
    assert should_partition(499) is False
    assert should_partition(500) is True
    assert should_partition(501) is True
    assert should_partition(3000) is True


def test_should_partition_custom_threshold():
    """Custom threshold is respected."""
    assert should_partition(200, threshold=200) is True
    assert should_partition(199, threshold=200) is False
