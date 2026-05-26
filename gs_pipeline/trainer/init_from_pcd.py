"""Load and voxel-downsample a Metashape dense ``.ply`` for GS initialization.

The trainer never feeds the raw dense cloud to the optimizer:

- Even a "small" Metashape dense cloud is often 10-50 million points; MCMC's
  init quality plateaus well below 1 M points (the strategy redistributes
  splats aggressively in the first ~5k iterations).
- Loading the raw cloud into VRAM during init costs memory we want available
  for training. We voxel-downsample on CPU first.

The downsample is a pure-NumPy voxel grid keyed on per-axis integer voxel
indices (no open3d dependency in CI), with adaptive voxel sizing: if the first
pass leaves more than ``target_max_points`` points, voxel size doubles and the
pass repeats. Doubling is bounded so a degenerate cloud (e.g. a line) can't
loop forever.

Colors are kept (mean per voxel) and normalized to ``[0, 1]`` float32 — that's
the convention the trainer's init expects. If the PLY has no color, all points
get mid-gray (0.5).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData

_log = logging.getLogger(__name__)

# Default ceiling on points handed to the trainer. Overridden by
# vram_adaptive_max_points() when GPU info is available.
DEFAULT_TARGET_MAX_POINTS = 1_000_000

# VRAM → max init points. Bigger GPU = keep more of the dense cloud.
_VRAM_INIT_POINTS_TABLE: list[tuple[int, int]] = [
    (48_000_000_000, 5_000_000),   # A6000 / H100: 5M init pts
    (24_000_000_000, 3_000_000),   # A5000 / RTX 4090: 3M init pts
    (16_000_000_000, 2_000_000),   # 16 GB class: 2M
    (12_000_000_000, 1_500_000),   # 12 GB class: 1.5M
    (0,              1_000_000),   # 8-10 GB: 1M (original default)
]


def vram_adaptive_max_points(total_vram_bytes: int) -> int:
    for vram_thresh, max_pts in _VRAM_INIT_POINTS_TABLE:
        if total_vram_bytes >= vram_thresh:
            return max_pts
    return DEFAULT_TARGET_MAX_POINTS
# scene_extent / VOXEL_SIZE_FACTOR is the initial voxel edge length.
DEFAULT_VOXEL_SIZE_FACTOR = 1024.0
# Bound on adaptive voxel-size doubling.
MAX_VOXEL_DOUBLE_ITERATIONS = 12


@dataclass
class InitCloud:
    xyz: np.ndarray            # (N, 3) float32 in the dense cloud's frame
    rgb: np.ndarray            # (N, 3) float32 in [0, 1]
    scene_extent: float        # AABB diagonal (used for near/far plane, voxel size)
    n_loaded: int              # raw point count before downsample
    voxel_size: float          # final voxel size used
    aabb_min: np.ndarray       # (3,) float32
    aabb_max: np.ndarray       # (3,) float32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_downsample(
    ply_path: Path,
    *,
    target_max_points: int = DEFAULT_TARGET_MAX_POINTS,
    voxel_size_factor: float = DEFAULT_VOXEL_SIZE_FACTOR,
    fixed_voxel_size: Optional[float] = None,
) -> InitCloud:
    """Load a PLY, voxel-downsample, and return an InitCloud.

    Args:
        ply_path: path to a Metashape dense cloud PLY (ascii or binary).
        target_max_points: hard ceiling on the returned point count.
        voxel_size_factor: voxel size is scene_extent / this factor on the
            first pass. Ignored if ``fixed_voxel_size`` is given.
        fixed_voxel_size: pin the voxel size in world units (skips adaptive
            sizing). Useful for tests.

    Raises:
        FileNotFoundError, ValueError on malformed PLY or missing vertices.
    """
    xyz, rgb = _load_ply_xyz_rgb(ply_path)
    n_loaded = xyz.shape[0]
    if n_loaded == 0:
        raise ValueError(f"{ply_path}: 0 vertices loaded")

    aabb_min = xyz.min(axis=0)
    aabb_max = xyz.max(axis=0)
    scene_extent = float(np.linalg.norm(aabb_max - aabb_min))
    if scene_extent <= 0.0 or not math.isfinite(scene_extent):
        raise ValueError(f"{ply_path}: degenerate AABB, scene_extent={scene_extent}")

    if fixed_voxel_size is not None:
        voxel_size = float(fixed_voxel_size)
    else:
        voxel_size = scene_extent / voxel_size_factor

    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size={voxel_size} (<=0)")

    out_xyz, out_rgb, final_voxel = _adaptive_voxel_downsample(
        xyz, rgb, voxel_size, target_max_points,
    )

    return InitCloud(
        xyz=out_xyz,
        rgb=out_rgb,
        scene_extent=scene_extent,
        n_loaded=n_loaded,
        voxel_size=final_voxel,
        aabb_min=aabb_min.astype(np.float32),
        aabb_max=aabb_max.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _load_ply_xyz_rgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLY and return (xyz float32 [N,3], rgb float32 [N,3] in [0,1]).

    Tolerates PLYs with:
    - no color (returns mid-gray 0.5),
    - uint8 (red/green/blue 0..255) or float (0..1) colors,
    - lower-case (red/green/blue) or compact (r/g/b) color element names.
    """
    ply_path = Path(ply_path)
    if not ply_path.is_file():
        raise FileNotFoundError(ply_path)
    data = PlyData.read(str(ply_path))
    if "vertex" not in [el.name for el in data.elements]:
        raise ValueError(f"{ply_path}: no 'vertex' element")
    v = data["vertex"]
    names = set(v.data.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{ply_path}: vertex element missing x/y/z (have {names})")

    xyz = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)

    color_keys: Optional[tuple[str, str, str]] = None
    for keys in (("red", "green", "blue"), ("r", "g", "b"), ("diffuse_red", "diffuse_green", "diffuse_blue")):
        if set(keys).issubset(names):
            color_keys = keys
            break
    if color_keys is None:
        rgb = np.full_like(xyz, 0.5, dtype=np.float32)
        return xyz, rgb

    r = np.asarray(v[color_keys[0]])
    g = np.asarray(v[color_keys[1]])
    b = np.asarray(v[color_keys[2]])
    rgb = np.stack([r, g, b], axis=1).astype(np.float32)
    # Detect integer colors: uint8 (0..255) or uint16 (0..65535) vs float (0..1).
    max_val = rgb.max(initial=0.0)
    if max_val > 255.5:
        rgb = rgb / 65535.0
    elif max_val > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return xyz, rgb


def _adaptive_voxel_downsample(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
    target_max_points: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxel-downsample, doubling the voxel size until the count fits."""
    current = voxel_size
    for _ in range(MAX_VOXEL_DOUBLE_ITERATIONS):
        out_xyz, out_rgb = _voxel_downsample_once(xyz, rgb, current)
        if out_xyz.shape[0] <= target_max_points:
            return out_xyz, out_rgb, current
        _log.debug("voxel %.4g produced %d > target %d; doubling",
                   current, out_xyz.shape[0], target_max_points)
        current *= 2.0
    # If we're here, the cloud refuses to compress (e.g. all points colinear).
    # Truncate randomly to target_max_points.
    rng = np.random.default_rng(0)
    pick = rng.choice(out_xyz.shape[0], size=target_max_points, replace=False)
    return out_xyz[pick], out_rgb[pick], current


def _voxel_downsample_once(
    xyz: np.ndarray,
    rgb: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One pass of voxel-grid downsample (mean per voxel)."""
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)
    # np.unique on rows is O(N log N) but easy to read and fast enough for ~10M points.
    _, inv, counts = np.unique(voxel_idx, axis=0, return_inverse=True, return_counts=True)
    n_voxels = counts.shape[0]
    sum_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
    sum_rgb = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sum_xyz, inv, xyz)
    np.add.at(sum_rgb, inv, rgb)
    inv_count = (1.0 / counts).astype(np.float64)[:, None]
    out_xyz = (sum_xyz * inv_count).astype(np.float32)
    out_rgb = (sum_rgb * inv_count).astype(np.float32)
    return out_xyz, out_rgb
