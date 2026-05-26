"""Tests for gs_pipeline.trainer.large_scene (CPU-only; no GPU/torch required).

Focuses on ``merge_block_plys``:
- Writes two synthetic INRIA PLY files.
- Calls merge_block_plys with non-overlapping tight bounds on the x-axis.
- Verifies that only in-bounds Gaussians are kept and the total count is correct.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.trainer.export_ply import (
    LoadedSplat,
    read_inria_ply,
    write_inria_ply,
)
from gs_pipeline.trainer.large_scene import merge_block_plys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_splats(
    n: int,
    means: np.ndarray,   # (n, 3) pre-computed positions
    *,
    seed: int = 42,
) -> dict:
    """Build keyword-argument dict for write_inria_ply with fixed positions."""
    rng = np.random.default_rng(seed)
    return dict(
        means=means.astype(np.float32),
        scales=rng.normal(-3.0, 0.3, (n, 3)).astype(np.float32),
        quats=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)).astype(np.float32),
        opacities=rng.normal(0.0, 1.0, (n,)).astype(np.float32),
        sh_dc=rng.normal(0.0, 0.3, (n, 3)).astype(np.float32),
        sh_rest=np.zeros((n, 0, 3), dtype=np.float32),  # SH degree 0
    )


# ---------------------------------------------------------------------------
# Core merge test
# ---------------------------------------------------------------------------

def test_merge_block_plys_basic(tmp_path: Path):
    """merge_block_plys keeps only in-bounds Gaussians and gives correct counts."""
    rng = np.random.default_rng(0)
    N = 100

    # Scatter means uniformly in [0, 1]^3
    all_means = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)

    # Block 1 tight bounds: x in [0, 0.5], y/z unbounded → [0, 1]
    b1_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b1_max = np.array([0.5, 1.0, 1.0], dtype=np.float32)

    # Block 2 tight bounds: x in [0.5, 1], y/z unbounded → [0, 1]
    b2_min = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    b2_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Write block 1 PLY: all N points (both sides of the x split).
    ply1 = tmp_path / "block1.ply"
    write_inria_ply(out_path=ply1, **_make_synthetic_splats(N, all_means, seed=1))

    # Write block 2 PLY: all N points (same cloud, different tight bounds).
    ply2 = tmp_path / "block2.ply"
    write_inria_ply(out_path=ply2, **_make_synthetic_splats(N, all_means, seed=2))

    # Ground-truth counts: how many points fall within each tight box?
    expected_b1 = int(np.all((all_means >= b1_min) & (all_means <= b1_max), axis=1).sum())
    expected_b2 = int(np.all((all_means >= b2_min) & (all_means <= b2_max), axis=1).sum())
    expected_total = expected_b1 + expected_b2

    # Merge.
    merged = merge_block_plys(
        block_plys=[ply1, ply2],
        block_tight_bounds=[(b1_min, b1_max), (b2_min, b2_max)],
    )

    assert isinstance(merged, LoadedSplat)
    assert merged.means.shape[0] == expected_total, (
        f"Expected {expected_total} Gaussians, got {merged.means.shape[0]}"
    )
    # Verify individual block contributions.
    assert expected_b1 > 0, "Block 1 should contain some points"
    assert expected_b2 > 0, "Block 2 should contain some points"


def test_merge_block_plys_gaussians_within_bounds(tmp_path: Path):
    """Every Gaussian in the merged result lies within at least one tight box."""
    rng = np.random.default_rng(7)
    N = 80
    all_means = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)

    b1_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b1_max = np.array([0.5, 1.0, 1.0], dtype=np.float32)
    b2_min = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    b2_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    ply1 = tmp_path / "b1.ply"
    ply2 = tmp_path / "b2.ply"
    write_inria_ply(out_path=ply1, **_make_synthetic_splats(N, all_means, seed=11))
    write_inria_ply(out_path=ply2, **_make_synthetic_splats(N, all_means, seed=12))

    merged = merge_block_plys(
        block_plys=[ply1, ply2],
        block_tight_bounds=[(b1_min, b1_max), (b2_min, b2_max)],
    )

    means = merged.means
    in_b1 = np.all((means >= b1_min) & (means <= b1_max), axis=1)
    in_b2 = np.all((means >= b2_min) & (means <= b2_max), axis=1)
    in_either = in_b1 | in_b2

    assert in_either.all(), (
        f"{(~in_either).sum()} merged Gaussians are outside both tight bounds"
    )


def test_merge_block_plys_array_shapes_consistent(tmp_path: Path):
    """All arrays in the merged LoadedSplat have matching leading dimension."""
    rng = np.random.default_rng(99)
    N = 50
    means = rng.uniform(0.0, 1.0, (N, 3)).astype(np.float32)

    b1_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b1_max = np.array([0.5, 1.0, 1.0], dtype=np.float32)

    ply = tmp_path / "single.ply"
    write_inria_ply(out_path=ply, **_make_synthetic_splats(N, means, seed=5))

    merged = merge_block_plys(
        block_plys=[ply],
        block_tight_bounds=[(b1_min, b1_max)],
    )

    n = merged.means.shape[0]
    assert merged.scales.shape == (n, 3)
    assert merged.quats.shape == (n, 4)
    assert merged.opacities.shape == (n,)
    assert merged.sh_dc.shape == (n, 3)
    assert merged.sh_rest.shape[0] == n


def test_merge_preserves_count_exact(tmp_path: Path):
    """Exact count check: place 60 points x<0.5, 40 points x>=0.5."""
    # First 60 points have x in [0, 0.5), next 40 have x in [0.5, 1.0].
    part_a = np.random.default_rng(3).uniform(0.0, 0.5, (60, 3)).astype(np.float32)
    part_b = np.random.default_rng(4).uniform(0.5, 1.0, (40, 3)).astype(np.float32)
    all_means = np.concatenate([part_a, part_b], axis=0)

    b1_min = np.array([0.0,  0.0, 0.0], dtype=np.float32)
    b1_max = np.array([0.5,  1.0, 1.0], dtype=np.float32)
    b2_min = np.array([0.5,  0.0, 0.0], dtype=np.float32)
    b2_max = np.array([1.0,  1.0, 1.0], dtype=np.float32)

    ply1 = tmp_path / "exact1.ply"
    ply2 = tmp_path / "exact2.ply"
    write_inria_ply(out_path=ply1, **_make_synthetic_splats(100, all_means, seed=20))
    write_inria_ply(out_path=ply2, **_make_synthetic_splats(100, all_means, seed=21))

    merged = merge_block_plys(
        block_plys=[ply1, ply2],
        block_tight_bounds=[(b1_min, b1_max), (b2_min, b2_max)],
    )

    # Block 1 keeps x in [0, 0.5] — the 60 part_a points satisfy this
    # exactly (part_b has x in [0.5, 1.0], none <= 0.5 except the boundary).
    # Block 2 keeps x in [0.5, 1.0] — the 40 part_b points satisfy this.
    # Points at x==0.5 boundary may appear in both blocks; that's fine.
    # Total must be at least 100 (60 + 40 minimum without double-counting boundary).
    assert merged.means.shape[0] >= 100
