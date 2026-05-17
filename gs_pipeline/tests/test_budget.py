"""Tests for ``gs_pipeline.trainer.budget``.

Mocks the GPU directly via ``GPUInfo`` — no torch import. Covers boundary
cases that historically broke similar trainers: huge scenes, tiny scenes,
oversized images, malformed inputs, quality presets.
"""
from __future__ import annotations

import math

import pytest

from gs_pipeline.trainer.budget import (
    DEFAULT_FIXED_OVERHEAD_BYTES,
    DEFAULT_PER_SPLAT_BYTES,
    DEFAULT_TARGET_CAP_RATIO,
    DEFAULT_TARGET_FLOOR,
    DEFAULT_VRAM_SAFETY,
    GPUInfo,
    QUALITY_ITERATIONS,
    QUALITY_TARGET_MULT,
    compute_budget,
    compute_hard_cap_splats,
    compute_image_downscale,
    compute_target_splats,
)


GB = 1_000_000_000

GPU_24GB = GPUInfo(name="RTX A5000", total_vram_bytes=24 * GB)
GPU_48GB = GPUInfo(name="RTX A6000", total_vram_bytes=48 * GB)


# ---------------------------------------------------------------------------
# hard_cap_splats
# ---------------------------------------------------------------------------

def test_hard_cap_scales_with_vram():
    cap_24 = compute_hard_cap_splats(24 * GB)
    cap_48 = compute_hard_cap_splats(48 * GB)
    assert cap_48 > cap_24
    # 48 GB has ~2.4x the headroom after fixed overhead (overhead doesn't scale).
    ratio = cap_48 / cap_24
    assert 2.0 < ratio < 2.7


def test_hard_cap_at_24gb_in_expected_range():
    """Sanity: 24 GB card should land near ~14.7 M splats per the plan math."""
    cap = compute_hard_cap_splats(24 * GB)
    assert 12_000_000 < cap < 17_000_000


def test_hard_cap_at_48gb_in_expected_range():
    cap = compute_hard_cap_splats(48 * GB)
    assert 30_000_000 < cap < 36_000_000


def test_hard_cap_rejects_too_small_gpu():
    # 2 GB total < 3 GB fixed overhead -> impossible.
    with pytest.raises(ValueError, match="GPU too small"):
        compute_hard_cap_splats(2 * GB)


def test_hard_cap_overrides_constants():
    """Custom fixed_overhead reduces the cap proportionally."""
    cap_normal = compute_hard_cap_splats(24 * GB)
    cap_more_overhead = compute_hard_cap_splats(24 * GB, fixed_overhead_bytes=5 * GB)
    assert cap_more_overhead < cap_normal


# ---------------------------------------------------------------------------
# target_splats
# ---------------------------------------------------------------------------

def test_target_uses_geom_when_dense_dominates():
    hard_cap = compute_hard_cap_splats(48 * GB)
    target, exceeded = compute_target_splats(
        dense_pts=1_000_000, total_megapixels=100.0, n_cameras=50,
        hard_cap_splats=hard_cap,
    )
    # geom = 12e6, texture = 3e6, coverage = 2.5e6 -> geom wins.
    expected_natural = 12 * 1_000_000
    assert target == min(expected_natural, int(hard_cap * DEFAULT_TARGET_CAP_RATIO))
    assert exceeded is False


def test_target_uses_texture_when_high_mp_low_dense():
    hard_cap = compute_hard_cap_splats(48 * GB)
    # dense small, MP huge -> texture wins.
    target, _ = compute_target_splats(
        dense_pts=10_000, total_megapixels=10_000.0, n_cameras=10,
        hard_cap_splats=hard_cap,
    )
    expected_texture = int(300_000 * math.sqrt(10_000.0))
    assert target == min(max(expected_texture, DEFAULT_TARGET_FLOOR),
                          int(hard_cap * DEFAULT_TARGET_CAP_RATIO))


def test_target_uses_coverage_when_many_cameras():
    hard_cap = compute_hard_cap_splats(48 * GB)
    target, _ = compute_target_splats(
        dense_pts=10_000, total_megapixels=1.0, n_cameras=400,
        hard_cap_splats=hard_cap,
    )
    expected_coverage = 50_000 * 400  # 20M
    assert target == min(expected_coverage, int(hard_cap * DEFAULT_TARGET_CAP_RATIO))


def test_target_respects_floor_for_tiny_scenes():
    hard_cap = compute_hard_cap_splats(24 * GB)
    target, _ = compute_target_splats(
        dense_pts=100, total_megapixels=0.1, n_cameras=1,
        hard_cap_splats=hard_cap,
    )
    assert target == DEFAULT_TARGET_FLOOR


def test_target_clips_to_cap_ratio_for_huge_scenes():
    hard_cap = compute_hard_cap_splats(24 * GB)
    target, exceeded = compute_target_splats(
        dense_pts=50_000_000, total_megapixels=10_000.0, n_cameras=1000,
        hard_cap_splats=hard_cap,
    )
    assert target == int(hard_cap * DEFAULT_TARGET_CAP_RATIO)
    assert exceeded is True


def test_target_quality_multiplier_increases_target():
    hard_cap = compute_hard_cap_splats(48 * GB)  # big GPU so we don't clip
    auto, _ = compute_target_splats(
        dense_pts=500_000, total_megapixels=100.0, n_cameras=50,
        hard_cap_splats=hard_cap, quality_mult=1.0,
    )
    maximum, _ = compute_target_splats(
        dense_pts=500_000, total_megapixels=100.0, n_cameras=50,
        hard_cap_splats=hard_cap, quality_mult=1.25,
    )
    assert maximum > auto
    assert maximum == int(auto * 1.25) or maximum == DEFAULT_TARGET_FLOOR or auto == DEFAULT_TARGET_FLOOR


def test_target_rejects_negative_inputs():
    hard_cap = compute_hard_cap_splats(24 * GB)
    with pytest.raises(ValueError):
        compute_target_splats(dense_pts=-1, total_megapixels=1.0, n_cameras=1, hard_cap_splats=hard_cap)
    with pytest.raises(ValueError):
        compute_target_splats(dense_pts=1, total_megapixels=-1.0, n_cameras=1, hard_cap_splats=hard_cap)
    with pytest.raises(ValueError):
        compute_target_splats(dense_pts=1, total_megapixels=1.0, n_cameras=-1, hard_cap_splats=hard_cap)
    with pytest.raises(ValueError):
        compute_target_splats(dense_pts=1, total_megapixels=1.0, n_cameras=1, hard_cap_splats=0)


# ---------------------------------------------------------------------------
# image_downscale
# ---------------------------------------------------------------------------

def test_image_downscale_below_cap_is_identity():
    side, factor = compute_image_downscale(1600)
    assert (side, factor) == (1600, 1.0)


def test_image_downscale_above_cap_scales_to_max():
    side, factor = compute_image_downscale(8000, max_image_side=2000)
    assert side == 2000
    assert factor == pytest.approx(0.25, rel=1e-6)


def test_image_downscale_rejects_zero_side():
    with pytest.raises(ValueError):
        compute_image_downscale(0)


# ---------------------------------------------------------------------------
# compute_budget (end-to-end)
# ---------------------------------------------------------------------------

def _typical_scene():
    """80 photos at 4000x3000 (12 MP)."""
    return [(4000, 3000)] * 80


def test_typical_scene_on_24gb_returns_sane_numbers():
    b = compute_budget(
        gpu=GPU_24GB,
        image_sizes=_typical_scene(),
        dense_pts=1_500_000,
        quality_preset="Auto",
    )
    assert b.n_cameras == 80
    # Image side 4000 > 2000 cap -> downscale to 0.5.
    assert b.image_max_side == 2000
    assert b.downscale_factor == pytest.approx(0.5)
    # total_mp after downscale = 80 * (2000*1500)/1e6 = 240 MP
    assert b.total_megapixels == pytest.approx(240.0, rel=0.01)
    # 12 * 1.5M dense = 18M; clipped at 0.85 * hard_cap_24gb ~12.5M
    assert b.target_splats == int(b.hard_cap_splats * DEFAULT_TARGET_CAP_RATIO)
    assert any("exceeds VRAM budget" in n for n in b.notes)
    assert b.iterations == QUALITY_ITERATIONS["Auto"]


def test_same_scene_on_48gb_uses_more_splats():
    b24 = compute_budget(
        gpu=GPU_24GB, image_sizes=_typical_scene(), dense_pts=1_500_000,
    )
    b48 = compute_budget(
        gpu=GPU_48GB, image_sizes=_typical_scene(), dense_pts=1_500_000,
    )
    assert b48.target_splats > b24.target_splats


def test_simple_scene_under_full_budget_no_exceed_note():
    b = compute_budget(
        gpu=GPU_48GB,
        image_sizes=[(1024, 1024)] * 30,
        dense_pts=200_000,
        quality_preset="Auto",
    )
    assert b.downscale_factor == 1.0
    assert b.target_splats < b.hard_cap_splats * DEFAULT_TARGET_CAP_RATIO
    assert not any("exceeds VRAM budget" in n for n in b.notes)


def test_quality_preset_maximum_more_iterations_and_splats():
    auto = compute_budget(
        gpu=GPU_48GB, image_sizes=[(1024, 1024)] * 30, dense_pts=500_000,
        quality_preset="Auto",
    )
    maximum = compute_budget(
        gpu=GPU_48GB, image_sizes=[(1024, 1024)] * 30, dense_pts=500_000,
        quality_preset="Maximum",
    )
    assert maximum.iterations == QUALITY_ITERATIONS["Maximum"]
    assert auto.iterations == QUALITY_ITERATIONS["Auto"]
    assert maximum.target_splats >= auto.target_splats


def test_unknown_quality_preset_raises():
    with pytest.raises(ValueError, match="unknown quality preset"):
        compute_budget(
            gpu=GPU_24GB, image_sizes=[(1024, 1024)], dense_pts=1000,
            quality_preset="Ultra",
        )


def test_empty_image_list_raises():
    with pytest.raises(ValueError, match="image_sizes is empty"):
        compute_budget(gpu=GPU_24GB, image_sizes=[], dense_pts=1000)


def test_zero_dense_pts_raises():
    with pytest.raises(ValueError, match="dense_pts must be positive"):
        compute_budget(gpu=GPU_24GB, image_sizes=[(800, 600)], dense_pts=0)


def test_tiny_scene_returns_floor():
    b = compute_budget(
        gpu=GPU_24GB, image_sizes=[(640, 480)], dense_pts=200,
    )
    assert b.target_splats == DEFAULT_TARGET_FLOOR


def test_will_be_undersized_flag():
    b = compute_budget(
        gpu=GPU_24GB,
        image_sizes=[(8000, 6000)] * 400,
        dense_pts=12_000_000,
        quality_preset="Maximum",
    )
    assert b.will_be_undersized is True


def test_constants_used_consistently():
    """Sanity: the public constants are the ones the math actually uses."""
    cap_default = compute_hard_cap_splats(24 * GB)
    cap_explicit = compute_hard_cap_splats(
        24 * GB,
        safety=DEFAULT_VRAM_SAFETY,
        fixed_overhead_bytes=DEFAULT_FIXED_OVERHEAD_BYTES,
        per_splat_bytes=DEFAULT_PER_SPLAT_BYTES,
    )
    assert cap_default == cap_explicit
