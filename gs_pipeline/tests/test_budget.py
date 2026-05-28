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
# estimate_vram_bytes / fits_in_vram — must reflect *training* peak, not
# inference. The old 250 B/px estimate let the budget pick combos that OOM'd.
# ---------------------------------------------------------------------------

def test_vram_estimate_accounts_for_training_per_pixel_cost():
    """A 7.7 MP image's rasterizer term must be GBs, not a rounding error."""
    from gs_pipeline.trainer.budget import (
        estimate_vram_bytes, DEFAULT_FIXED_OVERHEAD_BYTES, PER_PIXEL_BYTES,
    )
    # per_pixel must be a *training*-scale number, well above the old 250.
    assert PER_PIXEL_BYTES >= 500
    # Image-only contribution (zero splats) for 3200x2400 must exceed ~3 GB.
    est = estimate_vram_bytes(0, 3200, 2400)
    image_term = est - DEFAULT_FIXED_OVERHEAD_BYTES
    assert image_term > 3 * GB


def test_vram_estimate_includes_intersection_density_term():
    """More splats at the same resolution must raise the estimate beyond the
    plain per-splat term (tile-intersection lists scale with density)."""
    from gs_pipeline.trainer.budget import estimate_vram_bytes, DEFAULT_PER_SPLAT_BYTES
    low = estimate_vram_bytes(1_000_000, 2000, 2000)
    high = estimate_vram_bytes(10_000_000, 2000, 2000)
    delta = high - low
    plain_splat_delta = 9_000_000 * DEFAULT_PER_SPLAT_BYTES
    assert delta > plain_splat_delta  # intersection term adds on top


def test_high_splat_highres_combo_correctly_rejected_on_24gb():
    """13 M splats + 3200x2400 must NOT be reported as fitting 24 GB.

    This is the exact production scenario that OOM'd six shipped images: the
    splat memory alone (~16.6 GB) plus a full-res image cannot coexist in the
    22 GB safety budget. The estimator must catch it.
    """
    from gs_pipeline.trainer.budget import fits_in_vram
    assert not fits_in_vram(24 * GB, 13_000_000, 3200, 2400)
    # ...but a sensibly reduced image side for that splat count does fit.
    assert fits_in_vram(24 * GB, 13_000_000, 1600, 1200)


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
    # 24 GB card's table value is 3200px, but this scene wants 0.87*hard_cap
    # (~13M) splats. Those splats alone consume ~16.6 GB, so the *training*
    # VRAM estimate (corrected to account for the forward+backward rasterizer
    # buffers + tile-intersection lists) no longer fits a 3200px image. The
    # budget correctly steps the image side DOWN so the run actually fits —
    # this is the fix for the repeated production OOMs. We assert the side is
    # reduced below the table value and that every camera shares one factor.
    assert b.image_max_side < 3200
    assert b.image_max_side >= 1600  # not collapsed to the floor
    expected_factor = b.image_max_side / 4000
    assert b.downscale_factor == pytest.approx(expected_factor, rel=0.02)
    assert len(b.downscale_per_camera) == 80
    assert all(abs(f - expected_factor) < 0.02 for f in b.downscale_per_camera)
    # 12 * 1.5M dense = 18M; clipped at 0.87 * hard_cap_24gb
    assert b.target_splats == int(b.hard_cap_splats * DEFAULT_TARGET_CAP_RATIO)
    assert any("exceeds VRAM budget" in n for n in b.notes)
    assert b.iterations == QUALITY_ITERATIONS["Auto"]
    # The chosen (splats, image) combo must actually fit the training estimate.
    from gs_pipeline.trainer.budget import fits_in_vram
    _tw = int(4000 * b.downscale_factor)
    _th = int(3000 * b.downscale_factor)
    assert fits_in_vram(GPU_24GB.total_vram_bytes, b.target_splats, _tw, _th)


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
    assert all(f == 1.0 for f in b.downscale_per_camera)
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


def test_mixed_camera_sizes_per_camera_downscale():
    """Sony Alpha (8000x6000) + GoPro (3840x2160) get independent downscale factors."""
    b = compute_budget(
        gpu=GPUInfo("RTX A5000", 24 * GB),
        image_sizes=[(8000, 6000)] * 5 + [(3840, 2160)] * 10,
        dense_pts=500_000,
        quality_preset="Auto",
    )
    assert len(b.downscale_per_camera) == 15
    side = b.image_max_side  # whatever side actually fits the training estimate
    # Each camera is independently scaled so its longest edge lands at `side`.
    # Sony (8000px longest) gets a more aggressive factor than GoPro (3840px).
    assert all(
        f == pytest.approx(side / 8000, rel=0.01) for f in b.downscale_per_camera[:5]
    )
    assert all(
        f == pytest.approx(side / 3840, rel=0.01) for f in b.downscale_per_camera[5:]
    )
    # Per-camera factors are genuinely different (the whole point of per-camera).
    assert b.downscale_per_camera[0] < b.downscale_per_camera[5]
    # Global downscale_factor is the smallest (Sony's, since 8000 >> side).
    assert b.downscale_factor == pytest.approx(side / 8000, rel=0.01)
