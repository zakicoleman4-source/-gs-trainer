"""Tests for the post-training Gaussian splat filter module.

All tests are CPU-only and use small synthetic numpy arrays — no torch, no GPU,
no disk I/O.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from gs_pipeline.trainer.filter_splats import (
    FilterReport,
    filter_opacity,
    filter_scale,
    filter_scene,
    filter_statistical_outlier,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic splat arrays
# ---------------------------------------------------------------------------

def _inverse_sigmoid(x: float) -> float:
    """Convert a probability to its logit (inverse sigmoid)."""
    x = max(1e-7, min(1 - 1e-7, x))
    return math.log(x / (1.0 - x))


def _make_splats(
    n: int,
    *,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    means: np.ndarray | None = None,
    sh_degree: int = 0,
):
    """Return (means, scales, quats, opacities, sh_dc, sh_rest) for *n* splats.

    Caller can override opacities (logits), scales (log-scale), or means;
    everything else gets sensible defaults.
    """
    if means is None:
        means = np.random.default_rng(42).standard_normal((n, 3)).astype(np.float32)
    if opacities is None:
        opacities = np.full(n, _inverse_sigmoid(0.5), dtype=np.float32)
    if scales is None:
        scales = np.full((n, 3), np.log(0.01), dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # identity quaternion
    sh_dc = np.random.default_rng(7).standard_normal((n, 3)).astype(np.float32)
    k = (sh_degree + 1) ** 2 - 1
    sh_rest = np.zeros((n, k, 3), dtype=np.float32)
    return means, scales, quats, opacities, sh_dc, sh_rest


# ---------------------------------------------------------------------------
# test_opacity_filter_removes_transparent
# ---------------------------------------------------------------------------

def test_opacity_filter_removes_transparent():
    """Splats below the opacity threshold are removed; those above are kept."""
    n = 100
    # 50 with actual opacity 0.001 (should be removed at threshold 0.01)
    # 50 with actual opacity 0.9 (should be kept)
    logits = np.array(
        [_inverse_sigmoid(0.001)] * 50 + [_inverse_sigmoid(0.9)] * 50,
        dtype=np.float32,
    )
    means, scales, quats, _, sh_dc, sh_rest = _make_splats(n)
    out = filter_opacity(means, scales, quats, logits, sh_dc, sh_rest, min_opacity=0.01)
    assert out[0].shape[0] == 50, f"expected 50, got {out[0].shape[0]}"
    # The kept splats should all have high opacity logits.
    kept_opacities = 1.0 / (1.0 + np.exp(-out[3]))
    assert np.all(kept_opacities >= 0.01)


# ---------------------------------------------------------------------------
# test_sor_removes_outliers
# ---------------------------------------------------------------------------

def test_sor_removes_outliers():
    """A tight cluster with a few distant outliers — outliers should be removed."""
    rng = np.random.default_rng(123)
    # 200 splats in a tight cluster around the origin.
    cluster = rng.normal(loc=0.0, scale=0.1, size=(200, 3)).astype(np.float32)
    # 5 outliers far away.
    outliers = rng.normal(loc=50.0, scale=0.1, size=(5, 3)).astype(np.float32)
    all_means = np.concatenate([cluster, outliers], axis=0)
    n = all_means.shape[0]
    means, scales, quats, opacities, sh_dc, sh_rest = _make_splats(n, means=all_means)
    out = filter_statistical_outlier(
        means, scales, quats, opacities, sh_dc, sh_rest,
        k=20, std_ratio=2.0,
    )
    n_kept = out[0].shape[0]
    # All 5 outliers should be gone (maybe a few cluster edge splats too,
    # but the count must be strictly less than the input).
    assert n_kept <= 200, f"expected <=200 kept, got {n_kept}"
    # The outlier means (>40) should not appear in the result.
    assert np.all(np.linalg.norm(out[0], axis=1) < 40.0)


# ---------------------------------------------------------------------------
# test_scale_filter_removes_oversized
# ---------------------------------------------------------------------------

def test_scale_filter_removes_oversized():
    """Splats with extreme scales are removed; normal ones are kept."""
    n = 100
    # 90 normal splats: exp(log(0.01)) = 0.01
    normal_scales = np.full((90, 3), np.log(0.01), dtype=np.float32)
    # 10 giant splats: exp(log(100)) = 100  (way above 10 * median)
    giant_scales = np.full((10, 3), np.log(100.0), dtype=np.float32)
    all_scales = np.concatenate([normal_scales, giant_scales], axis=0)
    means, _, quats, opacities, sh_dc, sh_rest = _make_splats(n)
    out = filter_scale(means, all_scales, quats, opacities, sh_dc, sh_rest, max_scale_factor=10.0)
    n_kept = out[0].shape[0]
    assert n_kept == 90, f"expected 90, got {n_kept}"


# ---------------------------------------------------------------------------
# test_filter_scene_chains_all
# ---------------------------------------------------------------------------

def test_filter_scene_chains_all():
    """The full chain applies all three filters and returns correct counts."""
    rng = np.random.default_rng(99)
    # 100 transparent (will be removed by opacity filter).
    # 100 normal (good cluster).
    # 50 oversized (will be removed by scale filter).
    # 5 scattered outliers (each far from everything; will be removed by SOR).
    n = 255
    opacities = np.concatenate([
        np.full(100, _inverse_sigmoid(0.001), dtype=np.float32),  # transparent
        np.full(100, _inverse_sigmoid(0.5), dtype=np.float32),    # normal
        np.full(50, _inverse_sigmoid(0.5), dtype=np.float32),     # oversized (opacity OK)
        np.full(5, _inverse_sigmoid(0.5), dtype=np.float32),      # outliers (opacity OK)
    ])
    scales = np.concatenate([
        np.full((100, 3), np.log(0.01), dtype=np.float32),   # transparent (scale OK)
        np.full((100, 3), np.log(0.01), dtype=np.float32),   # normal
        np.full((50, 3), np.log(1000.0), dtype=np.float32),  # oversized
        np.full((5, 3), np.log(0.01), dtype=np.float32),     # outliers (scale OK)
    ])
    cluster_means = rng.normal(0.0, 0.1, size=(250, 3)).astype(np.float32)
    # 5 isolated outliers — each at a unique distant location, not a cluster.
    outlier_means = np.array([
        [500, 0, 0], [0, 500, 0], [0, 0, 500], [-500, 0, 0], [0, -500, 0],
    ], dtype=np.float32)
    all_means = np.concatenate([cluster_means, outlier_means], axis=0)

    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    sh_dc = rng.standard_normal((n, 3)).astype(np.float32)
    sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    m, sc, q, o, dc, rest, report = filter_scene(
        means=all_means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        min_opacity=0.01,
        sor_k=20,
        sor_std_ratio=2.0,
        max_scale_factor=10.0,
    )

    assert report.n_input == 255
    # After opacity: 100 transparent removed -> 155 left.
    assert report.n_after_opacity == 155
    # After scale: 50 oversized removed -> 105 left.
    assert report.n_after_scale == 105
    # After SOR: 5 isolated outliers removed -> ~100 (may lose a few cluster-edge).
    assert report.n_after_sor <= 102
    assert report.n_after_sor >= 95
    assert report.n_output == report.n_after_sor
    assert m.shape[0] == report.n_output
    assert isinstance(report.summary, str)
    assert "Filter:" in report.summary


# ---------------------------------------------------------------------------
# test_filter_preserves_good_splats
# ---------------------------------------------------------------------------

def test_filter_preserves_good_splats():
    """When all splats are well-behaved, none are removed."""
    rng = np.random.default_rng(55)
    # Use a full uniform grid so every point has near-identical k-NN distances
    # (no boundary truncation artefacts).
    side = 8
    n = side ** 3  # 512
    grid = np.stack(np.meshgrid(
        np.linspace(0, 1, side),
        np.linspace(0, 1, side),
        np.linspace(0, 1, side),
    ), axis=-1).reshape(-1, 3).astype(np.float32)
    means = grid
    opacities = np.full(n, _inverse_sigmoid(0.8), dtype=np.float32)
    scales = np.full((n, 3), np.log(0.05), dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    sh_dc = rng.standard_normal((n, 3)).astype(np.float32)
    sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    m, sc, q, o, dc, rest, report = filter_scene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        sor_std_ratio=5.0,  # lenient — grid corners have larger k-NN dists than interior
    )
    assert report.n_output == n, f"expected {n} kept, got {report.n_output}"


# ---------------------------------------------------------------------------
# test_filter_empty_input
# ---------------------------------------------------------------------------

def test_filter_empty_input():
    """Edge case: 0 splats should not crash."""
    means = np.zeros((0, 3), dtype=np.float32)
    scales = np.zeros((0, 3), dtype=np.float32)
    quats = np.zeros((0, 4), dtype=np.float32)
    opacities = np.zeros((0,), dtype=np.float32)
    sh_dc = np.zeros((0, 3), dtype=np.float32)
    sh_rest = np.zeros((0, 0, 3), dtype=np.float32)

    m, sc, q, o, dc, rest, report = filter_scene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
    )
    assert report.n_input == 0
    assert report.n_output == 0
    assert m.shape[0] == 0
