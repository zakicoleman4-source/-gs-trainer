"""Post-training Gaussian splat filtering.

Removes low-quality splats from a trained scene to reduce file size and
improve rendering quality. Three independent filters are chained:

1. **Opacity** — removes near-transparent splats (sigmoid(logit) < threshold).
2. **Scale** — removes splats with extreme axis scales (likely floaters/blobs).
3. **Statistical Outlier Removal (SOR)** — removes spatially isolated splats
   using a k-NN distance test (global_mean + std_ratio * global_std cutoff).

All functions operate on numpy arrays in optimizer space (opacities are logits,
scales are log-scale) matching the INRIA PLY layout from ``export_ply.py``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_log = logging.getLogger(__name__)

# Type alias for the full set of per-splat arrays.
_Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _apply_mask(
    mask: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
) -> _Arrays:
    """Index all splat arrays by a boolean mask, returning the filtered subset."""
    return (
        means[mask],
        scales[mask],
        quats[mask],
        opacities[mask],
        sh_dc[mask],
        sh_rest[mask],
    )


# ---------------------------------------------------------------------------
# Individual filters
# ---------------------------------------------------------------------------

def filter_opacity(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    *,
    min_opacity: float = 0.01,
) -> _Arrays:
    """Remove splats with ``sigmoid(opacity_logit) < min_opacity``.

    Opacities in the PLY are stored as logits (pre-sigmoid).
    """
    if opacities.shape[0] == 0:
        return means, scales, quats, opacities, sh_dc, sh_rest
    actual_opacity = 1.0 / (1.0 + np.exp(-opacities))
    mask = actual_opacity >= min_opacity
    return _apply_mask(mask, means, scales, quats, opacities, sh_dc, sh_rest)


def filter_scale(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    *,
    max_scale_factor: float = 10.0,
    scene_extent: float | None = None,
) -> _Arrays:
    """Remove splats with extreme scales.

    ``max_scale = median(scale) * max_scale_factor``.
    A splat is removed if *any* of its three axes exceeds ``max_scale``
    (after applying ``exp()`` to convert from log-space).
    """
    if scales.shape[0] == 0:
        return means, scales, quats, opacities, sh_dc, sh_rest
    actual_scales = np.exp(scales)  # (N, 3)
    median_scale = np.median(actual_scales)
    max_scale = median_scale * max_scale_factor
    mask = np.all(actual_scales <= max_scale, axis=1)
    return _apply_mask(mask, means, scales, quats, opacities, sh_dc, sh_rest)


def filter_statistical_outlier(
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    *,
    k: int = 20,
    std_ratio: float = 2.0,
) -> _Arrays:
    """Remove spatially isolated splats using Statistical Outlier Removal (SOR).

    For each splat, compute mean distance to its ``k`` nearest neighbors.
    Remove splats where ``mean_dist > global_mean + std_ratio * global_std``.
    """
    n = means.shape[0]
    if n == 0:
        return means, scales, quats, opacities, sh_dc, sh_rest

    from scipy.spatial import KDTree

    k_eff = min(k, n - 1)
    if k_eff <= 0:
        # Only 1 splat — nothing to compare against.
        return means, scales, quats, opacities, sh_dc, sh_rest

    tree = KDTree(means)
    # k_eff+1 because the closest neighbor is the point itself (distance 0).
    dists, _ = tree.query(means, k=k_eff + 1)
    if dists.ndim == 1:
        dists = dists[:, None]
    # Drop self-distance (column 0).
    mean_dists = np.mean(dists[:, 1:], axis=1)

    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_ratio * global_std
    mask = mean_dists <= threshold
    return _apply_mask(mask, means, scales, quats, opacities, sh_dc, sh_rest)


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterReport:
    """Summary of how many splats each filter stage removed."""
    n_input: int
    n_after_opacity: int
    n_after_scale: int
    n_after_sor: int
    n_output: int

    @property
    def summary(self) -> str:
        lines = [
            f"Filter: {self.n_input} -> {self.n_output} splats "
            f"({self.n_input - self.n_output} removed, "
            f"{100 * (1 - self.n_output / max(self.n_input, 1)):.1f}%)",
            f"  opacity: -{self.n_input - self.n_after_opacity}",
            f"  scale:   -{self.n_after_opacity - self.n_after_scale}",
            f"  SOR:     -{self.n_after_scale - self.n_after_sor}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chained filter
# ---------------------------------------------------------------------------

def filter_scene(
    *,
    means: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    opacities: np.ndarray,
    sh_dc: np.ndarray,
    sh_rest: np.ndarray,
    scene_extent: float | None = None,
    min_opacity: float = 0.005,
    sor_k: int = 20,
    sor_std_ratio: float = 2.0,
    max_scale_factor: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, FilterReport]:
    """Chain all filters: opacity -> scale -> SOR.

    Returns the six filtered arrays plus a :class:`FilterReport`.
    """
    n_input = means.shape[0]

    # 1. Opacity filter.
    means, scales, quats, opacities, sh_dc, sh_rest = filter_opacity(
        means, scales, quats, opacities, sh_dc, sh_rest,
        min_opacity=min_opacity,
    )
    n_after_opacity = means.shape[0]

    # 2. Scale filter.
    means, scales, quats, opacities, sh_dc, sh_rest = filter_scale(
        means, scales, quats, opacities, sh_dc, sh_rest,
        max_scale_factor=max_scale_factor,
        scene_extent=scene_extent,
    )
    n_after_scale = means.shape[0]

    # 3. Statistical Outlier Removal.
    means, scales, quats, opacities, sh_dc, sh_rest = filter_statistical_outlier(
        means, scales, quats, opacities, sh_dc, sh_rest,
        k=sor_k,
        std_ratio=sor_std_ratio,
    )
    n_after_sor = means.shape[0]

    report = FilterReport(
        n_input=n_input,
        n_after_opacity=n_after_opacity,
        n_after_scale=n_after_scale,
        n_after_sor=n_after_sor,
        n_output=n_after_sor,
    )
    _log.info("Post-training filter:\n%s", report.summary)

    return means, scales, quats, opacities, sh_dc, sh_rest, report
