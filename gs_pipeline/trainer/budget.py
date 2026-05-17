"""Per-scene splat budget + image-downscale heuristic.

The point of this module is to answer two questions before training starts:

1. **How many splats can this GPU support?** The optimizer's per-splat
   training footprint (params + grads + Adam moments + MCMC relocate scratch)
   is about ~1.2 KB at SH degree 3. Rasterizer scratch + image tensors add
   a roughly fixed ~3 GB overhead. So::

       hard_cap_splats = (total_vram * safety - fixed_overhead) / per_splat_bytes

   On a 24 GB card this is ~14.7 M, on 48 GB ~33.5 M — i.e. the cap scales
   with the GPU and no scene is artificially limited by a magic number.

2. **How many splats does this scene want?** Whichever scene-complexity
   signal is strongest wins::

       geom     = 12 * dense_pts                     # MCMC subdivides each photogrammetry pt
       texture  = 300_000 * sqrt(total_megapixels)   # sub-linear in MP
       coverage = 50_000 * n_cameras
       target   = clip(max(geom, texture, coverage),
                       lo=500_000, hi=0.85 * hard_cap_splats)

   This answers "why not more gaussians for complex scenes": we do — bounded
   only by VRAM, not by a fixed ceiling.

It also picks an image-downscale factor: full-res training is great but very
high-resolution photos chew VRAM in the rasterizer's image-batch path. Default
cap is ``max_side=2000`` (configurable per-job and per-GPU class).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Constants (tunable; mirror trainer/config.yaml defaults)
# ---------------------------------------------------------------------------

# torch.cuda.set_per_process_memory_fraction we'll use at trainer startup.
DEFAULT_VRAM_SAFETY = 0.92
# Fixed overhead independent of splat count: rasterizer tiles + image batch +
# PyTorch/runtime scratch. Empirically ~2.5-3.5 GB; we budget 3.0 GB.
DEFAULT_FIXED_OVERHEAD_BYTES = 3_000_000_000
# Per-splat training footprint at SH degree 3 (params + grads + Adam + MCMC).
# ~236 B params + ~236 B grads + ~472 B Adam m/v + ~336 B scratch = ~1.28 KB.
DEFAULT_PER_SPLAT_BYTES = 1280
# Floor on target splat count: never train an under-populated scene.
DEFAULT_TARGET_FLOOR = 500_000
# Fraction of hard_cap reserved as relocate headroom.
DEFAULT_TARGET_CAP_RATIO = 0.85
# Image side cap (longest dim, pixels). Above this, we downscale per-job.
DEFAULT_MAX_IMAGE_SIDE = 2000

# Quality-preset multipliers. UI exposes only "Auto" / "Maximum".
QUALITY_ITERATIONS = {"Auto": 30_000, "Maximum": 50_000}
QUALITY_TARGET_MULT = {"Auto": 1.0, "Maximum": 1.25}


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUInfo:
    """Minimal snapshot of the GPU; lets tests inject without importing torch."""
    name: str
    total_vram_bytes: int


@dataclass
class Budget:
    # GPU
    gpu: GPUInfo
    hard_cap_splats: int                # absolute ceiling from VRAM
    # Scene signals
    n_cameras: int
    total_megapixels: float
    dense_pts: int
    # Decisions
    target_splats: int                  # what we'll tell MCMCStrategy as cap_max
    iterations: int                     # total training iterations
    image_max_side: int                 # per-job image side cap (pixels)
    downscale_factor: float             # 1.0 = full-res, 0.5 = half, etc.
    quality_preset: str
    notes: list[str] = field(default_factory=list)

    @property
    def will_be_undersized(self) -> bool:
        """True if the scene complexity exceeds the VRAM budget."""
        return any("exceeds VRAM budget" in n for n in self.notes)


# ---------------------------------------------------------------------------
# VRAM detection (mockable)
# ---------------------------------------------------------------------------

def detect_gpu() -> Optional[GPUInfo]:
    """Return GPU info via torch, or None if torch+CUDA is unavailable.

    Pure side-effect-free helper. Tests inject GPUInfo directly via
    ``compute_budget(..., gpu=...)`` and never need this.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return GPUInfo(name=props.name, total_vram_bytes=int(props.total_memory))


# ---------------------------------------------------------------------------
# Math (pure)
# ---------------------------------------------------------------------------

def compute_hard_cap_splats(
    total_vram_bytes: int,
    *,
    safety: float = DEFAULT_VRAM_SAFETY,
    fixed_overhead_bytes: int = DEFAULT_FIXED_OVERHEAD_BYTES,
    per_splat_bytes: int = DEFAULT_PER_SPLAT_BYTES,
) -> int:
    """Maximum splats that fit in VRAM at SH=3, given the constants above.

    Raises ValueError if the GPU is too small to hold even the fixed overhead.
    """
    usable = total_vram_bytes * safety - fixed_overhead_bytes
    if usable <= 0:
        raise ValueError(
            f"GPU too small: {total_vram_bytes / 1e9:.1f} GB VRAM after "
            f"{safety:.0%} safety leaves no room beyond {fixed_overhead_bytes / 1e9:.1f} GB overhead."
        )
    return int(usable // per_splat_bytes)


def compute_target_splats(
    dense_pts: int,
    total_megapixels: float,
    n_cameras: int,
    hard_cap_splats: int,
    *,
    cap_ratio: float = DEFAULT_TARGET_CAP_RATIO,
    floor: int = DEFAULT_TARGET_FLOOR,
    quality_mult: float = 1.0,
) -> tuple[int, bool]:
    """Choose target_splats from scene-complexity signals.

    Returns (target_splats, scene_exceeds_budget). The second value is True
    when the "natural" target was clipped by the VRAM cap — i.e. the scene
    wants more splats than the GPU can hold.
    """
    if dense_pts < 0 or total_megapixels < 0 or n_cameras < 0:
        raise ValueError("dense_pts / total_megapixels / n_cameras must all be non-negative")
    if hard_cap_splats <= 0:
        raise ValueError(f"hard_cap_splats must be positive; got {hard_cap_splats}")

    geom = 12 * dense_pts
    texture = int(300_000 * math.sqrt(max(total_megapixels, 0.0)))
    coverage = 50_000 * n_cameras
    natural = max(geom, texture, coverage)
    natural_with_quality = int(natural * quality_mult)

    cap = int(hard_cap_splats * cap_ratio)
    exceeded = natural_with_quality > cap
    target = min(natural_with_quality, cap)
    target = max(target, floor)
    return target, exceeded


def compute_image_downscale(
    longest_side_px: int,
    *,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
) -> tuple[int, float]:
    """Decide a per-job image downscale.

    Returns (max_side_used, downscale_factor). downscale_factor of 1.0 means
    "use full resolution"; 0.5 means "halve both axes".
    """
    if longest_side_px <= 0:
        raise ValueError(f"longest_side_px must be positive; got {longest_side_px}")
    if longest_side_px <= max_image_side:
        return longest_side_px, 1.0
    factor = max_image_side / longest_side_px
    return max_image_side, factor


# ---------------------------------------------------------------------------
# Top-level: pull it all together
# ---------------------------------------------------------------------------

def compute_budget(
    *,
    gpu: GPUInfo,
    image_sizes: Sequence[tuple[int, int]],
    dense_pts: int,
    quality_preset: str = "Auto",
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    safety: float = DEFAULT_VRAM_SAFETY,
    fixed_overhead_bytes: int = DEFAULT_FIXED_OVERHEAD_BYTES,
    per_splat_bytes: int = DEFAULT_PER_SPLAT_BYTES,
) -> Budget:
    """Top-level: turn (gpu, parsed scene, init cloud) into a Budget.

    Args:
        gpu: result of ``detect_gpu()`` or an explicit mock.
        image_sizes: list of (width_px, height_px) per camera, as the parser
            returns in ``ParsedScene.image_sizes``.
        dense_pts: point count *after* voxel-downsample (from InitCloud.xyz).
        quality_preset: ``"Auto"`` or ``"Maximum"`` (see QUALITY_*).
        max_image_side: longest image edge after downscale.

    Raises:
        ValueError on unknown preset, empty image list, or non-positive counts.
    """
    if quality_preset not in QUALITY_ITERATIONS:
        raise ValueError(
            f"unknown quality preset {quality_preset!r}; "
            f"must be one of {sorted(QUALITY_ITERATIONS)}"
        )
    if not image_sizes:
        raise ValueError("image_sizes is empty; cannot compute budget")
    if dense_pts <= 0:
        raise ValueError(f"dense_pts must be positive; got {dense_pts}")

    n_cameras = len(image_sizes)
    longest_side = max(max(w, h) for w, h in image_sizes)
    side_used, downscale_factor = compute_image_downscale(
        longest_side, max_image_side=max_image_side,
    )
    # Megapixel count after downscale (what actually feeds the trainer).
    total_mp = 0.0
    for w, h in image_sizes:
        eff_w = w * downscale_factor
        eff_h = h * downscale_factor
        total_mp += (eff_w * eff_h) / 1e6

    hard_cap = compute_hard_cap_splats(
        gpu.total_vram_bytes,
        safety=safety,
        fixed_overhead_bytes=fixed_overhead_bytes,
        per_splat_bytes=per_splat_bytes,
    )
    target, exceeded = compute_target_splats(
        dense_pts=dense_pts,
        total_megapixels=total_mp,
        n_cameras=n_cameras,
        hard_cap_splats=hard_cap,
        quality_mult=QUALITY_TARGET_MULT[quality_preset],
    )

    notes: list[str] = []
    if downscale_factor < 1.0:
        notes.append(
            f"Image longest side {longest_side}px > cap {max_image_side}px; "
            f"downscaling by {downscale_factor:.3f} for training. "
            f"Final eval renders run at full resolution."
        )
    if exceeded:
        notes.append(
            f"Scene complexity exceeds VRAM budget; target splats clipped to "
            f"{target / 1e6:.1f}M (85% of {hard_cap / 1e6:.1f}M hard cap). "
            f"Splat density will be reduced ~"
            f"{100 - 100 * (hard_cap * DEFAULT_TARGET_CAP_RATIO) / max(12 * dense_pts, 1):.0f}%."
        )

    return Budget(
        gpu=gpu,
        hard_cap_splats=hard_cap,
        n_cameras=n_cameras,
        total_megapixels=total_mp,
        dense_pts=dense_pts,
        target_splats=target,
        iterations=QUALITY_ITERATIONS[quality_preset],
        image_max_side=side_used,
        downscale_factor=downscale_factor,
        quality_preset=quality_preset,
        notes=notes,
    )
