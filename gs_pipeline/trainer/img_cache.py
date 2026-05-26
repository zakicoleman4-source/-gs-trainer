"""Pre-decode and resize training images to work_dir before the training loop.

Converts full-resolution source images to their training resolution once
so the training loop can do a cheap JPEG load instead of a 50MP decode +
resize on every step.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

_log = logging.getLogger(__name__)


def build_image_cache(
    image_paths: Sequence[Path],
    downscale_per_camera: Sequence[float],
    work_dir: Path,
    *,
    quality: int = 95,
) -> list[Path]:
    """Pre-resize all training images and write to work_dir/img_cache/.

    Returns a list of cached file paths (same order as image_paths).
    If a cached file already exists with the same size, it is reused.

    Args:
        image_paths: source image paths (may be full-res, e.g. 8000x6000 JPEG)
        downscale_per_camera: per-camera downscale factors (same length)
        work_dir: job work directory; cache goes under work_dir/img_cache/
        quality: JPEG quality for cached images (95 = visually lossless)

    Returns: list of Path objects pointing to cached images (same order).
    """
    from PIL import Image

    cache_dir = Path(work_dir) / "img_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_paths: list[Path] = []
    n = len(image_paths)

    for i, (src, ds) in enumerate(zip(image_paths, downscale_per_camera)):
        src = Path(src)
        # Cache filename: index + downscale factor in name so stale caches are detected
        ds_tag = f"{ds:.4f}".replace(".", "p")
        cache_name = f"{i:05d}_ds{ds_tag}.jpg"
        cache_path = cache_dir / cache_name

        if not cache_path.is_file():
            if i % 50 == 0 or i == n - 1:
                _log.info("img_cache: pre-decoding %d/%d ...", i + 1, n)
            img = Image.open(src).convert("RGB")
            if ds < 1.0:
                w, h = img.size
                new_w = max(1, int(round(w * ds)))
                new_h = max(1, int(round(h * ds)))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(cache_path, "JPEG", quality=quality, optimize=True)

        cached_paths.append(cache_path)

    return cached_paths
