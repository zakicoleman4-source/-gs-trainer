"""UI-side wrapper around ``trainer.pipeline._preflight`` for the dashboard.

Given an uploaded bundle zip, returns a ``PreflightReport`` the page can
render directly. Splits the heavy lifting (unzip + parse + downsample) out
of ``app.py`` so the UI module itself stays a thin Streamlit shell.

The bundle is extracted to a per-upload temp directory; the caller is
responsible for cleanup (e.g. by hashing the zip into the inbox path later).
"""
from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gs_pipeline.trainer.budget import Budget, GPUInfo, compute_budget, detect_gpu
from gs_pipeline.trainer.init_from_pcd import load_and_downsample
from gs_pipeline.trainer.parse_metashape import parse_cameras_xml


@dataclass
class PreflightReport:
    n_cameras: int
    image_max_side_orig: int    # longest side before downscale
    image_max_side: int         # cap applied for training
    downscale_factor: float
    total_megapixels: float     # post-downscale
    dense_pts_loaded: int
    dense_pts_after_downsample: int
    scene_extent: float
    target_splats: int
    hard_cap_splats: int
    iterations: int
    quality_preset: str
    gpu_name: str
    gpu_total_vram_bytes: int
    warnings: list[str]
    budget: Budget              # full Budget for downstream use


def estimate_training_minutes(target_splats: int, iterations: int) -> float:
    """Very rough wall-clock estimate to anchor the UI's "ETA" line.

    Calibrated against a single A5000: ~1.5M splats * 30k iters ≈ 30 min,
    with linear-ish scaling in both. The user-facing copy says "approximately"
    so we don't need this to be tight.
    """
    if target_splats <= 0 or iterations <= 0:
        return 0.0
    minutes_per_iter_per_msplat = 30.0 / (1_500_000 * 30_000) * 1_000_000
    return (iterations * (target_splats / 1_000_000) * minutes_per_iter_per_msplat)


def run_preflight(
    bundle_zip_path: Path,
    *,
    quality_preset: str = "Auto",
    gpu: Optional[GPUInfo] = None,
    extract_to: Optional[Path] = None,
) -> PreflightReport:
    """Extract + parse the uploaded bundle, compute a Budget, return a report.

    The bundle is **always** extracted into a fresh directory (either the
    caller-supplied ``extract_to`` or a tempdir). Path-traversal entries are
    rejected up front.
    """
    bundle_zip_path = Path(bundle_zip_path)
    if not bundle_zip_path.is_file():
        raise FileNotFoundError(bundle_zip_path)

    if extract_to is None:
        extract_to = Path(tempfile.mkdtemp(prefix="gs_preflight_"))
    extract_to = Path(extract_to)
    if extract_to.exists() and any(extract_to.iterdir()):
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_zip_path) as zf:
        for name in zf.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"bundle contains unsafe path {name!r}")
        zf.extractall(extract_to)

    cameras_xml = extract_to / "cameras.xml"
    dense_ply = extract_to / "dense.ply"
    images_dir = extract_to / "images"
    if not cameras_xml.is_file():
        raise FileNotFoundError(f"bundle missing cameras.xml")
    if not dense_ply.is_file():
        raise FileNotFoundError(f"bundle missing dense.ply")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"bundle missing images/ directory")

    scene = parse_cameras_xml(cameras_xml, image_dir=images_dir)
    init_cloud = load_and_downsample(dense_ply)
    gpu_info = gpu or detect_gpu() or GPUInfo(
        name="(preflight stub) 24GB", total_vram_bytes=24_000_000_000,
    )
    budget = compute_budget(
        gpu=gpu_info,
        image_sizes=scene.image_sizes,
        dense_pts=int(init_cloud.xyz.shape[0]),
        quality_preset=quality_preset,
    )
    longest_side = max(max(w, h) for w, h in scene.image_sizes)

    warnings: list[str] = []
    warnings.extend(scene.warnings)
    warnings.extend(budget.notes)
    return PreflightReport(
        n_cameras=budget.n_cameras,
        image_max_side_orig=int(longest_side),
        image_max_side=budget.image_max_side,
        downscale_factor=budget.downscale_factor,
        total_megapixels=budget.total_megapixels,
        dense_pts_loaded=init_cloud.n_loaded,
        dense_pts_after_downsample=int(init_cloud.xyz.shape[0]),
        scene_extent=init_cloud.scene_extent,
        target_splats=budget.target_splats,
        hard_cap_splats=budget.hard_cap_splats,
        iterations=budget.iterations,
        quality_preset=budget.quality_preset,
        gpu_name=budget.gpu.name,
        gpu_total_vram_bytes=budget.gpu.total_vram_bytes,
        warnings=warnings,
        budget=budget,
    )
