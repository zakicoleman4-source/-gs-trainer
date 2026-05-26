"""Post-training quality report.

Generates a human-readable and machine-parseable quality assessment covering
data quality, camera coverage, Gaussian splat quality, and filtering summary.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


def _grade(value: float, thresholds: list[tuple[float, str]]) -> str:
    for thresh, label in thresholds:
        if value >= thresh:
            return label
    return thresholds[-1][1]


@dataclass
class QualityReport:
    # Data quality
    n_cameras: int = 0
    cameras_grade: str = ""
    avg_image_megapixels: float = 0.0
    training_resolution: str = ""
    dense_pts: int = 0
    pts_per_camera: float = 0.0
    data_grade: str = ""

    # Camera coverage
    camera_spread_m: float = 0.0
    mean_baseline_m: float = 0.0
    coverage_grade: str = ""

    # GS quality
    final_psnr: float = 0.0
    final_ssim: float = 0.0
    psnr_grade: str = ""
    final_splat_count: int = 0
    splats_per_camera: float = 0.0

    # Filtering
    filter_input: int = 0
    filter_output: int = 0
    filter_removed_pct: float = 0.0
    opacity_removed: int = 0
    scale_removed: int = 0
    sor_removed: int = 0
    filter_grade: str = ""

    # Overall
    overall_grade: str = ""
    notes: list[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["notes"] is None:
            d["notes"] = []
        return d

    def summary_text(self) -> str:
        lines = [
            "=" * 60,
            "QUALITY REPORT",
            "=" * 60,
            "",
            f"Overall grade: {self.overall_grade}",
            "",
            "--- Data Quality ---",
            f"  Cameras:    {self.n_cameras} ({self.cameras_grade})",
            f"  Resolution: {self.avg_image_megapixels:.1f} MP avg, training at {self.training_resolution}",
            f"  Dense cloud: {self.dense_pts:,} pts ({self.pts_per_camera:.0f} pts/camera)",
            f"  Data grade: {self.data_grade}",
            "",
            "--- Camera Coverage ---",
            f"  Scene spread:    {self.camera_spread_m:.2f} m",
            f"  Mean baseline:   {self.mean_baseline_m:.3f} m",
            f"  Coverage grade:  {self.coverage_grade}",
            "",
            "--- Gaussian Splat Quality ---",
            f"  Final PSNR:  {self.final_psnr:.2f} dB ({self.psnr_grade})",
            f"  Final SSIM:  {self.final_ssim:.4f}",
            f"  Splat count: {self.final_splat_count:,} ({self.splats_per_camera:.0f} per camera)",
            "",
            "--- Filtering ---",
            f"  {self.filter_input:,} -> {self.filter_output:,} ({self.filter_removed_pct:.1f}% removed)",
            f"    Opacity: -{self.opacity_removed:,}  Scale: -{self.scale_removed:,}  SOR: -{self.sor_removed:,}",
            f"  Filter grade: {self.filter_grade}",
        ]
        if self.notes:
            lines.append("")
            lines.append("--- Notes ---")
            for n in self.notes:
                lines.append(f"  - {n}")
        lines.append("=" * 60)
        return "\n".join(lines)


def generate_quality_report(
    *,
    scene,
    init_cloud,
    budget,
    psnr_history: list,
    ssim_history: list,
    final_splat_count: int,
    filter_report: Optional[dict] = None,
    image_max_side: int = 0,
) -> QualityReport:
    notes = []

    # --- Data quality ---
    n_cam = len(scene)
    cameras_grade = _grade(n_cam, [(200, "Excellent"), (100, "Good"), (50, "Fair"), (0, "Sparse")])

    total_px = sum(w * h for w, h in scene.image_sizes)
    avg_mp = (total_px / max(n_cam, 1)) / 1e6

    dense_pts = int(init_cloud.xyz.shape[0])
    pts_per_cam = dense_pts / max(n_cam, 1)
    data_grade = _grade(pts_per_cam, [(2000, "Excellent"), (500, "Good"), (200, "Fair"), (0, "Sparse")])

    training_res = f"{image_max_side}px max side" if image_max_side else "full"

    # --- Camera coverage ---
    positions = []
    for i in range(n_cam):
        w2c = scene.w2c_per_camera[i]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        pos = -R.T @ t
        positions.append(pos)
    positions = np.array(positions)

    spread = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))

    dists = []
    for i in range(min(n_cam, 200)):
        for j in range(i + 1, min(n_cam, 200)):
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    mean_baseline = float(np.mean(dists)) if dists else 0.0

    if n_cam >= 50 and spread > 0.5:
        coverage_grade = "Good"
    elif n_cam >= 20:
        coverage_grade = "Fair"
    else:
        coverage_grade = "Sparse"
        notes.append("Few cameras — expect gaps in coverage and lower geometry quality")

    # --- GS quality ---
    final_psnr = psnr_history[-1][1] if psnr_history else 0.0
    final_ssim = ssim_history[-1][1] if ssim_history else 0.0
    psnr_grade = _grade(final_psnr, [(28, "Excellent"), (24, "Good"), (20, "Fair"), (0, "Poor")])

    splats_per_cam = final_splat_count / max(n_cam, 1)

    if final_psnr < 20:
        notes.append("Low PSNR — check input photo quality, camera alignment, or increase iterations")
    if final_psnr >= 28:
        notes.append("High PSNR — publication quality output")

    # --- Filtering ---
    f_in = filter_report.get("n_input", 0) if filter_report else 0
    f_out = filter_report.get("n_output", 0) if filter_report else 0
    f_pct = 100.0 * (1 - f_out / max(f_in, 1)) if f_in else 0.0
    opa_rm = f_in - filter_report.get("n_after_opacity", f_in) if filter_report else 0
    scale_rm = filter_report.get("n_after_opacity", 0) - filter_report.get("n_after_scale", 0) if filter_report else 0
    sor_rm = filter_report.get("n_after_scale", 0) - filter_report.get("n_after_sor", 0) if filter_report else 0

    if f_pct < 5:
        filter_grade = "Clean (minimal filtering needed)"
    elif f_pct < 15:
        filter_grade = "Normal"
    elif f_pct < 30:
        filter_grade = "Heavy (some data quality issues)"
        notes.append(f"Filtered {f_pct:.0f}% of splats — may indicate floaters from sparse regions or reflections")
    else:
        filter_grade = "Excessive (check input data)"
        notes.append(f"Filtered {f_pct:.0f}% of splats — likely poor input data or training issues")

    # --- Overall ---
    scores = {"Excellent": 4, "Good": 3, "Fair": 2, "Sparse": 1, "Poor": 0}
    avg_score = np.mean([
        scores.get(cameras_grade, 1),
        scores.get(data_grade, 1),
        scores.get(psnr_grade, 1),
    ])
    overall = _grade(avg_score, [(3.5, "Excellent"), (2.5, "Good"), (1.5, "Fair"), (0, "Poor")])

    return QualityReport(
        n_cameras=n_cam,
        cameras_grade=cameras_grade,
        avg_image_megapixels=round(avg_mp, 1),
        training_resolution=training_res,
        dense_pts=dense_pts,
        pts_per_camera=round(pts_per_cam, 0),
        data_grade=data_grade,
        camera_spread_m=round(spread, 2),
        mean_baseline_m=round(mean_baseline, 3),
        coverage_grade=coverage_grade,
        final_psnr=round(final_psnr, 2),
        final_ssim=round(final_ssim, 4),
        psnr_grade=psnr_grade,
        final_splat_count=final_splat_count,
        splats_per_camera=round(splats_per_cam, 0),
        filter_input=f_in,
        filter_output=f_out,
        filter_removed_pct=round(f_pct, 1),
        opacity_removed=opa_rm,
        scale_removed=scale_rm,
        sor_removed=sor_rm,
        filter_grade=filter_grade,
        overall_grade=overall,
        notes=notes,
    )
