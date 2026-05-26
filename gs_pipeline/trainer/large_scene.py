"""Block-based training orchestrator for large-scale scenes (drone 500+ cameras).

Strategy (VastGaussian-derived):
1. Partition scene into spatial blocks via scene_partition.partition_scene()
2. For each block: create a sub-scene (filtered cameras + points), compute a
   per-block budget, run the standard MCMC training loop
3. Merge: crop each block's Gaussians to its tight bounds, concatenate

The merged PLY is a valid INRIA-format scene.ply ready for SuperSplat/Polycam.
"""
from __future__ import annotations

import json
import logging
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gs_pipeline.trainer.parse_metashape import ParsedScene
    from gs_pipeline.trainer.init_from_pcd import InitCloud
    from gs_pipeline.trainer.budget import Budget
    from gs_pipeline.trainer.job_state import JobState, OutputsSnapshot
    from gs_pipeline.trainer.train_mcmc import TrainerConfig

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-scene / sub-cloud construction
# ---------------------------------------------------------------------------

def subset_scene(scene: "ParsedScene", camera_indices: list[int]) -> "ParsedScene":
    """Return a new ParsedScene restricted to the given camera indices.

    The returned ParsedScene has:
    - ``image_labels``, ``image_sizes``, ``K_per_camera``, ``w2c_per_camera``,
      ``image_paths`` all sliced to ``camera_indices``
    - ``sensors`` and ``chunk_transform`` copied from the original
    - ``warnings = []``
    """
    from gs_pipeline.trainer.parse_metashape import ParsedScene

    idx = list(camera_indices)
    return ParsedScene(
        sensors=scene.sensors,
        image_labels=[scene.image_labels[i] for i in idx],
        image_sizes=[scene.image_sizes[i] for i in idx],
        K_per_camera=scene.K_per_camera[idx],
        w2c_per_camera=scene.w2c_per_camera[idx],
        chunk_transform=scene.chunk_transform,
        image_paths=[scene.image_paths[i] for i in idx] if scene.image_paths else [],
        warnings=[],
    )


def subset_cloud(init_cloud: "InitCloud", point_mask: np.ndarray) -> "InitCloud":
    """Return a new InitCloud for points selected by a boolean mask.

    ``scene_extent`` is preserved from the original so near/far plane scale
    remains consistent across blocks.
    """
    from gs_pipeline.trainer.init_from_pcd import InitCloud

    mask = np.asarray(point_mask, dtype=bool)
    sub_xyz = init_cloud.xyz[mask]
    sub_rgb = init_cloud.rgb[mask]

    return InitCloud(
        xyz=sub_xyz,
        rgb=sub_rgb,
        scene_extent=init_cloud.scene_extent,   # preserve global scale
        n_loaded=init_cloud.n_loaded,
        voxel_size=init_cloud.voxel_size,
        aabb_min=init_cloud.aabb_min,
        aabb_max=init_cloud.aabb_max,
    )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_block_plys(
    block_plys: list[Path],
    block_tight_bounds: list[tuple[np.ndarray, np.ndarray]],  # list of (min3, max3)
) -> "LoadedSplat":
    """Load each block PLY, crop Gaussians to tight block bounds, concatenate."""
    from gs_pipeline.trainer.export_ply import read_inria_ply, LoadedSplat

    filtered: list[LoadedSplat] = []
    for ply_path, (tight_min, tight_max) in zip(block_plys, block_tight_bounds):
        loaded = read_inria_ply(ply_path)
        tight_min = np.asarray(tight_min, dtype=np.float32)
        tight_max = np.asarray(tight_max, dtype=np.float32)
        mask = np.all(
            (loaded.means >= tight_min) & (loaded.means <= tight_max),
            axis=1,
        )
        # Apply mask to all array fields; pass scalar fields through unchanged.
        field_values: dict = {}
        for f in dc_fields(loaded):
            val = getattr(loaded, f.name)
            if isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == loaded.means.shape[0]:
                field_values[f.name] = val[mask]
            else:
                field_values[f.name] = val
        filtered.append(LoadedSplat(**field_values))
        _log.debug(
            "Block %s: %d/%d Gaussians within tight bounds",
            ply_path.name, int(mask.sum()), loaded.means.shape[0],
        )

    if not filtered:
        raise RuntimeError("merge_block_plys: no blocks to merge")

    # Concatenate all filtered LoadedSplats.
    def _cat(field_name: str) -> np.ndarray:
        return np.concatenate([getattr(b, field_name) for b in filtered], axis=0)

    # sh_degree: take the maximum across blocks (pads are harmless; all blocks
    # should have the same degree in practice).
    sh_degree = max(b.sh_degree for b in filtered)

    # If blocks have different sh_degree, zero-pad the lower-degree blocks.
    merged_parts: list[LoadedSplat] = []
    for b in filtered:
        if b.sh_degree == sh_degree:
            merged_parts.append(b)
        else:
            # Pad sh_rest with zeros along the coefficient dimension.
            target_k = (sh_degree + 1) ** 2 - 1
            n = b.sh_rest.shape[0]
            padded_rest = np.zeros((n, target_k, 3), dtype=np.float32)
            padded_rest[:, : b.sh_rest.shape[1], :] = b.sh_rest
            from dataclasses import replace
            merged_parts.append(replace(b, sh_rest=padded_rest, sh_degree=sh_degree))

    means = np.concatenate([b.means for b in merged_parts], axis=0)
    scales = np.concatenate([b.scales for b in merged_parts], axis=0)
    quats = np.concatenate([b.quats for b in merged_parts], axis=0)
    opacities = np.concatenate([b.opacities for b in merged_parts], axis=0)
    sh_dc = np.concatenate([b.sh_dc for b in merged_parts], axis=0)
    sh_rest = np.concatenate([b.sh_rest for b in merged_parts], axis=0)

    return LoadedSplat(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        sh_degree=sh_degree,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_large_scene(
    *,
    scene,               # ParsedScene (K matrices already pre-scaled by pipeline.py)
    init_cloud,          # InitCloud
    budget,              # Budget (original full-scene budget, for GPU info)
    config,              # TrainerConfig
    job_state: "JobState",
    job_state_path: Path,
    work_dir: Path,
    outbox_dir: Path,
    target_cameras_per_block: int = 150,
    overlap_factor: float = 0.20,
    visibility_threshold: float = 0.25,
) -> "OutputsSnapshot":
    """Train a large scene by partitioning into spatial blocks and merging.

    Args:
        scene: ParsedScene with pre-scaled K matrices.
        init_cloud: downsampled InitCloud from the full dense cloud.
        budget: full-scene Budget (used for GPU info and quality_preset).
        config: TrainerConfig (will be adapted per block).
        job_state: mutable JobState for progress reporting.
        job_state_path: path where job_state is persisted.
        work_dir: root work directory for this job.
        outbox_dir: final output directory (scene.ply written here).
        target_cameras_per_block: target cameras per spatial block.
        overlap_factor: fractional overlap between adjacent blocks.
        visibility_threshold: minimum projected coverage for camera assignment.

    Returns:
        OutputsSnapshot with ``final_ply`` pointing at the merged scene.ply.
    """
    from gs_pipeline.trainer.job_state import OutputsSnapshot, write_state
    from gs_pipeline.trainer.scene_partition import partition_scene

    # 1. Partition the scene.
    blocks = partition_scene(
        scene, init_cloud,
        target_cameras_per_block=target_cameras_per_block,
        overlap_factor=overlap_factor,
        visibility_threshold=visibility_threshold,
    )

    if blocks is None:
        # partition_scene returns None when only one block would result — the
        # caller should have used standard single-scene training.  We handle
        # it gracefully by raising so the pipeline can fall through.
        raise RuntimeError(
            "run_large_scene: partition_scene returned None (single block); "
            "use standard training instead."
        )

    _log.info(
        "Large scene: %d blocks, cameras per block: %s",
        len(blocks), [b.n_cameras for b in blocks],
    )

    # 2. Update job_state to reflect block-based progress.
    job_state.tick(current_step=0, current_splats=0, loss=0.0)
    write_state(job_state, job_state_path)

    # 3. Train each block sequentially.
    block_plys: list[Path] = []
    block_bounds: list[tuple[np.ndarray, np.ndarray]] = []

    for block_idx, block in enumerate(blocks):
        _log.info(
            "Training block %d/%d (%d cameras, %d points)",
            block_idx + 1, len(blocks), block.n_cameras, block.n_points,
        )

        block_work_dir = work_dir / f"block_{block.block_id:02d}"
        block_work_dir.mkdir(parents=True, exist_ok=True)
        block_outbox = block_work_dir / "out"
        block_outbox.mkdir(exist_ok=True)

        sub_scene = subset_scene(scene, block.camera_indices)
        sub_cloud = subset_cloud(init_cloud, block.point_mask)

        # Compute a block-specific budget with the sub-scene.
        from gs_pipeline.trainer.budget import compute_budget
        sub_budget = compute_budget(
            gpu=budget.gpu,
            image_sizes=sub_scene.image_sizes,
            dense_pts=max(1, sub_cloud.xyz.shape[0]),
            quality_preset=budget.quality_preset,
        )

        # Use a lighter iteration count per block:
        # full global iterations ÷ (n_blocks / 2), minimum 20k.
        from dataclasses import replace
        block_config = replace(
            config,
            iterations=max(20_000, config.iterations // max(1, len(blocks) // 2)),
            checkpoint_every=config.iterations,   # checkpoint only at end for blocks
            timelapse_enabled=False,              # skip per-block timelapse
        )

        try:
            from gs_pipeline.trainer.train_mcmc import train
            block_outputs = train(
                scene=sub_scene,
                init_cloud=sub_cloud,
                budget=sub_budget,
                config=block_config,
                job_state=job_state,
                job_state_path=job_state_path,
                work_dir=block_work_dir,
                outbox_dir=block_outbox,
            )
            block_ply = Path(block_outputs.final_ply)
            if block_ply.is_file():
                block_plys.append(block_ply)
                block_bounds.append((block.tight_min, block.tight_max))
            else:
                _log.warning("Block %d produced no PLY; skipping", block_idx)
        except Exception as exc:
            _log.error("Block %d training failed: %s", block_idx, exc, exc_info=True)
            # Continue with other blocks even if one fails.

    # 4. Merge blocks.
    if not block_plys:
        raise RuntimeError("All blocks failed — no Gaussians to merge")

    _log.info("Merging %d block PLYs...", len(block_plys))
    merged = merge_block_plys(block_plys, block_bounds)

    final_ply = outbox_dir / "scene.ply"
    outbox_dir.mkdir(parents=True, exist_ok=True)
    from gs_pipeline.trainer.export_ply import write_inria_ply
    write_inria_ply(
        out_path=final_ply,
        means=merged.means,
        scales=merged.scales,
        quats=merged.quats,
        opacities=merged.opacities,
        sh_dc=merged.sh_dc,
        sh_rest=merged.sh_rest,
    )
    _log.info(
        "Merged %d Gaussians from %d blocks → %s",
        merged.means.shape[0], len(block_plys), final_ply,
    )

    # 5. Write report and return OutputsSnapshot.
    report = {
        "job_id": job_state.job_id,
        "mode": "large_scene_blocks",
        "n_blocks": len(blocks),
        "n_blocks_succeeded": len(block_plys),
        "final_splat_count": int(merged.means.shape[0]),
    }
    report_path = work_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return OutputsSnapshot(
        checkpoints=[],
        preview_png=job_state.outputs.preview_strip_png or "",
        preview_strip_png=job_state.outputs.preview_strip_png or "",
        final_ply=str(final_ply),
        metrics_csv="",
        report_json=str(report_path),
    )
