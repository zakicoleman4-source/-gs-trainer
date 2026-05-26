"""Visibility-aware block partitioning for large Gaussian Splat scenes.

Implements the VastGaussian approach: divide a large drone capture into a grid
of spatial blocks, assign cameras to blocks based on visibility (projection
coverage), then let the trainer handle each block independently.

Typical usage::

    from gs_pipeline.trainer.scene_partition import partition_scene, should_partition

    if should_partition(len(scene)):
        blocks = partition_scene(scene, init_cloud)
        if blocks is not None:
            for block in blocks:
                train_block(scene, init_cloud, block)
        else:
            train_single(scene, init_cloud)   # fell back: too few cameras
    else:
        train_single(scene, init_cloud)

Only numpy, math, dataclasses, typing, and logging are used — no torch, no
gsplat. All operations are pure CPU.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SceneBlock:
    """One spatial block in a partitioned scene.

    Cameras may appear in multiple blocks (the visibility-aware reassignment
    step deliberately allows overlap so adjacent blocks share boundary cameras).
    """
    block_id: int
    row: int                    # grid row index
    col: int                    # grid col index
    camera_indices: list[int]   # indices into the full ParsedScene
    point_mask: np.ndarray      # bool shape (N_points,) — which InitCloud points belong here
    tight_min: np.ndarray       # shape (3,) world-space crop boundary (no overlap)
    tight_max: np.ndarray       # shape (3,) world-space crop boundary
    expanded_min: np.ndarray    # shape (3,) with overlap margin (for training)
    expanded_max: np.ndarray    # shape (3,) with overlap margin

    @property
    def n_cameras(self) -> int:
        return len(self.camera_indices)

    @property
    def n_points(self) -> int:
        return int(self.point_mask.sum())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_camera_positions(scene) -> np.ndarray:
    """Return (N, 3) array of camera centres in world space.

    ``w2c`` is the world-to-camera transform; its inverse is c2w, and
    ``c2w[:3, 3]`` is the camera's origin in world coordinates.
    """
    positions = []
    for w2c in scene.w2c_per_camera:   # shape (N, 4, 4)
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
    return np.stack(positions)          # (N, 3)


def _up_axis(cam_positions: np.ndarray) -> int:
    """Return the axis index (0=X, 1=Y, 2=Z) with minimum variance.

    For aerial/drone data the 'up' direction has low positional variance
    across cameras because they all fly at roughly the same altitude.
    """
    return int(np.argmin(np.var(cam_positions, axis=0)))


def _auto_grid_size(n_cameras: int, target_per_block: int = 150) -> tuple[int, int]:
    """Return (rows, cols) for a sensible partition grid.

    Rules:
    - n_blocks = max(2, round(n_cameras / target_per_block))
    - If n_blocks <= 6: single row, multiple columns (drone scenes are
      horizontally elongated).
    - Otherwise: attempt a roughly square grid with 2+ rows.
    """
    n_blocks = max(2, round(n_cameras / target_per_block))
    if n_blocks <= 6:
        return 1, n_blocks
    cols = int(math.ceil(math.sqrt(n_blocks)))
    rows = int(math.ceil(n_blocks / cols))
    return rows, cols


def _camera_block_visibility(
    K: np.ndarray,
    w2c: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> float:
    """Fraction of image area covered by the projected AABB bounding box.

    Projects the 8 corners of the axis-aligned bounding box through the camera
    model and returns ``(projected_bbox_area) / (image_area)``.

    Returns 0.0 if all corners are behind the camera (z <= 0).

    Args:
        K: (3, 3) intrinsic matrix at training resolution.
        w2c: (4, 4) world-to-camera extrinsic.
        aabb_min: (3,) world-space AABB lower bound.
        aabb_max: (3,) world-space AABB upper bound.
    """
    # Build the 8 corners: all combinations of min/max per axis.
    xs = (aabb_min[0], aabb_max[0])
    ys = (aabb_min[1], aabb_max[1])
    zs = (aabb_min[2], aabb_max[2])
    corners = np.array(
        [[x, y, z] for x in xs for y in ys for z in zs],
        dtype=np.float64,
    )  # (8, 3)

    # Transform corners into camera space.
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    cam_pts = (R @ corners.T).T + t  # (8, 3)

    # Discard corners behind the camera.
    valid = cam_pts[:, 2] > 0.0
    if not valid.any():
        return 0.0

    cam_pts = cam_pts[valid]

    # Project to image coordinates.
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    z = cam_pts[:, 2]
    u = fx * (cam_pts[:, 0] / z) + cx
    v = fy * (cam_pts[:, 1] / z) + cy

    # Image dimensions from K (principal point is at the image centre).
    W = 2.0 * cx
    H = 2.0 * cy
    if W <= 0.0 or H <= 0.0:
        return 0.0

    # Clamp to image bounds.
    u = np.clip(u, 0.0, W)
    v = np.clip(v, 0.0, H)

    bbox_w = u.max() - u.min()
    bbox_h = v.max() - v.min()
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return 0.0

    return float((bbox_w * bbox_h) / (W * H))


def _merge_adjacent_block(
    block_idx: int,
    blocks: list[SceneBlock],
    rows: int,
    cols: int,
) -> Optional[int]:
    """Return the index into ``blocks`` of the best neighbour to merge into.

    Checks the four cardinal neighbours (row±1, col±1) and returns the one
    with the most cameras, or None if no valid neighbour exists.
    """
    b = blocks[block_idx]
    id_map: dict[tuple[int, int], int] = {(bl.row, bl.col): i for i, bl in enumerate(blocks)}
    candidates = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = b.row + dr, b.col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            idx = id_map.get((nr, nc))
            if idx is not None and idx != block_idx:
                candidates.append(idx)
    if not candidates:
        return None
    return max(candidates, key=lambda i: blocks[i].n_cameras)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def partition_scene(
    scene,
    init_cloud,
    *,
    target_cameras_per_block: int = 150,
    overlap_factor: float = 0.20,
    visibility_threshold: float = 0.25,
    min_cameras_per_block: int = 20,
) -> Optional[list[SceneBlock]]:
    """Partition a large scene into spatial blocks for independent training.

    Implements the VastGaussian visibility-aware block assignment strategy:

    1. Extract camera positions from w2c matrices.
    2. Auto-compute a rows × cols grid based on camera count.
    3. Assign cameras to cells by spatial position (initial pass).
    4. Compute per-cell AABBs from camera positions + overlap margin.
    5. Reassign cameras by visibility: a camera belongs to every block whose
       AABB projects to more than ``visibility_threshold`` of its image area.
    6. Assign InitCloud points to blocks via expanded AABB membership.
    7. Merge any block that ends up below ``min_cameras_per_block`` cameras.

    Args:
        scene: ParsedScene — must have ``w2c_per_camera`` (N, 4, 4) and
            ``K_per_camera`` (N, 3, 3).
        init_cloud: InitCloud — must have ``xyz`` (M, 3).
        target_cameras_per_block: desired number of cameras per block; used
            to auto-compute the grid dimensions.
        overlap_factor: fraction of cell size added as margin to each side of
            the tight AABB when constructing ``expanded_min/max``.
        visibility_threshold: minimum projected area fraction for a camera to
            be assigned to a block (VastGaussian visibility criterion).
        min_cameras_per_block: blocks with fewer cameras after reassignment
            are merged into their largest adjacent neighbour.

    Returns:
        A list of ``SceneBlock`` objects, or ``None`` if partitioning would
        yield only one block (caller should do single-shot training instead).
    """
    n_cameras = len(scene.w2c_per_camera)
    _log.info("partition_scene: %d cameras", n_cameras)

    # ------------------------------------------------------------------
    # Step 1: Extract camera positions
    # ------------------------------------------------------------------
    cam_positions = _extract_camera_positions(scene)  # (N, 3)

    # ------------------------------------------------------------------
    # Step 2: Auto-compute grid
    # ------------------------------------------------------------------
    rows, cols = _auto_grid_size(n_cameras, target_cameras_per_block)
    _log.info("partition_scene: grid %d × %d (%d blocks)", rows, cols, rows * cols)

    # ------------------------------------------------------------------
    # Step 3: Determine horizontal axes and project camera positions
    # ------------------------------------------------------------------
    up_ax = _up_axis(cam_positions)
    horiz_axes = [ax for ax in range(3) if ax != up_ax]  # two horizontal axes

    h_pos = cam_positions[:, horiz_axes]  # (N, 2)
    h0_min, h0_max = h_pos[:, 0].min(), h_pos[:, 0].max()
    h1_min, h1_max = h_pos[:, 1].min(), h_pos[:, 1].max()

    # Guard against degenerate extents.
    h0_extent = max(h0_max - h0_min, 1e-6)
    h1_extent = max(h1_max - h1_min, 1e-6)

    cell_h0 = h0_extent / cols
    cell_h1 = h1_extent / rows

    def _cell(cam_idx: int) -> tuple[int, int]:
        """Return (row, col) for a camera index (clamped to grid bounds)."""
        c = int((h_pos[cam_idx, 0] - h0_min) / cell_h0)
        r = int((h_pos[cam_idx, 1] - h1_min) / cell_h1)
        c = min(c, cols - 1)
        r = min(r, rows - 1)
        return r, c

    # Initial assignment: group cameras by grid cell.
    cell_cameras: dict[tuple[int, int], list[int]] = {}
    for rc in ((r, c) for r in range(rows) for c in range(cols)):
        cell_cameras[rc] = []
    for cam_idx in range(n_cameras):
        cell_cameras[_cell(cam_idx)].append(cam_idx)

    # ------------------------------------------------------------------
    # Step 4: Compute tight and expanded AABB for each cell
    # ------------------------------------------------------------------
    # cell_size: the larger of the two cell dimensions (used for overlap margin)
    cell_size = max(cell_h0, cell_h1)

    cell_tight_min: dict[tuple[int, int], np.ndarray] = {}
    cell_tight_max: dict[tuple[int, int], np.ndarray] = {}
    cell_expanded_min: dict[tuple[int, int], np.ndarray] = {}
    cell_expanded_max: dict[tuple[int, int], np.ndarray] = {}

    for rc, indices in cell_cameras.items():
        r, c = rc
        # Tight bounds from the grid cell geometry (in the horizontal plane).
        t_min_3d = np.full(3, -np.inf, dtype=np.float64)
        t_max_3d = np.full(3, np.inf, dtype=np.float64)

        h0_lo = h0_min + c * cell_h0
        h0_hi = h0_lo + cell_h0
        h1_lo = h1_min + r * cell_h1
        h1_hi = h1_lo + cell_h1

        t_min_3d[horiz_axes[0]] = h0_lo
        t_max_3d[horiz_axes[0]] = h0_hi
        t_min_3d[horiz_axes[1]] = h1_lo
        t_max_3d[horiz_axes[1]] = h1_hi
        # Vertical axis spans the full scene extent.
        t_min_3d[up_ax] = cam_positions[:, up_ax].min()
        t_max_3d[up_ax] = cam_positions[:, up_ax].max()

        margin = overlap_factor * cell_size
        e_min = t_min_3d.copy()
        e_max = t_max_3d.copy()
        e_min[horiz_axes[0]] -= margin
        e_max[horiz_axes[0]] += margin
        e_min[horiz_axes[1]] -= margin
        e_max[horiz_axes[1]] += margin
        # Vertical axis: no clipping at all.
        e_min[up_ax] = -np.inf
        e_max[up_ax] = np.inf

        cell_tight_min[rc] = t_min_3d
        cell_tight_max[rc] = t_max_3d
        cell_expanded_min[rc] = e_min
        cell_expanded_max[rc] = e_max

    # ------------------------------------------------------------------
    # Step 5: Visibility-aware camera reassignment (VastGaussian)
    # ------------------------------------------------------------------
    # Determine the vertical (up-axis) extent from the point cloud, with the
    # camera altitude range as a fallback.  This ensures the AABB used for
    # visibility checks has non-zero thickness even when all cameras fly at
    # exactly the same altitude (common for drone grids).
    xyz = init_cloud.xyz  # (M, 3)
    cloud_up_min = float(xyz[:, up_ax].min()) if xyz.shape[0] > 0 else float(cam_positions[:, up_ax].min())
    cloud_up_max = float(xyz[:, up_ax].max()) if xyz.shape[0] > 0 else float(cam_positions[:, up_ax].max())
    cam_up_min = float(cam_positions[:, up_ax].min())
    cam_up_max = float(cam_positions[:, up_ax].max())
    # Vertical range: span from the lower of (cloud, cameras) to the higher.
    vis_up_min = min(cloud_up_min, cam_up_min)
    vis_up_max = max(cloud_up_max, cam_up_max)
    # Guard: if still degenerate (all co-planar), pad by cell_size.
    if vis_up_max - vis_up_min < cell_size * 1e-3:
        vis_up_min -= cell_size
        vis_up_max += cell_size

    # For each block, collect all cameras that see it sufficiently well.
    block_cameras: dict[tuple[int, int], list[int]] = {
        (r, c): [] for r in range(rows) for c in range(cols)
    }
    for cam_idx in range(n_cameras):
        K = scene.K_per_camera[cam_idx]
        w2c = scene.w2c_per_camera[cam_idx]
        for rc in ((r, c) for r in range(rows) for c in range(cols)):
            e_min = cell_expanded_min[rc]
            e_max = cell_expanded_max[rc]
            # Use finite bounds for the visibility check.
            vis_min = e_min.copy()
            vis_max = e_max.copy()
            vis_min[up_ax] = vis_up_min
            vis_max[up_ax] = vis_up_max
            vis = _camera_block_visibility(K, w2c, vis_min, vis_max)
            if vis >= visibility_threshold:
                block_cameras[rc].append(cam_idx)

    # ------------------------------------------------------------------
    # Step 6: Build initial SceneBlock list and assign points
    # ------------------------------------------------------------------
    # xyz was extracted above (init_cloud.xyz) for use in the visibility step.
    blocks: list[SceneBlock] = []
    block_id = 0
    for r in range(rows):
        for c in range(cols):
            rc = (r, c)
            cam_list = block_cameras[rc]

            e_min = cell_expanded_min[rc]
            e_max = cell_expanded_max[rc]

            # Point mask: all axes must be within expanded bounds.
            mask = np.ones(xyz.shape[0], dtype=bool)
            for ax in range(3):
                lo = e_min[ax]
                hi = e_max[ax]
                if np.isfinite(lo):
                    mask &= xyz[:, ax] >= lo
                if np.isfinite(hi):
                    mask &= xyz[:, ax] <= hi

            t_min = cell_tight_min[rc]
            t_max = cell_tight_max[rc]
            # Replace ±inf in tight bounds with actual camera-position extremes.
            t_min_out = t_min.copy()
            t_max_out = t_max.copy()
            t_min_out[up_ax] = cam_positions[:, up_ax].min()
            t_max_out[up_ax] = cam_positions[:, up_ax].max()

            e_min_out = e_min.copy()
            e_max_out = e_max.copy()
            e_min_out[up_ax] = cam_positions[:, up_ax].min() - overlap_factor * cell_size
            e_max_out[up_ax] = cam_positions[:, up_ax].max() + overlap_factor * cell_size

            blocks.append(SceneBlock(
                block_id=block_id,
                row=r,
                col=c,
                camera_indices=list(cam_list),
                point_mask=mask,
                tight_min=t_min_out.astype(np.float64),
                tight_max=t_max_out.astype(np.float64),
                expanded_min=e_min_out.astype(np.float64),
                expanded_max=e_max_out.astype(np.float64),
            ))
            block_id += 1

    # ------------------------------------------------------------------
    # Step 7: Merge sparse blocks and validate
    # ------------------------------------------------------------------
    # Iteratively find the sparsest block and merge it with its best neighbour.
    changed = True
    while changed:
        changed = False
        sparse_idx = None
        for i, b in enumerate(blocks):
            if b.n_cameras < min_cameras_per_block:
                sparse_idx = i
                break
        if sparse_idx is None:
            break

        target_idx = _merge_adjacent_block(sparse_idx, blocks, rows, cols)
        if target_idx is None:
            _log.warning(
                "Block (row=%d, col=%d) has only %d cameras but has no neighbour "
                "to merge with; keeping as-is.",
                blocks[sparse_idx].row, blocks[sparse_idx].col,
                blocks[sparse_idx].n_cameras,
            )
            break

        tb = blocks[target_idx]
        sb = blocks[sparse_idx]
        _log.warning(
            "Merging sparse block (row=%d, col=%d, %d cams) into "
            "(row=%d, col=%d, %d cams).",
            sb.row, sb.col, sb.n_cameras,
            tb.row, tb.col, tb.n_cameras,
        )

        # Merge: absorb sparse block's cameras and points into target.
        merged_cam_indices = sorted(set(tb.camera_indices) | set(sb.camera_indices))
        merged_point_mask = tb.point_mask | sb.point_mask
        merged_tight_min = np.minimum(tb.tight_min, sb.tight_min)
        merged_tight_max = np.maximum(tb.tight_max, sb.tight_max)
        merged_exp_min = np.minimum(tb.expanded_min, sb.expanded_min)
        merged_exp_max = np.maximum(tb.expanded_max, sb.expanded_max)

        blocks[target_idx] = SceneBlock(
            block_id=tb.block_id,
            row=tb.row,
            col=tb.col,
            camera_indices=merged_cam_indices,
            point_mask=merged_point_mask,
            tight_min=merged_tight_min,
            tight_max=merged_tight_max,
            expanded_min=merged_exp_min,
            expanded_max=merged_exp_max,
        )
        blocks.pop(sparse_idx)
        changed = True

    if len(blocks) <= 1:
        _log.info(
            "partition_scene: only 1 block after merging — "
            "caller should use single-shot training."
        )
        return None

    _log.info(
        "partition_scene: %d blocks; camera counts: %s",
        len(blocks),
        [b.n_cameras for b in blocks],
    )
    return blocks


def should_partition(n_cameras: int, *, threshold: int = 500) -> bool:
    """Return True if the scene is large enough to warrant block partitioning."""
    return n_cameras >= threshold
