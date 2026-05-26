"""Scaffold-GS training loop: anchor-based neural Gaussian splatting.

Scaffold-GS (CVPR 2024 Highlight) replaces flat per-Gaussian optimization with
a two-level hierarchy of anchors + neural Gaussians predicted by MLPs.  Each
anchor carries a learnable feature vector; three small MLPs decode per-view
opacity, covariance, and colour from the feature concatenated with the viewing
direction.  Densification grows/prunes *anchors* on a voxel grid rather than
individual Gaussians, producing a more structured and compact representation.

Output: standard INRIA PLY via "baking" neural Gaussians from a canonical
viewpoint.  The baked file is compatible with SuperSplat / Polycam viewers.

This module imports torch and gsplat at function-call time so the rest of the
package stays importable on CPU CI.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gs_pipeline.trainer.budget import Budget
from gs_pipeline.trainer.init_from_pcd import InitCloud
from gs_pipeline.trainer.job_state import (
    JobState,
    OutputsSnapshot,
    write_state,
)
from gs_pipeline.trainer.oom_guard import (
    ProgressWatchdog,
    clear_cuda_cache,
    set_memory_fraction,
)
from gs_pipeline.trainer.parse_metashape import ParsedScene

_log = logging.getLogger(__name__)

C0 = 0.28209479177387814  # SH band-0 constant: 1 / (2 * sqrt(pi))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScaffoldConfig:
    iterations: int = 40_000
    n_offsets: int = 10
    anchor_feat_dim: int = 32
    voxel_size_factor: float = 1.0
    mlp_hidden_dim: int = 64

    lr_anchor_pos: float = 0.0
    lr_anchor_feat: float = 0.0075
    lr_anchor_offsets: float = 0.01
    lr_anchor_scales: float = 0.007
    lr_mlp_opacity: float = 0.002
    lr_mlp_cov: float = 0.004
    lr_mlp_color: float = 0.008
    lr_appearance: float = 0.05

    lr_final_factor: float = 0.01

    grow_start_iter: int = 1500
    grow_stop_iter: int = 15_000
    grow_every: int = 100
    grow_grad_threshold: float = 0.0002
    grow_voxel_levels: int = 3
    prune_opacity_threshold: float = 0.005
    prune_min_visits: int = 80

    appearance_embed_dim: int = 32

    opacity_reg: float = 0.01
    scale_reg: float = 0.01

    ssim_lambda: float = 0.2
    near_plane_extent_ratio: float = 0.01
    far_plane_extent_ratio: float = 100.0
    random_bg_per_step: bool = True
    holdout_stride: int = 8
    eval_every: int = 1000
    preview_every: int = 250
    checkpoint_every: int = 5000
    divergence_min_psnr: float = 12.0
    divergence_check_at_step: int = 15_000
    memory_fraction: float = 0.92
    watchdog_timeout_s: float = 7200.0
    watchdog_poll_interval_s: float = 30.0
    filter_enabled: bool = True
    filter_min_opacity: float = 0.005
    filter_sor_k: int = 20
    filter_sor_std_ratio: float = 2.0
    filter_max_scale_factor: float = 10.0
    timelapse_enabled: bool = True
    timelapse_fps: int = 10
    preview_panel_height: int = 400
    antialias: bool = True


def load_scaffold_config(yaml_path: Path, *, iterations_override: Optional[int] = None) -> ScaffoldConfig:
    import yaml
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    quality_preset = raw.get("quality", {}).get("preset", "Auto")
    iters_map = raw.get("iterations", {})
    iterations = int(iters_map.get(quality_preset, 40_000))
    if iterations_override is not None:
        iterations = int(iterations_override)

    sc = raw.get("scaffold", {})
    ev = raw.get("eval", {})
    ck = raw.get("checkpoint", {})
    div = raw.get("divergence_abort", {})
    wd = raw.get("watchdog", {})
    filt = raw.get("filter", {})
    tl = raw.get("timelapse", {})
    pv = raw.get("preview", {})
    rast = raw.get("rasterizer", {})

    return ScaffoldConfig(
        iterations=iterations,
        n_offsets=int(sc.get("n_offsets", 10)),
        anchor_feat_dim=int(sc.get("anchor_feat_dim", 32)),
        voxel_size_factor=float(sc.get("voxel_size_factor", 1.0)),
        mlp_hidden_dim=int(sc.get("mlp_hidden_dim", 64)),
        lr_anchor_pos=float(sc.get("lr_anchor_pos", 0.0)),
        lr_anchor_feat=float(sc.get("lr_anchor_feat", 0.0075)),
        lr_anchor_offsets=float(sc.get("lr_anchor_offsets", 0.01)),
        lr_anchor_scales=float(sc.get("lr_anchor_scales", 0.007)),
        lr_mlp_opacity=float(sc.get("lr_mlp_opacity", 0.002)),
        lr_mlp_cov=float(sc.get("lr_mlp_cov", 0.004)),
        lr_mlp_color=float(sc.get("lr_mlp_color", 0.008)),
        lr_appearance=float(sc.get("lr_appearance", 0.05)),
        lr_final_factor=float(sc.get("lr_final_factor", 0.01)),
        grow_start_iter=int(sc.get("grow_start_iter", 1500)),
        grow_stop_iter=int(sc.get("grow_stop_iter", 15000)),
        grow_every=int(sc.get("grow_every", 100)),
        grow_grad_threshold=float(sc.get("grow_grad_threshold", 0.0002)),
        grow_voxel_levels=int(sc.get("grow_voxel_levels", 3)),
        prune_opacity_threshold=float(sc.get("prune_opacity_threshold", 0.005)),
        prune_min_visits=int(sc.get("prune_min_visits", 80)),
        appearance_embed_dim=int(sc.get("appearance_embed_dim", 32)),
        opacity_reg=float(sc.get("opacity_reg", 0.01)),
        scale_reg=float(sc.get("scale_reg", 0.01)),
        ssim_lambda=float(sc.get("ssim_lambda", 0.2)),
        holdout_stride=int(ev.get("holdout_stride", 8)),
        eval_every=int(ev.get("eval_every", 1000)),
        preview_every=int(ev.get("preview_every", 250)),
        checkpoint_every=int(ck.get("every", 5000)),
        divergence_min_psnr=float(div.get("min_psnr_at_step", 12.0)),
        divergence_check_at_step=int(div.get("check_at_step", 15000)),
        memory_fraction=float(wd.get("memory_fraction", 0.92)),
        watchdog_timeout_s=float(wd.get("per_window_seconds", 7200.0)),
        watchdog_poll_interval_s=float(wd.get("poll_interval_seconds", 30.0)),
        filter_enabled=bool(filt.get("enabled", True)),
        filter_min_opacity=float(filt.get("min_opacity", 0.005)),
        filter_sor_k=int(filt.get("sor_k", 20)),
        filter_sor_std_ratio=float(filt.get("sor_std_ratio", 2.0)),
        filter_max_scale_factor=float(filt.get("max_scale_factor", 10.0)),
        timelapse_enabled=bool(tl.get("enabled", True)),
        timelapse_fps=int(tl.get("fps", 10)),
        preview_panel_height=int(pv.get("panel_height", 400)),
        antialias=bool(rast.get("antialias", True)),
    )


def auto_adjust_scaffold_config_for_scene(config: ScaffoldConfig, n_cameras: int) -> ScaffoldConfig:
    if n_cameras < 30:
        return replace(config,
            holdout_stride=2,
            divergence_check_at_step=3_000,
            divergence_min_psnr=10.0,
        )
    elif n_cameras < 80:
        return replace(config,
            holdout_stride=3,
            divergence_check_at_step=6_000,
        )
    elif n_cameras < 150:
        return replace(config, holdout_stride=5)
    elif n_cameras > 1000:
        return replace(config, holdout_stride=16, eval_every=2000)
    return config


# ---------------------------------------------------------------------------
# Scaffold Model
# ---------------------------------------------------------------------------

def _voxel_quantize(xyz: np.ndarray, voxel_size: float):
    coords = np.floor(xyz / voxel_size).astype(np.int64)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    anchor_pos = (unique_coords.astype(np.float64) + 0.5) * voxel_size
    return anchor_pos.astype(np.float32), inverse, voxel_size


def _auto_voxel_size(xyz: np.ndarray, factor: float = 1.0) -> float:
    from scipy.spatial import KDTree
    n = min(xyz.shape[0], 50_000)
    if n < 2:
        return 0.01
    rng = np.random.default_rng(42)
    idx = rng.choice(xyz.shape[0], size=n, replace=False) if xyz.shape[0] > n else np.arange(n)
    tree = KDTree(xyz[idx])
    dists, _ = tree.query(xyz[idx], k=2)
    median_nn = float(np.median(dists[:, 1]))
    return max(median_nn * 3.0 * factor, 1e-6)


def _knn_mean_distance_np(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    from scipy.spatial import KDTree
    n = xyz.shape[0]
    if n <= 1:
        return np.full(n, 1e-3, dtype=np.float32)
    k_eff = min(k, n - 1)
    tree = KDTree(xyz)
    dists, _ = tree.query(xyz, k=k_eff + 1)
    if dists.ndim == 1:
        dists = dists[:, None]
    return np.mean(dists[:, 1:], axis=1).astype(np.float32)


class ScaffoldModel:
    """Anchor-based neural Gaussian model.

    Pure-PyTorch; only the rasterization call (outside this class) uses CUDA
    via gsplat.  This class is *not* an nn.Module because its parameter set
    changes during anchor growing/pruning — we manage optimizer param groups
    manually.
    """

    def __init__(
        self,
        anchor_pos,     # (A, 3) tensor
        anchor_scales,  # (A, 3) tensor (log-scale)
        n_offsets: int,
        feat_dim: int,
        appearance_dim: int,
        n_cameras: int,
        mlp_hidden_dim: int,
        device,
    ):
        import torch
        import torch.nn as nn

        self.n_offsets = n_offsets
        self.feat_dim = feat_dim
        self.device = device
        A = anchor_pos.shape[0]

        self.anchor_pos = anchor_pos.clone().to(device).requires_grad_(True)
        self.anchor_feat = (torch.randn(A, feat_dim, device=device) * 0.01).requires_grad_(True)
        self.anchor_offsets = (torch.zeros(A, n_offsets, 3, device=device).uniform_(-1, 1) * 0.1).requires_grad_(True)
        self.anchor_scales = anchor_scales.clone().to(device).requires_grad_(True)

        self.appearance_embeds = (torch.zeros(max(n_cameras, 1), appearance_dim, device=device) * 0.01).requires_grad_(True)

        inp_dim = feat_dim + 3  # feat + view_dir
        self.opacity_mlp = nn.Sequential(
            nn.Linear(inp_dim, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_offsets), nn.Tanh(),
        ).to(device)

        self.cov_mlp = nn.Sequential(
            nn.Linear(inp_dim, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_offsets * 7),
        ).to(device)

        color_inp = inp_dim + appearance_dim
        self.color_mlp = nn.Sequential(
            nn.Linear(color_inp, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_offsets * 3), nn.Sigmoid(),
        ).to(device)

        self.opacity_accum = torch.zeros(A, device=device)
        self.visit_count = torch.zeros(A, device=device)
        self.offset_gradient_accum = torch.zeros(A, device=device)
        self.offset_gradient_count = torch.zeros(A, device=device)

    @staticmethod
    def from_point_cloud(
        xyz: np.ndarray,
        rgb: np.ndarray,
        *,
        n_offsets: int,
        feat_dim: int,
        appearance_dim: int,
        n_cameras: int,
        mlp_hidden_dim: int,
        voxel_size_factor: float,
        device,
    ) -> "ScaffoldModel":
        import torch

        voxel_size = _auto_voxel_size(xyz, factor=voxel_size_factor)
        anchor_pos_np, _inverse, _vs = _voxel_quantize(xyz, voxel_size)

        scale_init = _knn_mean_distance_np(anchor_pos_np, k=3) * 0.5
        anchor_scales_np = np.log(np.clip(scale_init, 1e-6, None))
        anchor_scales = np.stack([anchor_scales_np] * 3, axis=-1)

        _log.info(
            "Scaffold init: %d points -> %d anchors (voxel_size=%.4f)",
            xyz.shape[0], anchor_pos_np.shape[0], voxel_size,
        )

        return ScaffoldModel(
            anchor_pos=torch.from_numpy(anchor_pos_np).float(),
            anchor_scales=torch.from_numpy(anchor_scales).float(),
            n_offsets=n_offsets,
            feat_dim=feat_dim,
            appearance_dim=appearance_dim,
            n_cameras=n_cameras,
            mlp_hidden_dim=mlp_hidden_dim,
            device=device,
        )

    @property
    def n_anchors(self) -> int:
        return self.anchor_pos.shape[0]

    def all_params(self) -> list:
        return [self.anchor_pos, self.anchor_feat, self.anchor_offsets, self.anchor_scales, self.appearance_embeds]

    def mlp_params(self) -> list:
        params = []
        for mlp in (self.opacity_mlp, self.cov_mlp, self.color_mlp):
            params.extend(mlp.parameters())
        return params

    def generate_neural_gaussians(self, camera_center, cam_idx=None):
        import torch
        import torch.nn.functional as F

        A = self.anchor_pos.shape[0]
        view_dir = F.normalize(self.anchor_pos.detach() - camera_center[None], dim=-1)
        feat_view = torch.cat([self.anchor_feat, view_dir], dim=-1)

        opacity_raw = self.opacity_mlp(feat_view)
        opacity = (opacity_raw + 1.0) * 0.5  # Tanh output [-1,1] -> [0,1]

        cov_raw = self.cov_mlp(feat_view).view(A, self.n_offsets, 7)
        pred_scales = cov_raw[..., :3]
        pred_quats = cov_raw[..., 3:]

        if cam_idx is not None and self.appearance_embeds is not None:
            appear = self.appearance_embeds[cam_idx].unsqueeze(0).expand(A, -1)
        else:
            appear = self.appearance_embeds.mean(dim=0, keepdim=True).expand(A, -1)

        feat_view_appear = torch.cat([feat_view, appear], dim=-1)
        colors = self.color_mlp(feat_view_appear).view(A, self.n_offsets, 3)

        base_scale = torch.exp(self.anchor_scales)
        offset_pos = self.anchor_offsets * base_scale.unsqueeze(1)
        means = self.anchor_pos.unsqueeze(1) + offset_pos

        scales = pred_scales + self.anchor_scales.unsqueeze(1)

        means = means.reshape(-1, 3)
        scales = scales.reshape(-1, 3)
        quats = pred_quats.reshape(-1, 4)
        opacity = opacity.reshape(-1)
        colors = colors.reshape(-1, 3)

        mask = opacity > 0.005
        return means[mask], scales[mask], quats[mask], opacity[mask], colors[mask]

    def accumulate_stats(self, visible_anchor_mask, opacity_vals, offset_grads):
        import torch
        with torch.no_grad():
            if visible_anchor_mask is not None:
                self.visit_count[visible_anchor_mask] += 1
            if opacity_vals is not None and visible_anchor_mask is not None:
                max_opa = opacity_vals.view(-1, self.n_offsets).max(dim=1).values
                self.opacity_accum[visible_anchor_mask] += max_opa[:visible_anchor_mask.sum()]
            if offset_grads is not None and visible_anchor_mask is not None:
                grad_norm = offset_grads.view(-1, self.n_offsets, 3).norm(dim=-1).mean(dim=1)
                self.offset_gradient_accum[visible_anchor_mask] += grad_norm[:visible_anchor_mask.sum()]
                self.offset_gradient_count[visible_anchor_mask] += 1

    def grow_anchors(self, threshold: float, voxel_size: float):
        import torch

        mask = self.offset_gradient_count > 0
        if mask.sum() == 0:
            return 0

        avg_grad = self.offset_gradient_accum.clone()
        avg_grad[mask] /= self.offset_gradient_count[mask]

        grow_mask = avg_grad > threshold
        if grow_mask.sum() == 0:
            return 0

        candidate_pos = self.anchor_pos[grow_mask]
        coords = torch.floor(candidate_pos / voxel_size).long()
        unique_coords = torch.unique(coords, dim=0)

        existing_coords = torch.floor(self.anchor_pos / voxel_size).long()
        existing_set = set(map(tuple, existing_coords.cpu().numpy()))
        new_mask = torch.tensor(
            [tuple(c.cpu().numpy()) not in existing_set for c in unique_coords],
            dtype=torch.bool, device=self.device,
        )
        if new_mask.sum() == 0:
            return 0

        new_coords = unique_coords[new_mask]
        new_pos = (new_coords.float() + 0.5) * voxel_size
        n_new = new_pos.shape[0]

        new_feat = torch.randn(n_new, self.feat_dim, device=self.device) * 0.01
        new_offsets = torch.zeros(n_new, self.n_offsets, 3, device=self.device).uniform_(-1, 1) * 0.1
        median_scale = self.anchor_scales.median(dim=0).values
        new_scales = median_scale.unsqueeze(0).expand(n_new, -1).clone()

        self.anchor_pos = torch.cat([self.anchor_pos, new_pos.requires_grad_(True)])
        self.anchor_feat = torch.cat([self.anchor_feat, new_feat.requires_grad_(True)])
        self.anchor_offsets = torch.cat([self.anchor_offsets, new_offsets.requires_grad_(True)])
        self.anchor_scales = torch.cat([self.anchor_scales, new_scales.requires_grad_(True)])
        self.opacity_accum = torch.cat([self.opacity_accum, torch.zeros(n_new, device=self.device)])
        self.visit_count = torch.cat([self.visit_count, torch.zeros(n_new, device=self.device)])
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, torch.zeros(n_new, device=self.device)])
        self.offset_gradient_count = torch.cat([self.offset_gradient_count, torch.zeros(n_new, device=self.device)])

        return n_new

    def prune_anchors(self, min_opacity: float, min_visits: int):
        import torch

        visited = self.visit_count > min_visits
        if visited.sum() == 0:
            return 0

        avg_opacity = self.opacity_accum.clone()
        avg_opacity[visited] /= self.visit_count[visited]

        prune = visited & (avg_opacity < min_opacity)
        keep = ~prune
        if keep.all():
            return 0

        n_before = self.n_anchors
        self.anchor_pos = self.anchor_pos[keep].detach().requires_grad_(True)
        self.anchor_feat = self.anchor_feat[keep].detach().requires_grad_(True)
        self.anchor_offsets = self.anchor_offsets[keep].detach().requires_grad_(True)
        self.anchor_scales = self.anchor_scales[keep].detach().requires_grad_(True)
        self.opacity_accum = self.opacity_accum[keep]
        self.visit_count = self.visit_count[keep]
        self.offset_gradient_accum = self.offset_gradient_accum[keep]
        self.offset_gradient_count = self.offset_gradient_count[keep]

        return n_before - self.n_anchors

    def reset_stats(self):
        import torch
        self.opacity_accum.zero_()
        self.visit_count.zero_()
        self.offset_gradient_accum.zero_()
        self.offset_gradient_count.zero_()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _camera_center_from_w2c(w2c):
    import torch
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    return -R.T @ t


def _render_scaffold_view(model, K, w2c, width, height, near, far, cam_idx, device, *, bg=None, antialias=True):
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization

    camera_center = _camera_center_from_w2c(w2c)
    means, scales, quats, opacities, colors = model.generate_neural_gaussians(camera_center, cam_idx)

    if means.shape[0] == 0:
        return torch.zeros(height, width, 3, device=device), {}

    if bg is None:
        bg = torch.zeros(3, device=device)

    rendered, alphas, info = rasterization(
        means=means,
        quats=F.normalize(quats, dim=-1),
        scales=torch.exp(scales),
        opacities=opacities,
        colors=colors.unsqueeze(-2),  # (N, 1, 3) for sh_degree=None with D=3
        viewmats=w2c[None].to(device),
        Ks=K[None].to(device),
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        backgrounds=bg[None],
        sh_degree=None,
        rasterize_mode="antialiased" if antialias else "classic",
    )

    return rendered[0], info


def _evaluate_scaffold_holdout(model, scene, holdout_idx, near, far, device, *, antialias=True):
    import torch
    from gs_pipeline.trainer.render_eval import psnr as psnr_fn, _load_image_np, _load_camera

    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    for cam_i in holdout_idx:
        K, w2c, image_path = _load_camera(scene, cam_i)
        target_np = _load_image_np(image_path, 1.0)
        target = torch.from_numpy(target_np).to(device)
        h, w_img = target.shape[:2]
        with torch.no_grad():
            pred, _ = _render_scaffold_view(model, K, w2c, w_img, h, near, far, cam_i, device, antialias=antialias)
        mse = ((pred - target) ** 2).mean()
        p = psnr_fn(mse)
        total_psnr += p
        n += 1

    return (total_psnr / max(n, 1), 0.0)  # SSIM omitted for speed during scaffold eval


def _save_scaffold_preview_strip(model, scene, holdout_idx, near, far, device, out_path, panel_height=400, *, antialias=True):
    import torch
    from PIL import Image
    from gs_pipeline.trainer.render_eval import _load_camera, _load_image_np

    indices = [holdout_idx[0], holdout_idx[len(holdout_idx) // 2], holdout_idx[-1]]
    panels = []
    for cam_i in indices:
        K, w2c, image_path = _load_camera(scene, cam_i)
        target_np = _load_image_np(image_path, 1.0)
        h, w_img = target_np.shape[:2]
        with torch.no_grad():
            pred, _ = _render_scaffold_view(model, K, w2c, w_img, h, near, far, cam_i, device, antialias=antialias)
        img_np = (pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        panel = Image.fromarray(img_np)
        ratio = panel_height / panel.height
        panel = panel.resize((int(panel.width * ratio), panel_height), Image.LANCZOS)
        panels.append(panel)

    sep = 4
    total_w = sum(p.width for p in panels) + sep * (len(panels) - 1)
    strip = Image.new("RGB", (total_w, panel_height), (255, 255, 255))
    x = 0
    for p in panels:
        strip.paste(p, (x, 0))
        x += p.width + sep
    strip.save(out_path)


# ---------------------------------------------------------------------------
# Baking (neural Gaussians → static INRIA PLY arrays)
# ---------------------------------------------------------------------------

def _bake_to_static(model, scene, device):
    import torch

    camera_centers = []
    for i in range(len(scene)):
        w2c = torch.from_numpy(scene.w2c_per_camera[i]).float().to(device)
        camera_centers.append(_camera_center_from_w2c(w2c))
    canonical_center = torch.stack(camera_centers).mean(dim=0)

    with torch.no_grad():
        means, scales, quats, opacities, colors = model.generate_neural_gaussians(
            canonical_center, cam_idx=None,
        )

    means_np = means.cpu().numpy()
    scales_np = scales.cpu().numpy()  # already log-scale
    quats_np = quats.cpu().numpy()
    opacities_np = _inverse_sigmoid(opacities).cpu().numpy()
    sh_dc_np = ((colors - 0.5) / C0).cpu().numpy()
    sh_rest_np = np.zeros((means_np.shape[0], 0, 3), dtype=np.float32)

    return means_np, scales_np, quats_np, opacities_np, sh_dc_np, sh_rest_np


def _inverse_sigmoid(x):
    import torch
    x = x.clamp(1e-6, 1 - 1e-6)
    return torch.log(x / (1.0 - x))


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------

def _build_optimizer(model, config: ScaffoldConfig):
    import torch
    return torch.optim.Adam([
        {"params": [model.anchor_pos], "lr": config.lr_anchor_pos, "name": "anchor_pos"},
        {"params": [model.anchor_feat], "lr": config.lr_anchor_feat, "name": "anchor_feat"},
        {"params": [model.anchor_offsets], "lr": config.lr_anchor_offsets, "name": "anchor_offsets"},
        {"params": [model.anchor_scales], "lr": config.lr_anchor_scales, "name": "anchor_scales"},
        {"params": [model.appearance_embeds], "lr": config.lr_appearance, "name": "appearance"},
        {"params": list(model.opacity_mlp.parameters()), "lr": config.lr_mlp_opacity, "name": "mlp_opacity"},
        {"params": list(model.cov_mlp.parameters()), "lr": config.lr_mlp_cov, "name": "mlp_cov"},
        {"params": list(model.color_mlp.parameters()), "lr": config.lr_mlp_color, "name": "mlp_color"},
    ])


def _lr_at_step(step, total_steps, lr_init, final_factor):
    if total_steps <= 1 or final_factor >= 1.0:
        return lr_init
    t = (step - 1) / (total_steps - 1)
    return lr_init * (final_factor ** t)


def _update_lr(optimizer, step, total_steps, config: ScaffoldConfig):
    decay_groups = {
        "anchor_offsets": config.lr_anchor_offsets,
        "mlp_opacity": config.lr_mlp_opacity,
        "mlp_color": config.lr_mlp_color,
        "appearance": config.lr_appearance,
    }
    for pg in optimizer.param_groups:
        if pg["name"] in decay_groups:
            pg["lr"] = _lr_at_step(step, total_steps, decay_groups[pg["name"]], config.lr_final_factor)


# ---------------------------------------------------------------------------
# SSIM loss (same as train_mcmc.py)
# ---------------------------------------------------------------------------

def _ssim_loss(pred, target, window_size: int = 11):
    import torch
    import torch.nn.functional as F

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channels = pred.shape[-1]

    pred_4d = pred.permute(2, 0, 1).unsqueeze(0)
    target_4d = target.permute(2, 0, 1).unsqueeze(0)

    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device) - window_size // 2
    gauss_1d = torch.exp(-coords.pow(2) / (2.0 * 1.5 ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1).contiguous()

    pad = window_size // 2
    mu1 = F.conv2d(pred_4d, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(target_4d, kernel, padding=pad, groups=channels)
    mu1_sq, mu2_sq, mu12 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = F.conv2d(pred_4d.pow(2), kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target_4d.pow(2), kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred_4d * target_4d, kernel, padding=pad, groups=channels) - mu12

    ssim_map = ((2.0 * mu12 + C1) * (2.0 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return (1.0 - ssim_map).mean()


# ---------------------------------------------------------------------------
# Timelapse compiler (shared logic with train_mcmc)
# ---------------------------------------------------------------------------

def _compile_timelapse(frame_paths: list, out_path: Path, fps: int = 10) -> bool:
    import subprocess
    if not frame_paths or shutil.which("ffmpeg") is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = out_path.parent / "_timelapse_list.txt"
    try:
        with list_file.open("w", encoding="utf-8") as f:
            for p in frame_paths:
                f.write(f"file '{p.absolute()}'\n")
                f.write(f"duration {1.0 / fps:.4f}\n")
            f.write(f"file '{frame_paths[-1].absolute()}'\n")
        result = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0
    except Exception as exc:
        _log.warning("timelapse compilation failed: %s", exc)
        return False
    finally:
        list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main train entry point
# ---------------------------------------------------------------------------

class _DivergenceAbort(RuntimeError):
    pass


def train(
    *,
    scene: ParsedScene,
    init_cloud: InitCloud,
    budget: Budget,
    config: ScaffoldConfig,
    job_state: JobState,
    job_state_path: Path,
    work_dir: Path,
    outbox_dir: Path,
    on_tick: Optional[Any] = None,
) -> OutputsSnapshot:
    import torch
    from gsplat import rasterization
    from gs_pipeline.trainer.render_eval import _load_camera, _load_image_np

    set_memory_fraction(fraction=config.memory_fraction, device=0)
    device = torch.device("cuda:0")

    n_cam = len(scene)
    holdout_idx = list(range(0, n_cam, max(config.holdout_stride, 1)))
    train_idx = [i for i in range(n_cam) if i not in set(holdout_idx)]
    if not train_idx:
        raise ValueError(f"holdout_stride={config.holdout_stride} leaves no training cameras")

    model = ScaffoldModel.from_point_cloud(
        xyz=init_cloud.xyz.copy(),
        rgb=init_cloud.rgb.copy(),
        n_offsets=config.n_offsets,
        feat_dim=config.anchor_feat_dim,
        appearance_dim=config.appearance_embed_dim,
        n_cameras=n_cam,
        mlp_hidden_dim=config.mlp_hidden_dim,
        voxel_size_factor=config.voxel_size_factor,
        device=device,
    )

    near = init_cloud.scene_extent * config.near_plane_extent_ratio
    far = init_cloud.scene_extent * config.far_plane_extent_ratio
    voxel_size = _auto_voxel_size(init_cloud.xyz, factor=config.voxel_size_factor)

    optimizer = _build_optimizer(model, config)
    rng = np.random.default_rng(0)

    metrics_path = work_dir / "metrics.csv"
    metrics_path.write_text("step,loss,holdout_psnr,holdout_ssim\n", encoding="utf-8")
    timelapse_frames: list[Path] = []

    watchdog = ProgressWatchdog(timeout_s=config.watchdog_timeout_s, poll_interval_s=config.watchdog_poll_interval_s)
    watchdog.start()

    try:
        for step in range(1, config.iterations + 1):
            _update_lr(optimizer, step, config.iterations, config)

            cam_i = int(train_idx[rng.integers(0, len(train_idx))])
            K, w2c, image_path = _load_camera(scene, cam_i)
            target_img = torch.from_numpy(_load_image_np(image_path, 1.0)).to(device)
            h, w_img = target_img.shape[:2]

            bg = torch.rand(3, device=device) if config.random_bg_per_step else torch.zeros(3, device=device)

            pred, info = _render_scaffold_view(
                model, K, w2c, w_img, h, near, far, cam_i, device,
                bg=bg, antialias=config.antialias,
            )

            loss = (1.0 - config.ssim_lambda) * (pred - target_img).abs().mean()
            loss = loss + config.ssim_lambda * _ssim_loss(pred, target_img)
            loss = loss + config.opacity_reg * torch.exp(model.anchor_scales).mean()
            loss = loss + config.scale_reg * torch.exp(model.anchor_scales).prod(dim=-1).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            # Anchor growing / pruning
            in_grow_window = config.grow_start_iter <= step <= config.grow_stop_iter
            if in_grow_window and step % config.grow_every == 0:
                for level in range(config.grow_voxel_levels):
                    level_vs = voxel_size / (2 ** level)
                    level_thresh = config.grow_grad_threshold * (2 ** (config.grow_voxel_levels - level - 1))
                    n_grown = model.grow_anchors(level_thresh, level_vs)
                    if n_grown > 0:
                        _log.debug("step %d level %d: grew %d anchors", step, level, n_grown)

                n_pruned = model.prune_anchors(config.prune_opacity_threshold, config.prune_min_visits)
                if n_pruned > 0:
                    _log.debug("step %d: pruned %d anchors", step, n_pruned)

                # Rebuild optimizer after grow/prune
                optimizer = _build_optimizer(model, config)
                model.reset_stats()

            watchdog.tick(step)

            cur_neural = model.n_anchors * config.n_offsets
            if step % 50 == 0 or step == 1:
                job_state.tick(current_step=step, current_splats=cur_neural, loss=float(loss.item()))
                write_state(job_state, job_state_path)

            # Eval
            if step % config.eval_every == 0:
                holdout_psnr, holdout_ssim = _evaluate_scaffold_holdout(
                    model, scene, holdout_idx, near, far, device, antialias=config.antialias,
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(f"{step},{float(loss.item()):.6f},{holdout_psnr:.4f},{holdout_ssim:.4f}\n")
                job_state.tick(current_step=step, current_splats=cur_neural,
                               psnr=holdout_psnr, ssim=holdout_ssim, loss=float(loss.item()))
                write_state(job_state, job_state_path)

                if (config.divergence_check_at_step and step >= config.divergence_check_at_step
                        and holdout_psnr < config.divergence_min_psnr):
                    raise _DivergenceAbort(
                        f"holdout PSNR {holdout_psnr:.2f} < min {config.divergence_min_psnr:.1f} at step {step}"
                    )

            # Preview strip
            if step % config.preview_every == 0:
                strip_path = work_dir / "preview_strip.png"
                _save_scaffold_preview_strip(
                    model, scene, holdout_idx, near, far, device, strip_path,
                    panel_height=config.preview_panel_height, antialias=config.antialias,
                )
                job_state.outputs.preview_strip_png = str(strip_path)
                job_state.outputs.preview_png = str(strip_path)
                write_state(job_state, job_state_path)

                if step % config.checkpoint_every == 0 and config.timelapse_enabled:
                    tl_dir = work_dir / "timelapse_frames"
                    tl_dir.mkdir(exist_ok=True)
                    frame = tl_dir / f"strip_{step:06d}.png"
                    shutil.copy2(strip_path, frame)
                    timelapse_frames.append(frame)

            # Checkpoint
            if step % config.checkpoint_every == 0:
                ckpt = work_dir / f"ckpt_{step}.pt"
                torch.save({
                    "anchor_pos": model.anchor_pos.detach().cpu(),
                    "anchor_feat": model.anchor_feat.detach().cpu(),
                    "anchor_offsets": model.anchor_offsets.detach().cpu(),
                    "anchor_scales": model.anchor_scales.detach().cpu(),
                    "appearance_embeds": model.appearance_embeds.detach().cpu(),
                    "opacity_mlp": model.opacity_mlp.state_dict(),
                    "cov_mlp": model.cov_mlp.state_dict(),
                    "color_mlp": model.color_mlp.state_dict(),
                    "step": step,
                    "trainer_backend": "scaffold",
                }, ckpt)
                if str(ckpt) not in job_state.outputs.checkpoints:
                    job_state.outputs.checkpoints.append(str(ckpt))
                write_state(job_state, job_state_path)

        # ----- Clean finish -----
        _log.info("Baking %d anchors to static Gaussians...", model.n_anchors)
        means_np, scales_np, quats_np, opa_np, sh_dc_np, sh_rest_np = _bake_to_static(model, scene, device)

        from gs_pipeline.trainer.export_ply import write_inria_ply, read_inria_ply
        final_ply = outbox_dir / "scene.ply"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        write_inria_ply(
            out_path=final_ply,
            means=means_np, scales=scales_np, quats=quats_np,
            opacities=opa_np, sh_dc=sh_dc_np, sh_rest=sh_rest_np,
        )
        _log.info("Baked %d neural Gaussians -> %s", means_np.shape[0], final_ply)

        # Post-training filter
        filter_report_dict: dict[str, Any] = {}
        if config.filter_enabled:
            from gs_pipeline.trainer.filter_splats import filter_scene
            loaded = read_inria_ply(final_ply)
            f_means, f_scales, f_quats, f_opacities, f_sh_dc, f_sh_rest, f_report = filter_scene(
                means=loaded.means, scales=loaded.scales, quats=loaded.quats,
                opacities=loaded.opacities, sh_dc=loaded.sh_dc, sh_rest=loaded.sh_rest,
                scene_extent=init_cloud.scene_extent,
                min_opacity=config.filter_min_opacity, sor_k=config.filter_sor_k,
                sor_std_ratio=config.filter_sor_std_ratio, max_scale_factor=config.filter_max_scale_factor,
            )
            _log.info("Post-training filter:\n%s", f_report.summary)
            unfiltered_ply = outbox_dir / "scene_unfiltered.ply"
            shutil.copy2(str(final_ply), str(unfiltered_ply))
            write_inria_ply(
                out_path=final_ply,
                means=f_means, scales=f_scales, quats=f_quats,
                opacities=f_opacities, sh_dc=f_sh_dc, sh_rest=f_sh_rest,
            )
            filter_report_dict = {
                "n_input": f_report.n_input, "n_output": f_report.n_output,
                "summary": f_report.summary,
            }

        report = {
            "job_id": job_state.job_id,
            "trainer_backend": "scaffold",
            "final_step": config.iterations,
            "n_anchors": model.n_anchors,
            "n_neural_gaussians_baked": int(means_np.shape[0]),
            "filter": filter_report_dict or None,
        }
        (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        timelapse_path: Optional[str] = None
        if config.timelapse_enabled and timelapse_frames:
            tl_out = outbox_dir / "training_timelapse.mp4"
            if _compile_timelapse(timelapse_frames, tl_out, fps=config.timelapse_fps):
                timelapse_path = str(tl_out)

        strip_str = str(work_dir / "preview_strip.png")
        return OutputsSnapshot(
            checkpoints=job_state.outputs.checkpoints,
            preview_png=strip_str,
            preview_strip_png=strip_str,
            final_ply=str(final_ply),
            metrics_csv=str(metrics_path),
            report_json=str(work_dir / "report.json"),
            timelapse_mp4=timelapse_path,
        )
    finally:
        watchdog.stop()
        clear_cuda_cache()
