"""gsplat MCMC training loop for one Metashape-bundle job.

This is the heart of the trainer. It runs **inside** the worker subprocess
launched by ``watcher.py`` so a CUDA crash never kills the daemon or the UI.
The loop:

1. Loads ``config.yaml`` defaults (+ optional per-job override).
2. Initializes splats from the voxel-downsampled dense cloud
   (``init_from_pcd.load_and_downsample``).
3. Constructs gsplat's MCMCStrategy with ``cap_max=budget.target_splats``,
   ``noise_lr``, ``refine_*``, and the anti-Milky-Way regularization knobs.
4. Iterates: per-step, pick a training camera, rasterize, photometric loss
   (L1 + SSIM), backprop, optimizer step, MCMC refine, SH warmup.
5. Every ``eval.eval_every`` steps: PSNR/SSIM on the held-out cameras, write
   a preview PNG of one holdout view, update JobState progress.
6. Every ``checkpoint.every`` steps: save ``ckpt_<step>.pt`` and an
   intermediate ``.ply`` so the UI can offer mid-training downloads.
7. Divergence abort: if PSNR < ``min_psnr_at_step`` by ``check_at_step``,
   ``mark_failed``.
8. On clean finish: final eval, write ``scene.ply``, ``metrics.csv``,
   ``report.json``, ``finish(outputs=...)``.

This module imports torch and gsplat at function-call time so the rest of
the package (parser, budget, oom_guard, job_state) stays importable on CPU
CI. The unit tests in tests/test_train_mcmc_smoke.py are GPU-gated.

Axis-flip auto-detect lives here (after init, before training): render one
training view at step 0, correlate against the loaded source image; if the
correlation is significantly higher with ``diag(1,-1,-1)`` applied on the
camera-side rotation block, bake that flip into all w2c matrices.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gs_pipeline.trainer.budget import Budget
from gs_pipeline.trainer.init_from_pcd import InitCloud
from gs_pipeline.trainer.job_state import (
    JobState,
    OutputsSnapshot,
    State,
    write_state,
)
from gs_pipeline.trainer.oom_guard import (
    ProgressWatchdog,
    clear_cuda_cache,
    is_cuda_oom,
    set_memory_fraction,
)
from gs_pipeline.trainer.parse_metashape import ParsedScene
from gs_pipeline.trainer.render_eval import (
    RenderInputs,
    _gather_sh,
    _load_camera,
    _load_image_np,
    evaluate_holdout,
    psnr,
    render_view,
    save_preview_png,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain dataclass for trainer config (typed view of config.yaml)
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    iterations: int = 40_000
    noise_lr: float = 5.0e5
    refine_start_iter: int = 100
    refine_stop_iter_ratio: float = 0.90
    refine_every: int = 100
    prune_opa: float = 0.003
    sh_degree: int = 3
    sh_warmup_interval: int = 1000
    opacity_reg: float = 0.005
    scale_reg: float = 0.005
    init_opacity: float = 0.1
    init_scale_knn_k: int = 3
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
    ssim_lambda: float = 0.2
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
    appearance_enabled: bool = False     # enable per-camera exposure compensation
    appearance_lr: float = 0.01          # Adam learning rate for exposure scalars
    antialias: bool = True               # rasterize_mode="antialiased" (Mip-Splatting)
    means_lr_final_factor: float = 0.01  # exponential decay: ends at init_lr * this
    depth_reg_weight: float = 0.001      # edge-aware depth smoothness; 0 to disable
    depth_lap_weight: float = 0.0002     # depth Laplacian (planar surface regulariser); 0 = off
    multiscale_loss_weight: float = 0.5  # 0.5× downsampled L1 alongside full-res; 0 = off
    prog_res_enabled: bool = True        # coarse-to-fine resolution schedule
    prog_res_warmup_fraction: float = 0.15   # first N% of iters at 0.25× res
    prog_res_mid_fraction: float = 0.40      # then N% of iters at 0.5× res; rest = full
    export_splat_binary: bool = True     # write scene.splat alongside scene.ply
    sh_freq_reg_weight: float = 0.01    # L2 on SH rest; linearly decays to 0 at 50% of iters
    taming_opacity_enabled: bool = True   # "Taming 3DGS": abs().clamp() opacity after midpoint
    taming_start_frac: float = 0.5        # training fraction when taming activates
    fisher_prune_enabled: bool = False    # Fisher info pruning post-training (off by default)
    fisher_prune_ratio: float = 0.5       # keep this fraction of splats by gradient importance
    fisher_prune_n_views: int = 20        # training views sampled for gradient accumulation


def load_trainer_config(yaml_path: Path, *, iterations_override: Optional[int] = None) -> TrainerConfig:
    """Load and flatten ``config.yaml`` into a TrainerConfig."""
    import yaml  # cpu dep
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    quality_preset = raw.get("quality", {}).get("preset", "Auto")
    iters_map = raw.get("iterations", {})
    iterations = int(iters_map.get(quality_preset, 30_000))
    if iterations_override is not None:
        iterations = int(iterations_override)

    mcmc = raw.get("mcmc", {})
    sh = raw.get("sh", {})
    reg = raw.get("regularization", {})
    init_cfg = raw.get("init", {})
    planes = raw.get("planes", {})
    bg = raw.get("background", {})
    ev = raw.get("eval", {})
    ck = raw.get("checkpoint", {})
    div = raw.get("divergence_abort", {})
    wd = raw.get("watchdog", {})
    filt = raw.get("filter", {})
    tl = raw.get("timelapse", {})
    pv = raw.get("preview", {})
    app_cfg = raw.get("appearance", {})
    rasterizer_cfg = raw.get("rasterizer", {})
    means_lr_cfg = raw.get("means_lr", {})
    export_cfg = raw.get("export", {})

    return TrainerConfig(
        iterations=iterations,
        noise_lr=float(mcmc.get("noise_lr", 5.0e5)),
        refine_start_iter=int(mcmc.get("refine_start_iter", 500)),
        refine_stop_iter_ratio=float(mcmc.get("refine_stop_iter_ratio", 0.85)),
        refine_every=int(mcmc.get("refine_every", 100)),
        prune_opa=float(mcmc.get("prune_opa", 0.005)),
        sh_degree=int(sh.get("degree", 3)),
        sh_warmup_interval=int(sh.get("warmup_interval", 1500)),
        opacity_reg=float(reg.get("opacity_reg", 0.01)),
        scale_reg=float(reg.get("scale_reg", 0.01)),
        init_opacity=float(init_cfg.get("init_opacity", 0.1)),
        init_scale_knn_k=int(init_cfg.get("init_scale_knn_k", 3)),
        near_plane_extent_ratio=float(planes.get("near_plane_extent_ratio", 0.01)),
        far_plane_extent_ratio=float(planes.get("far_plane_extent_ratio", 100.0)),
        random_bg_per_step=bool(bg.get("random_per_step", True)),
        holdout_stride=int(ev.get("holdout_stride", 8)),
        eval_every=int(ev.get("eval_every", 1000)),
        preview_every=int(ev.get("preview_every", 250)),
        checkpoint_every=int(ck.get("every", 5000)),
        divergence_min_psnr=float(div.get("min_psnr_at_step", 12.0)),
        divergence_check_at_step=int(div.get("check_at_step", 15_000)),
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
        appearance_enabled=bool(app_cfg.get("enabled", False)),
        appearance_lr=float(app_cfg.get("lr", 0.01)),
        antialias=bool(rasterizer_cfg.get("antialias", True)),
        means_lr_final_factor=float(means_lr_cfg.get("final_factor", 0.01)),
        depth_reg_weight=float(reg.get("depth_reg_weight", 0.001)),
        depth_lap_weight=float(reg.get("depth_lap_weight", 0.0002)),
        multiscale_loss_weight=float(reg.get("multiscale_loss_weight", 0.5)),
        prog_res_enabled=bool(rasterizer_cfg.get("prog_res_enabled", True)),
        prog_res_warmup_fraction=float(rasterizer_cfg.get("prog_res_warmup_fraction", 0.15)),
        prog_res_mid_fraction=float(rasterizer_cfg.get("prog_res_mid_fraction", 0.40)),
        export_splat_binary=bool(export_cfg.get("splat_binary", True)),
        sh_freq_reg_weight=float(reg.get("sh_freq_reg_weight", 0.01)),
        taming_opacity_enabled=bool(raw.get("taming", {}).get("enabled", True)),
        taming_start_frac=float(raw.get("taming", {}).get("start_frac", 0.5)),
        fisher_prune_enabled=bool(raw.get("fisher_prune", {}).get("enabled", False)),
        fisher_prune_ratio=float(raw.get("fisher_prune", {}).get("keep_ratio", 0.5)),
        fisher_prune_n_views=int(raw.get("fisher_prune", {}).get("n_views", 20)),
    )


def auto_adjust_config_for_scene(config: TrainerConfig, n_cameras: int) -> TrainerConfig:
    """Auto-tune config settings based on camera count.

    Small scenes need smaller holdout_stride (don't hold out too much data),
    faster divergence detection, and lower splat cap (underconstrained geometry).
    Large scenes need larger holdout_stride (evaluation is expensive).
    """
    from dataclasses import replace
    if n_cameras < 30:
        return replace(config,
            holdout_stride=2,
            divergence_check_at_step=3_000,
            divergence_min_psnr=10.0,   # lower bar for tiny scenes
        )
    elif n_cameras < 80:
        return replace(config,
            holdout_stride=3,
            divergence_check_at_step=6_000,
        )
    elif n_cameras < 150:
        return replace(config, holdout_stride=5)
    elif n_cameras > 1000:
        # Large scenes: eval is expensive; do it less often
        return replace(config,
            holdout_stride=16,
            eval_every=2000,
        )
    return config


# ---------------------------------------------------------------------------
# k-NN init scale (CPU; one-shot at init)
# ---------------------------------------------------------------------------

def knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Per-point mean distance to its k nearest neighbors (excluding itself).

    Used to seed Gaussian scales: a splat in a dense region should be small,
    a splat in a sparse region large.

    Uses scipy's KDTree when available (O(N log N), memory-efficient) and
    falls back to a blocked brute-force approach for small clouds or when
    scipy is missing. The brute-force path builds (block, N, 3) intermediates
    so it is only safe for N <= ~50k; for larger clouds without scipy, a
    random subsample is used to estimate distances.
    """
    n = xyz.shape[0]
    if n <= 1:
        return np.full(n, 1e-3, dtype=np.float32)
    k_eff = min(k, n - 1)

    # Fast path: scipy KDTree — O(N log N) and memory-friendly.
    try:
        from scipy.spatial import KDTree
        tree = KDTree(xyz)
        # query k+1 because the closest neighbor is the point itself (distance 0).
        dists, _ = tree.query(xyz, k=k_eff + 1)
        # dists shape is (N, k_eff+1); drop the self-distance (column 0).
        if dists.ndim == 1:
            dists = dists[:, None]
        return np.mean(dists[:, 1:], axis=1).astype(np.float32)
    except ImportError:
        pass

    # Fallback: blocked brute-force. Safe for moderate N; for very large
    # clouds, subsample to keep memory bounded (block * N * 3 * 4 bytes).
    _MAX_BRUTEFORCE_N = 50_000
    if n > _MAX_BRUTEFORCE_N:
        # Subsample: build a small reference set for distance estimation,
        # then use blocked queries against the subsample.
        rng = np.random.default_rng(42)
        ref_idx = rng.choice(n, size=_MAX_BRUTEFORCE_N, replace=False)
        ref = xyz[ref_idx]
        return _knn_mean_distance_bruteforce(xyz, ref, k_eff)
    else:
        return _knn_mean_distance_bruteforce(xyz, xyz, k_eff)


def _knn_mean_distance_bruteforce(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> np.ndarray:
    """Blocked brute-force k-NN. ``query`` points are looked up against ``reference``."""
    n_q = query.shape[0]
    n_r = reference.shape[0]
    same_set = query is reference
    block = min(4096, n_q)
    out = np.empty(n_q, dtype=np.float32)
    for i in range(0, n_q, block):
        chunk = query[i:i + block]                               # (B, 3)
        diff = chunk[:, None, :] - reference[None, :, :]         # (B, N_r, 3)
        d2 = np.einsum("ijk,ijk->ij", diff, diff)                # (B, N_r)
        if same_set:
            rows = np.arange(chunk.shape[0])
            d2[rows, np.arange(i, i + chunk.shape[0])] = np.inf
        k_safe = min(k, n_r - (1 if same_set else 0))
        if k_safe <= 0:
            out[i:i + block] = 1e-3
            continue
        nearest = np.partition(d2, k_safe - 1, axis=1)[:, :k_safe]
        out[i:i + block] = np.sqrt(np.mean(nearest, axis=1)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Axis-flip auto-detect (CPU-friendly diagnostic)
# ---------------------------------------------------------------------------

def detect_camera_axis_flip(
    *,
    rendered_no_flip: np.ndarray,
    rendered_flipped: np.ndarray,
    target: np.ndarray,
) -> bool:
    """Return True if applying the camera-side axis flip improves alignment.

    Comparison: mean negative L1 against target. The variant with the
    *smaller* L1 wins; ties go to no-flip. All inputs are HxWx3 in [0,1].
    """
    err_no = float(np.mean(np.abs(rendered_no_flip - target)))
    err_yes = float(np.mean(np.abs(rendered_flipped - target)))
    # 5% margin: only flip if it's noticeably better.
    return err_yes < err_no * 0.95


# ---------------------------------------------------------------------------
# Main entry point (GPU-only; gated by torch import)
# ---------------------------------------------------------------------------

def train(
    *,
    scene: ParsedScene,
    init_cloud: InitCloud,
    budget: Budget,
    config: TrainerConfig,
    job_state: JobState,
    job_state_path: Path,
    work_dir: Path,
    outbox_dir: Path,
    on_tick: Optional[Any] = None,
) -> OutputsSnapshot:
    """Run the gsplat MCMC training loop. **Requires CUDA + gsplat.**

    Side-effects (per-job paths under ``work_dir`` and ``outbox_dir``):
      - work_dir/ckpt_<step>.pt every ``config.checkpoint_every`` steps
      - work_dir/preview.png updated every ``config.preview_every`` steps
      - work_dir/metrics.csv (step, loss, holdout_psnr, holdout_ssim)
      - outbox_dir/scene.ply written at the end
      - work_dir/state.json updated via ``write_state`` on each progress tick
      - work_dir/report.json final summary

    Returns the OutputsSnapshot the caller will pass to ``job_state.finish``.
    """
    # Heavy imports deferred so CPU CI never has to install them.
    import torch
    import torch.nn.functional as F
    from gsplat import rasterization
    from gsplat.strategy import MCMCStrategy

    set_memory_fraction(fraction=config.memory_fraction, device=0)
    device = torch.device("cuda:0")

    # Hold-out split.
    n_cam = len(scene)
    holdout_idx = list(range(0, n_cam, max(config.holdout_stride, 1)))
    train_idx = [i for i in range(n_cam) if i not in set(holdout_idx)]
    if not train_idx:
        raise ValueError(f"holdout_stride={config.holdout_stride} leaves no training cameras")

    # Build Gaussian state from init cloud.
    means = torch.from_numpy(init_cloud.xyz.copy()).float().to(device)
    rgb = torch.from_numpy(init_cloud.rgb.copy()).float().to(device)
    n = means.shape[0]

    # Scales from k-NN distance on CPU (one-shot).
    scale_init_np = knn_mean_distance(init_cloud.xyz, k=config.init_scale_knn_k) * 0.5
    scales = torch.from_numpy(np.log(np.clip(scale_init_np, 1e-6, None)))[:, None].repeat(1, 3).float().to(device)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1.0
    opacities = torch.full((n,), float(_inverse_sigmoid(config.init_opacity)), device=device)

    # SH coefficients (DC + rest). Initialize SH-DC from RGB via inverse-SH transfer.
    sh_dc = _rgb_to_sh_dc(rgb)
    sh_rest = torch.zeros(n, ((config.sh_degree + 1) ** 2 - 1), 3, device=device)
    active_sh_degree = 0  # warmup from 0

    means.requires_grad_(True); scales.requires_grad_(True)
    quats.requires_grad_(True); opacities.requires_grad_(True)
    sh_dc.requires_grad_(True); sh_rest.requires_grad_(True)

    # gsplat MCMCStrategy expects a dict of per-parameter optimizers.
    optimizers = {
        "means": torch.optim.Adam([means], lr=0.00016 * init_cloud.scene_extent),
        "scales": torch.optim.Adam([scales], lr=0.005),
        "quats": torch.optim.Adam([quats], lr=0.001),
        "opacities": torch.optim.Adam([opacities], lr=0.05),
        "sh_dc": torch.optim.Adam([sh_dc], lr=0.0025),
        "sh_rest": torch.optim.Adam([sh_rest], lr=0.0025 / 20.0),
    }

    # Per-camera appearance: learnable log-exposure (R, G, B) per image.
    # Applied only to loss computation; discarded after training.
    if config.appearance_enabled:
        log_exposure = torch.zeros(len(scene), 3, device=device, requires_grad=True)
        app_optimizer = torch.optim.Adam([log_exposure], lr=config.appearance_lr)
    else:
        log_exposure = None
        app_optimizer = None

    strategy = MCMCStrategy(
        cap_max=int(budget.target_splats),
        noise_lr=config.noise_lr,
        refine_start_iter=config.refine_start_iter,
        refine_stop_iter=int(config.refine_stop_iter_ratio * config.iterations),
        refine_every=config.refine_every,
        min_opacity=config.prune_opa,
    )
    strategy_state = strategy.initialize_state()

    near = init_cloud.scene_extent * config.near_plane_extent_ratio
    far = init_cloud.scene_extent * config.far_plane_extent_ratio

    # --- Axis-flip auto-detect -----------------------------------------------
    # Metashape chunks sometimes need a diag(1,-1,-1) flip on camera rotations.
    # Render one training view both ways and pick whichever matches the GT.
    _apply_axis_flip_if_needed(
        means=means, scales=scales, quats=quats, opacities=opacities,
        sh_dc=sh_dc, sh_rest=sh_rest, active_sh_degree=0,
        full_sh_degree=config.sh_degree, scene=scene, train_idx=train_idx,
        near=near, far=far, downscale=1.0, device=device,
    )

    rng = np.random.default_rng(0)
    means_lr_init = 0.00016 * init_cloud.scene_extent
    metrics_path = work_dir / "metrics.csv"
    metrics_path.write_text("step,loss,holdout_psnr,holdout_ssim\n", encoding="utf-8")

    # Timelapse: collect one preview strip frame per checkpoint for final MP4.
    _timelapse_frames: list[Path] = []

    watchdog = ProgressWatchdog(timeout_s=config.watchdog_timeout_s, poll_interval_s=config.watchdog_poll_interval_s)
    watchdog.start()

    try:
        for step in range(1, config.iterations + 1):
            # SH degree warmup.
            target_deg = min(config.sh_degree, step // max(config.sh_warmup_interval, 1))
            active_sh_degree = max(active_sh_degree, target_deg)

            # Pick a training camera.
            # K is pre-scaled by pipeline.py; images are pre-cached at training resolution.
            cam_i = int(train_idx[rng.integers(0, len(train_idx))])
            K, w2c, image_path = _load_camera(scene, cam_i)
            target_img = _load_image_tensor(image_path, 1.0, device)  # pre-cached at training res
            mask_path = scene.mask_paths[cam_i] if scene.mask_paths else None
            ds_cam = (budget.downscale_per_camera[cam_i] if budget.downscale_per_camera
                      else budget.downscale_factor)
            valid = _load_valid_mask(mask_path, ds_cam, device)

            # Progressive coarse-to-fine resolution: scale K and target image
            # for early training, then ramp to full res.  K was already pre-scaled
            # by pipeline.py; here we apply an additional step-dependent factor.
            if config.prog_res_enabled:
                _pds = _progressive_ds(step, config.iterations,
                                       config.prog_res_warmup_fraction,
                                       config.prog_res_mid_fraction)
                if _pds < 1.0:
                    _H = max(4, int(target_img.shape[0] * _pds))
                    _W = max(4, int(target_img.shape[1] * _pds))
                    K_r = K.clone(); K_r[:2, :] = K_r[:2, :] * _pds
                    target_r = F.interpolate(
                        target_img.permute(2, 0, 1).unsqueeze(0),
                        size=(_H, _W), mode="bilinear", align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                    valid_r = (F.interpolate(
                        valid.permute(2, 0, 1).unsqueeze(0),
                        size=(_H, _W), mode="nearest",
                    ).squeeze(0).permute(1, 2, 0) if valid is not None else None)
                else:
                    K_r, target_r, valid_r = K, target_img, valid
            else:
                K_r, target_r, valid_r = K, target_img, valid

            bg = torch.rand(3, device=device) if config.random_bg_per_step else torch.zeros(3, device=device)

            # Compose SH = [dc, rest[:active_dim]]
            sh_active = _gather_sh(sh_dc, sh_rest, active_sh_degree, config.sh_degree)

            # Exponential means LR decay: full rate early, 1% of init by final step.
            lr_means = _means_lr_at_step(
                step, config.iterations, means_lr_init, config.means_lr_final_factor,
            )
            for pg in optimizers["means"].param_groups:
                pg["lr"] = lr_means

            # Taming 3DGS: after the training midpoint, use abs().clamp() instead of
            # sigmoid so gradient flow through near-zero opacities is stronger, which
            # forces low-value Gaussians to either grow or die rather than accumulate
            # as hazy "Milky Way" artifacts.
            _opa_for_render = (
                opacities.abs().clamp(0.0, 1.0)
                if config.taming_opacity_enabled
                and step >= int(config.iterations * config.taming_start_frac)
                else torch.sigmoid(opacities)
            )
            try:
                colors, alphas, info = rasterization(
                    means=means,
                    quats=F.normalize(quats, dim=-1),
                    scales=torch.exp(scales),
                    opacities=_opa_for_render,
                    colors=sh_active,
                    viewmats=w2c[None].to(device),
                    Ks=K_r[None].to(device),
                    width=target_r.shape[1],
                    height=target_r.shape[0],
                    near_plane=near,
                    far_plane=far,
                    backgrounds=bg[None],
                    sh_degree=active_sh_degree,
                    render_mode="RGB+D",
                    rasterize_mode="antialiased" if config.antialias else "classic",
                )
            except torch.cuda.OutOfMemoryError:
                clear_cuda_cache()
                raise

            pred = colors[0, :, :, :3]    # (H, W, 3) RGB
            depth = colors[0, :, :, 3:4]  # (H, W, 1) camera-space depth

            # Apply per-camera exposure compensation to loss (not to the rendered output).
            if log_exposure is not None:
                pred_for_loss = pred * torch.exp(log_exposure[cam_i]).clamp(0.1, 10.0)
            else:
                pred_for_loss = pred

            if valid_r is not None:
                # valid_r: (H,W,1) float32 tensor, 1=compute loss, 0=masked out.
                # L1 on zeroed-masked pixels; SSIM computed on full image with mask
                # applied per-pixel so the convolution window stays clean.
                n_valid = valid_r.sum().clamp(min=1.0)
                pred_m = pred_for_loss * valid_r
                target_m = target_r * valid_r
                loss = (1.0 - config.ssim_lambda) * (pred_m - target_m).abs().sum() / n_valid
                loss = loss + config.ssim_lambda * _ssim_loss(pred_for_loss, target_r, mask=valid_r)
            else:
                loss = (1.0 - config.ssim_lambda) * (pred_for_loss - target_r).abs().mean()
                loss = loss + config.ssim_lambda * _ssim_loss(pred_for_loss, target_r)

            # Multi-scale L1: 0.5× downsampled term captures coarse structure.
            if config.multiscale_loss_weight > 0.0 and target_r.shape[0] >= 8 and target_r.shape[1] >= 8:
                pred_half = F.avg_pool2d(
                    pred_for_loss.permute(2, 0, 1).unsqueeze(0), 2, 2,
                ).squeeze(0).permute(1, 2, 0)
                gt_half = F.avg_pool2d(
                    target_r.permute(2, 0, 1).unsqueeze(0), 2, 2,
                ).squeeze(0).permute(1, 2, 0)
                loss = loss + config.multiscale_loss_weight * (pred_half - gt_half).abs().mean()

            # Regularization (gsplat MCMC-style anti-floater terms).
            loss = loss + config.opacity_reg * torch.sigmoid(opacities).mean()
            loss = loss + config.scale_reg * torch.exp(scales).mean()
            if config.depth_reg_weight > 0.0 or config.depth_lap_weight > 0.0:
                depth_norm = depth / max(float(init_cloud.scene_extent), 1e-6)
                if config.depth_reg_weight > 0.0:
                    loss = loss + config.depth_reg_weight * _depth_smooth_loss(depth_norm, target_r)
                if config.depth_lap_weight > 0.0 and depth_norm.shape[0] > 2 and depth_norm.shape[1] > 2:
                    loss = loss + config.depth_lap_weight * _depth_laplacian_loss(depth_norm)

            # SH frequency regularization: decaying L2 on rest bands.
            # Prevents high-frequency view-dependent noise from baking in before
            # coarse structure is established.  Ramps linearly from full weight
            # to zero at the midpoint (50%) of training.
            if config.sh_freq_reg_weight > 0.0 and sh_rest.shape[1] > 0:
                _sh_decay = max(0.0, 1.0 - step / max(config.iterations * 0.5, 1))
                if _sh_decay > 0.0:
                    loss = loss + config.sh_freq_reg_weight * _sh_decay * (sh_rest ** 2).mean()

            for opt in optimizers.values():
                opt.zero_grad(set_to_none=True)

            strategy.step_pre_backward(
                params=_strategy_params(means, scales, quats, opacities, sh_dc, sh_rest),
                optimizers=optimizers, state=strategy_state, step=step, info=info,
            )
            loss.backward()
            for opt in optimizers.values():
                opt.step()
            strategy.step_post_backward(
                params=_strategy_params(means, scales, quats, opacities, sh_dc, sh_rest),
                optimizers=optimizers, state=strategy_state, step=step, info=info,
                lr=lr_means,
            )

            if app_optimizer is not None:
                app_optimizer.step()
                app_optimizer.zero_grad(set_to_none=True)

            watchdog.tick(step)

            # Progress tick.
            cur_splats = means.shape[0]
            if step % 50 == 0 or step == 1:
                _tick(job_state, job_state_path, step=step, splats=cur_splats, loss=float(loss.item()))

            # Eval / preview.
            if step % config.eval_every == 0:
                render_inputs = RenderInputs(
                    means=means, scales=scales, quats=quats, opacities=opacities,
                    sh_dc=sh_dc, sh_rest=sh_rest,
                    sh_degree=active_sh_degree, full_sh_degree=config.sh_degree,
                )
                holdout_psnr, holdout_ssim = evaluate_holdout(
                    render_inputs, scene=scene, holdout_idx=holdout_idx,
                    near_plane=near, far_plane=far, downscale=1.0,
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(f"{step},{float(loss.item()):.6f},{holdout_psnr:.4f},{holdout_ssim:.4f}\n")
                job_state.tick(
                    current_step=step,
                    current_splats=means.shape[0],
                    psnr=holdout_psnr,
                    ssim=holdout_ssim,
                    loss=float(loss.item()),
                )
                write_state(job_state, job_state_path)

                # Divergence abort.
                if (config.divergence_check_at_step and step >= config.divergence_check_at_step
                        and holdout_psnr < config.divergence_min_psnr):
                    raise _DivergenceAbort(
                        f"holdout PSNR {holdout_psnr:.2f} < min {config.divergence_min_psnr:.1f} "
                        f"at step {step}"
                    )

            if step % config.preview_every == 0:
                preview_inputs = RenderInputs(
                    means=means, scales=scales, quats=quats, opacities=opacities,
                    sh_dc=sh_dc, sh_rest=sh_rest,
                    sh_degree=active_sh_degree, full_sh_degree=config.sh_degree,
                )
                strip_path = work_dir / "preview_strip.png"
                from gs_pipeline.trainer.render_eval import save_preview_strip
                save_preview_strip(
                    preview_inputs, scene=scene, holdout_idx=holdout_idx,
                    near_plane=near, far_plane=far, downscale=1.0,
                    out_path=strip_path,
                    target_height_px=config.preview_panel_height,
                )
                # Expose the strip to the live dashboard immediately.
                job_state.outputs.preview_strip_png = str(strip_path)
                job_state.outputs.preview_png = str(strip_path)
                write_state(job_state, job_state_path)
                # Archive one frame per checkpoint window for the timelapse.
                if step % config.checkpoint_every == 0 and config.timelapse_enabled:
                    import shutil as _shutil
                    tl_dir = work_dir / "timelapse_frames"
                    tl_dir.mkdir(exist_ok=True)
                    frame = tl_dir / f"strip_{step:06d}.png"
                    _shutil.copy2(strip_path, frame)
                    _timelapse_frames.append(frame)

            if step % config.checkpoint_every == 0:
                ckpt = work_dir / f"ckpt_{step}.pt"
                torch.save({
                    "means": means.detach().cpu(),
                    "scales": scales.detach().cpu(),
                    "quats": quats.detach().cpu(),
                    "opacities": opacities.detach().cpu(),
                    "sh_dc": sh_dc.detach().cpu(),
                    "sh_rest": sh_rest.detach().cpu(),
                    "step": step,
                    "active_sh_degree": active_sh_degree,
                }, ckpt)
                _add_checkpoint(job_state, str(ckpt))
                # Also export an intermediate .ply for the UI's mid-training download button.
                from gs_pipeline.trainer.export_ply import write_inria_ply  # local import
                write_inria_ply(
                    out_path=work_dir / f"scene_step_{step}.ply",
                    means=means.detach().cpu().numpy(),
                    scales=scales.detach().cpu().numpy(),
                    quats=quats.detach().cpu().numpy(),
                    opacities=opacities.detach().cpu().numpy(),
                    sh_dc=sh_dc.detach().cpu().numpy(),
                    sh_rest=sh_rest.detach().cpu().numpy(),
                )
                write_state(job_state, job_state_path)

        # Clean finish.
        from gs_pipeline.trainer.export_ply import write_inria_ply, read_inria_ply  # local import
        final_ply = outbox_dir / "scene.ply"
        outbox_dir.mkdir(parents=True, exist_ok=True)
        write_inria_ply(
            out_path=final_ply,
            means=means.detach().cpu().numpy(),
            scales=scales.detach().cpu().numpy(),
            quats=quats.detach().cpu().numpy(),
            opacities=opacities.detach().cpu().numpy(),
            sh_dc=sh_dc.detach().cpu().numpy(),
            sh_rest=sh_rest.detach().cpu().numpy(),
        )

        # --- Post-training filter ------------------------------------------------
        filter_report_dict: dict[str, Any] = {}
        if config.filter_enabled:
            from gs_pipeline.trainer.filter_splats import filter_scene  # local import
            loaded = read_inria_ply(final_ply)
            (
                f_means, f_scales, f_quats, f_opacities, f_sh_dc, f_sh_rest,
                f_report,
            ) = filter_scene(
                means=loaded.means,
                scales=loaded.scales,
                quats=loaded.quats,
                opacities=loaded.opacities,
                sh_dc=loaded.sh_dc,
                sh_rest=loaded.sh_rest,
                scene_extent=init_cloud.scene_extent,
                min_opacity=config.filter_min_opacity,
                sor_k=config.filter_sor_k,
                sor_std_ratio=config.filter_sor_std_ratio,
                max_scale_factor=config.filter_max_scale_factor,
            )
            _log.info("Post-training filter:\n%s", f_report.summary)
            # Save unfiltered backup, then overwrite with filtered.
            unfiltered_ply = outbox_dir / "scene_unfiltered.ply"
            import shutil as _shutil
            _shutil.copy2(str(final_ply), str(unfiltered_ply))
            write_inria_ply(
                out_path=final_ply,
                means=f_means,
                scales=f_scales,
                quats=f_quats,
                opacities=f_opacities,
                sh_dc=f_sh_dc,
                sh_rest=f_sh_rest,
            )
            filter_report_dict = {
                "n_input": f_report.n_input,
                "n_after_opacity": f_report.n_after_opacity,
                "n_after_scale": f_report.n_after_scale,
                "n_after_sor": f_report.n_after_sor,
                "n_output": f_report.n_output,
                "summary": f_report.summary,
            }
            _final_np = (f_means, f_scales, f_quats, f_opacities, f_sh_dc, f_sh_rest)
        else:
            _final_np = (
                means.detach().cpu().numpy(),
                scales.detach().cpu().numpy(),
                quats.detach().cpu().numpy(),
                opacities.detach().cpu().numpy(),
                sh_dc.detach().cpu().numpy(),
                sh_rest.detach().cpu().numpy(),
            )

        # Optional Fisher information pruning: remove Gaussians that contribute
        # little to the reconstruction loss across training views.  Disabled by
        # default because it adds GPU time; useful for client demos where file
        # size matters more than marginal completeness.
        if config.fisher_prune_enabled:
            _before_fisher = int(_final_np[0].shape[0])
            _log.info(
                "Fisher pruning: accumulating gradients over %d views, keep %.0f%%...",
                config.fisher_prune_n_views, config.fisher_prune_ratio * 100,
            )
            _final_np = _fisher_prune(
                *_final_np,
                scene=scene, train_idx=train_idx, near=near, far=far, device=device,
                keep_ratio=config.fisher_prune_ratio, n_views=config.fisher_prune_n_views,
                sh_degree=config.sh_degree,
            )
            _log.info(
                "Fisher pruning: %d → %d splats (%.1f%% kept)",
                _before_fisher, _final_np[0].shape[0],
                100.0 * _final_np[0].shape[0] / max(_before_fisher, 1),
            )
            write_inria_ply(
                out_path=final_ply, means=_final_np[0], scales=_final_np[1],
                quats=_final_np[2], opacities=_final_np[3],
                sh_dc=_final_np[4], sh_rest=_final_np[5],
            )
            if filter_report_dict:
                filter_report_dict["fisher_n_input"] = _before_fisher
                filter_report_dict["fisher_n_output"] = int(_final_np[0].shape[0])

        # Final per-camera evaluation with the filtered splats.
        _per_cam_stats: list[dict] = []
        try:
            _final_ri = RenderInputs(
                means=torch.from_numpy(_final_np[0]).float().to(device),
                scales=torch.from_numpy(_final_np[1]).float().to(device),
                quats=torch.from_numpy(_final_np[2]).float().to(device),
                opacities=torch.from_numpy(_final_np[3]).float().to(device),
                sh_dc=torch.from_numpy(_final_np[4]).float().to(device),
                sh_rest=torch.from_numpy(_final_np[5]).float().to(device),
                sh_degree=config.sh_degree, full_sh_degree=config.sh_degree,
            )
            _final_psnr, _final_ssim, _per_cam_stats = evaluate_holdout(
                _final_ri, scene=scene, holdout_idx=holdout_idx,
                near_plane=near, far_plane=far, downscale=1.0,
                per_camera=True,
            )
            _log.info("Final holdout PSNR %.2f dB  SSIM %.4f", _final_psnr, _final_ssim)
        except Exception as _eval_exc:
            _log.warning("Final evaluation failed: %s", _eval_exc)
            _final_psnr = job_state.progress.psnr_history[-1][1] if job_state.progress.psnr_history else None
            _final_ssim = job_state.progress.ssim_history[-1][1] if job_state.progress.ssim_history else None

        # Compact .splat binary — 32 bytes/splat, web-compatible with SuperSplat.
        if config.export_splat_binary:
            from gs_pipeline.trainer.export_ply import write_splat_binary  # local import
            write_splat_binary(
                out_path=outbox_dir / "scene.splat",
                means=_final_np[0], scales=_final_np[1], quats=_final_np[2],
                opacities=_final_np[3], sh_dc=_final_np[4],
            )

        _final_count = int(_final_np[0].shape[0])

        report = {
            "job_id": job_state.job_id,
            "final_step": config.iterations,
            "final_splat_count": _final_count,
            "final_psnr_db": round(_final_psnr, 4) if _final_psnr is not None else None,
            "final_ssim": round(_final_ssim, 4) if _final_ssim is not None else None,
            "holdout_per_camera": _per_cam_stats,
            "preflight": job_state.preflight.__dict__ if job_state.preflight else None,
            "filter": filter_report_dict if filter_report_dict else None,
        }
        (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        # Compile timelapse if we collected frames.
        timelapse_path: Optional[str] = None
        if config.timelapse_enabled and _timelapse_frames:
            tl_out = outbox_dir / "training_timelapse.mp4"
            if _compile_timelapse(_timelapse_frames, tl_out, fps=config.timelapse_fps):
                timelapse_path = str(tl_out)
                _log.info("timelapse written: %s", tl_out)

        _splat_path = outbox_dir / "scene.splat"
        _unfiltered_path = outbox_dir / "scene_unfiltered.ply"

        strip_str = str(work_dir / "preview_strip.png")
        return OutputsSnapshot(
            checkpoints=job_state.outputs.checkpoints,
            preview_png=strip_str,
            preview_strip_png=strip_str,
            final_ply=str(final_ply),
            final_splat=str(_splat_path) if _splat_path.is_file() else None,
            final_ply_unfiltered=str(_unfiltered_path) if _unfiltered_path.is_file() else None,
            final_psnr=_final_psnr,
            final_ssim=_final_ssim,
            final_splat_count=_final_count,
            metrics_csv=str(metrics_path),
            report_json=str(work_dir / "report.json"),
            timelapse_mp4=timelapse_path,
        )
    finally:
        watchdog.stop()
        clear_cuda_cache()


# ---------------------------------------------------------------------------
# Small helpers (most are GPU-side; collected here for readability)
# ---------------------------------------------------------------------------

class _DivergenceAbort(RuntimeError):
    """Raised inside the training loop to abort on persistent low PSNR."""


def _inverse_sigmoid(x: float) -> float:
    x = max(1e-6, min(1 - 1e-6, x))
    return math.log(x / (1.0 - x))


def _rgb_to_sh_dc(rgb):  # tensor in [0,1]
    # SH band 0 coefficient: c0 = 1/(2*sqrt(pi))
    import torch
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def _strategy_params(means, scales, quats, opacities, sh_dc, sh_rest):
    return {
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh_dc": sh_dc, "sh_rest": sh_rest,
    }


def _load_image_tensor(image_path: Path, downscale: float, device):
    """Load a training image as a torch tensor on ``device`` in ``[0, 1]``."""
    import torch
    from gs_pipeline.trainer.render_eval import _load_image_np
    return torch.from_numpy(_load_image_np(image_path, downscale)).to(device)


def _load_valid_mask(mask_path, downscale: float, device):
    """Load a Metashape mask PNG as a (H, W, 1) validity tensor on ``device``.

    Inverts Metashape's convention (white=excluded → 0, black=keep → 1) so the
    result can be multiplied directly against pred/target in the loss.
    Returns None if mask_path is None (no mask for this camera).
    """
    if mask_path is None:
        return None
    import torch
    import numpy as np
    try:
        from PIL import Image
        img = Image.open(mask_path).convert("L")
        w, h = img.size
        if downscale != 1.0:
            nw = max(1, int(w * downscale))
            nh = max(1, int(h * downscale))
            img = img.resize((nw, nh), Image.NEAREST)
        arr = np.array(img, dtype=np.float32) / 255.0
        # Invert: Metashape white=excluded → we want 0=excluded, 1=valid.
        valid = torch.from_numpy(1.0 - arr).unsqueeze(-1).to(device)  # (H, W, 1)
        return valid
    except Exception:
        return None


def _ssim_loss(pred, target, *, mask=None, window_size: int = 11):
    """Differentiable SSIM loss (1 - SSIM) via Gaussian-windowed convolutions on GPU.

    mask: optional (H, W, 1) validity tensor (1=valid, 0=masked). When provided,
    the per-pixel SSIM loss is averaged only over valid pixels — the convolution
    window is computed on the full image so masked regions don't corrupt neighbors.
    """
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
    loss_map = 1.0 - ssim_map  # (1, C, H, W)
    if mask is not None:
        m = mask.permute(2, 0, 1).unsqueeze(0)  # (1, 1, H, W)
        return (loss_map * m).sum() / m.sum().clamp(min=1.0) / channels
    return loss_map.mean()


def _depth_smooth_loss(depth, image):
    """Edge-aware depth smoothness. depth: (H,W,1), image: (H,W,3) in [0,1].

    Penalises depth changes in smooth image regions; allows discontinuities where
    there are strong image edges. Pass depth / scene_extent for scale invariance.
    """
    import torch
    depth_dx = (depth[1:, :, :] - depth[:-1, :, :]).abs()
    depth_dy = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
    img_dx = (image[1:, :, :] - image[:-1, :, :]).abs().mean(dim=-1, keepdim=True)
    img_dy = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(dim=-1, keepdim=True)
    return (
        (torch.exp(-img_dx * 10.0) * depth_dx).mean()
        + (torch.exp(-img_dy * 10.0) * depth_dy).mean()
    )


def _progressive_ds(step: int, total_steps: int, warmup_frac: float, mid_frac: float) -> float:
    """Coarse-to-fine resolution factor.

    Returns 0.25 for the first ``warmup_frac`` of training, 0.5 for the next
    ``mid_frac - warmup_frac`` portion, and 1.0 for the remainder.  Rendering
    at reduced resolution during early training speeds up densification and
    avoids fitting high-frequency noise before coarse structure is established.
    """
    frac = step / max(total_steps, 1)
    if frac < warmup_frac:
        return 0.25
    if frac < mid_frac:
        return 0.5
    return 1.0


def _depth_laplacian_loss(depth):
    """Second-order depth smoothness: penalises non-planar depth variation.

    Computes the discrete Laplacian (d²D/dx² + d²D/dy²) and returns its
    mean squared magnitude.  Planar surfaces → Laplacian ≈ 0.  This is
    complementary to ``_depth_smooth_loss`` (first-order): together they
    encourage piece-wise planar geometry.  Input: (H, W, 1) depth tensor.
    """
    lap_x = depth[2:, 1:-1, :] - 2.0 * depth[1:-1, 1:-1, :] + depth[:-2, 1:-1, :]
    lap_y = depth[1:-1, 2:, :] - 2.0 * depth[1:-1, 1:-1, :] + depth[1:-1, :-2, :]
    return (lap_x ** 2 + lap_y ** 2).mean()


def _means_lr_at_step(step, total_steps, lr_init, final_factor):
    """Exponential decay: lr_init at step 1, lr_init*final_factor at total_steps."""
    if total_steps <= 1 or final_factor >= 1.0:
        return lr_init
    t = (step - 1) / (total_steps - 1)
    return lr_init * (final_factor ** t)


def _tick(job_state: JobState, path: Path, *, step: int, splats: int, loss: float) -> None:
    """Lightweight progress update (no eval) — every 50 steps."""
    job_state.tick(current_step=step, current_splats=splats, loss=loss)
    write_state(job_state, path)


def _add_checkpoint(job_state: JobState, ckpt_path: str) -> None:
    if ckpt_path not in job_state.outputs.checkpoints:
        job_state.outputs.checkpoints.append(ckpt_path)


def _compile_timelapse(frame_paths: list[Path], out_path: Path, fps: int = 10) -> bool:
    """Compile preview strip PNGs into an MP4 via ffmpeg. Returns True on success."""
    import shutil
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
            # Repeat last frame so ffmpeg concat doesn't drop it.
            f.write(f"file '{frame_paths[-1].absolute()}'\n")
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(list_file),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                str(out_path),
            ],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0
    except Exception as exc:
        _log.warning("timelapse compilation failed: %s", exc)
        return False
    finally:
        list_file.unlink(missing_ok=True)


def _apply_axis_flip_if_needed(
    *, means, scales, quats, opacities, sh_dc, sh_rest,
    active_sh_degree, full_sh_degree, scene, train_idx,
    near, far, downscale, device,
) -> None:
    """Render one training view with and without a camera-axis flip; apply if it helps.

    ``downscale`` should be 1.0 when K is already pre-scaled by pipeline.py.
    """
    import torch

    cam_i = train_idx[0]
    K, w2c, image_path = _load_camera(scene, cam_i)
    target_np = _load_image_np(image_path, downscale)
    target = torch.from_numpy(target_np).to(device)
    h, w = target.shape[:2]

    ri = RenderInputs(
        means=means, scales=scales, quats=quats, opacities=opacities,
        sh_dc=sh_dc, sh_rest=sh_rest,
        sh_degree=active_sh_degree, full_sh_degree=full_sh_degree,
    )

    flip_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=device))

    with torch.no_grad():
        pred_no_flip, _ = render_view(
            ri, K=K.to(device), w2c=w2c.to(device),
            width=w, height=h, near_plane=near, far_plane=far,
        )
        w2c_flipped = (flip_mat @ w2c.to(device)).float()
        pred_flipped, _ = render_view(
            ri, K=K.to(device), w2c=w2c_flipped,
            width=w, height=h, near_plane=near, far_plane=far,
        )

    needs_flip = detect_camera_axis_flip(
        rendered_no_flip=pred_no_flip.cpu().numpy(),
        rendered_flipped=pred_flipped.cpu().numpy(),
        target=target_np,
    )

    if needs_flip:
        _log.warning("axis-flip auto-detect: applying diag(1,-1,-1) to all w2c matrices")
        flip_np = np.diag([1.0, -1.0, -1.0, 1.0])
        scene.w2c_per_camera = (flip_np @ scene.w2c_per_camera).astype(np.float64)
    else:
        _log.info("axis-flip auto-detect: no flip needed")


def _fisher_prune(
    means_np: np.ndarray,
    scales_np: np.ndarray,
    quats_np: np.ndarray,
    opacities_np: np.ndarray,
    sh_dc_np: np.ndarray,
    sh_rest_np: np.ndarray,
    *,
    scene,
    train_idx: list,
    near: float,
    far: float,
    device,
    keep_ratio: float = 0.5,
    n_views: int = 20,
    sh_degree: int = 3,
) -> tuple:
    """Fisher information pruning: keep splats by positional gradient importance.

    Accumulates (∂L/∂means)² per Gaussian over ``n_views`` training images.
    Gaussians with low gradient magnitude are rarely "seen" or not needed for
    accurate reconstruction — pruning them reduces file size with minimal loss.
    Returns a 6-tuple of numpy arrays (same layout as input) for the survivors.
    """
    import torch
    import torch.nn.functional as F
    from gs_pipeline.trainer.render_eval import _load_camera, _load_image_np, _gather_sh

    means = torch.from_numpy(means_np).float().to(device).requires_grad_(True)
    scales_t = torch.from_numpy(scales_np).float().to(device)
    quats_t = torch.from_numpy(quats_np).float().to(device)
    opacities_t = torch.from_numpy(opacities_np).float().to(device)
    sh_dc_t = torch.from_numpy(sh_dc_np).float().to(device)
    sh_rest_t = torch.from_numpy(sh_rest_np).float().to(device)

    n = means.shape[0]
    fisher_scores = torch.zeros(n, device=device)

    rng_np = np.random.default_rng(99)
    view_indices = list(train_idx)
    if len(view_indices) > n_views:
        view_indices = list(rng_np.choice(len(view_indices), n_views, replace=False))
        view_indices = [train_idx[i] for i in view_indices]

    sh_active = _gather_sh(sh_dc_t, sh_rest_t, sh_degree, sh_degree)

    from gsplat import rasterization as _gs_rasterize
    n_accumulated = 0
    for cam_i in view_indices:
        K, w2c, image_path = _load_camera(scene, cam_i)
        target_np = _load_image_np(image_path, 1.0)
        target = torch.from_numpy(target_np).to(device)
        try:
            with torch.enable_grad():
                if means.grad is not None:
                    means.grad.zero_()
                colors, _alphas, _info = _gs_rasterize(
                    means=means,
                    quats=F.normalize(quats_t, dim=-1),
                    scales=torch.exp(scales_t),
                    opacities=torch.sigmoid(opacities_t),
                    colors=sh_active,
                    viewmats=w2c[None].to(device),
                    Ks=K[None].to(device),
                    width=target.shape[1],
                    height=target.shape[0],
                    near_plane=near,
                    far_plane=far,
                    sh_degree=sh_degree,
                )
                pred = colors[0, :, :, :3].clamp(0.0, 1.0)
                loss = (pred - target).abs().mean()
                loss.backward()
            if means.grad is not None:
                fisher_scores += means.grad.detach().pow(2).sum(dim=1)
                n_accumulated += 1
        except Exception as _e:
            _log.debug("Fisher prune skipped view %d: %s", cam_i, _e)
            continue

    if n_accumulated == 0:
        _log.warning("Fisher pruning: no views succeeded; returning all splats unchanged")
        return means_np, scales_np, quats_np, opacities_np, sh_dc_np, sh_rest_np

    k_keep = max(1, int(n * keep_ratio))
    _, top_idx = torch.topk(fisher_scores, k_keep)
    mask_gpu = torch.zeros(n, dtype=torch.bool, device=device)
    mask_gpu[top_idx] = True
    mask = mask_gpu.cpu().numpy()

    return (
        means_np[mask], scales_np[mask], quats_np[mask],
        opacities_np[mask], sh_dc_np[mask], sh_rest_np[mask],
    )
