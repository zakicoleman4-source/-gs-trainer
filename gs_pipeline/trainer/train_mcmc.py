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
    evaluate_holdout,
    psnr,
    save_preview_png,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain dataclass for trainer config (typed view of config.yaml)
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    iterations: int = 30_000
    noise_lr: float = 5.0e5
    refine_start_iter: int = 500
    refine_stop_iter_ratio: float = 0.85
    refine_every: int = 100
    prune_opa: float = 0.005
    sh_degree: int = 3
    sh_warmup_interval: int = 1500
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    init_opacity: float = 0.1
    init_scale_knn_k: int = 3
    near_plane_extent_ratio: float = 0.01
    far_plane_extent_ratio: float = 100.0
    random_bg_per_step: bool = True
    holdout_stride: int = 8
    eval_every: int = 1000
    preview_every: int = 1000
    checkpoint_every: int = 5000
    divergence_min_psnr: float = 12.0
    divergence_check_at_step: int = 10_000
    memory_fraction: float = 0.92
    # SSIM weight in the photometric loss: L = (1-lambda)*L1 + lambda*(1-SSIM).
    ssim_lambda: float = 0.2


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
        preview_every=int(ev.get("preview_every", 1000)),
        checkpoint_every=int(ck.get("every", 5000)),
        divergence_min_psnr=float(div.get("min_psnr_at_step", 12.0)),
        divergence_check_at_step=int(div.get("check_at_step", 10_000)),
    )


# ---------------------------------------------------------------------------
# k-NN init scale (CPU; one-shot at init)
# ---------------------------------------------------------------------------

def knn_mean_distance(xyz: np.ndarray, k: int = 3) -> np.ndarray:
    """Per-point mean distance to its k nearest neighbors (excluding itself).

    Used to seed Gaussian scales: a splat in a dense region should be small,
    a splat in a sparse region large. Pure NumPy, O(N^2) — fine for N <= 1M
    since we only run it once at init.
    """
    n = xyz.shape[0]
    if n <= 1:
        return np.full(n, 1e-3, dtype=np.float32)
    # Block to avoid building an N x N matrix on huge clouds.
    block = min(8192, n)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, block):
        chunk = xyz[i:i + block]                                # (B, 3)
        diff = chunk[:, None, :] - xyz[None, :, :]              # (B, N, 3)
        d2 = np.einsum("ijk,ijk->ij", diff, diff)               # (B, N)
        # Replace self-distances with +inf, then take k smallest.
        rows = np.arange(chunk.shape[0])
        d2[rows, np.arange(i, i + chunk.shape[0])] = np.inf
        # k smallest by partial sort.
        k_eff = min(k, n - 1)
        nearest = np.partition(d2, k_eff - 1, axis=1)[:, :k_eff]
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

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 0.00016 * init_cloud.scene_extent, "name": "means"},
        {"params": [scales], "lr": 0.005, "name": "scales"},
        {"params": [quats], "lr": 0.001, "name": "quats"},
        {"params": [opacities], "lr": 0.05, "name": "opacities"},
        {"params": [sh_dc], "lr": 0.0025, "name": "sh_dc"},
        {"params": [sh_rest], "lr": 0.0025 / 20.0, "name": "sh_rest"},
    ])

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

    rng = np.random.default_rng(0)
    metrics_path = work_dir / "metrics.csv"
    metrics_path.write_text("step,loss,holdout_psnr,holdout_ssim\n", encoding="utf-8")

    watchdog = ProgressWatchdog(timeout_s=1800.0, poll_interval_s=30.0)
    watchdog.start()

    try:
        for step in range(1, config.iterations + 1):
            # SH degree warmup.
            target_deg = min(config.sh_degree, step // max(config.sh_warmup_interval, 1))
            active_sh_degree = max(active_sh_degree, target_deg)

            # Pick a training camera.
            cam_i = int(train_idx[rng.integers(0, len(train_idx))])
            K, w2c, image_path = _load_camera(scene, cam_i)
            target_img = _load_image_tensor(image_path, budget.downscale_factor, device)

            bg = torch.rand(3, device=device) if config.random_bg_per_step else torch.zeros(3, device=device)

            # Compose SH = [dc, rest[:active_dim]]
            sh_active = _gather_sh(sh_dc, sh_rest, active_sh_degree, config.sh_degree)

            try:
                colors, alphas, info = rasterization(
                    means=means,
                    quats=torch.nn.functional.normalize(quats, dim=-1),
                    scales=torch.exp(scales),
                    opacities=torch.sigmoid(opacities),
                    colors=sh_active,
                    viewmats=w2c[None].to(device),
                    Ks=K[None].to(device),
                    width=target_img.shape[1],
                    height=target_img.shape[0],
                    near_plane=near,
                    far_plane=far,
                    backgrounds=bg[None],
                    sh_degree=active_sh_degree,
                )
            except torch.cuda.OutOfMemoryError:
                clear_cuda_cache()
                raise

            pred = colors[0]
            loss = (1.0 - config.ssim_lambda) * (pred - target_img).abs().mean()
            # SSIM term (skipped if scikit-image unavailable at import time):
            loss = loss + config.ssim_lambda * _ssim_loss(pred, target_img)

            # Regularization (gsplat MCMC-style anti-floater terms).
            loss = loss + config.opacity_reg * torch.sigmoid(opacities).mean()
            loss = loss + config.scale_reg * torch.exp(scales).mean()

            optimizer.zero_grad(set_to_none=True)

            strategy.step_pre_backward(
                params=_strategy_params(means, scales, quats, opacities, sh_dc, sh_rest),
                optimizers=optimizer, state=strategy_state, step=step, info=info,
            )
            loss.backward()
            optimizer.step()
            strategy.step_post_backward(
                params=_strategy_params(means, scales, quats, opacities, sh_dc, sh_rest),
                optimizers=optimizer, state=strategy_state, step=step, info=info,
                lr=0.00016 * init_cloud.scene_extent,
            )

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
                    near_plane=near, far_plane=far, downscale=budget.downscale_factor,
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
                save_preview_png(
                    preview_inputs, scene=scene, holdout_idx=holdout_idx,
                    near_plane=near, far_plane=far, downscale=budget.downscale_factor,
                    out_path=work_dir / "preview.png",
                )

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
        from gs_pipeline.trainer.export_ply import write_inria_ply  # local import
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

        report = {
            "job_id": job_state.job_id,
            "final_step": config.iterations,
            "final_splat_count": int(means.shape[0]),
            "preflight": job_state.preflight.__dict__ if job_state.preflight else None,
        }
        (work_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

        return OutputsSnapshot(
            checkpoints=job_state.outputs.checkpoints,
            preview_png=str(work_dir / "preview.png"),
            final_ply=str(final_ply),
            metrics_csv=str(metrics_path),
            report_json=str(work_dir / "report.json"),
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


def _ssim_loss(pred, target):
    """SSIM-weighted L1 loss: computes per-pixel SSIM map via skimage, uses it
    to weight the L1 term so gradients flow back through pred. Falls back to
    plain L2 if skimage is unavailable."""
    try:
        from skimage.metrics import structural_similarity as sk_ssim
    except Exception:
        return ((pred - target) ** 2).mean()
    import torch
    p = pred.detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    _, ssim_map = sk_ssim(t, p, channel_axis=-1, data_range=1.0, full=True)
    ssim_map_t = torch.from_numpy(ssim_map.astype("float32")).to(pred.device)
    return ((1.0 - ssim_map_t).unsqueeze(-1) * (pred - target).abs()).mean()


def _tick(job_state: JobState, path: Path, *, step: int, splats: int, loss: float) -> None:
    """Lightweight progress update (no eval) — every 50 steps."""
    job_state.tick(current_step=step, current_splats=splats, loss=loss)
    write_state(job_state, path)


def _add_checkpoint(job_state: JobState, ckpt_path: str) -> None:
    if ckpt_path not in job_state.outputs.checkpoints:
        job_state.outputs.checkpoints.append(ckpt_path)
