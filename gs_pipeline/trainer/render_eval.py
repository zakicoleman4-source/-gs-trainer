"""Holdout PSNR/SSIM evaluation and preview renders for the trainer.

The training loop calls into here every ``eval_every`` steps; the same
functions also produce the live preview PNG the Streamlit UI shows mid-run.

Everything GPU-side (the actual rasterization) is gated behind lazy
``torch`` / ``gsplat`` imports inside each function so the module imports
cleanly on CPU CI. The CPU-only utilities (``psnr``, ``ssim``) are unit-
tested without touching CUDA.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU metrics (deterministic, used in unit tests and training loop)
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, gt: np.ndarray, *, max_val: float = 1.0) -> float:
    """Per-image PSNR in dB. Inputs in ``[0, max_val]``. Returns +inf on identical inputs."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0.0:
        return float("inf")
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def ssim(pred: np.ndarray, gt: np.ndarray, *, data_range: float = 1.0) -> float:
    """SSIM, falling back to a 1 - normalised-MSE proxy when scikit-image is absent.

    The fallback is good enough for divergence detection (the trainer uses
    SSIM only to flag whether quality is regressing, not for a leaderboard).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred={pred.shape} gt={gt.shape}")
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        return float(sk_ssim(gt, pred, channel_axis=-1, data_range=data_range))
    except Exception:
        # Bounded fallback in [-1, 1]: 1 - normalised MSE.
        denom = float(data_range) ** 2
        mse = float(np.mean((pred - gt) ** 2))
        return max(-1.0, min(1.0, 1.0 - mse / max(denom, 1e-12)))


# ---------------------------------------------------------------------------
# GPU render (lazy imports; only callable on a CUDA host)
# ---------------------------------------------------------------------------

@dataclass
class RenderInputs:
    """Bag of the gsplat Gaussian state needed to render one view."""
    means: object        # torch.Tensor (N, 3)
    scales: object       # torch.Tensor (N, 3)  -- log-scale parameters
    quats: object        # torch.Tensor (N, 4)  -- w, x, y, z (unnormalised ok)
    opacities: object    # torch.Tensor (N,)    -- logits (pre-sigmoid)
    sh_dc: object        # torch.Tensor (N, 3)
    sh_rest: object      # torch.Tensor (N, K, 3), K = (sh_degree+1)^2 - 1
    sh_degree: int       # active SH degree (warmup)
    full_sh_degree: int  # full SH degree used to size sh_rest


def render_view(
    inputs: RenderInputs,
    *,
    K,                   # torch.Tensor (3, 3)
    w2c,                 # torch.Tensor (4, 4)
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    background=None,     # torch.Tensor (3,) or None
):
    """Run a single gsplat rasterization and return (image, info_dict).

    ``image`` is a HxWx3 ``torch.Tensor`` in ``[0, 1]``. Caller is responsible
    for any ``.detach()`` / ``.cpu()`` it needs.
    """
    import torch
    from gsplat import rasterization

    sh_active = _gather_sh(inputs.sh_dc, inputs.sh_rest, inputs.sh_degree, inputs.full_sh_degree)
    if background is None:
        background = torch.zeros(3, device=inputs.means.device)
    colors, _alphas, info = rasterization(
        means=inputs.means,
        quats=torch.nn.functional.normalize(inputs.quats, dim=-1),
        scales=torch.exp(inputs.scales),
        opacities=torch.sigmoid(inputs.opacities),
        colors=sh_active,
        viewmats=w2c[None],
        Ks=K[None],
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        backgrounds=background[None],
        sh_degree=inputs.sh_degree,
    )
    return colors[0].clamp(0.0, 1.0), info


def evaluate_holdout(
    inputs: RenderInputs,
    *,
    scene,
    holdout_idx: Sequence[int],
    near_plane: float,
    far_plane: float,
    downscale: float = 1.0,
    downscale_per_camera: Optional[list[float]] = None,
) -> tuple[float, float]:
    """Mean PSNR / SSIM over the holdout cameras. Returns (psnr, ssim)."""
    import torch
    psnrs: list[float] = []
    ssims: list[float] = []
    with torch.no_grad():
        for list_pos, cam_i in enumerate(holdout_idx):
            ds = downscale_per_camera[cam_i] if downscale_per_camera is not None else downscale
            K, w2c, image_path = _load_camera(scene, cam_i, downscale=ds)
            target_np = _load_image_np(image_path, ds)
            target = torch.from_numpy(target_np).to(inputs.means.device)
            pred, _ = render_view(
                inputs, K=K.to(inputs.means.device), w2c=w2c.to(inputs.means.device),
                width=target.shape[1], height=target.shape[0],
                near_plane=near_plane, far_plane=far_plane,
            )
            pred_np = pred.detach().cpu().numpy()
            psnrs.append(psnr(pred_np, target_np))
            ssims.append(ssim(pred_np, target_np))
    return float(np.mean(psnrs)) if psnrs else float("nan"), float(np.mean(ssims)) if ssims else float("nan")


def save_preview_png(
    inputs: RenderInputs,
    *,
    scene,
    holdout_idx: Sequence[int],
    near_plane: float,
    far_plane: float,
    downscale: float = 1.0,
    downscale_per_camera: Optional[list[float]] = None,
    out_path: Path,
) -> None:
    """Render one holdout view and save as PNG for the UI live-preview tile."""
    import torch
    from PIL import Image
    if not holdout_idx:
        raise ValueError("save_preview_png needs at least one holdout camera")
    i = int(holdout_idx[0])
    ds = downscale_per_camera[i] if downscale_per_camera is not None else downscale
    K, w2c, image_path = _load_camera(scene, i, downscale=ds)
    target_np = _load_image_np(image_path, ds)
    with torch.no_grad():
        pred, _ = render_view(
            inputs, K=K.to(inputs.means.device), w2c=w2c.to(inputs.means.device),
            width=target_np.shape[1], height=target_np.shape[0],
            near_plane=near_plane, far_plane=far_plane,
        )
    arr = (pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path)


def save_preview_strip(
    inputs: RenderInputs,
    *,
    scene,
    holdout_idx: Sequence[int],
    near_plane: float,
    far_plane: float,
    downscale: float = 1.0,
    downscale_per_camera: Optional[list[float]] = None,
    out_path: Path,
    target_height_px: int = 400,
) -> None:
    """Render up to 3 holdout cameras and save side-by-side as a strip PNG.

    Picks cameras from start, middle, end of holdout_idx for scene coverage.
    Scales each panel to target_height_px tall (maintaining aspect ratio).
    """
    import torch
    from PIL import Image

    if not holdout_idx:
        raise ValueError("save_preview_strip needs at least one holdout camera")

    # Pick up to 3 indices: first, middle, last (deduplicated)
    idx_list = list(holdout_idx)
    n = len(idx_list)
    candidates = [idx_list[0], idx_list[n // 2], idx_list[-1]]
    # Deduplicate while preserving order
    seen: set[int] = set()
    selected: list[int] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            selected.append(c)

    SEPARATOR_WIDTH = 4
    panels: list[np.ndarray] = []

    with torch.no_grad():
        for cam_i in selected:
            ds = downscale_per_camera[cam_i] if downscale_per_camera is not None else downscale
            K, w2c, image_path = _load_camera(scene, cam_i, downscale=ds)
            target_np = _load_image_np(image_path, ds)
            pred, _ = render_view(
                inputs,
                K=K.to(inputs.means.device),
                w2c=w2c.to(inputs.means.device),
                width=target_np.shape[1],
                height=target_np.shape[0],
                near_plane=near_plane,
                far_plane=far_plane,
            )
            arr = (pred.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            # Resize to target_height_px maintaining aspect ratio
            h, w = arr.shape[:2]
            new_w = max(1, int(round(w * target_height_px / h)))
            resized = np.asarray(
                Image.fromarray(arr).resize((new_w, target_height_px), Image.LANCZOS),
                dtype=np.uint8,
            )
            panels.append(resized)

    # Build strip: panels separated by white vertical bars
    strip_parts: list[np.ndarray] = []
    separator = np.full((target_height_px, SEPARATOR_WIDTH, 3), 255, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        if idx > 0:
            strip_parts.append(separator)
        strip_parts.append(panel)

    strip = np.concatenate(strip_parts, axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(out_path)


# ---------------------------------------------------------------------------
# Internals (shared with train_mcmc.py)
# ---------------------------------------------------------------------------

def _gather_sh(sh_dc, sh_rest, active_deg: int, full_deg: int):
    import torch
    active_dim = (active_deg + 1) ** 2 - 1
    full_dim = (full_deg + 1) ** 2 - 1
    rest = sh_rest
    if active_dim < full_dim:
        rest = sh_rest.clone()
        rest[:, active_dim:] = 0.0
    return torch.cat([sh_dc[:, None, :], rest], dim=1)


def _load_camera(scene, i: int, downscale: float = 1.0):
    import torch
    K = torch.from_numpy(scene.K_per_camera[i]).float()
    if downscale != 1.0:
        K = K.clone()
        K[0] *= downscale  # scale row 0: fx, skew, cx — all scale together
        K[1] *= downscale  # scale row 1: 0, fy, cy — all scale together
        # K[2] = [0, 0, 1] — unchanged
    w2c = torch.from_numpy(scene.w2c_per_camera[i]).float()
    return K, w2c, scene.image_paths[i]


def _load_image_np(image_path: Path, downscale: float) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if downscale < 1.0:
        w, h = img.size
        img = img.resize(
            (max(1, int(round(w * downscale))), max(1, int(round(h * downscale)))),
            Image.LANCZOS,
        )
    return (np.asarray(img, dtype=np.float32) / 255.0)
