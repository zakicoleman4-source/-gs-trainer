"""CPU tests for ``gs_pipeline.trainer.render_eval``.

GPU render paths (``render_view``, ``evaluate_holdout``, ``save_preview_png``)
are exercised by the GPU-gated smoke test in ``test_pipeline_smoke.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from gs_pipeline.trainer.render_eval import psnr, ssim, _load_camera


# ---------------------------------------------------------------------------
# psnr — re-tested here so render_eval owns the canonical contract
# ---------------------------------------------------------------------------

def test_psnr_identical_inputs_returns_inf():
    img = np.full((4, 4, 3), 0.25, dtype=np.float32)
    assert np.isinf(psnr(img, img))


def test_psnr_known_offset_value():
    """Uniform 0.1 offset => PSNR = -20 log10(0.1) = 20 dB."""
    gt = np.zeros((8, 8, 3), dtype=np.float32)
    pred = np.full_like(gt, 0.1)
    assert psnr(pred, gt) == pytest.approx(20.0, rel=1e-6)


def test_psnr_shape_mismatch_raises():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.zeros((5, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        psnr(a, b)


def test_psnr_max_val_scaling_consistent():
    gt = np.zeros((4, 4, 3), dtype=np.float32)
    pred = np.full_like(gt, 25.5)  # 10% of 255
    a = psnr(pred, gt, max_val=255.0)
    b = psnr(pred / 255.0, gt / 255.0, max_val=1.0)
    assert a == pytest.approx(b, abs=1e-6)


# ---------------------------------------------------------------------------
# ssim
# ---------------------------------------------------------------------------

def test_ssim_identical_inputs_returns_one():
    img = np.full((16, 16, 3), 0.4, dtype=np.float32)
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_bounded_when_skimage_missing(monkeypatch):
    """If scikit-image isn't importable, fall back to a bounded proxy."""
    real_import = __import__
    def fake_import(name, *args, **kwargs):
        if name.startswith("skimage"):
            raise ImportError("forced for test")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", fake_import)

    img = np.full((4, 4, 3), 0.4, dtype=np.float32)
    # Identical inputs: fallback returns 1 (MSE=0 -> 1 - 0).
    assert ssim(img, img) == pytest.approx(1.0, abs=1e-9)
    # Maximally different: fallback returns -inf-clamped to -1? Use 1.0 vs 0.0
    # mse=1, data_range=1 -> 1 - 1/1 = 0. Bound: in [-1, 1].
    a = np.ones((4, 4, 3), dtype=np.float32)
    b = np.zeros((4, 4, 3), dtype=np.float32)
    val = ssim(a, b)
    assert -1.0 <= val <= 1.0
    assert val == pytest.approx(0.0, abs=1e-6)


def test_ssim_shape_mismatch_raises():
    a = np.zeros((4, 4, 3), dtype=np.float32)
    b = np.zeros((4, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim(a, b)


# ---------------------------------------------------------------------------
# RenderInputs / GPU paths
# ---------------------------------------------------------------------------

def test_render_view_requires_torch_and_gsplat():
    """Calling render_view on CPU host must fail with an import error, not silently."""
    from gs_pipeline.trainer.render_eval import RenderInputs, render_view
    inputs = RenderInputs(
        means=np.zeros((1, 3)), scales=np.zeros((1, 3)),
        quats=np.zeros((1, 4)), opacities=np.zeros((1,)),
        sh_dc=np.zeros((1, 3)), sh_rest=np.zeros((1, 0, 3)),
        sh_degree=0, full_sh_degree=0,
    )
    with pytest.raises((ImportError, ModuleNotFoundError, AttributeError, TypeError)):
        render_view(
            inputs, K=np.eye(3), w2c=np.eye(4),
            width=8, height=8, near_plane=0.1, far_plane=10.0,
        )


# ---------------------------------------------------------------------------
# _load_camera — K scaling (CPU-only, no rasterization)
# ---------------------------------------------------------------------------

def test_load_camera_scales_K_by_downscale():
    """_load_camera must scale the K matrix when downscale < 1.0."""
    K_native = np.array(
        [[1000.0, 0.0, 500.0],
         [0.0, 1000.0, 400.0],
         [0.0,    0.0,   1.0]],
        dtype=np.float64,
    )
    w2c_identity = np.eye(4, dtype=np.float64)
    scene = SimpleNamespace(
        K_per_camera=[K_native],
        w2c_per_camera=[w2c_identity],
        image_paths=["dummy_path.jpg"],
    )

    K, w2c, path = _load_camera(scene, 0, downscale=0.5)

    assert K[0, 0].item() == pytest.approx(500.0)   # fx * 0.5
    assert K[0, 2].item() == pytest.approx(250.0)   # cx * 0.5
    assert K[1, 1].item() == pytest.approx(500.0)   # fy * 0.5
    assert K[1, 2].item() == pytest.approx(200.0)   # cy * 0.5
    assert K[2, 2].item() == pytest.approx(1.0)     # homogeneous row unchanged
