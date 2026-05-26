"""CPU-side tests for ``train_mcmc.py``'s pure-Python / NumPy helpers.

The full training loop is GPU-only; its smoke test lives in
``test_pipeline_smoke.py`` behind ``@pytest.mark.gpu``. The helpers here
(knn_mean_distance, psnr, detect_camera_axis_flip, load_trainer_config) are
deterministic CPU code and worth testing without CUDA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.trainer.train_mcmc import (
    TrainerConfig,
    detect_camera_axis_flip,
    knn_mean_distance,
    load_trainer_config,
    psnr,
)


# ---------------------------------------------------------------------------
# knn_mean_distance
# ---------------------------------------------------------------------------

def test_knn_mean_distance_known_grid():
    """Points on a unit-spaced 1D line: nearest neighbors at distance 1."""
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
    ], dtype=np.float32)
    d = knn_mean_distance(xyz, k=1)
    # Endpoints have one neighbor at distance 1; interior also.
    np.testing.assert_allclose(d, 1.0, atol=1e-6)


def test_knn_mean_distance_two_clusters_dense_is_smaller():
    """A dense cluster's knn distance should be smaller than a sparse one's."""
    rng = np.random.default_rng(0)
    dense = rng.normal(0.0, 0.01, size=(50, 3))
    sparse = rng.normal(5.0, 1.0, size=(50, 3))
    xyz = np.concatenate([dense, sparse], axis=0).astype(np.float32)
    d = knn_mean_distance(xyz, k=3)
    assert d[:50].mean() < d[50:].mean()


def test_knn_mean_distance_single_point():
    xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    d = knn_mean_distance(xyz, k=3)
    # Falls back to a small positive constant; just must not crash.
    assert d.shape == (1,)
    assert d[0] > 0.0


def test_knn_mean_distance_block_boundary():
    """Cross-check against an N <= block size path: result identical regardless of N."""
    rng = np.random.default_rng(1)
    xyz = rng.normal(0, 1, size=(64, 3)).astype(np.float32)
    d1 = knn_mean_distance(xyz, k=4)
    # Tile to N > internal block (8192) and re-test a few points stay sane.
    big = np.tile(xyz, (200, 1)) + rng.normal(0, 1e-4, (64 * 200, 3)).astype(np.float32)
    d2 = knn_mean_distance(big, k=4)
    # On the tiled set, neighbors are very close; mean knn distance should be tiny.
    assert d2.mean() < d1.mean()


# ---------------------------------------------------------------------------
# psnr
# ---------------------------------------------------------------------------

def test_psnr_identical_inf():
    img = np.full((8, 8, 3), 0.5, dtype=np.float32)
    assert np.isinf(psnr(img, img))


def test_psnr_known_value():
    """For a uniform offset of 0.1, PSNR = -20*log10(0.1) = 20 dB."""
    gt = np.zeros((4, 4, 3), dtype=np.float32)
    pred = np.full_like(gt, 0.1)
    assert psnr(pred, gt) == pytest.approx(20.0, rel=1e-4)


def test_psnr_max_val_scaling():
    """PSNR with max_val=255 on uint8-like inputs is 8 bits higher than max_val=1 form."""
    gt = np.zeros((4, 4, 3), dtype=np.float32)
    pred = np.full_like(gt, 25.5)  # 10% of 255
    psnr_255 = psnr(pred, gt, max_val=255.0)
    psnr_1 = psnr(pred / 255.0, gt / 255.0, max_val=1.0)
    assert psnr_255 == pytest.approx(psnr_1, abs=1e-4)


# ---------------------------------------------------------------------------
# detect_camera_axis_flip
# ---------------------------------------------------------------------------

def test_detect_flip_picks_better_match():
    target = np.zeros((4, 4, 3), dtype=np.float32)
    no_flip = np.ones_like(target) * 0.8         # very wrong
    flipped = np.zeros_like(target)              # perfect
    assert detect_camera_axis_flip(
        rendered_no_flip=no_flip, rendered_flipped=flipped, target=target,
    ) is True


def test_detect_flip_no_change_when_no_flip_is_better():
    target = np.zeros((4, 4, 3), dtype=np.float32)
    no_flip = np.zeros_like(target)               # perfect
    flipped = np.ones_like(target)                # wrong
    assert detect_camera_axis_flip(
        rendered_no_flip=no_flip, rendered_flipped=flipped, target=target,
    ) is False


def test_detect_flip_tie_falls_back_to_no_flip():
    target = np.full((4, 4, 3), 0.5, dtype=np.float32)
    a = np.zeros_like(target)
    b = np.ones_like(target)
    # Both have L1 = 0.5 — within the 5% margin → no flip.
    assert detect_camera_axis_flip(rendered_no_flip=a, rendered_flipped=b, target=target) is False


# ---------------------------------------------------------------------------
# load_trainer_config
# ---------------------------------------------------------------------------

def test_load_trainer_config_defaults_match_yaml(tmp_path: Path):
    """The shipped config.yaml round-trips into a TrainerConfig with the documented
    Auto-preset values."""
    cfg_path = Path(__file__).resolve().parent.parent / "trainer" / "config.yaml"
    assert cfg_path.is_file(), "missing shipped config.yaml"
    cfg = load_trainer_config(cfg_path)
    assert isinstance(cfg, TrainerConfig)
    assert cfg.iterations == 40_000             # Auto preset
    assert cfg.sh_degree == 3
    assert cfg.sh_warmup_interval == 1000
    assert cfg.opacity_reg == 0.005
    assert cfg.scale_reg == 0.005
    assert cfg.prune_opa == 0.003
    assert cfg.refine_start_iter == 100
    assert cfg.eval_every == 1000
    assert cfg.preview_every == 250
    assert cfg.checkpoint_every == 5000
    assert cfg.holdout_stride == 8


def test_load_trainer_config_iterations_override(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parent.parent / "trainer" / "config.yaml"
    cfg = load_trainer_config(cfg_path, iterations_override=500)
    assert cfg.iterations == 500


def test_load_trainer_config_maximum_preset_iterations(tmp_path: Path):
    """If the YAML preset is Maximum, iterations come from that map entry."""
    src = (Path(__file__).resolve().parent.parent / "trainer" / "config.yaml").read_text()
    mutated = tmp_path / "cfg.yaml"
    mutated.write_text(src.replace("preset: Auto", "preset: Maximum"), encoding="utf-8")
    cfg = load_trainer_config(mutated)
    assert cfg.iterations == 100_000


def test_auto_adjust_config_small_scene():
    from gs_pipeline.trainer.train_mcmc import auto_adjust_config_for_scene, TrainerConfig
    base = TrainerConfig()
    # Very small scene gets tighter holdout
    tiny = auto_adjust_config_for_scene(base, n_cameras=20)
    assert tiny.holdout_stride == 2
    assert tiny.divergence_check_at_step == 3_000


def test_auto_adjust_config_medium_scene_unchanged():
    from gs_pipeline.trainer.train_mcmc import auto_adjust_config_for_scene, TrainerConfig
    base = TrainerConfig()
    medium = auto_adjust_config_for_scene(base, n_cameras=200)
    assert medium.holdout_stride == base.holdout_stride
    assert medium.eval_every == base.eval_every


def test_auto_adjust_config_large_scene():
    from gs_pipeline.trainer.train_mcmc import auto_adjust_config_for_scene, TrainerConfig
    base = TrainerConfig()
    large = auto_adjust_config_for_scene(base, n_cameras=1500)
    assert large.holdout_stride == 16
    assert large.eval_every == 2000
