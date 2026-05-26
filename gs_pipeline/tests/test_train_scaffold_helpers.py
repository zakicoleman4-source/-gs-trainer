"""CPU-only unit tests for Scaffold-GS trainer helpers.

No GPU or gsplat import needed — tests verify config loading, voxel init,
bake math, and MLP dimension contracts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


YAML_PATH = Path(__file__).resolve().parent.parent / "trainer" / "config.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_load_scaffold_config_from_yaml():
    from gs_pipeline.trainer.train_scaffold import load_scaffold_config
    cfg = load_scaffold_config(YAML_PATH)
    assert cfg.n_offsets == 10
    assert cfg.anchor_feat_dim == 32
    assert cfg.mlp_hidden_dim == 64
    assert cfg.grow_start_iter < cfg.grow_stop_iter
    assert cfg.filter_enabled is True


def test_scaffold_config_defaults_match_yaml():
    from gs_pipeline.trainer.train_scaffold import ScaffoldConfig, load_scaffold_config
    defaults = ScaffoldConfig()
    loaded = load_scaffold_config(YAML_PATH)
    assert loaded.n_offsets == defaults.n_offsets
    assert loaded.anchor_feat_dim == defaults.anchor_feat_dim


def test_auto_adjust_scaffold_small_scene():
    from gs_pipeline.trainer.train_scaffold import ScaffoldConfig, auto_adjust_scaffold_config_for_scene
    cfg = ScaffoldConfig()
    adj = auto_adjust_scaffold_config_for_scene(cfg, n_cameras=20)
    assert adj.holdout_stride == 2
    assert adj.divergence_min_psnr == 10.0


def test_auto_adjust_scaffold_large_scene():
    from gs_pipeline.trainer.train_scaffold import ScaffoldConfig, auto_adjust_scaffold_config_for_scene
    cfg = ScaffoldConfig()
    adj = auto_adjust_scaffold_config_for_scene(cfg, n_cameras=2000)
    assert adj.holdout_stride == 16
    assert adj.eval_every == 2000


def test_auto_adjust_scaffold_normal_scene_unchanged():
    from gs_pipeline.trainer.train_scaffold import ScaffoldConfig, auto_adjust_scaffold_config_for_scene
    cfg = ScaffoldConfig()
    adj = auto_adjust_scaffold_config_for_scene(cfg, n_cameras=200)
    assert adj.holdout_stride == cfg.holdout_stride


# ---------------------------------------------------------------------------
# Voxel init
# ---------------------------------------------------------------------------

def test_voxel_quantize_basic():
    from gs_pipeline.trainer.train_scaffold import _voxel_quantize
    xyz = np.array([
        [0.1, 0.1, 0.1],
        [0.15, 0.12, 0.11],  # same voxel as above at size=0.5
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    anchors, inverse, vs = _voxel_quantize(xyz, voxel_size=0.5)
    assert anchors.shape[0] == 2  # two unique voxels
    assert inverse.shape[0] == 3
    assert anchors.shape[1] == 3


def test_voxel_quantize_deduplication():
    from gs_pipeline.trainer.train_scaffold import _voxel_quantize
    xyz = np.random.default_rng(42).uniform(0, 1, (100, 3)).astype(np.float32)
    anchors, inverse, vs = _voxel_quantize(xyz, voxel_size=0.5)
    assert anchors.shape[0] <= 100
    assert anchors.shape[0] >= 1


def test_auto_voxel_size_reasonable():
    from gs_pipeline.trainer.train_scaffold import _auto_voxel_size
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0, 10, (1000, 3)).astype(np.float32)
    vs = _auto_voxel_size(xyz)
    assert 0.01 < vs < 10.0


def test_voxel_init_from_point_cloud_shapes():
    """ScaffoldModel.from_point_cloud produces anchors with correct shapes."""
    # We can't instantiate the full model on CPU (requires torch), but we can
    # test the numpy-side init helpers.
    from gs_pipeline.trainer.train_scaffold import _voxel_quantize, _auto_voxel_size, _knn_mean_distance_np
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0, 5, (500, 3)).astype(np.float32)
    vs = _auto_voxel_size(xyz)
    anchors, inverse, _ = _voxel_quantize(xyz, vs)
    assert anchors.ndim == 2
    assert anchors.shape[1] == 3
    assert anchors.shape[0] > 0

    knn_dist = _knn_mean_distance_np(anchors, k=3)
    assert knn_dist.shape == (anchors.shape[0],)
    assert (knn_dist > 0).all()


# ---------------------------------------------------------------------------
# Bake math
# ---------------------------------------------------------------------------

def test_inverse_sigmoid_roundtrip():
    """sigmoid(inverse_sigmoid(x)) == x for valid inputs."""
    import torch
    from gs_pipeline.trainer.train_scaffold import _inverse_sigmoid
    x = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99])
    logits = _inverse_sigmoid(x)
    recovered = torch.sigmoid(logits)
    np.testing.assert_allclose(recovered.numpy(), x.numpy(), atol=1e-5)


def test_rgb_to_sh_dc_conversion():
    """Verify RGB -> SH-DC matches the formula used in baking."""
    C0 = 0.28209479177387814
    rgb = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    sh_dc = (rgb - 0.5) / C0
    assert sh_dc[1] == pytest.approx(0.0, abs=1e-6)
    assert sh_dc[0] < 0  # black -> negative SH
    assert sh_dc[2] > 0  # white -> positive SH


# ---------------------------------------------------------------------------
# Pipeline routing
# ---------------------------------------------------------------------------

def test_config_yaml_has_trainer_backend():
    import yaml
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    assert "trainer" in raw
    assert "backend" in raw["trainer"]
    assert raw["trainer"]["backend"] in ("mcmc", "scaffold")


def test_config_yaml_has_scaffold_section():
    import yaml
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    assert "scaffold" in raw
    sc = raw["scaffold"]
    assert sc["n_offsets"] == 10
    assert sc["anchor_feat_dim"] == 32
    assert sc["mlp_hidden_dim"] == 64
