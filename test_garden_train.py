"""
400-iteration integration test on the real garden_3 bundle.
Trains a quick smoke scene, then runs filter_splats at 5 aggressiveness levels
and reports the splat count + per-stage breakdown for each.

Run (requires GPU + bundle on external drive):
    pytest test_garden_train.py -v -m gpu -s

Skip automatically when the bundle isn't mounted.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

BUNDLE = Path("/media/tarbut/cAVS-132/garden_3_splat_bundle.zip")

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def garden_ply(tmp_path_factory):
    """Train 400 iterations on the garden bundle; yield the raw output PLY."""
    if not BUNDLE.exists():
        pytest.skip(f"Bundle not found: {BUNDLE}")

    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from gs_pipeline.trainer.pipeline import run_pipeline
    from gs_pipeline.trainer.train_mcmc import TrainConfig

    work = tmp_path_factory.mktemp("garden_work")
    outbox = tmp_path_factory.mktemp("garden_out")

    # Minimal config override: only 400 iters, no preview strip, filter disabled
    # (we apply filters manually below to compare levels).
    overrides = {
        "iterations": 400,
        "refine_start_iter": 50,
        "eval_every": 400,
        "preview_every": 9999,
        "filter": {"enabled": False},
    }

    result = run_pipeline(
        bundle_path=BUNDLE,
        work_dir=work,
        outbox_dir=outbox,
        config_overrides=overrides,
    )

    assert result is not None, "pipeline returned None"
    ply = outbox / "scene.ply"
    assert ply.exists(), f"scene.ply not written to {outbox}"
    return ply


# ---------------------------------------------------------------------------
# Filter level definitions (none → maximum)
# ---------------------------------------------------------------------------
FILTER_LEVELS = [
    {
        "label": "none",
        "enabled": False,
    },
    {
        "label": "light",
        "enabled": True,
        "min_opacity": 0.001,
        "sor_std_ratio": 3.5,
        "max_scale_factor": 15.0,
    },
    {
        "label": "medium (default)",
        "enabled": True,
        "min_opacity": 0.005,
        "sor_std_ratio": 2.0,
        "max_scale_factor": 10.0,
    },
    {
        "label": "aggressive",
        "enabled": True,
        "min_opacity": 0.01,
        "sor_std_ratio": 1.5,
        "max_scale_factor": 7.0,
    },
    {
        "label": "maximum",
        "enabled": True,
        "min_opacity": 0.02,
        "sor_std_ratio": 1.0,
        "max_scale_factor": 5.0,
    },
]


def _load_splat_arrays(ply_path: Path):
    """Return (means, scales, quats, opacities, sh_dc, sh_rest) from a PLY."""
    from gs_pipeline.trainer.export_ply import load_ply
    return load_ply(ply_path)


@pytest.mark.parametrize("level", FILTER_LEVELS, ids=[l["label"] for l in FILTER_LEVELS])
def test_filter_level(garden_ply, level):
    """Apply one filter level; assert splat count is sane and nothing crashes."""
    from gs_pipeline.trainer.filter_splats import filter_scene

    arrays = _load_splat_arrays(garden_ply)
    means, scales, quats, opacities, sh_dc, sh_rest = arrays
    n_raw = len(means)

    if not level["enabled"]:
        print(f"\n[none]  raw splats = {n_raw:,}")
        assert n_raw > 0
        return

    filtered, report = filter_scene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh_dc=sh_dc,
        sh_rest=sh_rest,
        min_opacity=level["min_opacity"],
        sor_k=20,
        sor_std_ratio=level["sor_std_ratio"],
        max_scale_factor=level["max_scale_factor"],
    )

    n_after = len(filtered[0])
    pct_kept = 100.0 * n_after / n_raw

    print(
        f"\n[{level['label']}]  raw={n_raw:,}  "
        f"after_opacity={report.n_after_opacity:,}  "
        f"after_scale={report.n_after_scale:,}  "
        f"after_sor={report.n_after_sor:,}  "
        f"kept={pct_kept:.1f}%"
    )

    # Sanity: should keep at least 30% even on maximum settings
    assert n_after > 0, "filter removed everything"
    assert pct_kept > 30.0, f"filter too aggressive: only {pct_kept:.1f}% kept"

    # Counts must be monotonically non-increasing through filter stages
    assert report.n_after_opacity <= n_raw
    assert report.n_after_scale <= report.n_after_opacity
    assert report.n_after_sor <= report.n_after_scale
    assert n_after == report.n_after_sor
