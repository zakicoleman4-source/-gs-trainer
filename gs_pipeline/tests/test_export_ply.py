"""Round-trip + schema tests for ``gs_pipeline.trainer.export_ply``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData

from gs_pipeline.trainer.export_ply import (
    LoadedSplat,
    num_sh_rest_coeffs,
    read_inria_ply,
    write_inria_ply,
)


# ---------------------------------------------------------------------------
# num_sh_rest_coeffs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg,expected", [(0, 0), (1, 3), (2, 8), (3, 15)])
def test_num_sh_rest_coeffs(deg: int, expected: int):
    assert num_sh_rest_coeffs(deg) == expected


def test_num_sh_rest_coeffs_negative_raises():
    with pytest.raises(ValueError):
        num_sh_rest_coeffs(-1)


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------

def _make_splat_arrays(n: int, sh_degree: int, *, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 1, (n, 3)).astype(np.float32)
    scales = rng.normal(-3.0, 0.5, (n, 3)).astype(np.float32)  # log-scale
    quats = rng.normal(0, 1, (n, 4)).astype(np.float32)
    opacities = rng.normal(0.0, 1.0, (n,)).astype(np.float32)  # logits
    sh_dc = rng.normal(0, 0.3, (n, 3)).astype(np.float32)
    k = num_sh_rest_coeffs(sh_degree)
    sh_rest = rng.normal(0, 0.05, (n, k, 3)).astype(np.float32)
    return means, scales, quats, opacities, sh_dc, sh_rest


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_property_layout_at_sh3(tmp_path: Path):
    """Every expected property is present in the written PLY, in order."""
    n, deg = 50, 3
    out = tmp_path / "scene.ply"
    write_inria_ply(out_path=out, **_to_kwargs(*_make_splat_arrays(n, deg)))
    data = PlyData.read(str(out))
    v = data["vertex"]
    names = list(v.data.dtype.names)
    expected = (
        ["x", "y", "z", "nx", "ny", "nz",
         "f_dc_0", "f_dc_1", "f_dc_2"]
        + [f"f_rest_{i}" for i in range(3 * num_sh_rest_coeffs(deg))]
        + ["opacity",
           "scale_0", "scale_1", "scale_2",
           "rot_0", "rot_1", "rot_2", "rot_3"]
    )
    assert names == expected
    assert len(v) == n


def test_normals_are_zero(tmp_path: Path):
    out = tmp_path / "scene.ply"
    arrs = _make_splat_arrays(20, 3)
    write_inria_ply(out_path=out, **_to_kwargs(*arrs))
    v = PlyData.read(str(out))["vertex"]
    assert np.all(np.asarray(v["nx"]) == 0)
    assert np.all(np.asarray(v["ny"]) == 0)
    assert np.all(np.asarray(v["nz"]) == 0)


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0, 1, 2, 3])
def test_round_trip_preserves_all_arrays(deg: int, tmp_path: Path):
    n = 120
    means, scales, quats, opacities, sh_dc, sh_rest = _make_splat_arrays(n, deg, seed=deg)
    out = tmp_path / f"scene_deg{deg}.ply"
    write_inria_ply(
        out_path=out, means=means, scales=scales, quats=quats,
        opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest,
    )
    loaded = read_inria_ply(out)
    assert isinstance(loaded, LoadedSplat)
    assert loaded.sh_degree == deg
    np.testing.assert_allclose(loaded.means, means, atol=1e-6)
    np.testing.assert_allclose(loaded.scales, scales, atol=1e-6)
    np.testing.assert_allclose(loaded.quats, quats, atol=1e-6)
    np.testing.assert_allclose(loaded.opacities, opacities, atol=1e-6)
    np.testing.assert_allclose(loaded.sh_dc, sh_dc, atol=1e-6)
    np.testing.assert_allclose(loaded.sh_rest, sh_rest, atol=1e-6)


def test_sh_rest_layout_is_channel_major(tmp_path: Path):
    """Verify INRIA's f_rest_* ordering: R[0..K-1], G[0..K-1], B[0..K-1]."""
    n, deg = 5, 1  # K = 3 -> f_rest_0..f_rest_8 (9 numbers)
    sh_rest = np.zeros((n, num_sh_rest_coeffs(deg), 3), dtype=np.float32)
    # Encode the channel index into the value so we can spot the order in the file.
    for ch in range(3):
        sh_rest[:, :, ch] = (ch + 1) * 10 + np.arange(num_sh_rest_coeffs(deg))
    means = np.zeros((n, 3), dtype=np.float32)
    scales = np.zeros((n, 3), dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32); quats[:, 0] = 1.0
    opacities = np.zeros((n,), dtype=np.float32)
    sh_dc = np.zeros((n, 3), dtype=np.float32)
    out = tmp_path / "layout.ply"
    write_inria_ply(out_path=out, means=means, scales=scales, quats=quats,
                    opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest)
    v = PlyData.read(str(out))["vertex"]
    # f_rest_0 .. f_rest_2 should be R[0], R[1], R[2] = 10, 11, 12
    # f_rest_3 .. f_rest_5 should be G[0], G[1], G[2] = 20, 21, 22
    # f_rest_6 .. f_rest_8 should be B[0], B[1], B[2] = 30, 31, 32
    row0 = [float(v[f"f_rest_{i}"][0]) for i in range(9)]
    assert row0 == [10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0]


def test_ascii_form_also_round_trips(tmp_path: Path):
    n, deg = 10, 0  # tiny
    arrs = _make_splat_arrays(n, deg)
    out = tmp_path / "scene.ply"
    write_inria_ply(out_path=out, **_to_kwargs(*arrs), binary=False)
    text = out.read_text()
    assert "format ascii 1.0" in text
    loaded = read_inria_ply(out)
    np.testing.assert_allclose(loaded.means, arrs[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_wrong_means_shape_raises(tmp_path: Path):
    n, deg = 4, 2
    means, scales, quats, opacities, sh_dc, sh_rest = _make_splat_arrays(n, deg)
    bad_means = np.zeros((n, 4), dtype=np.float32)  # 4 cols, not 3
    with pytest.raises(ValueError, match="means shape"):
        write_inria_ply(
            out_path=tmp_path / "x.ply", means=bad_means, scales=scales,
            quats=quats, opacities=opacities, sh_dc=sh_dc, sh_rest=sh_rest,
        )


def test_wrong_sh_rest_dim_raises(tmp_path: Path):
    """sh_rest with non-integer-sqrt K is rejected."""
    n = 4
    means, scales, quats, opacities, sh_dc, _ = _make_splat_arrays(n, 0)
    bad = np.zeros((n, 4, 3), dtype=np.float32)  # K=4 -> deg sqrt(5)-1 not integer
    with pytest.raises(ValueError, match="not \\(deg\\+1\\)\\^2"):
        write_inria_ply(
            out_path=tmp_path / "x.ply", means=means, scales=scales,
            quats=quats, opacities=opacities, sh_dc=sh_dc, sh_rest=bad,
        )


def test_read_missing_required_property_raises(tmp_path: Path):
    """A PLY without the required INRIA properties should be rejected."""
    out = tmp_path / "minimal.ply"
    # Write a PLY with just xyz (no f_dc, no scale_*, no rot_*).
    from plyfile import PlyElement
    data = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(out))
    with pytest.raises(ValueError, match="missing properties"):
        read_inria_ply(out)


# ---------------------------------------------------------------------------
# Glue
# ---------------------------------------------------------------------------

def _to_kwargs(means, scales, quats, opacities, sh_dc, sh_rest) -> dict:
    return {
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "sh_dc": sh_dc, "sh_rest": sh_rest,
    }
