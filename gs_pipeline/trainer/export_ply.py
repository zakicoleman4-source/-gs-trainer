"""Write / read 3DGS-style ``.ply`` files in the INRIA layout that SuperSplat,
Polycam, gsplat.js and the original Inria viewer all consume.

The on-disk schema is per-vertex little-endian binary floats with these
properties, in this order::

    x, y, z, nx, ny, nz,
    f_dc_0, f_dc_1, f_dc_2,                 # SH DC band (R, G, B)
    f_rest_0 ... f_rest_(3K-1),             # SH bands 1..deg, K = (deg+1)^2 - 1
    opacity,                                # logit (pre-sigmoid)
    scale_0, scale_1, scale_2,              # log-scale (pre-exp)
    rot_0, rot_1, rot_2, rot_3              # quaternion (w, x, y, z)

The SH-rest layout is **channel-major within each splat**: for K rest coeffs
that's ``(R[0], R[1], ..., R[K-1], G[0], ..., G[K-1], B[0], ..., B[K-1])``.
This matches the original INRIA ``gaussian_model.save_ply`` (which does
``features_rest.transpose(1, 2).flatten(start_dim=1)``).

Scales and opacities are written as the raw optimiser parameters (log /
logit); viewers apply ``exp()`` / ``sigmoid()`` at render time. Normals are
written as zero (3DGS doesn't carry true surface normals — the field exists
only to keep some legacy tools happy).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

_log = logging.getLogger(__name__)


def num_sh_rest_coeffs(sh_degree: int) -> int:
    """Number of higher-band SH coefficients per channel for a given degree.

    Total band count is ``(deg+1)^2``; the DC band takes 1, the rest take
    ``(deg+1)^2 - 1``. We write ``3 * that`` floats per splat under
    ``f_rest_*`` (channel-major across R, G, B).
    """
    if sh_degree < 0:
        raise ValueError(f"sh_degree must be >= 0; got {sh_degree}")
    return (sh_degree + 1) ** 2 - 1


def _rest_property_names(sh_degree: int) -> list[str]:
    k = num_sh_rest_coeffs(sh_degree)
    return [f"f_rest_{i}" for i in range(3 * k)]


def _validate_shapes(
    means: np.ndarray, scales: np.ndarray, quats: np.ndarray,
    opacities: np.ndarray, sh_dc: np.ndarray, sh_rest: np.ndarray,
) -> tuple[int, int]:
    """Return (n_splats, sh_degree) inferred from sh_rest's middle dim."""
    n = means.shape[0]
    for name, arr, expected in (
        ("means", means, (n, 3)),
        ("scales", scales, (n, 3)),
        ("quats", quats, (n, 4)),
        ("opacities", opacities, (n,)),
        ("sh_dc", sh_dc, (n, 3)),
    ):
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape} != {expected}")
    if sh_rest.ndim != 3 or sh_rest.shape[0] != n or sh_rest.shape[2] != 3:
        raise ValueError(
            f"sh_rest shape {sh_rest.shape} != ({n}, K, 3) for some K"
        )
    k = int(sh_rest.shape[1])
    # K = (deg+1)^2 - 1 -> deg = sqrt(K+1) - 1; must be a non-negative integer.
    deg_f = (k + 1) ** 0.5 - 1
    deg = int(round(deg_f))
    if (deg + 1) ** 2 - 1 != k:
        raise ValueError(
            f"sh_rest has {k} coeffs per splat; not (deg+1)^2 - 1 for any integer degree"
        )
    return n, deg


def write_inria_ply(
    *,
    out_path: Path,
    means: np.ndarray,        # (N, 3) float
    scales: np.ndarray,       # (N, 3) float (log-scale)
    quats: np.ndarray,        # (N, 4) float (w, x, y, z; need not be normalised)
    opacities: np.ndarray,    # (N,) float (logit)
    sh_dc: np.ndarray,        # (N, 3) float
    sh_rest: np.ndarray,      # (N, K, 3) float
    binary: bool = True,
) -> Path:
    """Persist a Gaussian splat scene as an INRIA-layout PLY."""
    n, sh_degree = _validate_shapes(means, scales, quats, opacities, sh_dc, sh_rest)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rest_names = _rest_property_names(sh_degree)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        *[(name, "f4") for name in rest_names],
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]

    data = np.zeros(n, dtype=dtype)
    data["x"] = means[:, 0]; data["y"] = means[:, 1]; data["z"] = means[:, 2]
    # nx/ny/nz already zero from np.zeros.
    data["f_dc_0"] = sh_dc[:, 0]; data["f_dc_1"] = sh_dc[:, 1]; data["f_dc_2"] = sh_dc[:, 2]

    # Channel-major rest: (N, K, 3) -> (N, 3, K) -> (N, 3*K)
    if n > 0 and rest_names:
        rest_cm = np.transpose(sh_rest, (0, 2, 1)).reshape(n, -1)
        for idx, name in enumerate(rest_names):
            data[name] = rest_cm[:, idx]

    data["opacity"] = opacities
    data["scale_0"] = scales[:, 0]; data["scale_1"] = scales[:, 1]; data["scale_2"] = scales[:, 2]
    data["rot_0"] = quats[:, 0]; data["rot_1"] = quats[:, 1]
    data["rot_2"] = quats[:, 2]; data["rot_3"] = quats[:, 3]

    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=not binary).write(str(out_path))
    _log.info("wrote %d splats (sh_degree=%d) to %s", n, sh_degree, out_path)
    return out_path


@dataclass
class LoadedSplat:
    means: np.ndarray         # (N, 3) f32
    scales: np.ndarray        # (N, 3) f32
    quats: np.ndarray         # (N, 4) f32
    opacities: np.ndarray     # (N,)   f32
    sh_dc: np.ndarray         # (N, 3) f32
    sh_rest: np.ndarray       # (N, K, 3) f32
    sh_degree: int


def read_inria_ply(path: Path) -> LoadedSplat:
    """Inverse of ``write_inria_ply``. Mainly for tests / sanity checks."""
    path = Path(path)
    data = PlyData.read(str(path))
    if "vertex" not in [el.name for el in data.elements]:
        raise ValueError(f"{path}: no 'vertex' element")
    v = data["vertex"]
    names = set(v.data.dtype.names or ())
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = required - names
    if missing:
        raise ValueError(f"{path}: missing properties {sorted(missing)}")

    n = len(v)
    means = np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], axis=1).astype(np.float32)
    scales = np.stack([np.asarray(v["scale_0"]), np.asarray(v["scale_1"]), np.asarray(v["scale_2"])], axis=1).astype(np.float32)
    quats = np.stack([np.asarray(v["rot_0"]), np.asarray(v["rot_1"]),
                      np.asarray(v["rot_2"]), np.asarray(v["rot_3"])], axis=1).astype(np.float32)
    opacities = np.asarray(v["opacity"], dtype=np.float32)
    sh_dc = np.stack([np.asarray(v["f_dc_0"]), np.asarray(v["f_dc_1"]), np.asarray(v["f_dc_2"])], axis=1).astype(np.float32)

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        rest_cm = np.stack([np.asarray(v[name]) for name in rest_names], axis=1).astype(np.float32)  # (N, 3K)
        if rest_cm.shape[1] % 3 != 0:
            raise ValueError(f"f_rest_* count {rest_cm.shape[1]} is not divisible by 3")
        k = rest_cm.shape[1] // 3
        sh_rest = rest_cm.reshape(n, 3, k).transpose(0, 2, 1).astype(np.float32)
    else:
        k = 0
        sh_rest = np.zeros((n, 0, 3), dtype=np.float32)

    sh_degree = int(round((k + 1) ** 0.5 - 1))
    if (sh_degree + 1) ** 2 - 1 != k:
        raise ValueError(f"f_rest count {k} per channel does not match any sh_degree")

    return LoadedSplat(
        means=means, scales=scales, quats=quats, opacities=opacities,
        sh_dc=sh_dc, sh_rest=sh_rest, sh_degree=sh_degree,
    )


_SPLAT_BINARY_DTYPE = np.dtype([
    ("x",  "<f4"), ("y",  "<f4"), ("z",  "<f4"),
    ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
    ("r",  "u1"),  ("g",  "u1"),  ("b",  "u1"),  ("a",  "u1"),
    ("qw", "u1"),  ("qx", "u1"),  ("qy", "u1"),  ("qz", "u1"),
])  # 32 bytes per splat — antimatter15 / SuperSplat binary format


def write_splat_binary(
    *,
    out_path: Path,
    means: np.ndarray,       # (N, 3) float
    scales: np.ndarray,      # (N, 3) float (log-scale)
    quats: np.ndarray,       # (N, 4) float (w, x, y, z; need not be normalised)
    opacities: np.ndarray,   # (N,) float (logit)
    sh_dc: np.ndarray,       # (N, 3) float
) -> Path:
    """Write a 32-byte-per-splat ``.splat`` binary (antimatter15 / SuperSplat format).

    Smaller (2-3×) than PLY; loads directly in web viewers.  SH rest bands are
    discarded — the file encodes DC colour only, which is what the compact format
    supports.
    """
    C0 = 0.28209479177387814
    N = means.shape[0]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.zeros(N, dtype=_SPLAT_BINARY_DTYPE)

    data["x"] = means[:, 0].astype(np.float32)
    data["y"] = means[:, 1].astype(np.float32)
    data["z"] = means[:, 2].astype(np.float32)

    # Actual scale (exp applied — viewers do NOT apply exp again).
    data["sx"] = np.exp(scales[:, 0]).astype(np.float32)
    data["sy"] = np.exp(scales[:, 1]).astype(np.float32)
    data["sz"] = np.exp(scales[:, 2]).astype(np.float32)

    # RGB from DC SH: C0 * sh_dc + 0.5 clamped to [0, 1] then to uint8.
    rgb = np.clip(C0 * sh_dc.astype(np.float32) + 0.5, 0.0, 1.0)
    data["r"] = (rgb[:, 0] * 255.0 + 0.5).astype(np.uint8)
    data["g"] = (rgb[:, 1] * 255.0 + 0.5).astype(np.uint8)
    data["b"] = (rgb[:, 2] * 255.0 + 0.5).astype(np.uint8)

    # Alpha: sigmoid(logit) mapped to [0, 255].
    logits = opacities.astype(np.float32).clip(-20.0, 20.0)
    alpha_f = 1.0 / (1.0 + np.exp(-logits))
    data["a"] = (alpha_f * 255.0 + 0.5).astype(np.uint8)

    # Quaternion (w, x, y, z): normalise then map [-1, 1] → [0, 255].
    q = quats.astype(np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
    q /= norms
    data["qw"] = np.clip(q[:, 0] * 128.0 + 128.0, 0.0, 255.0).astype(np.uint8)
    data["qx"] = np.clip(q[:, 1] * 128.0 + 128.0, 0.0, 255.0).astype(np.uint8)
    data["qy"] = np.clip(q[:, 2] * 128.0 + 128.0, 0.0, 255.0).astype(np.uint8)
    data["qz"] = np.clip(q[:, 3] * 128.0 + 128.0, 0.0, 255.0).astype(np.uint8)

    data.tofile(str(out_path))
    _log.info("wrote %d splats to %s (%.1f MB)", N, out_path, out_path.stat().st_size / 1e6)
    return out_path
