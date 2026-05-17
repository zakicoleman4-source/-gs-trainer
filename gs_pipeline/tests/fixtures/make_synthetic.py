"""Synthetic Metashape-like export bundle for tests.

Builds, in a target directory:
    bundle/
        cameras.xml      # Agisoft schema, frame sensor, undistorted (k* = p* = 0)
        dense.ply        # ascii PLY, colored points on a unit cube
        masks/           # (empty, reserved)
        images/
            cam_000.png .. cam_(N-1).png   # rendered via pinhole projection
        manifest.json    # bundle metadata the trainer reads

The geometry is an 8-camera ring on radius `cam_radius` looking at the origin,
with images at `image_size x image_size`. The cube is centered at the origin
with half-extent `cube_extent`. f is chosen so the cube fills ~40% of the view.

This module is importable from tests (`from gs_pipeline.tests.fixtures.make_synthetic
import build_bundle`) and is also runnable as a script:

    python -m gs_pipeline.tests.fixtures.make_synthetic --out /tmp/synth_bundle
    python -m gs_pipeline.tests.fixtures.make_synthetic --out /tmp/bundle.zip --zip
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _look_at_w2c(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """OpenCV-convention world-to-camera matrix (+X right, +Y down, +Z forward).

    Returns a 4x4 homogeneous matrix.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # OpenCV: y points down. Right-handed: right = forward x up_world; down = forward x right.
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    R = np.stack([right, down, forward], axis=0)  # rows = camera axes in world
    t = -R @ eye
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _w2c_to_metashape_transform(w2c: np.ndarray) -> np.ndarray:
    """Agisoft `<transform>` is camera->chunk (i.e. c2w). Invert w2c."""
    return np.linalg.inv(w2c)


def _cube_point_cloud(n_points: int, half_extent: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample points on the 6 faces of a cube centered at origin.

    Returns (xyz [N,3] float32, rgb [N,3] uint8).
    """
    per_face = n_points // 6
    pts = []
    cols = []
    # Face colors: +X red, -X cyan, +Y green, -Y magenta, +Z blue, -Z yellow.
    face_colors = np.array([
        [220, 40, 40],
        [40, 220, 220],
        [40, 220, 40],
        [220, 40, 220],
        [40, 40, 220],
        [220, 220, 40],
    ], dtype=np.uint8)
    axes = [0, 0, 1, 1, 2, 2]
    signs = [+1, -1, +1, -1, +1, -1]
    for face_idx, (axis, sign) in enumerate(zip(axes, signs)):
        uv = rng.uniform(-half_extent, half_extent, size=(per_face, 2))
        face = np.zeros((per_face, 3), dtype=np.float32)
        # The fixed-axis coord is +/- half_extent; the other two are uv.
        fixed_val = sign * half_extent
        other = [i for i in range(3) if i != axis]
        face[:, axis] = fixed_val
        face[:, other[0]] = uv[:, 0]
        face[:, other[1]] = uv[:, 1]
        pts.append(face)
        cols.append(np.tile(face_colors[face_idx], (per_face, 1)))
    xyz = np.concatenate(pts, axis=0).astype(np.float32)
    rgb = np.concatenate(cols, axis=0).astype(np.uint8)
    # Shuffle so faces interleave (more representative downstream).
    idx = rng.permutation(xyz.shape[0])
    return xyz[idx], rgb[idx]


def _project_points(
    xyz: np.ndarray,
    rgb: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a colored PNG by splatting each cube point as a 2px disc.

    Background is gray (128). Points behind the camera (z<=0) are skipped.
    """
    N = xyz.shape[0]
    pts_h = np.concatenate([xyz, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (w2c @ pts_h.T).T  # [N,4]
    z = cam[:, 2]
    visible = z > 1e-3
    cam = cam[visible]
    cols = rgb[visible]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    img = np.full((height, width, 3), 128, dtype=np.uint8)
    radius = 1
    for ui, vi, c in zip(u, v, cols):
        ui_i = int(round(float(ui)))
        vi_i = int(round(float(vi)))
        if 0 <= ui_i < width and 0 <= vi_i < height:
            u0 = max(0, ui_i - radius)
            u1 = min(width, ui_i + radius + 1)
            v0 = max(0, vi_i - radius)
            v1 = min(height, vi_i + radius + 1)
            img[v0:v1, u0:u1] = c
    return img


# ---------------------------------------------------------------------------
# Agisoft cameras.xml writer
# ---------------------------------------------------------------------------

def _format_matrix(mat: np.ndarray) -> str:
    return " ".join(f"{v:.12e}" for v in mat.flatten())


def _write_cameras_xml(
    out_path: Path,
    *,
    width: int,
    height: int,
    f: float,
    cx: float,
    cy: float,
    camera_transforms_c2w: list[np.ndarray],
    camera_labels: list[str],
) -> None:
    """Write a minimal but Agisoft-shaped cameras.xml.

    Schema follows what `parse_metashape.py` consumes: one sensor (frame, adjusted),
    a list of <camera> elements with a <transform> child holding the 4x4 c2w matrix.
    The chunk-level <transform> is identity.
    """
    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<document version="2.0.0">')
    lines.append('  <chunk label="synthetic" enabled="true">')
    lines.append('    <sensors next_id="1">')
    lines.append('      <sensor id="0" label="synthetic_cam" type="frame">')
    lines.append(f'        <resolution width="{width}" height="{height}"/>')
    lines.append('        <property name="layer_index" value="0"/>')
    lines.append('        <calibration type="frame" class="adjusted">')
    lines.append(f'          <resolution width="{width}" height="{height}"/>')
    lines.append(f'          <f>{f:.12e}</f>')
    lines.append(f'          <cx>{cx:.12e}</cx>')
    lines.append(f'          <cy>{cy:.12e}</cy>')
    lines.append('          <k1>0</k1><k2>0</k2><k3>0</k3><k4>0</k4>')
    lines.append('          <p1>0</p1><p2>0</p2><p3>0</p3><p4>0</p4>')
    lines.append('          <b1>0</b1><b2>0</b2>')
    lines.append('        </calibration>')
    lines.append('      </sensor>')
    lines.append('    </sensors>')
    lines.append(f'    <cameras next_id="{len(camera_transforms_c2w)}">')
    for i, (M, label) in enumerate(zip(camera_transforms_c2w, camera_labels)):
        lines.append(f'      <camera id="{i}" sensor_id="0" label="{label}" enabled="true">')
        lines.append(f'        <transform>{_format_matrix(M)}</transform>')
        lines.append('      </camera>')
    lines.append('    </cameras>')
    # Chunk-level identity transform.
    identity_R = np.eye(3).flatten()
    lines.append('    <transform>')
    lines.append(f'      <rotation>{" ".join(f"{v:.1f}" for v in identity_R)}</rotation>')
    lines.append('      <translation>0 0 0</translation>')
    lines.append('      <scale>1</scale>')
    lines.append('    </transform>')
    lines.append('  </chunk>')
    lines.append('</document>')
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _write_dense_ply(out_path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(xyz.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["red"] = rgb[:, 0]
    data["green"] = rgb[:, 1]
    data["blue"] = rgb[:, 2]
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(out_path))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class SyntheticBundle:
    root: Path
    n_cameras: int
    image_size: int
    n_points: int
    K: np.ndarray
    w2c_per_camera: list[np.ndarray]


def build_bundle(
    out_dir: Path,
    *,
    n_cameras: int = 8,
    image_size: int = 512,
    n_points: int = 5000,
    cam_radius: float = 2.0,
    cube_extent: float = 0.5,
    seed: int = 0,
) -> SyntheticBundle:
    """Build a synthetic Metashape-like bundle under `out_dir/`.

    Returns metadata so tests can assert against known-good values.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "masks").mkdir()

    rng = np.random.default_rng(seed)

    # Intrinsics: principal point at image center => Agisoft cx=cy=0.
    f = image_size * 1.4  # cube ~40% of frame at radius 2
    K = np.array([
        [f, 0.0, image_size / 2.0],
        [0.0, f, image_size / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Cameras on a ring in the XY plane, looking at origin. Up is +Z.
    up = np.array([0.0, 0.0, 1.0])
    w2cs: list[np.ndarray] = []
    c2ws: list[np.ndarray] = []
    labels: list[str] = []
    for i in range(n_cameras):
        theta = 2 * math.pi * i / n_cameras
        eye = np.array([cam_radius * math.cos(theta), cam_radius * math.sin(theta), 0.0])
        target = np.zeros(3)
        w2c = _look_at_w2c(eye, target, up)
        w2cs.append(w2c)
        c2ws.append(_w2c_to_metashape_transform(w2c))
        labels.append(f"cam_{i:03d}.png")

    # Cube cloud.
    xyz, rgb = _cube_point_cloud(n_points=n_points, half_extent=cube_extent, rng=rng)

    # Render ground-truth images.
    for label, w2c in zip(labels, w2cs):
        img = _project_points(xyz, rgb, K, w2c, image_size, image_size)
        Image.fromarray(img).save(out_dir / "images" / label)

    # Write Metashape-shaped artifacts.
    _write_cameras_xml(
        out_dir / "cameras.xml",
        width=image_size,
        height=image_size,
        f=f,
        cx=0.0,
        cy=0.0,
        camera_transforms_c2w=c2ws,
        camera_labels=labels,
    )
    _write_dense_ply(out_dir / "dense.ply", xyz, rgb)

    manifest = {
        "format": "metashape_splat_bundle/v1",
        "source": "synthetic",
        "n_cameras": n_cameras,
        "image_size": [image_size, image_size],
        "dense_points": int(xyz.shape[0]),
        "intrinsics": {
            "f": float(f),
            "cx": 0.0,
            "cy": 0.0,
            "width": image_size,
            "height": image_size,
        },
        "scene_extent_hint": float(2 * cube_extent * math.sqrt(3)),
        "undistorted": True,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SyntheticBundle(
        root=out_dir,
        n_cameras=n_cameras,
        image_size=image_size,
        n_points=int(xyz.shape[0]),
        K=K,
        w2c_per_camera=w2cs,
    )


def zip_bundle(bundle_dir: Path, out_zip: Path) -> Path:
    """Zip the bundle directory into a single .zip suitable for inbox/ drop."""
    bundle_dir = Path(bundle_dir)
    out_zip = Path(out_zip)
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(bundle_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(bundle_dir))
    return out_zip


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="Output dir (or .zip path with --zip)")
    parser.add_argument("--zip", action="store_true",
                        help="Zip the bundle to --out (a .zip path)")
    parser.add_argument("--n-cameras", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.zip:
        bundle_dir = args.out.with_suffix("")
        build_bundle(
            bundle_dir,
            n_cameras=args.n_cameras,
            image_size=args.image_size,
            n_points=args.n_points,
            seed=args.seed,
        )
        zip_bundle(bundle_dir, args.out)
        shutil.rmtree(bundle_dir)
        print(f"wrote {args.out}")
    else:
        bundle = build_bundle(
            args.out,
            n_cameras=args.n_cameras,
            image_size=args.image_size,
            n_points=args.n_points,
            seed=args.seed,
        )
        print(f"wrote {bundle.root} ({bundle.n_cameras} cameras, {bundle.n_points} dense pts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
