"""Tests for the synthetic-bundle generator."""
from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData

from gs_pipeline.tests.fixtures.make_synthetic import (
    build_bundle,
    zip_bundle,
)


def test_default_bundle_layout(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b")
    root = bundle.root

    # Structural files
    assert (root / "cameras.xml").is_file()
    assert (root / "dense.ply").is_file()
    assert (root / "manifest.json").is_file()
    assert (root / "images").is_dir()
    assert (root / "masks").is_dir()  # reserved, currently empty

    # One image per camera, sized correctly
    image_paths = sorted((root / "images").iterdir())
    assert len(image_paths) == bundle.n_cameras == 8
    for p in image_paths:
        with Image.open(p) as im:
            assert im.size == (bundle.image_size, bundle.image_size)


def test_cameras_xml_has_expected_shape(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b")
    xml_text = (bundle.root / "cameras.xml").read_text(encoding="utf-8")
    # Minimal Agisoft schema invariants the parser will rely on.
    assert '<document version="2.0.0">' in xml_text
    assert '<sensor id="0"' in xml_text and 'type="frame"' in xml_text
    assert "<f>" in xml_text and "<cx>" in xml_text and "<cy>" in xml_text
    assert "<k1>0</k1>" in xml_text  # undistorted: all distortion zeroed
    assert xml_text.count("<camera id=") == bundle.n_cameras
    assert xml_text.count("<transform>") == bundle.n_cameras + 1  # cameras + chunk transform


def test_dense_ply_loadable_and_colored(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_points=600)
    data = PlyData.read(str(bundle.root / "dense.ply"))
    v = data["vertex"]
    # n_points is rounded down to a multiple of 6 (faces) inside the generator.
    assert len(v) >= 600 - 5
    assert {"x", "y", "z", "red", "green", "blue"}.issubset(set(v.data.dtype.names))
    # Coordinates fall on the cube faces, so at least one of |x|, |y|, |z| must
    # equal the half-extent (0.5) for every point.
    coords = np.stack([v["x"], v["y"], v["z"]], axis=1)
    on_face = np.any(np.isclose(np.abs(coords), 0.5, atol=1e-6), axis=1)
    assert on_face.all()


def test_manifest_matches_bundle(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=8, image_size=256, n_points=300)
    m = json.loads((bundle.root / "manifest.json").read_text())
    assert m["format"] == "metashape_splat_bundle/v1"
    assert m["n_cameras"] == 8
    assert m["image_size"] == [256, 256]
    assert m["undistorted"] is True
    assert m["dense_points"] == bundle.n_points
    # scene_extent_hint = cube diagonal = 2 * half_extent * sqrt(3) = sqrt(3) by default
    assert m["scene_extent_hint"] == pytest.approx(math.sqrt(3), rel=1e-9)


def test_cameras_form_ring_around_origin(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b")
    # Reconstruct camera centers from w2c -> -R^T t
    centers = []
    for w2c in bundle.w2c_per_camera:
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        centers.append(-R.T @ t)
    centers = np.stack(centers, axis=0)

    # All cameras at radius 2 in the XY plane (z=0).
    radii = np.linalg.norm(centers[:, :2], axis=1)
    np.testing.assert_allclose(radii, 2.0, atol=1e-6)
    np.testing.assert_allclose(centers[:, 2], 0.0, atol=1e-6)

    # Each w2c maps the world origin to a point at depth ~2 in camera frame.
    for w2c in bundle.w2c_per_camera:
        origin_cam = w2c @ np.array([0.0, 0.0, 0.0, 1.0])
        assert origin_cam[2] == pytest.approx(2.0, abs=1e-6)
        # Origin projects to the principal point (image center) horizontally.
        assert abs(origin_cam[0]) < 1e-6
        assert abs(origin_cam[1]) < 1e-6


def test_zip_bundle_round_trip(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=64, n_points=120)
    out_zip = tmp_path / "bundle.zip"
    zip_bundle(bundle.root, out_zip)
    assert out_zip.is_file()
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
    assert "cameras.xml" in names
    assert "dense.ply" in names
    assert "manifest.json" in names
    assert sum(1 for n in names if n.startswith("images/") and n.endswith(".png")) == 4


def test_image_has_some_non_background_pixels(tmp_path: Path):
    """Synthetic render should contain colored cube splats, not just gray."""
    bundle = build_bundle(tmp_path / "b", n_points=3000)
    img_path = sorted((bundle.root / "images").iterdir())[0]
    arr = np.array(Image.open(img_path))
    background = np.all(arr == 128, axis=-1)
    foreground_fraction = 1 - background.mean()
    assert foreground_fraction > 0.005, "render should contain visible cube points"
