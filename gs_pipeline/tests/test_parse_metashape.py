"""Tests for ``gs_pipeline.trainer.parse_metashape``.

The synthetic fixture is the source of truth: it writes a cameras.xml with
known intrinsics and extrinsics, and the parser must reproduce them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gs_pipeline.tests.fixtures.make_synthetic import build_bundle
from gs_pipeline.trainer.parse_metashape import (
    ParsedScene,
    SensorCalibration,
    parse_cameras_xml,
)


def _w2c_close(actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    """Compare 4x4 extrinsics, allowing tiny float jitter from inv() round-trip."""
    np.testing.assert_allclose(actual, expected, atol=atol)


def test_parses_synthetic_bundle(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=8, image_size=512)
    scene = parse_cameras_xml(bundle.root / "cameras.xml")
    assert isinstance(scene, ParsedScene)
    assert len(scene) == bundle.n_cameras

    # All cameras share one sensor.
    assert len(scene.sensors) == 1
    sensor = next(iter(scene.sensors.values()))
    assert isinstance(sensor, SensorCalibration)
    assert (sensor.width, sensor.height) == (512, 512)


def test_intrinsics_match_known_K(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=512)
    scene = parse_cameras_xml(bundle.root / "cameras.xml")
    for K in scene.K_per_camera:
        np.testing.assert_allclose(K, bundle.K, atol=1e-6)
        # Principal point at image center because synthetic fixture uses cx=cy=0.
        assert K[0, 2] == pytest.approx(256.0)
        assert K[1, 2] == pytest.approx(256.0)
        # No skew, no affinity.
        assert K[0, 1] == 0.0
        assert K[1, 0] == 0.0


def test_extrinsics_round_trip_within_tolerance(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=8, image_size=256)
    scene = parse_cameras_xml(bundle.root / "cameras.xml")
    assert scene.w2c_per_camera.shape == (8, 4, 4)
    # The fixture wrote the c2w (inv of its w2c); parser inverts back to w2c.
    for i, expected in enumerate(bundle.w2c_per_camera):
        _w2c_close(scene.w2c_per_camera[i], expected, atol=1e-6)


def test_image_paths_resolved_when_image_dir_given(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=128)
    scene = parse_cameras_xml(bundle.root / "cameras.xml", image_dir=bundle.root / "images")
    assert len(scene.image_paths) == 4
    for p in scene.image_paths:
        assert p.is_file()
    # No "image not found" warnings.
    assert not any("image not found" in w for w in scene.warnings)


def test_image_paths_empty_when_image_dir_omitted(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=128)
    scene = parse_cameras_xml(bundle.root / "cameras.xml")
    assert scene.image_paths == []


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_cameras_xml(tmp_path / "does_not_exist.xml")


def test_chunk_transform_identity_returns_none(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=128)
    scene = parse_cameras_xml(bundle.root / "cameras.xml")
    # Synthetic bundle writes an identity chunk transform.
    assert scene.chunk_transform is None
    assert not any("Chunk-level <transform>" in w for w in scene.warnings)


def test_chunk_transform_non_identity_warns(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=128)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text()
    # Replace the chunk-level rotation with something non-identity (180 deg around Z).
    text = text.replace(
        "<rotation>1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0</rotation>",
        "<rotation>-1.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 1.0</rotation>",
    )
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path)
    assert scene.chunk_transform is not None
    assert any("Chunk-level <transform>" in w for w in scene.warnings)


def test_b1_b2_included_in_K(tmp_path: Path):
    """Affinity/skew on the sensor should appear in K[0,0] and K[0,1]."""
    bundle = build_bundle(tmp_path / "b", n_cameras=2, image_size=128)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text()
    # Inject b1=2.5, b2=0.3 into the calibration block.
    text = text.replace("<b1>0</b1><b2>0</b2>", "<b1>2.5</b1><b2>0.3</b2>")
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path)
    K = scene.K_per_camera[0]
    assert K[0, 0] == pytest.approx(bundle.K[0, 0] + 2.5)
    assert K[0, 1] == pytest.approx(0.3)
    assert any("non-zero affinity/skew" in w for w in scene.warnings)


def test_distortion_warns_but_does_not_block(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=2, image_size=128)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text()
    text = text.replace(
        "<k1>0</k1><k2>0</k2><k3>0</k3><k4>0</k4>",
        "<k1>-0.12</k1><k2>0.03</k2><k3>0</k3><k4>0</k4>",
    )
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path)
    assert any("non-zero distortion" in w for w in scene.warnings)
    # K is unaffected by distortion.
    np.testing.assert_allclose(scene.K_per_camera[0], bundle.K, atol=1e-6)


def test_disabled_camera_dropped(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=4, image_size=128)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text()
    # Disable the first camera.
    text = text.replace(
        '<camera id="0" sensor_id="0" label="cam_000.png" enabled="true">',
        '<camera id="0" sensor_id="0" label="cam_000.png" enabled="false">',
    )
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path)
    assert len(scene) == 3
    assert "cam_000.png" not in scene.image_labels


def test_camera_without_transform_skipped(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=3, image_size=128)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text()
    # Strip the <transform> from cam_001 (unaligned camera).
    head, tail = text.split('<camera id="1"', 1)
    open_to_close = tail.split("</camera>", 1)
    body = open_to_close[0]
    # Remove the transform line; keep the camera element.
    cleaned = "\n".join(
        ln for ln in body.splitlines() if not ln.strip().startswith("<transform>")
    )
    text = head + '<camera id="1"' + cleaned + "</camera>" + open_to_close[1]
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path)
    assert len(scene) == 2
    assert "cam_001.png" not in scene.image_labels
    assert any("not aligned" in w for w in scene.warnings)


def test_all_cameras_disabled_raises(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=2, image_size=64)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text().replace('enabled="true"', 'enabled="false"')
    xml_path.write_text(text)
    with pytest.raises(ValueError, match="no aligned cameras"):
        parse_cameras_xml(xml_path)


def test_camera_with_unknown_sensor_raises(tmp_path: Path):
    bundle = build_bundle(tmp_path / "b", n_cameras=2, image_size=64)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text().replace('sensor_id="0"', 'sensor_id="99"', 1)
    xml_path.write_text(text)
    with pytest.raises(ValueError, match="unknown sensor_id"):
        parse_cameras_xml(xml_path)


def test_image_resolution_via_extension_search(tmp_path: Path):
    """Labels may omit the extension or refer to .jpg while files are .png."""
    bundle = build_bundle(tmp_path / "b", n_cameras=2, image_size=64)
    xml_path = bundle.root / "cameras.xml"
    text = xml_path.read_text().replace('cam_000.png', 'cam_000')
    xml_path.write_text(text)
    scene = parse_cameras_xml(xml_path, image_dir=bundle.root / "images")
    assert any(p.name == "cam_000.png" for p in scene.image_paths)
