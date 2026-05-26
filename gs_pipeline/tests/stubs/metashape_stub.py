"""Tiny in-process stand-in for Metashape's Python API.

Covers the surface our exporter calls: ``chunk.cameras``, ``chunk.sensors``,
``chunk.transform``, ``chunk.dense_cloud``, ``chunk.exportCameras``,
``chunk.exportPoints``, ``chunk.undistortPhotos``. Lets the export tests run
on CPU CI without a Metashape license.

Each stub *writes a valid file* on the export call (cameras.xml that the
real parser will load, dense.ply with a tiny point cloud, undistorted PNG
images). That way the bundle produced by ``export_chunk_to_zip(stub_chunk)``
round-trips through the trainer's parser end-to-end.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


@dataclass
class StubSensor:
    type: str = "frame"
    width: int = 256
    height: int = 256
    f: float = 350.0


@dataclass
class StubPhoto:
    path: str


@dataclass
class StubMask:
    """Minimal stand-in for Metashape.Mask; image() returns a solid-black PIL image."""
    width: int = 128
    height: int = 128

    def image(self):
        arr = np.zeros((self.height, self.width), dtype=np.uint8)
        return _PilSaveWrapper(arr)


class _PilSaveWrapper:
    """Wraps a numpy array so ``.save(path)`` writes a PNG via PIL."""
    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def save(self, path: str) -> None:
        Image.fromarray(self._arr, mode="L").save(path)


@dataclass
class StubCamera:
    label: str
    sensor: StubSensor
    photo: Optional[StubPhoto] = None
    transform: Optional[np.ndarray] = None  # 4x4 c2w; None = not aligned
    mask: Optional[StubMask] = None


@dataclass
class StubDenseCloud:
    n_points: int = 200


@dataclass
class StubChunk:
    label: str = "synthetic"
    enabled: bool = True
    cameras: list = field(default_factory=list)
    sensors: list = field(default_factory=list)
    dense_cloud: Optional[StubDenseCloud] = None
    transform: Optional[np.ndarray] = None  # 4x4 chunk-level transform (None = identity)
    _bundle_root_for_images: Optional[Path] = None

    def exportCameras(self, path: str, format: Any = None) -> None:
        """Write a tiny but parser-compatible cameras.xml."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        s = self.sensors[0]
        lines: list[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<document version="2.0.0">')
        lines.append(f'  <chunk label="{self.label}" enabled="true">')
        lines.append('    <sensors next_id="1">')
        lines.append(f'      <sensor id="0" label="stub_sensor" type="frame">')
        lines.append(f'        <resolution width="{s.width}" height="{s.height}"/>')
        lines.append('        <calibration type="frame" class="adjusted">')
        lines.append(f'          <resolution width="{s.width}" height="{s.height}"/>')
        lines.append(f'          <f>{s.f}</f><cx>0</cx><cy>0</cy>')
        lines.append('          <k1>0</k1><k2>0</k2><k3>0</k3><k4>0</k4>')
        lines.append('          <p1>0</p1><p2>0</p2><p3>0</p3><p4>0</p4>')
        lines.append('          <b1>0</b1><b2>0</b2>')
        lines.append('        </calibration>')
        lines.append('      </sensor>')
        lines.append('    </sensors>')
        lines.append(f'    <cameras next_id="{len(self.cameras)}">')
        for i, cam in enumerate(self.cameras):
            if cam.transform is None:
                continue
            mat_str = " ".join(f"{v:.12e}" for v in cam.transform.flatten())
            lines.append(f'      <camera id="{i}" sensor_id="0" label="{cam.label}" enabled="true">')
            lines.append(f'        <transform>{mat_str}</transform>')
            lines.append('      </camera>')
        lines.append('    </cameras>')
        lines.append('    <transform>')
        lines.append('      <rotation>1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0</rotation>')
        lines.append('      <translation>0 0 0</translation>')
        lines.append('      <scale>1</scale>')
        lines.append('    </transform>')
        lines.append('  </chunk>')
        lines.append('</document>')
        Path(path).write_text("\n".join(lines), encoding="utf-8")

    def exportPoints(self, path: str, *, source_data: Any = None,
                     save_colors: bool = True, save_normals: bool = False) -> None:
        """Write a tiny PLY (200 colored points along a line)."""
        n = self.dense_cloud.n_points if self.dense_cloud else 200
        rng = np.random.default_rng(0)
        xyz = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
        rgb = rng.integers(0, 256, size=(n, 3)).astype(np.uint8)
        data = np.empty(n, dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ])
        data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]
        data["red"] = rgb[:, 0]; data["green"] = rgb[:, 1]; data["blue"] = rgb[:, 2]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(data, "vertex")], text=False).write(str(path))

    def undistortPhotos(self, path: str) -> None:
        """Write one PNG per aligned camera (solid gray; size matches sensor)."""
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        for cam in self.cameras:
            if cam.transform is None:
                continue
            arr = np.full((cam.sensor.height, cam.sensor.width, 3), 128, dtype=np.uint8)
            Image.fromarray(arr).save(d / cam.label)

    def optimizeCameras(self, **kwargs) -> None:
        """No-op in the stub; real optimization happens in Metashape."""


# Top-level "module" attributes the exporter inspects.

class _DataSource:
    DenseCloudData = "DenseCloudData"


DataSource = _DataSource
DenseCloudData = _DataSource.DenseCloudData  # legacy
CamerasFormat = type("CamerasFormat", (), {"CamerasFormatXML": "xml"})


# ---------------------------------------------------------------------------
# Helpers for tests
# ---------------------------------------------------------------------------

def build_stub_chunk(
    n_cameras: int = 6,
    *,
    label: str = "synthetic",
    image_size: int = 128,
    chunk_transform: Optional[np.ndarray] = None,
    unaligned_count: int = 0,
    no_dense: bool = False,
) -> StubChunk:
    """Build a stub chunk with the requested camera count and properties."""
    sensor = StubSensor(width=image_size, height=image_size, f=image_size * 1.4)
    cameras: list[StubCamera] = []
    for i in range(n_cameras):
        # c2w: identity-ish (rotation + small translation along x).
        c2w = np.eye(4, dtype=np.float64)
        c2w[0, 3] = float(i)
        aligned = i >= unaligned_count
        cameras.append(StubCamera(
            label=f"cam_{i:03d}.png",
            sensor=sensor,
            transform=(c2w if aligned else None),
        ))
    return StubChunk(
        label=label,
        cameras=cameras,
        sensors=[sensor],
        dense_cloud=None if no_dense else StubDenseCloud(n_points=200),
        transform=chunk_transform,
    )
