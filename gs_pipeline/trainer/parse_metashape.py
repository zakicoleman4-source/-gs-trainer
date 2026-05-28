"""Agisoft Metashape `cameras.xml` parser for the GS trainer.

Produces a ``ParsedScene`` containing per-camera intrinsics (K) and
world-to-camera extrinsics (w2c) in the chunk's local frame, matching the
OpenCV / COLMAP / gsplat convention (+X right, +Y down, +Z forward).

Schema (relevant bits Agisoft writes; only `class="adjusted"` calibrations are
used for training):

    <document version="2.0.0">
      <chunk>
        <sensors>
          <sensor id="0" type="frame">
            <resolution width="W" height="H"/>
            <calibration type="frame" class="adjusted">
              <resolution width="W" height="H"/>
              <f>...</f><cx>...</cx><cy>...</cy>
              <k1>..</k1>...<p1>..</p1>...<b1>..</b1><b2>..</b2>
            </calibration>
          </sensor>
        </sensors>
        <cameras>
          <camera id="0" sensor_id="0" label="IMG_001.JPG" enabled="true">
            <transform>m00 m01 ... m33</transform>   <!-- 4x4 camera->chunk (c2w) -->
          </camera>
        </cameras>
        <transform>...</transform>   <!-- chunk->world; ignored for training -->
      </chunk>
    </document>

Conventions enforced here:

- ``cx``, ``cy`` are offsets from the image **center**. Principal point is
  ``(W/2 + cx, H/2 + cy)``.
- ``b1`` is the (fx - f) difference, ``b2`` is the skew term. The full
  intrinsic matrix is ``K = [[f+b1, b2, W/2+cx], [0, f, H/2+cy], [0, 0, 1]]``.
- After undistort-export, ``k1..k4`` and ``p1..p4`` and ``b1, b2`` are all
  zero (or near-zero). Non-zero distortion is recorded as a warning in
  ``ParsedScene.warnings`` — the trainer assumes undistorted input.
- Each ``<camera><transform>`` is a 4x4 matrix row-major. Agisoft treats it as
  camera-to-chunk (c2w). We invert to produce world-to-camera (w2c), in
  chunk-local meters. The chunk-level ``<transform>`` is **not** applied (we
  train in chunk-local space; this keeps things metric without dragging
  geo-referencing into the optimizer).
- Cameras with ``enabled="false"`` are dropped.

The parser does **not** detect coordinate-axis mismatches (some chunk frames
need a ``diag(1, -1, -1)`` flip on the camera-side rotation). That detection
happens at training start in ``train_mcmc.py`` by correlating a rendered
training view against the source image; if a flip is needed, it is applied to
all w2c matrices then.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np


# Distortion magnitude above which we consider the export non-undistorted.
_DISTORTION_TOL = 1.0e-3
# Affinity/skew magnitude above which we warn (and still use them in K).
_AFFINITY_TOL = 1.0e-3
# Chunk-transform deviation from identity above which we warn.
_CHUNK_IDENTITY_TOL = 1.0e-6


@dataclass
class SensorCalibration:
    """Calibrated intrinsics for one Metashape sensor."""
    sensor_id: int
    width: int
    height: int
    f: float
    cx: float
    cy: float
    b1: float = 0.0
    b2: float = 0.0
    # Distortion is reported for diagnostics only; the trainer assumes
    # undistorted input.
    k: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    p: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def K(self) -> np.ndarray:
        """Standard 3x3 intrinsics matrix in pixels."""
        return np.array([
            [self.f + self.b1, self.b2, self.width / 2.0 + self.cx],
            [0.0, self.f, self.height / 2.0 + self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


@dataclass
class ParsedScene:
    """Parsed result of a Metashape cameras.xml."""
    sensors: dict[int, SensorCalibration]
    image_labels: list[str]
    image_sizes: list[tuple[int, int]]  # (w, h) per camera, in declared order
    K_per_camera: np.ndarray            # shape (N, 3, 3)
    w2c_per_camera: np.ndarray          # shape (N, 4, 4)
    chunk_transform: Optional[np.ndarray]  # 4x4 c2w in world, None if identity
    image_paths: list[Path]             # resolved against image_dir (may be empty if image_dir is None)
    # Parallel to image_labels; None for cameras that have no mask file.
    mask_paths: list           # list[Optional[Path]]
    warnings: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.image_labels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_float(el: ET.Element, tag: str, default: float = 0.0) -> float:
    child = el.find(tag)
    if child is None or child.text is None:
        return default
    try:
        return float(child.text.strip())
    except ValueError:
        return default


def _parse_transform_matrix(text: str) -> np.ndarray:
    parts = text.replace(",", " ").split()
    if len(parts) != 16:
        raise ValueError(f"<transform> must contain 16 floats, got {len(parts)}")
    return np.array(parts, dtype=np.float64).reshape(4, 4)


def _parse_chunk_transform(chunk: ET.Element) -> Optional[np.ndarray]:
    """Return the chunk-level <transform> as a 4x4 (c2w in world) or None.

    Agisoft writes the chunk transform either as a flat 16-element <transform>
    (newer) or as <rotation>+<translation>+<scale> children (older). Returns
    None if absent or numerically identity.
    """
    # Newer flat form: a <transform> child of <chunk> whose text is 16 floats.
    for child in chunk:
        if child.tag != "transform":
            continue
        text = (child.text or "").strip()
        if text and not list(child):
            try:
                M = _parse_transform_matrix(text)
            except ValueError:
                continue
            return None if _is_identity(M) else M
        # Older form: rotation/translation/scale subelements.
        rot_el = child.find("rotation")
        trn_el = child.find("translation")
        scl_el = child.find("scale")
        if rot_el is None and trn_el is None:
            continue
        R = np.eye(3)
        t = np.zeros(3)
        s = 1.0
        if rot_el is not None and rot_el.text:
            R = np.array(rot_el.text.split(), dtype=np.float64).reshape(3, 3)
        if trn_el is not None and trn_el.text:
            t = np.array(trn_el.text.split(), dtype=np.float64)
        if scl_el is not None and scl_el.text:
            s = float(scl_el.text)
        M = np.eye(4)
        M[:3, :3] = R * s
        M[:3, 3] = t
        return None if _is_identity(M) else M
    return None


def _is_identity(M: np.ndarray) -> bool:
    return bool(np.allclose(M, np.eye(M.shape[0]), atol=_CHUNK_IDENTITY_TOL))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cameras_xml(
    xml_path: Path,
    image_dir: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
) -> ParsedScene:
    """Parse a Metashape ``cameras.xml`` into a ParsedScene.

    Args:
        xml_path: path to ``cameras.xml`` (or any path resolvable to one).
        image_dir: optional directory containing the photo files. If given,
            ``ParsedScene.image_paths`` will be filled with resolved paths and
            a warning is recorded for any label that cannot be found.
        masks_dir: optional directory containing per-camera mask PNGs exported
            by ``export_for_splat.py``. If given, ``ParsedScene.mask_paths``
            is a list parallel to ``image_labels`` with ``None`` for cameras
            that have no mask.

    Raises:
        FileNotFoundError: if ``xml_path`` does not exist.
        ValueError: if the XML lacks a chunk / sensors / cameras section, or
            if any camera references an unknown sensor.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    chunks = list(root.iter("chunk"))
    if not chunks:
        raise ValueError("cameras.xml: no <chunk> element found")
    if len(chunks) > 1:
        # We train one chunk at a time; pick the first enabled chunk.
        enabled = [c for c in chunks if c.attrib.get("enabled", "true").lower() != "false"]
        chunk = enabled[0] if enabled else chunks[0]
    else:
        chunk = chunks[0]

    warnings: list[str] = []

    sensor_reject_reasons: list[str] = []
    sensors = _parse_sensors(chunk, warnings, sensor_reject_reasons)
    if not sensors:
        # Distinguish "no sensor elements at all" from "sensors present but every
        # one was rejected" so the client gets an actionable message instead of a
        # misleading 'section missing'.
        if sensor_reject_reasons:
            raise ValueError(
                "cameras.xml: no usable sensor calibration. "
                + "; ".join(sensor_reject_reasons)
            )
        raise ValueError(
            "cameras.xml: <sensors> section empty or missing adjusted calibrations"
        )

    chunk_transform = _parse_chunk_transform(chunk)
    if chunk_transform is not None:
        warnings.append(
            "Chunk-level <transform> is non-identity. Training will use "
            "chunk-local coordinates and ignore this transform. Output splats "
            "will be in the same local frame as the dense cloud."
        )

    cameras_el = chunk.find("cameras")
    if cameras_el is None:
        raise ValueError("cameras.xml: <cameras> section missing")

    image_labels: list[str] = []
    image_sizes: list[tuple[int, int]] = []
    Ks: list[np.ndarray] = []
    w2cs: list[np.ndarray] = []
    image_paths: list[Path] = []
    mask_paths: list = []  # Optional[Path] per camera

    for cam_el in cameras_el.iter("camera"):
        if cam_el.attrib.get("enabled", "true").lower() == "false":
            continue
        label = cam_el.attrib.get("label")
        if not label:
            warnings.append(f"camera id={cam_el.attrib.get('id')} has no label; skipped")
            continue
        sensor_id_str = cam_el.attrib.get("sensor_id")
        if sensor_id_str is None:
            warnings.append(f"camera {label} has no sensor_id; skipped")
            continue
        try:
            sensor_id = int(sensor_id_str)
        except ValueError:
            warnings.append(f"camera {label} has non-integer sensor_id; skipped")
            continue
        sensor = sensors.get(sensor_id)
        if sensor is None:
            raise ValueError(f"camera {label} references unknown sensor_id={sensor_id}")

        transform_el = cam_el.find("transform")
        if transform_el is None or transform_el.text is None:
            # Cameras without a transform are not aligned; skip with a warning.
            warnings.append(f"camera {label} has no <transform> (not aligned); skipped")
            continue
        try:
            c2w = _parse_transform_matrix(transform_el.text)
        except ValueError as e:
            warnings.append(f"camera {label}: {e}; skipped")
            continue
        # Invert to get w2c (Agisoft <transform> is camera->chunk, i.e. c2w).
        try:
            w2c = np.linalg.inv(c2w)
        except np.linalg.LinAlgError:
            warnings.append(f"camera {label}: singular c2w matrix; skipped")
            continue

        K = sensor.K()

        image_labels.append(label)
        image_sizes.append((sensor.width, sensor.height))
        Ks.append(K)
        w2cs.append(w2c)

        if image_dir is not None:
            resolved = _resolve_image(image_dir, label, warnings)
            if resolved is not None:
                image_paths.append(resolved)

        if masks_dir is not None:
            mask_paths.append(_resolve_mask(masks_dir, label))

    if not image_labels:
        raise ValueError("cameras.xml: no aligned cameras found")

    K_arr = np.stack(Ks, axis=0)
    w2c_arr = np.stack(w2cs, axis=0)

    return ParsedScene(
        sensors=sensors,
        image_labels=image_labels,
        image_sizes=image_sizes,
        K_per_camera=K_arr,
        w2c_per_camera=w2c_arr,
        chunk_transform=chunk_transform,
        image_paths=image_paths,
        mask_paths=mask_paths,
        warnings=warnings,
    )


def _parse_sensors(
    chunk: ET.Element,
    warnings: list[str],
    reject_reasons: Optional[list[str]] = None,
) -> dict[int, SensorCalibration]:
    if reject_reasons is None:
        reject_reasons = []
    sensors_el = chunk.find("sensors")
    if sensors_el is None:
        return {}
    sensors: dict[int, SensorCalibration] = {}
    for sensor_el in sensors_el.iter("sensor"):
        sid_str = sensor_el.attrib.get("id")
        if sid_str is None:
            continue
        try:
            sid = int(sid_str)
        except ValueError:
            continue
        sensor_label = sensor_el.attrib.get("label", f"sensor_{sid}")
        sensor_type = sensor_el.attrib.get("type", "frame")
        if sensor_type != "frame":
            warnings.append(
                f"sensor {sensor_label}: type={sensor_type!r} is not 'frame'; "
                "fisheye/spherical sensors must be undistorted to frame on export."
            )

        # Resolution: prefer the calibration's resolution, fall back to the sensor's.
        sensor_res = sensor_el.find("resolution")
        # Find the first adjusted frame calibration.
        calib = None
        for c in sensor_el.iter("calibration"):
            if c.attrib.get("class") == "adjusted" and c.attrib.get("type", "frame") == "frame":
                calib = c
                break
        if calib is None:
            # Fall back to any calibration child.
            calib = sensor_el.find("calibration")
        if calib is None:
            msg = f"sensor {sensor_label}: no calibration"
            warnings.append(msg + "; skipped")
            reject_reasons.append(msg)
            continue

        calib_res = calib.find("resolution")
        res = calib_res if calib_res is not None else sensor_res
        if res is None:
            msg = f"sensor {sensor_label}: no <resolution>"
            warnings.append(msg + "; skipped")
            reject_reasons.append(msg)
            continue
        try:
            width = int(res.attrib["width"])
            height = int(res.attrib["height"])
        except (KeyError, ValueError):
            msg = f"sensor {sensor_label}: malformed <resolution>"
            warnings.append(msg + "; skipped")
            reject_reasons.append(msg)
            continue
        if width <= 0 or height <= 0:
            msg = f"sensor {sensor_label}: non-positive resolution {width}x{height}"
            warnings.append(msg + "; skipped")
            reject_reasons.append(msg)
            continue

        f = _read_float(calib, "f")
        if not math.isfinite(f) or f <= 0.0:
            msg = (
                f"sensor {sensor_label}: invalid focal length f={f} "
                f"(must be a positive number of pixels)"
            )
            warnings.append(msg + "; skipped")
            reject_reasons.append(msg)
            continue
        cx = _read_float(calib, "cx")
        cy = _read_float(calib, "cy")
        b1 = _read_float(calib, "b1")
        b2 = _read_float(calib, "b2")
        k = (
            _read_float(calib, "k1"),
            _read_float(calib, "k2"),
            _read_float(calib, "k3"),
            _read_float(calib, "k4"),
        )
        p = (
            _read_float(calib, "p1"),
            _read_float(calib, "p2"),
            _read_float(calib, "p3"),
            _read_float(calib, "p4"),
        )

        if max(abs(v) for v in k + p) > _DISTORTION_TOL:
            warnings.append(
                f"sensor {sensor_label}: non-zero distortion (k={k}, p={p}); "
                "the trainer assumes undistorted photos — re-export with "
                "File > Export > Undistort Photos."
            )
        if max(abs(b1), abs(b2)) > _AFFINITY_TOL:
            warnings.append(
                f"sensor {sensor_label}: non-zero affinity/skew (b1={b1}, b2={b2}); "
                "K will include these terms but quality may suffer."
            )

        sensors[sid] = SensorCalibration(
            sensor_id=sid, width=width, height=height,
            f=f, cx=cx, cy=cy, b1=b1, b2=b2, k=k, p=p,
        )
    return sensors


def _resolve_mask(masks_dir: Path, label: str) -> Optional[Path]:
    """Return the mask PNG for ``label`` under ``masks_dir``, or None.

    Only looks for PNG (the format _export_masks writes). No warning on miss —
    it is normal for some cameras to lack masks.
    """
    stem = Path(label).stem
    candidate = masks_dir / f"{stem}.png"
    if candidate.is_file():
        return candidate
    candidate_upper = masks_dir / f"{stem}.PNG"
    if candidate_upper.is_file():
        return candidate_upper
    return None


def _resolve_image(image_dir: Path, label: str, warnings: list[str]) -> Optional[Path]:
    """Resolve a camera ``label`` to an actual file under ``image_dir``.

    Tries the label as-is, then with common image extensions, then a recursive
    glob fallback. Returns None and appends a warning if nothing matches.

    Handles Windows-style absolute paths in labels (e.g.
    ``C:\\Users\\John\\Photos\\IMG_001.JPG``) by extracting the filename
    component before searching.
    """
    # Strip Windows-style absolute path prefix: if the label contains a
    # backslash it's almost certainly a Windows path baked into cameras.xml.
    # Extract just the filename for lookup.
    if "\\" in label:
        label = label.rsplit("\\", 1)[-1]
    # Also handle forward-slash absolute paths (less common but possible).
    if "/" in label and not (image_dir / label).is_file():
        label = label.rsplit("/", 1)[-1]

    p = image_dir / label
    if p.is_file():
        return p
    stem = Path(label).stem
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".PNG", ".JPG", ".JPEG", ".TIF", ".TIFF"):
        candidate = image_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    # Recursive fallback (slow on huge trees; only used when the flat lookup fails).
    matches = list(image_dir.rglob(label))
    if matches:
        return matches[0]
    matches = list(image_dir.rglob(f"{stem}.*"))
    if matches:
        return matches[0]
    warnings.append(f"image not found for camera label={label!r} under {image_dir}")
    return None
