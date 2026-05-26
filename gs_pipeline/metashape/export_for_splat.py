"""Metashape Pro script: export a chunk as a `*_splat_bundle.zip`.

User flow on Windows:
    1. Open .psx in Metashape Pro.
    2. Tools > Run Script > pick this file.
    3. Watch the progress dialog. When done, the bundle zip is sitting next
       to the .psx, ready to drag into the gs_pipeline web UI.

This file is structured so the heavy lifting lives in **pure** functions
that take a chunk-like object and a filesystem path — they import nothing
from ``Metashape`` themselves. The Metashape-coupled entry point in
``main()`` is the only piece that touches ``Metashape.app`` / GUI APIs.

Why: gives us CPU-runnable tests via a tiny stub chunk under
``tests/stubs/`` without needing an actual Metashape license.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


BUNDLE_FORMAT = "metashape_splat_bundle/v1"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class ChunkValidation:
    chunk_label: str
    n_cameras_total: int
    n_cameras_aligned: int
    has_dense_cloud: bool
    chunk_transform_is_identity: bool
    sensor_types: list[str]
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_chunk(chunk: Any) -> ChunkValidation:
    """Inspect a Metashape Chunk-like object; report errors + warnings.

    Required-for-export:
      - chunk.cameras non-empty and >=4 have transforms (aligned).
      - chunk has a dense_cloud (or dense_point_cloud, Metashape 2.x renamed).

    Warnings (non-blocking):
      - non-identity chunk-level transform.
      - non-frame sensor types (fisheye etc) — user must undistort export.
    """
    label = getattr(chunk, "label", "") or "(unlabeled)"
    cameras = list(getattr(chunk, "cameras", []) or [])
    n_aligned = sum(1 for c in cameras if getattr(c, "transform", None) is not None)
    dense = getattr(chunk, "dense_cloud", None)
    if dense is None:
        dense = getattr(chunk, "dense_point_cloud", None)
    if dense is None and getattr(chunk, "tie_points", None) is not None:
        # Metashape 2.x: dense_cloud was renamed to point_cloud;
        # tie_points existing confirms we're on 2.x (in 1.x point_cloud
        # was the sparse cloud).
        dense = getattr(chunk, "point_cloud", None)
    chunk_t = getattr(chunk, "transform", None)
    chunk_identity = _is_identity_chunk_transform(chunk_t)

    sensor_types: list[str] = []
    for s in getattr(chunk, "sensors", []) or []:
        t = getattr(s, "type", None)
        # Metashape exposes type as an enum-like; str() works for our purposes.
        sensor_types.append(str(t).split(".")[-1].lower())

    errors: list[str] = []
    warnings: list[str] = []
    if not cameras:
        errors.append("chunk has no cameras")
    elif n_aligned < 4:
        errors.append(
            f"only {n_aligned} cameras are aligned; need at least 4 for GS training"
        )
    if dense is None:
        errors.append("chunk has no dense cloud — build dense cloud first")
    if not chunk_identity:
        warnings.append(
            "chunk has a non-identity geo-transform; training will use chunk-local "
            "coordinates (output splats will be in the same local frame as the dense cloud)"
        )
    for st in sensor_types:
        if st not in ("frame", "spherical", "fisheye"):
            warnings.append(f"unknown sensor type {st!r} — bundle may not import cleanly")
        elif st in ("fisheye", "spherical"):
            warnings.append(
                f"sensor type {st!r}: bundle export will undistort photos to frame; "
                "verify the result before training"
            )
    return ChunkValidation(
        chunk_label=label,
        n_cameras_total=len(cameras),
        n_cameras_aligned=n_aligned,
        has_dense_cloud=dense is not None,
        chunk_transform_is_identity=chunk_identity,
        sensor_types=sensor_types,
        errors=errors,
        warnings=warnings,
    )


def _is_identity_chunk_transform(t: Any) -> bool:
    """Best-effort check for an identity Metashape Matrix transform."""
    if t is None:
        return True
    mat = getattr(t, "matrix", t)
    if mat is None:
        return True
    try:
        for i in range(4):
            for j in range(4):
                try:
                    val = float(mat[i, j])
                except (TypeError, KeyError):
                    val = float(mat[i][j])
                target = 1.0 if i == j else 0.0
                if abs(val - target) > 1.0e-6:
                    return False
    except Exception:
        return True
    return True


def select_chunk(doc: Any) -> Any:
    """Pick the right chunk to export.

    Priority order:
      1. ``doc.chunk`` (the currently selected chunk).
      2. The first enabled chunk in ``doc.chunks``.
      3. The first chunk.

    Raises ``ValueError`` if the doc has zero chunks.
    """
    chunk = getattr(doc, "chunk", None)
    if chunk is not None and getattr(chunk, "enabled", True):
        return chunk
    for c in getattr(doc, "chunks", []) or []:
        if getattr(c, "enabled", True):
            return c
    chunks = list(getattr(doc, "chunks", []) or [])
    if chunks:
        return chunks[0]
    raise ValueError("document has no chunks")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_chunk_to_bundle_dir(
    chunk: Any,
    bundle_dir: Path,
    *,
    progress: Optional[Callable[[str, float], None]] = None,
    metashape_module: Any = None,
) -> Path:
    """Run the export steps on ``chunk`` into ``bundle_dir``.

    Steps:
      1. ``chunk.exportCameras(bundle_dir/cameras.xml, ...)``
      2. ``chunk.exportPoints(bundle_dir/dense.ply, source_data=DenseCloudData)``
      3. Copy undistorted photos to ``bundle_dir/images/`` (uses
         ``chunk.undistortPhotos`` if available; otherwise falls back to
         ``camera.photo.path`` -> direct copy).
      4. Write ``manifest.json``.

    ``progress`` is a callback ``(message, fraction)`` so we can wire it to
    Metashape's progress dialog OR to a print loop in tests.
    """
    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "images").mkdir(parents=True)

    if progress is None:
        progress = lambda msg, frac: None  # noqa: E731

    progress("exporting cameras", 0.10)
    cameras_xml = bundle_dir / "cameras.xml"
    _call_export_cameras(chunk, cameras_xml, metashape_module=metashape_module)

    progress("exporting dense cloud", 0.40)
    dense_ply = bundle_dir / "dense.ply"
    _call_export_points(chunk, dense_ply, metashape_module=metashape_module)

    progress("exporting undistorted photos", 0.70)
    _export_undistorted_photos(chunk, bundle_dir / "images", metashape_module=metashape_module)

    progress("writing manifest", 0.95)
    manifest = _build_manifest(chunk, bundle_dir)
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    progress("done", 1.0)
    return bundle_dir


def zip_bundle(bundle_dir: Path, out_zip: Path) -> Path:
    """Zip ``bundle_dir`` into ``out_zip`` with deflate compression."""
    bundle_dir = Path(bundle_dir)
    out_zip = Path(out_zip)
    if out_zip.exists():
        out_zip.unlink()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(bundle_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(bundle_dir))
    return out_zip


def _call_export_cameras(chunk: Any, out_path: Path, *, metashape_module: Any) -> None:
    """Wrap Metashape's exportCameras API, tolerating its evolving signature."""
    # Metashape 1.8+: chunk.exportCameras(path, format=...) ; 2.x identical.
    fn = getattr(chunk, "exportCameras", None)
    if fn is None:
        raise AttributeError("chunk has no exportCameras method")
    try:
        fn(str(out_path))
    except TypeError:
        # Some versions require a format kwarg.
        if metashape_module is not None and hasattr(metashape_module, "CamerasFormat"):
            fn(str(out_path),
               format=metashape_module.CamerasFormat.CamerasFormatXML)
        else:
            raise


def _call_export_points(chunk: Any, out_path: Path, *, metashape_module: Any) -> None:
    """Wrap Metashape's exportPoints, asking for the dense cloud by default."""
    fn = getattr(chunk, "exportPoints", None)
    if fn is None:
        raise AttributeError("chunk has no exportPoints method")
    source = None
    if metashape_module is not None:
        # Newer Metashape uses .DataSource.DenseCloudData; older uses
        # .DenseCloudData top-level.
        # Metashape 2.x: DataSource.PointCloudData (was DenseCloudData)
        for path in ("DataSource.PointCloudData", "DataSource.DenseCloudData",
                      "PointCloudData", "DenseCloudData"):
            obj = metashape_module
            for attr in path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                source = obj
                break
<<<<<<< Updated upstream
    try:
        if source is not None:
            fn(str(out_path), source_data=source, save_colors=True, save_normals=False)
        else:
            fn(str(out_path))
    except TypeError:
        # Fallback to positional / minimal args.
        fn(str(out_path))
=======
    kwargs_attempts = []
    if source is not None:
        kwargs_attempts.append(dict(source_data=source, save_colors=True, save_normals=False))
        kwargs_attempts.append(dict(source_data=source))
    kwargs_attempts.append({})
    for kwargs in kwargs_attempts:
        try:
            fn(str(out_path), **kwargs)
            return
        except TypeError:
            continue
    fn(str(out_path))
>>>>>>> Stashed changes


def _export_undistorted_photos(
    chunk: Any,
    images_dir: Path,
    *,
    metashape_module: Any = None,
) -> None:
    """Copy or undistort each aligned camera's photo into ``images_dir``.

    Metashape 2.0+ has ``chunk.undistortPhotos(path, ...)``. Older versions
    require per-camera ``camera.image()`` calls. We try the modern API first.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    fn = getattr(chunk, "undistortPhotos", None)
    if callable(fn):
        try:
            fn(str(images_dir))
            return
        except Exception:
            # Fall through to the per-camera path on any failure.
            pass
    # Fallback: copy original photos verbatim. Trainer's parse will warn if
    # distortion is non-zero in the cameras.xml so the user sees the issue.
    for camera in (chunk.cameras or []):
        if getattr(camera, "transform", None) is None:
            continue  # unaligned
        photo = getattr(camera, "photo", None)
        path = getattr(photo, "path", None) if photo is not None else None
        if not path or not Path(path).is_file():
            continue
        dest = images_dir / Path(path).name
        try:
            shutil.copy2(path, dest)
        except OSError:
            continue


def _build_manifest(chunk: Any, bundle_dir: Path) -> dict:
    """Summarise the bundle for the trainer's preflight (and the UI)."""
    cameras = [c for c in (chunk.cameras or []) if getattr(c, "transform", None) is not None]
    return {
        "format": BUNDLE_FORMAT,
        "source": "metashape_export",
        "chunk_label": getattr(chunk, "label", None),
        "n_cameras": len(cameras),
        "image_size": _peek_image_size(chunk),
        "undistorted": True,
        "files": [p.name for p in sorted(bundle_dir.iterdir()) if p.is_file()],
    }


def _peek_image_size(chunk: Any) -> Optional[list[int]]:
    sensors = list(getattr(chunk, "sensors", []) or [])
    if not sensors:
        return None
    s = sensors[0]
    w = getattr(s, "width", None)
    h = getattr(s, "height", None)
    if w is None or h is None:
        return None
    return [int(w), int(h)]


# ---------------------------------------------------------------------------
# Top-level export = validate + export + zip
# ---------------------------------------------------------------------------

def export_chunk_to_zip(
    chunk: Any,
    *,
    out_zip: Path,
    work_dir: Optional[Path] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    metashape_module: Any = None,
) -> tuple[Path, ChunkValidation]:
    """Run validation, export to a temp bundle dir, zip, clean up.

    Returns ``(zip_path, validation)``. Validation is returned even on
    failure so callers can surface errors clearly; on validation failure the
    zip is **not** written and the function raises ``ValueError``.
    """
    v = validate_chunk(chunk)
    if not v.ok:
        raise ValueError("; ".join(v.errors))
    if work_dir is None:
        import tempfile
        work_dir = Path(tempfile.mkdtemp(prefix="gs_bundle_"))
    bundle_dir = Path(work_dir) / "bundle"
    export_chunk_to_bundle_dir(
        chunk, bundle_dir, progress=progress, metashape_module=metashape_module,
    )
    out_zip = Path(out_zip)
    zip_bundle(bundle_dir, out_zip)
    return out_zip, v


# ---------------------------------------------------------------------------
# Metashape GUI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run as a Metashape ``Tools > Run Script`` target.

    Lazy-imports Metashape so this module remains importable on CPU CI.
    """
    import Metashape  # type: ignore  # available only inside Metashape

    app = Metashape.app
    doc = app.document
    psx_path = Path(doc.path) if doc.path else None
    if psx_path is None:
        Metashape.app.messageBox("Save the project first.")
        return
    chunk = select_chunk(doc)
    v = validate_chunk(chunk)
    if not v.ok:
        Metashape.app.messageBox("Cannot export:\n" + "\n".join(v.errors))
        return
    for w in v.warnings:
        Metashape.app.messageBox(f"Warning: {w}")

    progress_dialog = getattr(Metashape.app, "progress", None)
    progress_ctx = None
    if callable(progress_dialog):
        try:
            progress_ctx = progress_dialog("Exporting splat bundle")
        except Exception:
            pass

    def report(msg: str, frac: float) -> None:
        try:
            if progress_ctx is not None:
                progress_ctx.update(msg, frac)
        except Exception:
            pass

    out_zip = psx_path.with_name(f"{psx_path.stem}_splat_bundle.zip")
    try:
        export_chunk_to_zip(
            chunk, out_zip=out_zip,
            progress=report,
            metashape_module=Metashape,
        )
    finally:
        if progress_ctx is not None:
            try:
                progress_ctx.finish()
            except Exception:
                pass
    Metashape.app.messageBox(f"Splat bundle written to:\n{out_zip}")


if __name__ == "__main__":
    main()
