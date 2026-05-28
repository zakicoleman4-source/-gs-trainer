"""Fuzz-test the gs_pipeline bundle ingestion path with malformed inputs.

Every malformed bundle is run through the REAL ``pipeline.run_job`` with a stub
trainer (no GPU). The contract under test:

    run_job() must NEVER raise. It must return a JobState that is either
    DONE (trained) or FAILED with a non-empty error_msg. Anything else --
    an unhandled exception escaping run_job, or a FAILED state with no
    error message -- is a BAD (ungraceful) outcome.

Run:  python3 fuzz_bundles.py
      python3 fuzz_bundles.py --big        # also run the slow ~2GB-zip case
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

# Ensure the package is importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gs_pipeline.tests.fixtures.make_synthetic import build_bundle, zip_bundle
from gs_pipeline.tests.test_pipeline import _stub_train
from gs_pipeline.trainer.budget import GPUInfo
from gs_pipeline.trainer.job_state import State
from gs_pipeline.trainer.pipeline import run_job

GPU_24 = GPUInfo(name="(fuzz) 24GB", total_vram_bytes=24_000_000_000)


# ---------------------------------------------------------------------------
# Bundle-building helpers
# ---------------------------------------------------------------------------

def _write_dense_ply(out_path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    data = np.empty(xyz.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    data["x"] = xyz[:, 0]; data["y"] = xyz[:, 1]; data["z"] = xyz[:, 2]
    data["red"] = rgb[:, 0]; data["green"] = rgb[:, 1]; data["blue"] = rgb[:, 2]
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(out_path))


def _zip_dir(src_dir: Path, out_zip: Path) -> Path:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(src_dir))
    return out_zip


def _good_bundle_dir(dst: Path, *, n_cameras=4, image_size=64, n_points=120) -> Path:
    """A known-good synthetic bundle directory (cameras.xml + dense.ply + images)."""
    build_bundle(dst, n_cameras=n_cameras, image_size=image_size, n_points=n_points)
    return dst


# Each case-builder returns the path to a .zip ready for run_job.

def case_empty_zip(work: Path) -> Path:
    z = work / "empty.zip"
    with zipfile.ZipFile(z, "w"):
        pass
    return z


def case_cameras_no_ply(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    (b / "dense.ply").unlink()
    return _zip_dir(b, work / "bundle.zip")


def case_ply_no_cameras(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    (b / "cameras.xml").unlink()
    return _zip_dir(b, work / "bundle.zip")


def case_empty_images(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    for f in (b / "images").iterdir():
        f.unlink()
    return _zip_dir(b, work / "bundle.zip")


def case_truncated_cameras_xml(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xml = (b / "cameras.xml").read_text(encoding="utf-8")
    (b / "cameras.xml").write_text(xml[: len(xml) // 2] + "<transform>1 0 0", encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


def case_zero_aligned_cameras(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xml = (b / "cameras.xml").read_text(encoding="utf-8")
    # Drop every <transform>...</transform> inside cameras so none are aligned.
    import re
    # Only strip transforms that are camera children (single-line form here).
    xml2 = re.sub(r"\s*<transform>[^<]*</transform>", "", xml)
    (b / "cameras.xml").write_text(xml2, encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


def case_missing_images_referenced(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    for f in (b / "images").iterdir():  # delete the actual image files; xml still lists them
        f.unlink()
    return _zip_dir(b, work / "bundle.zip")


def case_zero_point_ply(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    _write_dense_ply(b / "dense.ply",
                     np.zeros((0, 3), dtype=np.float32),
                     np.zeros((0, 3), dtype=np.uint8))
    return _zip_dir(b, work / "bundle.zip")


def case_one_point_ply(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    _write_dense_ply(b / "dense.ply",
                     np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
                     np.array([[200, 100, 50]], dtype=np.uint8))
    return _zip_dir(b, work / "bundle.zip")


def case_nan_inf_ply(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
        [np.inf, -np.inf, 1.0],
        [1.0, 2.0, 3.0],
    ], dtype=np.float32)
    rgb = np.array([[10, 20, 30]] * 4, dtype=np.uint8)
    _write_dense_ply(b / "dense.ply", xyz, rgb)
    return _zip_dir(b, work / "bundle.zip")


def case_path_traversal(work: Path) -> Path:
    z = work / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../etc/passwd", "root:x:0:0")
        zf.writestr("cameras.xml", "<doc/>")
    return z


def case_zero_byte_images(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    for f in (b / "images").iterdir():
        f.write_bytes(b"")  # truncate every image to 0 bytes
    return _zip_dir(b, work / "bundle.zip")


def case_duplicate_labels(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xml = (b / "cameras.xml").read_text(encoding="utf-8")
    # Force every camera to share the same label.
    import re
    xml2 = re.sub(r'label="cam_\d+\.png"', 'label="cam_000.png"', xml)
    (b / "cameras.xml").write_text(xml2, encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


def case_camera_no_sensor(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xml = (b / "cameras.xml").read_text(encoding="utf-8")
    # Remove the entire <sensors>...</sensors> block but keep the cameras.
    import re
    xml2 = re.sub(r"<sensors.*?</sensors>", "", xml, flags=re.DOTALL)
    (b / "cameras.xml").write_text(xml2, encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


def case_giant_random_zip(work: Path, *, size_bytes: int) -> Path:
    z = work / "random.zip"
    chunk = os.urandom(1024 * 1024)  # 1 MiB of random bytes
    written = 0
    with open(z, "wb") as fh:
        while written < size_bytes:
            fh.write(chunk)
            written += len(chunk)
    return z


def case_wild_resolutions(work: Path) -> Path:
    b = _good_bundle_dir(work / "src", n_cameras=2, image_size=64, n_points=120)
    imgs = sorted((b / "images").glob("*.png"))
    # Replace image files: one tiny (100px), one genuinely huge so that
    # width*height (50000*2000 = 1e8) exceeds PIL's decompression-bomb limit
    # (~89M px), exercising the DecompressionBombError path on decode.
    Image.new("RGB", (100, 100), (120, 120, 120)).save(imgs[0])
    big = Image.new("RGB", (50000, 2000), (120, 120, 120))  # > MAX_IMAGE_PIXELS
    big.save(imgs[1])
    # The bundle shares one sensor declaring 64px; leave xml as-is so the
    # actual image dimensions disagree with cameras.xml (resolution mismatch).
    return _zip_dir(b, work / "bundle.zip")


def case_negative_focal(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    xml = (b / "cameras.xml").read_text(encoding="utf-8")
    import re
    xml2 = re.sub(r"<f>[^<]*</f>", "<f>-1234.5</f>", xml)
    (b / "cameras.xml").write_text(xml2, encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


def case_corrupt_manifest(work: Path) -> Path:
    b = _good_bundle_dir(work / "src")
    (b / "manifest.json").write_text("{ this is not: valid json,,, ]", encoding="utf-8")
    return _zip_dir(b, work / "bundle.zip")


CASES = [
    ("1. empty zip (no files)", case_empty_zip),
    ("2. cameras.xml but no dense.ply", case_cameras_no_ply),
    ("3. dense.ply but no cameras.xml", case_ply_no_cameras),
    ("4. cameras.xml + dense.ply, empty images/", case_empty_images),
    ("5. corrupt cameras.xml (truncated mid-tag)", case_truncated_cameras_xml),
    ("6. cameras.xml with 0 aligned cameras", case_zero_aligned_cameras),
    ("7. cameras.xml referencing missing images", case_missing_images_referenced),
    ("8. dense.ply with 0 points", case_zero_point_ply),
    ("9. dense.ply with 1 point", case_one_point_ply),
    ("10. dense.ply with NaN/Inf coordinates", case_nan_inf_ply),
    ("11. zip with path traversal entries", case_path_traversal),
    ("12. valid structure, 0-byte images", case_zero_byte_images),
    ("13. cameras.xml with duplicate labels", case_duplicate_labels),
    ("14. camera but no sensor calibration", case_camera_no_sensor),
    ("16. images of wildly different resolutions", case_wild_resolutions),
    ("17. cameras.xml with negative focal length", case_negative_focal),
    ("18. corrupt manifest.json", case_corrupt_manifest),
]


def _run_one(name: str, builder, jobroot: Path) -> tuple[str, str, str]:
    """Return (name, verdict, detail). verdict in {GOOD-FAIL, GOOD-DONE, BAD-CRASH, BAD-NOERR}."""
    casedir = jobroot / "cases" / name.split(".")[0].strip()
    casedir.mkdir(parents=True, exist_ok=True)
    try:
        zpath = builder(casedir)
    except Exception:
        return (name, "BUILDER-ERROR", traceback.format_exc())

    jid = "job-" + name.split(".")[0].strip()
    try:
        state = run_job(
            job_id=jid, bundle_zip=zpath,
            work_root=jobroot / "work", outbox_root=jobroot / "outbox",
            log_root=jobroot / "logs", gpu=GPU_24, train_fn=_stub_train,
            bundle_filename=zpath.name,
        )
    except BaseException:  # run_job promised never to raise -> this is the worst BAD
        return (name, "BAD-CRASH", traceback.format_exc())

    if state.state is State.DONE:
        return (name, "GOOD-DONE", f"trained; ply={state.outputs.final_ply}")
    if state.state is State.FAILED:
        if state.error_msg:
            return (name, "GOOD-FAIL", state.error_msg)
        return (name, "BAD-NOERR", "FAILED state but error_msg is empty/None")
    return (name, "BAD-STATE", f"unexpected terminal state: {state.state}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="also run case 15 (slow ~2GB random .zip)")
    ap.add_argument("--big-size-mb", type=int, default=2048,
                    help="size of the case-15 random zip in MiB (default 2048 = 2GB)")
    args = ap.parse_args()

    jobroot = Path(tempfile.mkdtemp(prefix="gs_fuzz_"))
    results: list[tuple[str, str, str]] = []
    try:
        for name, builder in CASES:
            results.append(_run_one(name, builder, jobroot))

        if args.big:
            big_builder = lambda w: case_giant_random_zip(w, size_bytes=args.big_size_mb * 1024 * 1024)
            results.append(_run_one(
                f"15. ~{args.big_size_mb}MiB random bytes renamed to .zip", big_builder, jobroot))
        else:
            # Quick proxy: a small random-bytes .zip exercises the same code path
            # (zipfile.BadZipFile) without writing 2 GB to disk.
            small_builder = lambda w: case_giant_random_zip(w, size_bytes=2 * 1024 * 1024)
            results.append(_run_one(
                "15. random bytes renamed to .zip (2MiB proxy; use --big for 2GB)",
                small_builder, jobroot))
    finally:
        pass

    # Report
    print("\n" + "=" * 78)
    print("FUZZ RESULTS")
    print("=" * 78)
    bad = []
    for name, verdict, detail in sorted(results, key=lambda r: int(r[0].split(".")[0])):
        tag = "GOOD" if verdict.startswith("GOOD") else "BAD "
        print(f"[{tag}] {verdict:12s} {name}")
        first_line = (detail or "").strip().splitlines()[-1] if detail else ""
        if first_line:
            print(f"           -> {first_line[:160]}")
        if not verdict.startswith("GOOD"):
            bad.append((name, verdict, detail))

    print("=" * 78)
    if bad:
        print(f"\n{len(bad)} UNGRACEFUL case(s):\n")
        for name, verdict, detail in bad:
            print(f"### {name}  [{verdict}]")
            print(detail)
            print("-" * 78)
        return 1
    print("\nAll cases failed gracefully or trained. No ungraceful crashes.")
    # Clean up the temp dir only on full success to allow inspection on failure.
    shutil.rmtree(jobroot, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
