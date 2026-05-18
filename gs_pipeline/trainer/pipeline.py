"""Per-job orchestration: unzip -> parse -> init -> budget -> train -> export.

Called by ``watcher.py`` as a subprocess so a CUDA crash never kills the
daemon or the UI. The actual GPU training is reached through a single
injectable callable (``train_fn``), which lets CPU tests drive the full
orchestration with a stub trainer.

State transitions (also reflected in ``state.json``)::

    queued -> preflight -> training -> done       on success
                       \\-> failed                 on parse / budget failure
                          training -> failed      on trainer exception
"""
from __future__ import annotations

import logging
import shutil
import traceback
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Protocol

from gs_pipeline.trainer.budget import (
    Budget,
    GPUInfo,
    compute_budget,
    detect_gpu,
)
from gs_pipeline.trainer.init_from_pcd import InitCloud, load_and_downsample
from gs_pipeline.trainer.job_state import (
    JobState,
    OutputsSnapshot,
    PreflightSnapshot,
    new_job_state,
    state_path_for,
    write_state,
)
from gs_pipeline.trainer.parse_metashape import ParsedScene, parse_cameras_xml

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def job_work_dir(work_root: Path, job_id: str) -> Path:
    return Path(work_root) / job_id


def job_outbox_dir(outbox_root: Path, job_id: str) -> Path:
    return Path(outbox_root) / job_id


def job_log_dir(log_root: Path, job_id: str) -> Path:
    return Path(log_root) / job_id


# ---------------------------------------------------------------------------
# Train function protocol (lets tests inject a stub)
# ---------------------------------------------------------------------------

class TrainFn(Protocol):
    """Signature the GPU trainer must satisfy. See ``train_mcmc.train``."""
    def __call__(
        self,
        *,
        scene: ParsedScene,
        init_cloud: InitCloud,
        budget: Budget,
        config: object,           # TrainerConfig in production; opaque to orchestration
        job_state: JobState,
        job_state_path: Path,
        work_dir: Path,
        outbox_dir: Path,
    ) -> OutputsSnapshot: ...


def _default_train_fn(**kwargs) -> OutputsSnapshot:
    """Lazy wrapper around ``train_mcmc.train`` so importing this module
    doesn't pull in torch / gsplat."""
    from gs_pipeline.trainer.train_mcmc import train, load_trainer_config
    cfg_path = kwargs.pop("config_yaml", None)
    if cfg_path is not None and kwargs.get("config") is None:
        kwargs["config"] = load_trainer_config(cfg_path)
    return train(**kwargs)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def run_job(
    *,
    job_id: str,
    bundle_zip: Path,
    work_root: Path,
    outbox_root: Path,
    log_root: Path,
    config_yaml: Optional[Path] = None,
    gpu: Optional[GPUInfo] = None,
    quality_preset: str = "Auto",
    train_fn: Optional[TrainFn] = None,
    bundle_filename: Optional[str] = None,
) -> JobState:
    """Run one job end-to-end. Writes ``state.json`` updates along the way.

    On success: returns a JobState in ``DONE``, outputs populated.
    On failure: returns a JobState in ``FAILED`` with ``error_msg`` set; never
    raises (callers shouldn't have to wrap this in try/except).
    """
    work_root = Path(work_root)
    outbox_root = Path(outbox_root)
    log_root = Path(log_root)

    work_dir = job_work_dir(work_root, job_id)
    outbox_dir = job_outbox_dir(outbox_root, job_id)
    log_dir = job_log_dir(log_root, job_id)
    work_dir.mkdir(parents=True, exist_ok=True)
    outbox_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_path_for(work_root, job_id)

    js = new_job_state(job_id, bundle_filename=bundle_filename or Path(bundle_zip).name)
    write_state(js, state_path)

    try:
        js.start_preflight()
        write_state(js, state_path)

        bundle_dir = _unzip(bundle_zip, work_dir / "bundle")
        scene, init_cloud, budget = _preflight(
            bundle_dir=bundle_dir, gpu=gpu, quality_preset=quality_preset,
        )
        preflight = _preflight_snapshot(budget)
        js.start_training(preflight)
        write_state(js, state_path)

        trainer = train_fn or _default_train_fn
        outputs = trainer(
            scene=scene, init_cloud=init_cloud, budget=budget,
            config=None, config_yaml=config_yaml,
            job_state=js, job_state_path=state_path,
            work_dir=work_dir, outbox_dir=outbox_dir,
        )
        js.finish(outputs)
        write_state(js, state_path)
        _log.info("job %s done; outputs=%s", job_id, asdict(outputs))
        return js

    except BaseException as exc:  # broad: this function must never re-raise
        tb = traceback.format_exc()
        (log_dir / "pipeline_error.txt").write_text(tb, encoding="utf-8")
        try:
            js.mark_failed(f"{type(exc).__name__}: {exc}")
        except Exception:
            # If the state machine itself rejects (e.g. already terminal),
            # write the raw error and move on.
            js.error_msg = f"{type(exc).__name__}: {exc}"
        write_state(js, state_path)
        _log.exception("job %s failed", job_id)
        return js


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _unzip(bundle_zip: Path, dest: Path) -> Path:
    bundle_zip = Path(bundle_zip)
    if not bundle_zip.is_file():
        raise FileNotFoundError(bundle_zip)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(bundle_zip) as zf:
        # Reject path traversal entries before extracting anything.
        for name in zf.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"bundle contains unsafe path {name!r}")
        zf.extractall(dest)
    return dest


def _preflight(
    *,
    bundle_dir: Path,
    gpu: Optional[GPUInfo],
    quality_preset: str,
) -> tuple[ParsedScene, InitCloud, Budget]:
    bundle_dir = Path(bundle_dir)
    cameras_xml = bundle_dir / "cameras.xml"
    dense_ply = bundle_dir / "dense.ply"
    images_dir = bundle_dir / "images"
    if not cameras_xml.is_file():
        raise FileNotFoundError(f"missing cameras.xml in bundle {bundle_dir}")
    if not dense_ply.is_file():
        raise FileNotFoundError(f"missing dense.ply in bundle {bundle_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"missing images/ in bundle {bundle_dir}")

    scene = parse_cameras_xml(cameras_xml, image_dir=images_dir)
    if not scene.image_paths or len(scene.image_paths) != len(scene):
        raise ValueError(
            f"{len(scene)} aligned cameras but only {len(scene.image_paths)} "
            f"images resolved under {images_dir}"
        )
    init_cloud = load_and_downsample(dense_ply)
    gpu_info = gpu or detect_gpu()
    if gpu_info is None:
        # Fall back to a synthetic 24 GB GPU so preflight numbers are sensible
        # for previewing in tests / dry-runs. Real training will fail later if
        # there is no actual GPU.
        gpu_info = GPUInfo(name="(synthetic) 24GB", total_vram_bytes=24_000_000_000)
    budget = compute_budget(
        gpu=gpu_info,
        image_sizes=scene.image_sizes,
        dense_pts=int(init_cloud.xyz.shape[0]),
        quality_preset=quality_preset,
    )
    return scene, init_cloud, budget


def _preflight_snapshot(budget: Budget) -> PreflightSnapshot:
    return PreflightSnapshot(
        n_cameras=budget.n_cameras,
        total_megapixels=budget.total_megapixels,
        dense_pts=budget.dense_pts,
        target_splats=budget.target_splats,
        hard_cap_splats=budget.hard_cap_splats,
        iterations=budget.iterations,
        downscale_factor=budget.downscale_factor,
        image_max_side=budget.image_max_side,
        quality_preset=budget.quality_preset,
        gpu_name=budget.gpu.name,
        gpu_total_vram_bytes=budget.gpu.total_vram_bytes,
        notes=list(budget.notes),
    )


# ---------------------------------------------------------------------------
# CLI (subprocess target invoked by watcher.process_one_subprocess)
# ---------------------------------------------------------------------------

def _cli_main(argv: Optional[list] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Run one gs_pipeline job (subprocess target).")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--bundle-zip", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--outbox-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--quality-preset", default="Auto", choices=["Auto", "Maximum"])
    parser.add_argument("--config-yaml", type=Path, default=None)
    args = parser.parse_args(argv)

    state = run_job(
        job_id=args.job_id,
        bundle_zip=args.bundle_zip,
        work_root=args.work_root,
        outbox_root=args.outbox_root,
        log_root=args.log_root,
        config_yaml=args.config_yaml,
        quality_preset=args.quality_preset,
    )
    return 0 if state.state.value == "done" else 1


if __name__ == "__main__":
    raise SystemExit(_cli_main())
