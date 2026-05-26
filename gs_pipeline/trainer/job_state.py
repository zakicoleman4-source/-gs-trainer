"""Persisted, atomic job state shared between the worker and the Streamlit UI.

Every job has one ``state.json`` under ``work/<job_id>/``. The worker writes
to it (preflight, progress ticks, terminal state, error message); the UI reads
it on a 5-second auto-refresh to drive the dashboard.

Concurrency model:

- The worker is the *only* writer. The UI is read-only. We never need
  cross-process write locking.
- Each write is atomic: serialize to a sibling ``.tmp`` file, ``os.replace``
  it onto ``state.json``. UI readers therefore always see either the prior
  full snapshot or the new full snapshot — never a half-written file.

State machine (single linear flow with a failure sink and a resume edge):

    queued
       |  start_preflight()
       v
    preflight
       |  start_training(preflight=...)
       v
    training  --tick(step, splats, psnr?)-->  training
       |                                          ^
       |  mark_failed(msg)        mark_resuming() |
       |                          |               |
       v                          v               |
     failed                     resuming ---------+ (worker re-attaches)
       ^                                          |
       |                       finish(outputs=...)
       |                                          v
       +-----------------(no transition)----     done

Forward transitions are validated; ``done`` and ``failed`` are terminal.
Backward / invalid transitions raise ``InvalidStateTransition``.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# Schema version. Bump when the JSON layout changes in a non-back-compat way.
SCHEMA_VERSION = 1
STATE_FILENAME = "state.json"


class State(str, Enum):
    QUEUED = "queued"
    PREFLIGHT = "preflight"
    TRAINING = "training"
    RESUMING = "resuming"
    DONE = "done"
    FAILED = "failed"


# Allowed forward edges. Terminal states (DONE, FAILED) have no successors.
_ALLOWED: dict[State, frozenset[State]] = {
    State.QUEUED: frozenset({State.PREFLIGHT, State.FAILED}),
    State.PREFLIGHT: frozenset({State.TRAINING, State.FAILED}),
    State.TRAINING: frozenset({State.TRAINING, State.RESUMING, State.DONE, State.FAILED}),
    State.RESUMING: frozenset({State.TRAINING, State.FAILED}),
    State.DONE: frozenset(),
    State.FAILED: frozenset(),
}


class InvalidStateTransition(RuntimeError):
    """Raised when a transition would violate the state machine."""


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------

@dataclass
class PreflightSnapshot:
    n_cameras: int
    total_megapixels: float
    dense_pts: int
    target_splats: int
    hard_cap_splats: int
    iterations: int
    downscale_factor: float
    image_max_side: int
    quality_preset: str
    gpu_name: str
    gpu_total_vram_bytes: int
    notes: list[str] = field(default_factory=list)
    downscale_per_camera: list[float] = field(default_factory=list)


@dataclass
class ProgressSnapshot:
    current_step: int = 0
    current_splats: int = 0
    # List of (step, value) pairs; the UI plots these as curves.
    psnr_history: list[list[float]] = field(default_factory=list)
    ssim_history: list[list[float]] = field(default_factory=list)
    loss_history: list[list[float]] = field(default_factory=list)


@dataclass
class OutputsSnapshot:
    checkpoints: list[str] = field(default_factory=list)
    preview_png: Optional[str] = None
    preview_strip_png: Optional[str] = None   # path to latest 3-panel strip
    timelapse_mp4: Optional[str] = None       # path to training timelapse (written at end)
    final_ply: Optional[str] = None
    metrics_csv: Optional[str] = None
    report_json: Optional[str] = None


@dataclass
class JobState:
    job_id: str
    state: State = State.QUEUED
    status_msg: str = "queued"
    schema_version: int = SCHEMA_VERSION
    started_at: Optional[str] = None
    last_update_at: Optional[str] = None
    preflight: Optional[PreflightSnapshot] = None
    progress: ProgressSnapshot = field(default_factory=ProgressSnapshot)
    outputs: OutputsSnapshot = field(default_factory=OutputsSnapshot)
    error_msg: Optional[str] = None
    bundle_filename: Optional[str] = None  # original .zip name in inbox/

    # ----- transitions ----------------------------------------------------

    def _transition(self, new: State, *, msg: str) -> None:
        if new not in _ALLOWED[self.state]:
            raise InvalidStateTransition(
                f"cannot go from {self.state.value!r} to {new.value!r}"
            )
        self.state = new
        self.status_msg = msg
        now = _now_iso()
        if self.started_at is None and new != State.QUEUED:
            self.started_at = now
        self.last_update_at = now

    def start_preflight(self) -> None:
        self._transition(State.PREFLIGHT, msg="running preflight")

    def start_training(self, preflight: PreflightSnapshot) -> None:
        self.preflight = preflight
        self._transition(
            State.TRAINING,
            msg=f"training 0/{preflight.iterations}",
        )

    def tick(
        self,
        *,
        current_step: int,
        current_splats: int,
        psnr: Optional[float] = None,
        ssim: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> None:
        if self.state not in (State.TRAINING, State.RESUMING):
            raise InvalidStateTransition(
                f"tick() requires state TRAINING/RESUMING; current={self.state.value}"
            )
        # Coming out of RESUMING; flip to TRAINING.
        if self.state == State.RESUMING:
            self.state = State.TRAINING
        self.progress.current_step = int(current_step)
        self.progress.current_splats = int(current_splats)
        if psnr is not None:
            self.progress.psnr_history.append([int(current_step), float(psnr)])
        if ssim is not None:
            self.progress.ssim_history.append([int(current_step), float(ssim)])
        if loss is not None:
            self.progress.loss_history.append([int(current_step), float(loss)])
        total = self.preflight.iterations if self.preflight else 0
        self.status_msg = f"training {current_step}/{total}"
        self.last_update_at = _now_iso()

    def mark_resuming(self, *, msg: str = "resuming after restart") -> None:
        # Resuming is only meaningful from TRAINING (e.g. on container restart).
        if self.state == State.TRAINING:
            self.state = State.RESUMING
            self.status_msg = msg
            self.last_update_at = _now_iso()
            return
        raise InvalidStateTransition(
            f"cannot mark_resuming from {self.state.value!r}"
        )

    def finish(self, outputs: OutputsSnapshot) -> None:
        self.outputs = outputs
        self._transition(State.DONE, msg="done")

    def mark_failed(self, msg: str) -> None:
        # FAIL is allowed from any non-terminal state.
        if self.state in (State.DONE, State.FAILED):
            raise InvalidStateTransition(
                f"already terminal ({self.state.value}); cannot fail"
            )
        self.state = State.FAILED
        self.status_msg = "failed"
        self.error_msg = msg
        self.last_update_at = _now_iso()

    # ----- IO -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "state": self.state.value,
            "status_msg": self.status_msg,
            "started_at": self.started_at,
            "last_update_at": self.last_update_at,
            "preflight": asdict(self.preflight) if self.preflight else None,
            "progress": asdict(self.progress),
            "outputs": asdict(self.outputs),
            "error_msg": self.error_msg,
            "bundle_filename": self.bundle_filename,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobState":
        schema = int(d.get("schema_version", SCHEMA_VERSION))
        if schema != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported state.json schema_version={schema} "
                f"(expected {SCHEMA_VERSION})"
            )
        preflight_d = d.get("preflight")
        if preflight_d:
            valid_keys = {f.name for f in __import__("dataclasses").fields(PreflightSnapshot)}
            preflight = PreflightSnapshot(**{k: v for k, v in preflight_d.items() if k in valid_keys})
        else:
            preflight = None
        progress_d = d.get("progress") or {}
        outputs_d = d.get("outputs") or {}
        return cls(
            job_id=d["job_id"],
            state=State(d.get("state", "queued")),
            status_msg=d.get("status_msg", ""),
            schema_version=schema,
            started_at=d.get("started_at"),
            last_update_at=d.get("last_update_at"),
            preflight=preflight,
            progress=ProgressSnapshot(**progress_d) if progress_d else ProgressSnapshot(),
            outputs=OutputsSnapshot(**outputs_d) if outputs_d else OutputsSnapshot(),
            error_msg=d.get("error_msg"),
            bundle_filename=d.get("bundle_filename"),
        )


# ---------------------------------------------------------------------------
# Atomic file IO
# ---------------------------------------------------------------------------

def state_path_for(work_dir: Path, job_id: str) -> Path:
    return Path(work_dir) / job_id / STATE_FILENAME


def write_state(state: JobState, path: Path) -> None:
    """Atomically write JobState as JSON to ``path``.

    Writes to a sibling tempfile and ``os.replace`` onto the target — readers
    never see a half-written file. ``path.parent`` is created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.to_dict(), indent=2, ensure_ascii=False, sort_keys=False)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".state.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # tmpfs / overlayfs sometimes refuses fsync; the os.replace
                # below is still atomic at the directory entry level.
                pass
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_state(path: Path) -> JobState:
    """Load a JobState from disk. Raises FileNotFoundError / ValueError."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    d = json.loads(raw)
    return JobState.from_dict(d)


def safe_read_state(path: Path) -> Optional[JobState]:
    """Same as ``read_state`` but returns ``None`` on any parse / IO error.

    The UI polls this; we never want a transient partial read to crash the
    page. (In practice atomic writes mean we never see a partial read, but
    being defensive on the read side is cheap.)
    """
    try:
        return read_state(path)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_job_state(job_id: str, bundle_filename: Optional[str] = None) -> JobState:
    """Construct a fresh QUEUED JobState with current timestamps."""
    now = _now_iso()
    return JobState(
        job_id=job_id,
        state=State.QUEUED,
        status_msg="queued",
        last_update_at=now,
        bundle_filename=bundle_filename,
    )
