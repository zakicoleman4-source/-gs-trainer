"""OOM hardening + progress watchdog for the trainer.

Two failure modes this module exists to contain:

1. **CUDA OOM mid-train.** Even with a careful budget the rasterizer can spike
   memory during MCMC relocate or densify. We catch ``torch.cuda.OutOfMemoryError``
   (detected by type name so this module is importable without torch), run a
   user-supplied shrinker (e.g. halve cap_max, halve image resolution), free
   the cache, and retry. Shrinkers are tried in order; if all fail, the
   exception is re-raised and the caller marks the job failed in the UI.

2. **Silent stalls.** A trainer can hang on a degenerate batch or a CUDA
   driver bug without raising. ``ProgressWatchdog`` lets the trainer call
   ``tick(step)`` periodically; if no tick has happened in ``timeout_s``, the
   watchdog raises ``TrainingStalled`` from the polling thread. The pipeline
   subprocess gets killed by the parent watcher.

Both pieces are torch-optional: the OOM-detection predicate works on any
exception with the right type name or message, the memory-fraction setter is
a no-op when torch isn't installed, and the watchdog is pure Python. CPU
unit tests can exercise everything without CUDA.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, TypeVar

_log = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# CUDA OOM detection / cleanup (torch-optional)
# ---------------------------------------------------------------------------

def is_cuda_oom(exc: BaseException) -> bool:
    """Return True if ``exc`` looks like a CUDA out-of-memory error.

    Detection is by class name and message substring so this module doesn't
    have to import torch. Matches:
      - ``torch.cuda.OutOfMemoryError`` (class name ``OutOfMemoryError``)
      - older torch versions' ``RuntimeError`` with "CUDA out of memory" text
      - cuBLAS/cuDNN OOMs that surface as ``RuntimeError`` with "out of memory"
    """
    if exc is None:
        return False
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def clear_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()``. No-op without torch/CUDA."""
    try:
        import torch  # type: ignore
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    try:
        torch.cuda.synchronize()
    except Exception:
        pass


def set_memory_fraction(fraction: float = 0.92, device: int = 0) -> bool:
    """Set per-process VRAM cap. Returns True iff applied.

    Used once at trainer startup. We deliberately bound at 0.92 (default) to
    leave room for the rasterizer's tile-binning scratch, which torch's
    accounting can underestimate by ~1 GB at peak.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")
    try:
        import torch  # type: ignore
    except Exception:
        return False
    if not torch.cuda.is_available():
        return False
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return True


# ---------------------------------------------------------------------------
# Retry with shrinkers
# ---------------------------------------------------------------------------

@dataclass
class RetryAttempt:
    attempt_index: int       # 0 = first try, 1 = first retry, ...
    shrinker_name: Optional[str]  # None on the first attempt
    exception: Optional[BaseException]
    succeeded: bool


@dataclass
class RetryResult:
    succeeded: bool
    attempts: list[RetryAttempt] = field(default_factory=list)
    return_value: object = None

    @property
    def final_exception(self) -> Optional[BaseException]:
        for a in reversed(self.attempts):
            if a.exception is not None:
                return a.exception
        return None


Shrinker = Callable[[], None]


def run_with_oom_retry(
    fn: Callable[[], T],
    shrinkers: Iterable[Shrinker],
    *,
    clear_between: bool = True,
    catch: Callable[[BaseException], bool] = is_cuda_oom,
) -> RetryResult:
    """Run ``fn``; on OOM apply the next shrinker and retry. Returns a RetryResult.

    Args:
        fn: zero-arg callable; will be invoked once normally, then after each
            shrinker. Should be idempotent w.r.t. its own state (the trainer
            reads its config dict each call, so this is naturally true).
        shrinkers: ordered iterable of zero-arg callables. Each one mutates
            external state (config dict, target_splats, image_max_side, ...)
            and returns None. They are applied in order; after exhausting
            them, the last exception is preserved on the result.
        clear_between: whether to call ``clear_cuda_cache()`` before each retry.
        catch: predicate; only exceptions matching it trigger retry. Other
            exceptions propagate immediately.

    The function does **not** raise on OOM exhaustion; the caller inspects
    ``result.succeeded`` and ``result.final_exception`` and decides whether to
    log + fail the job.
    """
    shrinkers = list(shrinkers)
    result = RetryResult(succeeded=False)
    attempt_index = 0
    shrinker_name: Optional[str] = None

    while True:
        try:
            value = fn()
        except BaseException as exc:
            if not catch(exc):
                # Not an OOM-style error — propagate.
                raise
            _log.warning(
                "OOM-style failure on attempt %d (shrinker=%s): %s",
                attempt_index, shrinker_name, exc,
            )
            result.attempts.append(RetryAttempt(
                attempt_index=attempt_index,
                shrinker_name=shrinker_name,
                exception=exc,
                succeeded=False,
            ))
            if attempt_index >= len(shrinkers):
                # Out of shrinkers; fail the job.
                return result
            if clear_between:
                clear_cuda_cache()
            # Apply the next shrinker before retrying.
            next_shrinker = shrinkers[attempt_index]
            shrinker_name = getattr(next_shrinker, "__name__", "shrinker") or "shrinker"
            try:
                next_shrinker()
            except BaseException as shrink_exc:
                _log.error("shrinker %s itself raised %s; giving up", shrinker_name, shrink_exc)
                result.attempts.append(RetryAttempt(
                    attempt_index=attempt_index + 1,
                    shrinker_name=shrinker_name,
                    exception=shrink_exc,
                    succeeded=False,
                ))
                return result
            attempt_index += 1
            continue
        # Success path.
        result.attempts.append(RetryAttempt(
            attempt_index=attempt_index,
            shrinker_name=shrinker_name,
            exception=None,
            succeeded=True,
        ))
        result.return_value = value
        result.succeeded = True
        return result


# ---------------------------------------------------------------------------
# Progress watchdog
# ---------------------------------------------------------------------------

class TrainingStalled(RuntimeError):
    """Raised by ProgressWatchdog when no tick arrives in time."""


class ProgressWatchdog:
    """Detect stalled training.

    The trainer calls ``tick(step)`` periodically (e.g. every iteration).
    A background thread polls the wall-clock delta since the last tick; if it
    exceeds ``timeout_s``, the watchdog calls ``on_stall(last_step)``.

    The default ``on_stall`` callback raises ``TrainingStalled``, but because
    that runs on the watchdog thread it doesn't actually unwind the trainer.
    The wrapping subprocess in ``watcher.py`` listens for SIGTERM from the
    parent — the recommended pattern is::

        wd = ProgressWatchdog(timeout_s=900, on_stall=lambda step: os.kill(os.getpid(), signal.SIGTERM))
        wd.start()
        try:
            for step in train_loop():
                wd.tick(step)
        finally:
            wd.stop()
    """

    def __init__(
        self,
        timeout_s: float,
        *,
        poll_interval_s: float = 5.0,
        on_stall: Optional[Callable[[int], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0; got {timeout_s}")
        if poll_interval_s <= 0:
            raise ValueError(f"poll_interval_s must be > 0; got {poll_interval_s}")
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.on_stall = on_stall or _default_on_stall
        self._clock = clock
        self._lock = threading.Lock()
        self._last_step = 0
        self._last_tick = self._clock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stalled = False

    def tick(self, step: int) -> None:
        with self._lock:
            self._last_step = step
            self._last_tick = self._clock()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ProgressWatchdog", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            with self._lock:
                elapsed = self._clock() - self._last_tick
                last_step = self._last_step
            if elapsed > self.timeout_s:
                self.stalled = True
                _log.warning(
                    "Training stalled: no tick for %.1fs at step %d",
                    elapsed, last_step,
                )
                try:
                    self.on_stall(last_step)
                except BaseException:
                    _log.exception("on_stall callback raised")
                return  # one-shot

    # Context-manager sugar so callers can do `with ProgressWatchdog(...) as wd:`.
    def __enter__(self) -> "ProgressWatchdog":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def _default_on_stall(last_step: int) -> None:
    raise TrainingStalled(f"no progress past step {last_step}")


# ---------------------------------------------------------------------------
# Context manager combining the cache-clear and the memory fraction
# ---------------------------------------------------------------------------

@contextmanager
def trainer_memory_guard(*, fraction: float = 0.92, device: int = 0):
    """Set memory fraction on entry, clear cache on exit. Safe without CUDA."""
    set_memory_fraction(fraction=fraction, device=device)
    try:
        yield
    finally:
        clear_cuda_cache()
