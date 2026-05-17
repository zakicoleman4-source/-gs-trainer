"""Tests for ``gs_pipeline.trainer.oom_guard`` (no torch, no CUDA)."""
from __future__ import annotations

import threading
import time
from typing import Callable, List

import pytest

from gs_pipeline.trainer.oom_guard import (
    ProgressWatchdog,
    TrainingStalled,
    clear_cuda_cache,
    is_cuda_oom,
    run_with_oom_retry,
    set_memory_fraction,
    trainer_memory_guard,
)


# ---------------------------------------------------------------------------
# Fake OOMs (we don't have torch in this env)
# ---------------------------------------------------------------------------

class FakeCudaOOM(Exception):
    """Mirrors torch.cuda.OutOfMemoryError by class name."""
    pass


# Trick: rename the class to OutOfMemoryError so is_cuda_oom matches it by name.
FakeCudaOOM.__name__ = "OutOfMemoryError"


def make_runtime_error_oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 12.34 GiB.")


# ---------------------------------------------------------------------------
# is_cuda_oom
# ---------------------------------------------------------------------------

def test_is_cuda_oom_by_class_name():
    assert is_cuda_oom(FakeCudaOOM("boom"))


def test_is_cuda_oom_by_message():
    assert is_cuda_oom(make_runtime_error_oom())


def test_is_cuda_oom_false_for_other_errors():
    assert not is_cuda_oom(ValueError("nope"))
    assert not is_cuda_oom(RuntimeError("unrelated runtime failure"))


def test_is_cuda_oom_none_safe():
    assert not is_cuda_oom(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# clear / set
# ---------------------------------------------------------------------------

def test_clear_cuda_cache_no_torch_no_op():
    # Should not raise even without torch.
    clear_cuda_cache()


def test_set_memory_fraction_returns_false_without_cuda():
    # No torch+CUDA -> returns False, doesn't raise.
    assert set_memory_fraction(0.9) is False


def test_set_memory_fraction_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        set_memory_fraction(0.0)
    with pytest.raises(ValueError):
        set_memory_fraction(1.5)
    with pytest.raises(ValueError):
        set_memory_fraction(-0.1)


def test_trainer_memory_guard_contextmanager_no_op():
    # Should enter and exit cleanly with no torch.
    with trainer_memory_guard(fraction=0.9):
        pass


# ---------------------------------------------------------------------------
# run_with_oom_retry
# ---------------------------------------------------------------------------

def test_first_attempt_succeeds_no_shrinkers_called():
    calls: List[str] = []
    def shrink() -> None:
        calls.append("shrink")

    def fn() -> int:
        calls.append("fn")
        return 42

    result = run_with_oom_retry(fn, shrinkers=[shrink])
    assert result.succeeded
    assert result.return_value == 42
    assert calls == ["fn"]
    assert len(result.attempts) == 1
    assert result.attempts[0].shrinker_name is None


def test_oom_then_success_after_first_shrinker():
    state = {"size": 100}
    calls: List[str] = []

    def shrink_halve():
        state["size"] //= 2
        calls.append(f"shrink->{state['size']}")

    fails_remaining = {"n": 1}

    def fn() -> str:
        calls.append(f"fn({state['size']})")
        if fails_remaining["n"] > 0:
            fails_remaining["n"] -= 1
            raise FakeCudaOOM(f"size {state['size']} too big")
        return f"ok at {state['size']}"

    result = run_with_oom_retry(fn, shrinkers=[shrink_halve, shrink_halve])
    assert result.succeeded
    assert result.return_value == "ok at 50"
    assert state["size"] == 50
    assert calls == ["fn(100)", "shrink->50", "fn(50)"]


def test_all_shrinkers_exhausted_returns_failure():
    state = {"size": 100}
    def shrink():
        state["size"] //= 2

    def fn() -> None:
        raise FakeCudaOOM(f"still too big at {state['size']}")

    result = run_with_oom_retry(fn, shrinkers=[shrink, shrink])
    assert not result.succeeded
    assert result.return_value is None
    assert isinstance(result.final_exception, FakeCudaOOM)
    # 3 attempts: initial + 2 shrinkers.
    assert sum(1 for a in result.attempts if a.exception is not None) == 3


def test_non_oom_exception_propagates_immediately():
    def shrink():
        raise AssertionError("should not be called")

    def fn():
        raise ValueError("not an OOM")

    with pytest.raises(ValueError, match="not an OOM"):
        run_with_oom_retry(fn, shrinkers=[shrink])


def test_runtime_error_oom_message_also_triggers_retry():
    fails_remaining = {"n": 1}
    def shrink():
        fails_remaining["n"] = 0

    def fn():
        if fails_remaining["n"] > 0:
            raise make_runtime_error_oom()
        return "done"

    result = run_with_oom_retry(fn, shrinkers=[shrink])
    assert result.succeeded
    assert result.return_value == "done"


def test_shrinker_that_raises_aborts_retry():
    def shrink():
        raise RuntimeError("cannot shrink further")

    def fn():
        raise FakeCudaOOM("bang")

    result = run_with_oom_retry(fn, shrinkers=[shrink])
    assert not result.succeeded
    assert isinstance(result.final_exception, RuntimeError)
    assert "cannot shrink" in str(result.final_exception)


def test_no_shrinkers_provided_one_attempt_only():
    def fn():
        raise FakeCudaOOM("first and only failure")
    result = run_with_oom_retry(fn, shrinkers=[])
    assert not result.succeeded
    assert len(result.attempts) == 1


# ---------------------------------------------------------------------------
# ProgressWatchdog
# ---------------------------------------------------------------------------

class _FakeClock:
    """Manually advanced monotonic clock for deterministic watchdog tests."""
    def __init__(self) -> None:
        self.t = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.t

    def advance(self, dt: float) -> None:
        with self._lock:
            self.t += dt


def test_watchdog_does_not_fire_when_ticked():
    """Real-time, very short timeout. Tick every poll => no stall."""
    fired = threading.Event()
    wd = ProgressWatchdog(
        timeout_s=0.5, poll_interval_s=0.05,
        on_stall=lambda step: fired.set(),
    )
    wd.start()
    try:
        for s in range(20):
            wd.tick(s)
            time.sleep(0.02)
    finally:
        wd.stop()
    assert not fired.is_set()


def test_watchdog_fires_when_silent():
    """Real-time, short timeout, never tick => stall callback fires."""
    fired = threading.Event()
    seen_step: List[int] = []
    def on_stall(step: int) -> None:
        seen_step.append(step)
        fired.set()

    wd = ProgressWatchdog(timeout_s=0.2, poll_interval_s=0.05, on_stall=on_stall)
    wd.start()
    try:
        # Establish a starting step then go silent.
        wd.tick(7)
        assert fired.wait(timeout=2.0), "watchdog should have fired"
    finally:
        wd.stop()
    assert wd.stalled
    assert seen_step == [7]


def test_watchdog_context_manager():
    fired = threading.Event()
    with ProgressWatchdog(
        timeout_s=0.5, poll_interval_s=0.05,
        on_stall=lambda step: fired.set(),
    ) as wd:
        wd.tick(1)
        time.sleep(0.05)
        wd.tick(2)
    assert not fired.is_set()


def test_watchdog_default_callback_raises_in_thread():
    """Default on_stall raises TrainingStalled — must not crash the test thread."""
    wd = ProgressWatchdog(timeout_s=0.1, poll_interval_s=0.02)
    wd.start()
    try:
        time.sleep(0.5)
        assert wd.stalled, "watchdog should have stalled"
    finally:
        wd.stop()


def test_watchdog_validates_arguments():
    with pytest.raises(ValueError):
        ProgressWatchdog(timeout_s=0)
    with pytest.raises(ValueError):
        ProgressWatchdog(timeout_s=1, poll_interval_s=0)


def test_training_stalled_is_runtime_error_subclass():
    assert issubclass(TrainingStalled, RuntimeError)
