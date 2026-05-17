"""Shared pytest fixtures for the gs_pipeline test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `import gs_pipeline.<...>` resolve when pytest is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def pytest_collection_modifyitems(config, items):
    """Auto-skip @pytest.mark.gpu tests when CUDA isn't available."""
    if _has_cuda():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA / torch available in this environment")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def tmp_job_root(tmp_path: Path) -> Path:
    """A throwaway directory mimicking the docker-mounted job tree."""
    for sub in ("inbox", "outbox", "logs", "work", "config"):
        (tmp_path / sub).mkdir()
    return tmp_path
