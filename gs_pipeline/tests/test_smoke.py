"""Sanity check: the test harness imports the package and fixtures work."""
from __future__ import annotations

from pathlib import Path

import gs_pipeline


def test_package_importable():
    assert hasattr(gs_pipeline, "__all__")


def test_tmp_job_root_fixture(tmp_job_root: Path):
    for sub in ("inbox", "outbox", "logs", "work", "config"):
        assert (tmp_job_root / sub).is_dir()
