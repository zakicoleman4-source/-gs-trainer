"""Structural tests for the docker/ assets.

A real `docker compose build` needs the nvidia container toolkit and is run
on the GPU host as part of slice 14's acceptance. Here we verify that the
files exist, parse, reference the right entry points, expose the expected
ports / mounts / env, and stay in sync with each other (e.g. supervisord
points at the watcher entry that watcher.py actually exposes).
"""
from __future__ import annotations

import configparser
import re
import stat
from pathlib import Path

import pytest
import yaml


DOCKER_DIR = Path(__file__).resolve().parent.parent / "docker"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

def _read_dockerfile() -> str:
    return (DOCKER_DIR / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_present_and_nonempty():
    text = _read_dockerfile()
    assert text.strip(), "Dockerfile is empty"


def test_dockerfile_base_image_is_cuda_devel():
    text = _read_dockerfile()
    # ARG default points at a cuda devel image (devel, not runtime, because
    # we need nvcc to compile gsplat).
    assert re.search(r"ARG\s+CUDA_IMAGE=nvidia/cuda:.*devel-ubuntu", text), text
    assert "FROM ${CUDA_IMAGE}" in text


def test_dockerfile_pins_torch_and_gsplat():
    text = _read_dockerfile()
    assert "torch==2.4.1" in text
    assert "--index-url https://download.pytorch.org/whl/cu124" in text
    assert "gsplat==1.4.0" in text
    # No-build-isolation is required so setup.py can see torch's headers.
    assert "--no-build-isolation" in text


def test_dockerfile_compiles_gsplat_with_max_jobs():
    """We pin MAX_JOBS to keep nvcc memory in check on small build hosts."""
    text = _read_dockerfile()
    assert re.search(r"\bMAX_JOBS=\d+", text)


def test_dockerfile_creates_data_dirs():
    text = _read_dockerfile()
    for sub in ("/data/inbox", "/data/outbox", "/data/logs", "/data/work", "/data/config"):
        assert sub in text


def test_dockerfile_exposes_8501_and_sets_streamlit_headless():
    text = _read_dockerfile()
    assert "EXPOSE 8501" in text
    assert "STREAMLIT_SERVER_HEADLESS=true" in text
    assert "STREAMLIT_SERVER_PORT=8501" in text


def test_dockerfile_entrypoint_is_script():
    text = _read_dockerfile()
    assert 'ENTRYPOINT ["/entrypoint.sh"]' in text


def test_dockerfile_copy_sources_resolve_against_build_context():
    """Each COPY src must exist relative to the compose build context.

    docker-compose.yml sets `context: ..` from gs_pipeline/docker/, so the
    context root is the gs_pipeline/ package directory. A bad COPY path
    (e.g. `COPY gs_pipeline/` when no nested gs_pipeline/gs_pipeline/
    exists) only surfaces at image-build time; this test fails the unit
    suite instead.
    """
    text = _read_dockerfile()
    context_root = DOCKER_DIR.parent
    for raw in re.findall(r"^\s*COPY\s+(.+)$", text, flags=re.MULTILINE):
        tokens = raw.split()
        if not tokens or tokens[0].startswith("--"):
            continue  # skip COPY --from=... and other flagged forms
        srcs = tokens[:-1]
        for src in srcs:
            resolved = (context_root / src).resolve()
            assert resolved.exists(), (
                f"COPY source {src!r} does not exist under build context "
                f"{context_root} (resolved to {resolved})"
            )


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

def _read_compose() -> dict:
    return yaml.safe_load((DOCKER_DIR / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_defines_one_service():
    data = _read_compose()
    assert "services" in data
    assert "gs_pipeline" in data["services"]


def test_compose_publishes_streamlit_port():
    svc = _read_compose()["services"]["gs_pipeline"]
    assert "8501:8501" in svc.get("ports", [])


def test_compose_mounts_all_data_dirs():
    svc = _read_compose()["services"]["gs_pipeline"]
    mounts = svc.get("volumes", [])
    targets = {m.split(":")[1] for m in mounts}
    assert {"/data/inbox", "/data/outbox", "/data/logs", "/data/work", "/data/config"}.issubset(targets)


def test_compose_requests_nvidia_gpu():
    svc = _read_compose()["services"]["gs_pipeline"]
    devices = svc.get("deploy", {}).get("resources", {}).get("reservations", {}).get("devices", [])
    assert any(d.get("driver") == "nvidia" and "gpu" in d.get("capabilities", []) for d in devices)


def test_compose_default_env_knobs_present():
    svc = _read_compose()["services"]["gs_pipeline"]
    env = svc.get("environment", {})
    if isinstance(env, list):
        env = dict(e.split("=", 1) for e in env)
    for key in ("MAX_UPLOAD_GB", "QUALITY_PRESET_DEFAULT"):
        assert key in env


# ---------------------------------------------------------------------------
# supervisord.conf
# ---------------------------------------------------------------------------

def _read_supervisord() -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, strict=False)
    cp.read(DOCKER_DIR / "supervisord.conf")
    return cp


def test_supervisord_defines_ui_and_worker():
    cp = _read_supervisord()
    sections = set(cp.sections())
    assert "program:ui" in sections
    assert "program:worker" in sections


def test_supervisord_ui_runs_streamlit_on_app_py():
    cp = _read_supervisord()
    cmd = cp.get("program:ui", "command")
    assert cmd.startswith("streamlit run")
    assert "gs_pipeline/ui/app.py" in cmd


def test_supervisord_worker_runs_watcher_module():
    cp = _read_supervisord()
    cmd = " ".join(cp.get("program:worker", "command").split())
    assert "python -m gs_pipeline.trainer.watcher" in cmd
    # And passes through the four mount paths.
    for token in ("--inbox /data/inbox", "--work /data/work",
                  "--outbox /data/outbox", "--logs /data/logs"):
        assert token in cmd


def test_supervisord_processes_have_autorestart():
    cp = _read_supervisord()
    for sect in ("program:ui", "program:worker"):
        assert cp.get(sect, "autorestart") == "true"


def test_supervisord_worker_cli_matches_watcher_argparser():
    """Defence-in-depth: every CLI flag used by supervisord must also be
    accepted by watcher.watcher_main's argparse, so a typo on either side
    fails the build instead of silently breaking the container."""
    import argparse
    import inspect

    from gs_pipeline.trainer import watcher

    src = inspect.getsource(watcher.watcher_main)
    # Each --flag in supervisord must appear in watcher_main's argparser.
    cp = _read_supervisord()
    cmd_text = cp.get("program:worker", "command")
    for token in re.findall(r"--[a-z-]+", cmd_text):
        assert token in src, f"watcher_main has no argparse flag {token!r}"


# ---------------------------------------------------------------------------
# entrypoint.sh
# ---------------------------------------------------------------------------

def test_entrypoint_shebang_and_supervisord_exec():
    p = DOCKER_DIR / "entrypoint.sh"
    text = p.read_text(encoding="utf-8")
    assert text.startswith("#!/"), "entrypoint.sh missing shebang"
    assert "exec /usr/bin/supervisord" in text


def test_entrypoint_is_executable():
    p = DOCKER_DIR / "entrypoint.sh"
    mode = p.stat().st_mode
    assert mode & stat.S_IXUSR, "entrypoint.sh is not executable; chmod +x"
