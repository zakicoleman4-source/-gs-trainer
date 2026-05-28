# gs_pipeline Docker image: Streamlit UI + GS trainer on one NVIDIA GPU.
#
# Build:   docker compose -f gs_pipeline/docker/docker-compose.yml build
# Run:     docker compose -f gs_pipeline/docker/docker-compose.yml up -d
# URL:     http://<host>:8501
#
# Design notes (learned the hard way over many broken images):
#  - Uses Ubuntu 22.04's NATIVE python3.10. No python3.11 symlink gymnastics:
#    python3-pip works out of the box, python3-setuptools provides pkg_resources
#    (supervisor needs it), and the gsplat PREBUILT wheel only ships for cp310.
#  - gsplat is installed from its PREBUILT wheel index (docs.gsplat.studio),
#    which ships a real compiled csrc.so for sm_70..sm_90 (covers A5000 sm_86,
#    A6000 sm_80). This means NO JIT compile on first training run — the old
#    `pip install gsplat` was JIT-only and hung ~10 min on first use.
#  - setuptools is NOT upgraded past 80: setuptools 81 deprecated and 82 removed
#    pkg_resources, which made supervisord crash on startup. We keep the system
#    setuptools (59.x) which still ships pkg_resources.
#   torch  == 2.4.1 + cu124,  gsplat == 1.4.0+pt24cu124 (prebuilt)

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages (all for the native python3.10):
# - python3, python3-pip, python3-dev: interpreter + Python.h for any C build
# - python3-setuptools: provides pkg_resources (supervisor imports it)
# - supervisor: process manager for UI + worker
# - libopenblas: scipy/numpy linalg; Pillow image libs; ffmpeg for timelapse
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-pip python3-setuptools python3-venv \
    build-essential ninja-build git curl ca-certificates pkg-config \
    supervisor libgl1 libglib2.0-0 ffmpeg \
    libopenblas-dev \
    libjpeg-dev libpng-dev zlib1g-dev libtiff-dev libwebp-dev \
    libfreetype-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-side deps. python3-pip targets python3.10 natively (no symlink tricks).
# Upgrade pip/wheel only — NOT setuptools (keep the system 59.x that has
# pkg_resources; setuptools>=81 removed it and breaks supervisor).
COPY requirements-cpu.txt requirements-gpu.txt /app/
RUN python3 -m pip install --upgrade pip wheel \
 && python3 -m pip install -r requirements-cpu.txt

# Torch (cu124 wheels) for python3.10.
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1

# Other GPU deps (open3d, scikit-image, matplotlib, tyro).
RUN python3 -m pip install -r requirements-gpu.txt

# gsplat from the PREBUILT wheel index — ships a compiled csrc.so for
# sm_70/75/80/86/90, so the container never JIT-compiles. --extra-index-url
# (not --index-url) so PyPI deps like `rich` still resolve.
RUN python3 -m pip install gsplat==1.4.0 \
    --extra-index-url https://docs.gsplat.studio/whl/pt24cu124

# Comprehensive smoke test: import EVERY package the pipeline needs, confirm
# gsplat ships a precompiled .so (not JIT), and verify pkg_resources +
# supervisor import (the things that broke past images). Build aborts on any
# failure — no more silent breakage shipped to the client.
RUN python3 -c "\
import importlib, sys, glob; \
print(f'Python {sys.version}'); \
mods = ['numpy','scipy','scipy.spatial','PIL','yaml','plyfile','watchdog', \
        'streamlit','torch','torchvision','gsplat','open3d','matplotlib', \
        'skimage','tyro','pkg_resources','supervisor.supervisord']; \
[print(f'  {m} {getattr(importlib.import_module(m), \"__version__\", \"ok\")}') for m in mods]; \
from gsplat import rasterization; \
from gsplat.strategy import MCMCStrategy; \
import gsplat, os; \
so = glob.glob(os.path.dirname(gsplat.__file__) + '/*.so'); \
assert so, 'gsplat has NO precompiled .so — would JIT at runtime!'; \
print(f'  gsplat precompiled kernel: {so[0]}'); \
import torch; print(f'CUDA build: {torch.version.cuda}'); \
print('=== ALL IMPORTS OK ==='); \
"

# Application code. Build context is the gs_pipeline package directory
# (docker-compose.yml: context = ..), so `.` is the package root.
COPY . /app/gs_pipeline/
COPY docker/supervisord.conf /etc/supervisor/conf.d/gs_pipeline.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runtime mounts (one per concern). The compose file points host dirs here.
RUN mkdir -p /data/inbox /data/outbox /data/logs /data/work /data/config

ENV PYTHONPATH=/app \
    GS_INBOX=/data/inbox \
    GS_OUTBOX=/data/outbox \
    GS_LOGS=/data/logs \
    GS_WORK=/data/work \
    GS_CONFIG=/data/config \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]
