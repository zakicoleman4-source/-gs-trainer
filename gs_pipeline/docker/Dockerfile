# gs_pipeline Docker image: Streamlit UI + GS trainer on one NVIDIA GPU.
#
# Build:   docker compose -f gs_pipeline/docker/docker-compose.yml build
# Run:     docker compose -f gs_pipeline/docker/docker-compose.yml up -d
# URL:     http://<host>:8501
#
# gsplat's CUDA extension is precompiled into the image (MAX_JOBS=2, no
# build isolation) so first-run inside a container never JITs. Pinning:
#   torch  == 2.4.1 + cu124
#   gsplat == 1.4.0
#   cuda base image: 12.4.1 cudnn devel (matches torch's cu124 wheels)

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    MAX_JOBS=2

# System packages.
# python3.11-dev: provides Python.h — required to compile ANY C extension
#   (numpy, scipy, Pillow, gsplat CUDA kernels, etc.).
# libopenblas-dev: BLAS/LAPACK for scipy (and numpy's linalg).
# libjpeg-dev, libpng-dev, zlib1g-dev, libtiff-dev, libwebp-dev,
#   libfreetype-dev: Pillow build-time headers so it links against system
#   imaging libs instead of bundling its own (smaller image, faster build).
# libgomp1: OpenMP runtime needed by open3d and numpy BLAS at runtime.
# pkg-config: lets build systems locate the above libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    build-essential ninja-build git curl ca-certificates pkg-config \
    supervisor libgl1 libglib2.0-0 ffmpeg \
    libopenblas-dev \
    libjpeg-dev libpng-dev zlib1g-dev libtiff-dev libwebp-dev \
    libfreetype-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

WORKDIR /app

# CPU-side deps (Streamlit UI, parser, plyfile, numpy, etc.).
# IMPORTANT: use `python -m pip` (not bare `pip`) so packages install into
# the python3.11 that the symlink at /usr/local/bin/python points to.
# Bare `pip` / `pip3` may resolve to the system python3.10 shebang and
# install into the wrong site-packages, causing ImportError at runtime.
COPY requirements-cpu.txt requirements-gpu.txt /app/
RUN python -m pip install --upgrade pip wheel setuptools \
 && python -m pip install -r requirements-cpu.txt

# Torch (cu124 wheels). Pinned to the version gsplat 1.4.x was tested with.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.4.1 torchvision==0.19.1

# Other GPU deps (open3d, scikit-image, matplotlib).
RUN python -m pip install -r requirements-gpu.txt

# gsplat: precompile the CUDA extension at image build time so the container
# never has to JIT on first inference. `--no-build-isolation` makes setup.py
# see torch's headers; MAX_JOBS=2 keeps the parallel nvcc within memory bounds
# on smaller build hosts.
RUN python -m pip install --no-build-isolation gsplat==1.4.0

# Comprehensive smoke test: verify that EVERY package the pipeline needs is
# importable from the python3.11 runtime.  Catches:
#   - pip/Python version mismatches (e.g. pip installed to python3.10)
#   - missing system libraries (libopenblas, libjpeg, etc.)
#   - broken CUDA extension builds (gsplat)
# If ANY import fails the build aborts immediately — no more silent breakage.
RUN python -c "\
import sys; \
print(f'Python {sys.version}'); \
import numpy; print(f'numpy {numpy.__version__}'); \
import scipy; print(f'scipy {scipy.__version__}'); \
import scipy.spatial; \
from PIL import Image; import PIL; print(f'Pillow {PIL.__version__}'); \
import yaml; print(f'PyYAML {yaml.__version__}'); \
import plyfile; print(f'plyfile {plyfile.__version__}'); \
import watchdog; print(f'watchdog {watchdog.__version__}'); \
import streamlit; print(f'streamlit {streamlit.__version__}'); \
import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}'); \
import torchvision; print(f'torchvision {torchvision.__version__}'); \
import gsplat; print(f'gsplat {gsplat.__version__}'); \
from gsplat import rasterization; \
from gsplat.strategy import MCMCStrategy; \
import open3d; print(f'open3d {open3d.__version__}'); \
import matplotlib; print(f'matplotlib {matplotlib.__version__}'); \
import skimage; print(f'scikit-image {skimage.__version__}'); \
import tyro; print(f'tyro {tyro.__version__}'); \
print('=== ALL IMPORTS OK ==='); \
"

# Application code. Build context is the gs_pipeline package directory
# (see docker-compose.yml: context = ..), so `.` here is the package root
# and we copy its contents into /app/gs_pipeline/.
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
