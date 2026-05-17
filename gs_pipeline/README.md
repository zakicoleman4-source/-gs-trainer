# gs_pipeline — Metashape → Gaussian Splat pipeline

Drop a Metashape export zip into a web UI, watch it train, download a clean
`.ply` splat. Runs in a single Docker container on one NVIDIA GPU (24 GB or
48 GB cards).

This subsystem is independent of the GNSS dashboard in the rest of this repo.

## End-user flow

1. In Metashape Pro on Windows: open project → finish alignment + dense cloud →
   Tools → Run Script → pick `gs_pipeline/metashape/export_for_splat.py`.
   Script writes `<project>_splat_bundle.zip` next to the `.psx`.
2. Open `http://<docker-host>:8501` in any browser.
3. Drag the zip in. Confirm the pre-flight panel. Click **Start Training**.
4. Watch the live preview converge. Download the final `.ply` when done.
   Open in [SuperSplat](https://playcanvas.com/supersplat/editor) or any
   3DGS viewer.

## Quality posture

- Init from the Metashape dense cloud — never random — so MCMC starts on
  geometrically correct surfaces.
- Undistorted photos only (Metashape handles GoPro fisheye well).
- Splat count auto-scales to scene complexity, bounded only by VRAM.
- Anti-floater defaults (SH warmup, opacity reg, scale reg, far-plane sized
  to scene extent).
- Quality > speed: 30k iterations default, 50k on the `Maximum` preset, no
  aggressive early stopping.

## Run it

```bash
cd gs_pipeline/docker
docker compose build       # precompiles gsplat into the image
docker compose up -d       # starts UI on :8501 and the training worker
```

Mounts: `./inbox`, `./outbox`, `./logs`, `./work`, `./config`.

## Local development (CPU, no training)

```bash
pip install -r gs_pipeline/requirements-cpu.txt
pytest gs_pipeline                       # unit tests
pytest -m gpu gs_pipeline                # GPU-gated tests (host with CUDA)
```

## Layout

```
gs_pipeline/
├── docker/        # Dockerfile, compose, supervisord, entrypoint
├── metashape/     # export_for_splat.py — runs inside Metashape GUI
├── trainer/       # parser, init, budget, OOM guard, train loop, watcher
├── ui/            # Streamlit app: upload, preflight, live view, download
└── tests/         # CPU unit tests + GPU-gated smoke tests + synthetic fixtures
```
