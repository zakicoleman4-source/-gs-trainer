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

## Acceptance runbook (real Metashape, real GPU)

Run these on your Windows box (Metashape + RTX A5000 or A6000):

1. **Install / refresh the Docker image** on the GPU host (Linux preferred;
   on Windows, Docker Desktop with the NVIDIA Container Toolkit also works):

   ```bash
   cd gs_pipeline/docker
   docker compose build           # ~15 min first time (gsplat precompiles)
   docker compose up -d
   # UI at http://<gpu-host>:8501
   ```

2. **Export from Metashape**: open a known-good `.psx` in Metashape Pro →
   `Tools > Run Script` → pick `gs_pipeline/metashape/export_for_splat.py`.
   Wait for the "Splat bundle written to:" dialog. A `*_splat_bundle.zip`
   appears next to the project file.

3. **Drag the zip into the browser UI**. The preflight panel should appear
   in under 2 seconds with sensible numbers (cameras / MP / dense pts /
   target splats / GPU / ETA). Click **Start Training**.

4. **Observe the live dashboard**:
   - Progress bar advances within ~30 s.
   - Loss curve appears around step 100; PSNR curve at the first eval (1000).
   - Preview tile shows a recognisable scene by step ~5 000.
   - "Download intermediate .ply" appears at step 5 000; drop it into
     [SuperSplat](https://playcanvas.com/supersplat/editor) — no
     Milky-Way haze, recognisable surfaces.

5. **At completion** (~30 min for ~5 M splats on a 24 GB card, ~50 min on
   the Maximum preset): download `scene.ply`, open in SuperSplat. The
   acceptance bar:
   - Holdout PSNR ≥ your current manual MCMC-GS baseline on the same scene.
   - Fewer floaters in empty regions (compare side-by-side).
   - Splat count auto-scaled to the GPU: ~12 M on 24 GB, ~28 M on 48 GB
     for complex scenes (verify via the dashboard's preflight panel).

6. **GPU smoke check**, automatable, before / after any change:
   ```bash
   pip install -r gs_pipeline/requirements-gpu.txt
   pip install --no-build-isolation gsplat==1.4.0
   pytest -m gpu gs_pipeline/tests/test_pipeline_smoke.py
   ```
   Runs a 500-step training on the synthetic 8-camera bundle. Passes when:
   final scene.ply written in INRIA layout, splat count ≤ budget, state
   machine transitioned to DONE.

## Where state lives

| Path | What |
| --- | --- |
| `inbox/` | User-dropped bundle zips. Watcher renames to `claim__<id>__name.zip` when it picks one up. |
| `work/<job_id>/` | Per-job working dir: bundle/ extract, ckpt_*.pt, scene_step_*.ply, preview.png, metrics.csv, state.json, report.json |
| `outbox/<job_id>/` | Final `scene.ply`. |
| `logs/<job_id>/` | Worker stdout/stderr + pipeline_error.txt on failures. |
| `logs/supervisord.log`, `logs/ui.{out,err}`, `logs/worker.{out,err}` | Long-running process logs. |
