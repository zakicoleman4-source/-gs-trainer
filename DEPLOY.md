# gs-trainer — Client Deployment Guide (v0.2.0)

## What you're getting

A fully self-contained Docker image that turns a Metashape bundle export into
a high-quality Gaussian Splat scene. Drop the `.zip` file into the web UI,
click Start, and download `scene.ply` when training finishes.

**Docker Hub image:** `swdsfd/gs-trainer:latest` (or `:v0.2.0`)

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| NVIDIA GPU | 8 GB VRAM | 24 GB (A5000 / RTX 4090) |
| NVIDIA Driver | 525+ | latest |
| nvidia-container-toolkit | any | latest |
| Docker | 20.10+ | latest |
| Docker Compose | v2 | v2 |
| Disk | 50 GB free | 200 GB |

Verify GPU passthrough works before running:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

---

## Quick start (5 minutes)

### 1. Create a working directory

```bash
mkdir gs-trainer && cd gs-trainer
mkdir inbox outbox logs work config
```

### 2. Create `docker-compose.yml`

```yaml
services:
  gs_pipeline:
    image: swdsfd/gs-trainer:latest
    container_name: gs_trainer
    restart: unless-stopped
    ports:
      - "8501:8501"
    volumes:
      - ./inbox:/data/inbox
      - ./outbox:/data/outbox
      - ./logs:/data/logs
      - ./work:/data/work
      - ./config:/data/config
    environment:
      MAX_UPLOAD_GB: "20"
      QUALITY_PRESET_DEFAULT: "Auto"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    shm_size: "8gb"
```

### 3. Start

```bash
docker compose up -d
```

### 4. Open the UI

Navigate to `http://localhost:8501` (or `http://<server-ip>:8501`).

---

## Using the pipeline

### Metashape export

Run the provided Metashape script (`export_for_splat.py`) via **Tools > Run Script**.
This exports a `<project>_splat_bundle.zip` containing:
- `cameras.xml` — aligned camera positions and intrinsics
- `dense.ply` — dense point cloud (full Metashape density, not decimated)
- `images/` — undistorted photos at original resolution
- `masks/` *(optional)* — per-camera exclusion masks

**Dense cloud export guidance:**
- Export the full ROI density — do not manually decimate the point cloud.
  `init_from_pcd.py` voxel-downsamples to ≤1M points automatically.
- Include colors (`Save point colors` checked).
- Use Ultra High or High quality for best initialization.

### Training workflow

1. Open `http://<host>:8501`
2. Select **Quality preset** (Auto = 40k iterations, Maximum = 100k)
3. Select **Trainer backend** (MCMC = default/fast, Scaffold-GS = +1 dB quality, 2× slower)
4. Drop the bundle `.zip` onto the uploader
5. Review preflight notes (GPU info, splat budget, any warnings)
6. Click **Start training**
7. Watch live: progress bar, PSNR/SSIM charts, 3-panel preview render updates every 250 steps
8. When done: download `scene.ply` (full quality, SH degree 3) and/or `scene.splat` (web viewer)

### Output files

| File | Description |
|---|---|
| `scene.ply` | Full quality PLY — SuperSplat, Polycam, Inria viewer compatible |
| `scene.splat` | 32 bytes/splat binary — direct web viewer upload |
| `scene_unfiltered.ply` | Pre-filter backup (for custom re-filtering) |
| `metrics.csv` | Per-step loss, PSNR, SSIM |
| `report.json` | Final PSNR/SSIM, filter stats, per-camera quality breakdown |
| `training_timelapse.mp4` | Time-lapse of training progress |

### Re-filtering (interactive)

On the Done screen, expand **"Re-filter scene"** to apply a different filter
intensity (light / default / aggressive / extreme) and download immediately.
No re-training required.

---

## Quality settings

### Presets

| Preset | Iterations | Use when |
|---|---|---|
| Auto | 40,000 | Standard deliveries, < 2 hours on 24 GB GPU |
| Maximum | 100,000 | Publication quality, client showcases |

### Trainer backends

| Backend | Quality | Speed | Notes |
|---|---|---|---|
| MCMC | Baseline | Fast | gsplat MCMC; proven on all scene types |
| Scaffold-GS | +1 dB PSNR | ~2× slower | Anchor-based neural Gaussians; better for complex scenes |

---

## Monitoring and logs

```bash
# Live logs
docker compose logs -f

# Supervisor status
docker compose exec gs_trainer supervisorctl status

# Per-job error log (if a job failed)
cat logs/<job_id>/pipeline_error.txt
```

---

## VRAM guide (auto-tuned by the pipeline)

| GPU VRAM | Max image resolution | Approx splat count | Auto-preset time |
|---|---|---|---|
| 8 GB | 1600 px | ~3M | 30–45 min |
| 12 GB | 1800 px | ~5M | 45–60 min |
| 16 GB | 2200 px | ~7M | 60–90 min |
| 24 GB | 2800 px | ~10M | 60–90 min |
| 48 GB | 3200 px | ~30M | 90–120 min |

---

## Updating to a new version

```bash
docker compose pull
docker compose up -d
```

---

## Troubleshooting

**"No CUDA device" / training fails immediately**
→ Run `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
→ If that fails, reinstall `nvidia-container-toolkit` and restart Docker

**Out of memory during training**
→ The pipeline auto-retries with smaller image resolution (OOM guard).
→ If it still fails: set `QUALITY_PRESET_DEFAULT=Auto` and try again.

**PSNR stuck below 15 dB at step 15,000**
→ The divergence guard aborts and marks the job failed. Cause is usually
  very sparse image overlap. Check Metashape alignment quality first.

**Upload fails / progress stops at preflight**
→ Check `logs/<job_id>/pipeline_error.txt`
→ Verify `cameras.xml`, `dense.ply`, and `images/` are all in the zip

**Large scene (500+ cameras) takes very long**
→ The pipeline switches to block-partitioned training automatically (VastGaussian).
→ Each block trains independently; final PLY merges all blocks. Expect 2–4× longer.
