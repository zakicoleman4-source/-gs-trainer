# gs-trainer — Claude session context

## What this project is
Metashape Pro -> Gaussian Splat training pipeline. Users export a bundle zip
from Metashape (cameras.xml + dense.ply + undistorted images), drop it into a
Streamlit web UI running in Docker, and get a trained `scene.ply` output.

## Architecture (key files)
- `gs_pipeline/trainer/parse_metashape.py` — cameras.xml parser (intrinsics + extrinsics)
- `gs_pipeline/trainer/init_from_pcd.py` — voxel-downsample dense PLY to <=1M points
- `gs_pipeline/trainer/budget.py` — VRAM-aware splat budget + per-camera image downscale
- `gs_pipeline/trainer/img_cache.py` — pre-decode + resize images once before training
- `gs_pipeline/trainer/train_mcmc.py` — gsplat MCMC training loop (the core)
- `gs_pipeline/trainer/pipeline.py` — orchestrator: unzip -> parse -> init -> budget -> [partition?] -> train -> export
- `gs_pipeline/trainer/render_eval.py` — holdout PSNR/SSIM evaluation + preview strip renders
- `gs_pipeline/trainer/export_ply.py` — INRIA-layout PLY writer (SuperSplat/Polycam compatible)
- `gs_pipeline/trainer/scene_partition.py` — VastGaussian block partitioning (drone/large scenes)
- `gs_pipeline/trainer/large_scene.py` — block training orchestrator + merge
- `gs_pipeline/trainer/watcher.py` — inbox filesystem watcher, spawns training as subprocess
- `gs_pipeline/trainer/oom_guard.py` — CUDA OOM retry + progress watchdog
- `gs_pipeline/trainer/config.yaml` — all tunable training defaults
- `gs_pipeline/metashape/export_for_splat.py` — Metashape Pro script (Tools > Run Script)
- `gs_pipeline/ui/app.py` — Streamlit UI (upload, preflight, live dashboard, downloads)
- `gs_pipeline/docker/` — Dockerfile, docker-compose.yml, entrypoint.sh, supervisord.conf

## Multi-session protocol — READ THIS FIRST

Multiple Claude sessions may work on this repo in parallel. Follow these rules
every time, no exceptions. Conflicts and overwritten work happen when sessions
ignore them.

### 1. Always pull before you start
```bash
git pull origin main
```

### 2. Always work on a feature branch — never commit directly to main
```bash
git checkout -b feat/short-description   # e.g. feat/ui-filter-comparison
# ... do your work ...
git push origin feat/short-description
```
Then open a PR, or if you're the only session active right now and it's a tiny
fix, you may push directly to main **only after confirming no other session is
mid-work** (check `session_state.md` in memory/).

### 3. Claim your work in memory/session_state.md before starting
Edit `/home/tarbut/.claude/projects/-home-tarbut-Music-aj-aj-projects-gs-pipeline/memory/session_state.md`
and add a line: `[YOUR-BRANCH] — what you're doing`. Remove it when you merge.
This is the only way parallel sessions know not to touch the same files.

### 4. Check session_state.md before touching any file
If another session has claimed a file, work around it or coordinate via the
user. Never blindly overwrite in-progress work.

### 5. Run tests before every push
```bash
python3 -m pytest gs_pipeline/tests/ -q \
  --ignore=gs_pipeline/tests/test_smoke.py \
  --ignore=gs_pipeline/tests/test_pipeline_smoke.py \
  -k "not test_set_memory_fraction_returns_false and not test_ui"
```
Do not push a branch that breaks the test suite.

### 6. Update CLAUDE.md and memory when you finish
After merging: update the "Pending work" section in this file, and update
`memory/project_gs_trainer.md` with what changed. This keeps the next session
fully briefed.

### Docker
Docker only rebuilds on version tags. Do NOT manually trigger a Docker build
during active development. When a feature is client-ready:
```bash
git tag v0.X.Y && git push origin v0.X.Y
```

---

## Running tests
```bash
pip install -r gs_pipeline/requirements-cpu.txt
python3 -m pytest gs_pipeline/tests/ -v \
  --ignore=gs_pipeline/tests/test_smoke.py \
  --ignore=gs_pipeline/tests/test_pipeline_smoke.py \
  -k "not test_set_memory_fraction_returns_false and not test_ui"
```
The `test_set_memory_fraction` test fails when CUDA is present (expects no-GPU).
The `test_ui` tests require streamlit.

## Docker
- Image: `swdsfd/gs-trainer:edge` on Docker Hub (linux/amd64)
- Run: `docker run --gpus all -p 8501:8501 -v inbox:/data/inbox -v outbox:/data/outbox -v logs:/data/logs -v work:/data/work swdsfd/gs-trainer:edge`
- Tag a release: `git tag v0.1.0 && git push origin v0.1.0` -> workflow pushes :v0.1.0, :latest
- GPU arch support: `7.0;7.5;8.0;8.6;8.9;9.0` (Volta V100, Turing RTX 2000/T4, Ampere, Ada, Hopper)
- ffmpeg is installed in the image (required for timelapse MP4 generation)
- `MAX_IMAGE_SIDE` is NOT set in docker-compose.yml — budget.py picks it from VRAM at runtime

## Key architectural decisions

### K pre-scaling (CRITICAL correctness)
`pipeline.py` pre-scales K matrices in-place after preflight and before training:
```python
for i, ds in enumerate(budget.downscale_per_camera):
    if ds < 1.0:
        scene.K_per_camera[i, :2, :] *= ds  # scale fx,cx and fy,cy rows
```
The training loop always uses `downscale=1.0` after this. Never move this scaling
into the training loop — it would silently run at wrong resolution.

### Per-camera downscale
Sony Alpha (50 MP, ~8000px) and GoPro (4K, 3840px) get different scale factors.
`budget.downscale_per_camera` is a `list[float]` (one entry per camera). The global
`budget.downscale_factor` is the MAX of all per-camera factors (for display only).

### Image pre-cache (`img_cache.py`)
50 MP JPEGs take ~400ms each to decode. `build_image_cache()` decodes + resizes once
to a `work_dir/img_cache/` directory, then training loads cheap pre-sized files.
Keyed by `{i:05d}_ds{ds_tag}.jpg` so re-runs skip existing files.

### Large-scene routing (`scene_partition.py` + `large_scene.py`)
`pipeline.py` checks `should_partition(len(scene))` (threshold: 500 cameras).
If true and no custom `train_fn` injected: calls `run_large_scene()` which:
1. Partitions into a rows×cols spatial grid (VastGaussian strategy)
2. Assigns cameras via AABB projection visibility (threshold 0.25, 20% overlap)
3. Trains each block independently with lighter iteration count
4. Crops Gaussians to tight bounds, concatenates into final `scene.ply`

If `train_fn` is injected (test mode), always uses the standard path regardless of camera count.

### Auto-config for scene size (`auto_adjust_config_for_scene`)
Adjusts holdout_stride, eval_every, divergence_check based on n_cameras:
- < 30 cameras: holdout_stride=2, divergence_check=3000, min_psnr=10.0
- < 80: holdout_stride=3, divergence_check=6000
- < 150: holdout_stride=5
- > 1000: holdout_stride=16, eval_every=2000

### Preview strip + timelapse
`save_preview_strip()` renders 3 holdout cameras (first/middle/last), resizes to
`panel_height=400px`, concatenates with 4px white separators. Updated every 250
training steps; updates `job_state.outputs.preview_strip_png` live for the dashboard.
Each checkpoint copies the strip to `timelapse_frames/` → ffmpeg MP4 at job end.

### Appearance conditioning
Per-camera log-exposure (3 RGB scalars) for drone footage with auto-exposure variation.
Disabled by default (`appearance.enabled: false` in config.yaml). Applied only to
the photometric loss; discarded from final PLY.

### Post-training filter (`filter_splats.py`)
Three-stage chain applied after training when `filter.enabled: true` (default on):
1. **Opacity** — drops splats with `sigmoid(logit) < filter.min_opacity` (default 0.005)
2. **Scale** — drops splats whose max axis scale exceeds `filter.max_scale_factor × median_scale`
3. **SOR** — statistical outlier removal via k-NN (k=`filter.sor_k`, cutoff at
   `mean_dist + sor_std_ratio × std_dist`). Uses `scipy.spatial.KDTree` (O(N log N));
   falls back to brute-force block multiply for N ≤ 50k when scipy is absent.

Pipeline saves `scene_unfiltered.ply` as backup before overwriting `scene.ply`.
All filter params live under `filter:` in `config.yaml`; all default to sensible values.
scipy is required for SOR on large scenes — it is in `requirements-cpu.txt` (≥ 1.12).

## VRAM class → max_image_side table (budget.py)
| VRAM      | max_image_side |
|-----------|---------------|
| 48 GB+    | 3200 px       |
| 24 GB     | 2800 px       |
| 16 GB     | 2200 px       |
| 12 GB     | 1800 px       |
| < 12 GB   | 1600 px       |

## Quality preset defaults (config.yaml)
| Setting               | Auto     | Maximum  |
|-----------------------|----------|----------|
| iterations            | 40,000   | 100,000  |
| target_splats mult    | 1.0×     | 1.35×    |
| refine_start_iter     | 100      | 100      |
| prune_opa             | 0.003    | 0.003    |
| sh_warmup_interval    | 1000     | 1000     |

## Target hardware
- Minimum: 8 GB VRAM (GTX 1080 class) — heavily downscaled, ~3M splats
- Sweet spot: 24 GB (A5000/RTX 4090) — 2800px images, ~9-12M splats
- High-end: 48+ GB (A6000/H100) — 3200px images, ~30M+ splats

## Target input
- Sony Alpha photos: ~50 MP each, up to 500 photos per scene
- GoPro video frames: 4K-5.3K, potentially thousands of frames
- Drone datasets: 3000+ cameras — handled by block partitioning
- Metashape exports the bundle as `*_splat_bundle.zip` containing cameras.xml + dense.ply + images/

## Quality philosophy
"Even an idiot can train a good splat from a good Metashape project."
All quality knobs are auto-tuned. User only picks "Auto" vs "Maximum" preset.
The pipeline should produce publication-quality splats with zero manual tuning.

## Post-training filtering (`filter_splats.py`)
Three filters chained automatically after training (configurable in config.yaml `filter:` section):
1. **Opacity** — removes near-transparent splats (`sigmoid(logit) < min_opacity`)
2. **Scale** — removes oversized splats (`exp(scale) > median * max_scale_factor`)
3. **SOR** — Statistical Outlier Removal via scipy KDTree (`mean_dist > global_mean + std_ratio * std`)

Saves `scene_unfiltered.ply` backup + overwrites `scene.ply` with filtered result.
Filter stats included in `report.json`.

User can compare filter levels: no_filter, light, default, aggressive, extreme.

## Docker notes
- `scipy` added to requirements-cpu.txt (needed by knn_mean_distance + filter SOR)
- `ffmpeg` added to Dockerfile (needed for training timelapse)
- `TORCH_CUDA_ARCH_LIST` expanded to include Volta (7.0) and Turing (7.5)
- `MAX_IMAGE_SIDE` removed from docker-compose.yml (now VRAM-adaptive via budget.py)

## Scaffold-GS trainer (`train_scaffold.py`)
Alternative to MCMC for +1 dB PSNR (~30.13 vs ~29.18 on MipNeRF360). Selected via
`trainer.backend: scaffold` in config.yaml or the Streamlit UI dropdown.

Architecture: anchor-based neural Gaussian hierarchy.
- **Anchors** (~200K): voxel-quantized positions with learnable features (dim=32)
- **Neural Gaussians**: 3 MLPs (opacity, covariance, color) predict per-view Gaussian
  attributes from anchor features + view direction. Each anchor spawns `n_offsets=10` Gaussians.
- **Densification**: grows/prunes anchors on multi-resolution voxel grid (not individual Gaussians)
- **PLY export**: "bakes" neural Gaussians from canonical viewpoint (mean of all camera centers)
  to standard INRIA format. Baked PLY goes through existing `filter_splats.py` chain unchanged.

Uses gsplat `rasterization(sh_degree=None)` with raw RGB — no SH, no new CUDA deps.

Config: `scaffold:` section in config.yaml. `ScaffoldConfig` dataclass in `train_scaffold.py`.

## Pending work
- UI: filter comparison view (let user choose filter level and download)
- Validate `export_for_splat.py` Metashape script on actual Metashape Pro install
- GTX 1080 (sm_61) needs patched gsplat; Docker build on Volta+ is fine
- GPU smoke test Scaffold-GS on real data

## Repo
https://github.com/zakicoleman4-source/-gs-trainer (private)
Docker Hub: swdsfd/gs-trainer
