# gs-trainer

**Metashape → Gaussian Splat training pipeline with a Streamlit web UI and Docker.**

Drag a Metashape `*_splat_bundle.zip` into a browser, watch the train live,
download a clean `scene.ply`. Auto-tunes for your GPU (24 GB or 48 GB cards)
and your scene's complexity — anti-floater defaults, no command line, no
config parameters to tune.

The full subsystem lives under [`gs_pipeline/`](./gs_pipeline/). Start
there for the architecture, quality posture, and runtime layout.

## Quick start

```bash
# CPU unit tests (no GPU needed)
pip install -r gs_pipeline/requirements-cpu.txt
pytest gs_pipeline                           # 193 tests pass

# Full stack: GPU training + Streamlit UI in Docker
cd gs_pipeline/docker
docker compose build                         # ~15 min first time (gsplat precompiles)
docker compose up -d                         # UI at http://<host>:8501
```

End-user flow:

1. In Metashape Pro: open `.psx`, finish alignment + dense cloud,
   `Tools > Run Script` → `gs_pipeline/metashape/export_for_splat.py`.
2. Open `http://<docker-host>:8501`. Drag in the generated zip.
3. Confirm preflight. Click Start. Watch the live dashboard.
4. Download the final `scene.ply`. Open in
   [SuperSplat](https://playcanvas.com/supersplat/editor).

## Built in 14 vertical slices

Each commit is a self-contained slice with tests:

1. scaffold + pytest harness with `gpu` marker (auto-skips on CPU CI)
2. synthetic Metashape-like bundle for tests
3. Agisoft `cameras.xml` parser (intrinsics + extrinsics, 1e-6 round-trip)
4. voxel-downsample dense `.ply` to ≤1 M points (server-side)
5. VRAM-aware splat-budget heuristic (14.7M on 24 GB, 33.5M on 48 GB)
6. OOM retry + stall watchdog (torch-optional)
7. atomic JobState shared between worker + UI
8. MCMC training loop + `config.yaml` + CPU helper tests
9. `render_eval` + INRIA-layout `.ply` writer (SuperSplat / Polycam compatible)
10. `pipeline.run_job` + `watcher.process_one` + integration tests
11. Docker image — cuda 12.4 + precompiled gsplat + supervisord
12. Streamlit UI — upload, preflight, live dashboard, downloads
13. Metashape export script + CPU-runnable stub
14. GPU smoke test + acceptance runbook in `gs_pipeline/README.md`

## License

MIT (see [LICENSE](./LICENSE)).
