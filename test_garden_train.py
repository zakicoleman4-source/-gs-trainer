"""Quick 400-iteration training on the garden bundle + filter comparison.

Run with:
    CUDA_HOME=/home/tarbut/miniconda3 TORCH_CUDA_ARCH_LIST="6.1" python3 test_garden_train.py
"""
import json
import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

os.environ.setdefault("CUDA_HOME", "/home/tarbut/miniconda3")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")

BUNDLE_ZIP = Path("/media/tarbut/cAVS-132/garden_3_splat_bundle.zip")
WORK_DIR = Path("/tmp/gs_garden_test")
ITERATIONS = 400

def main():
    print("=" * 60)
    print(f"GARDEN TRAINING TEST — {ITERATIONS} iterations")
    print("=" * 60)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    # Extract bundle
    print("\n[1/5] Extracting bundle...")
    bundle_dir = WORK_DIR / "bundle"
    with zipfile.ZipFile(BUNDLE_ZIP) as zf:
        zf.extractall(bundle_dir)

    # Parse + init + budget
    print("\n[2/5] Preflight...")
    from gs_pipeline.trainer.parse_metashape import parse_cameras_xml
    from gs_pipeline.trainer.init_from_pcd import load_and_downsample
    from gs_pipeline.trainer.budget import compute_budget, detect_gpu

    scene = parse_cameras_xml(bundle_dir / "cameras.xml", image_dir=bundle_dir / "images")
    cloud = load_and_downsample(bundle_dir / "dense.ply")
    gpu = detect_gpu()
    budget = compute_budget(gpu=gpu, image_sizes=scene.image_sizes, dense_pts=cloud.xyz.shape[0])

    print(f"  {len(scene)} cameras, {cloud.xyz.shape[0]:,} init pts")
    print(f"  GPU: {gpu.name}, {gpu.total_vram_bytes/1e9:.1f} GB")
    print(f"  Target splats: {budget.target_splats:,}, downscale: {budget.downscale_factor:.3f}")

    # Pre-scale K matrices (critical — the other session's pipeline.py does this)
    if hasattr(budget, 'downscale_per_camera'):
        for i, ds in enumerate(budget.downscale_per_camera):
            if ds < 1.0:
                scene.K_per_camera[i, :2, :] *= ds
        effective_downscale = 1.0
    else:
        effective_downscale = budget.downscale_factor

    # Train
    print(f"\n[3/5] Training {ITERATIONS} iterations...")
    from gs_pipeline.trainer.train_mcmc import train, TrainerConfig
    from gs_pipeline.trainer.job_state import new_job_state, PreflightSnapshot, write_state

    config = TrainerConfig(
        iterations=ITERATIONS,
        eval_every=100,
        preview_every=200,
        checkpoint_every=200,
        divergence_check_at_step=0,
        memory_fraction=0.80,
        holdout_stride=20,
        filter_enabled=False,  # we'll filter manually after
        timelapse_enabled=False,
    )

    work_dir = WORK_DIR / "work"
    outbox_dir = WORK_DIR / "outbox"
    work_dir.mkdir(); outbox_dir.mkdir()

    js = new_job_state("garden_test", bundle_filename="garden_3_splat_bundle.zip")
    js.start_preflight()
    preflight = PreflightSnapshot(
        n_cameras=len(scene), total_megapixels=budget.total_megapixels,
        dense_pts=budget.dense_pts, target_splats=budget.target_splats,
        hard_cap_splats=budget.hard_cap_splats, iterations=ITERATIONS,
        downscale_factor=budget.downscale_factor,
        image_max_side=budget.image_max_side,
        quality_preset="Auto", gpu_name=gpu.name,
        gpu_total_vram_bytes=gpu.total_vram_bytes, notes=list(budget.notes),
    )
    js.start_training(preflight)
    state_path = work_dir / "state.json"
    write_state(js, state_path)

    # Override budget downscale since K is already pre-scaled
    if hasattr(budget, 'downscale_per_camera'):
        budget_for_train = budget
        # The training loop uses budget.downscale_factor; since K is pre-scaled,
        # set it to 1.0 for training image loading
        from dataclasses import replace
        try:
            budget_for_train = replace(budget, downscale_factor=1.0)
        except Exception:
            budget.downscale_factor = 1.0
    else:
        budget_for_train = budget

    t0 = time.time()
    try:
        outputs = train(
            scene=scene, init_cloud=cloud, budget=budget_for_train, config=config,
            job_state=js, job_state_path=state_path,
            work_dir=work_dir, outbox_dir=outbox_dir,
        )
    except Exception as e:
        print(f"  TRAINING FAILED: {e}")
        import traceback; traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({ITERATIONS/elapsed:.1f} it/s)")
    print(f"  Final PLY: {outputs.final_ply}")

    # Filter comparison
    print("\n[4/5] Testing filter configurations...")
    from gs_pipeline.trainer.export_ply import read_inria_ply, write_inria_ply
    from gs_pipeline.trainer.filter_splats import filter_scene

    splat = read_inria_ply(outputs.final_ply)
    n_total = splat.means.shape[0]
    print(f"  Unfiltered splats: {n_total:,}")

    filter_configs = {
        "no_filter": dict(min_opacity=0.0, sor_k=1, sor_std_ratio=999.0, max_scale_factor=9999.0),
        "light": dict(min_opacity=0.002, sor_k=10, sor_std_ratio=3.0, max_scale_factor=20.0),
        "default": dict(min_opacity=0.005, sor_k=20, sor_std_ratio=2.0, max_scale_factor=10.0),
        "aggressive": dict(min_opacity=0.01, sor_k=30, sor_std_ratio=1.5, max_scale_factor=5.0),
        "extreme": dict(min_opacity=0.05, sor_k=50, sor_std_ratio=1.0, max_scale_factor=3.0),
    }

    results = {}
    filter_dir = WORK_DIR / "filter_comparison"
    filter_dir.mkdir()

    # Copy unfiltered for reference
    shutil.copy2(outputs.final_ply, filter_dir / "scene_no_filter.ply")

    for name, cfg in filter_configs.items():
        if name == "no_filter":
            results[name] = {"n_input": n_total, "n_output": n_total, "removed": 0,
                             "pct_removed": "0.0%", "file_size_mb": f"{Path(outputs.final_ply).stat().st_size/1e6:.1f}",
                             "path": str(filter_dir / "scene_no_filter.ply")}
            continue

        m, sc, q, o, dc, rest, report = filter_scene(
            means=splat.means.copy(), scales=splat.scales.copy(),
            quats=splat.quats.copy(), opacities=splat.opacities.copy(),
            sh_dc=splat.sh_dc.copy(), sh_rest=splat.sh_rest.copy(),
            **cfg,
        )
        out_path = filter_dir / f"scene_{name}.ply"
        write_inria_ply(out_path=out_path, means=m, scales=sc, quats=q,
                        opacities=o, sh_dc=dc, sh_rest=rest)
        size_mb = out_path.stat().st_size / 1e6
        results[name] = {
            "n_input": report.n_input, "n_output": report.n_output,
            "removed": report.n_input - report.n_output,
            "pct_removed": f"{100*(1-report.n_output/max(report.n_input,1)):.1f}%",
            "opacity_removed": report.n_input - report.n_after_opacity,
            "scale_removed": report.n_after_opacity - report.n_after_scale,
            "sor_removed": report.n_after_scale - report.n_after_sor,
            "file_size_mb": f"{size_mb:.1f}",
            "path": str(out_path),
        }
        print(f"  [{name}] {report.n_input:,} -> {report.n_output:,} ({results[name]['pct_removed']} removed)")

    # Report
    print("\n[5/5] Writing report...")
    report_data = {
        "bundle": str(BUNDLE_ZIP), "iterations": ITERATIONS,
        "training_time_s": round(elapsed, 1), "gpu": gpu.name,
        "unfiltered_splats": n_total, "filter_results": results,
    }
    report_path = filter_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report_data, indent=2))

    print(f"\n{'='*60}")
    print(f"RESULTS in {filter_dir}")
    for name in filter_configs:
        r = results[name]
        print(f"  {name:12s}: {r.get('n_output', n_total):>8,} splats  {r['file_size_mb']:>6s} MB  {r['path']}")
    print(f"\nOpen PLY files in SuperSplat to compare visually.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
