"""Streamlit widgets that render the live job dashboard from ``state.json``.

The dashboard is a polling view (every ``DEFAULT_REFRESH_S`` seconds): the
worker is the only writer of ``state.json`` and ``preview.png``, the UI
reads. ``render_live_dashboard`` is callable from ``app.py`` or in isolation
(tests do the latter to inspect what's rendered).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from gs_pipeline.trainer.job_state import (
    JobState,
    OutputsSnapshot,
    PreflightSnapshot,
    State,
    safe_read_state,
    state_path_for,
)


DEFAULT_REFRESH_S = 5
TERMINAL_STATES = (State.DONE, State.FAILED)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without Streamlit)
# ---------------------------------------------------------------------------

def format_eta(current_step: int, total_steps: int, started_at_iso: Optional[str]) -> str:
    """Human-readable ETA, given progress + started_at timestamp.

    Returns "(estimating)" until we have at least 1 step of data. Falls back
    to "—" if the started_at timestamp isn't parseable.
    """
    if total_steps <= 0 or current_step <= 0:
        return "(estimating)"
    if started_at_iso is None:
        return "—"
    from datetime import datetime, timezone
    try:
        started = datetime.fromisoformat(started_at_iso)
    except ValueError:
        return "—"
    now = datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed = (now - started).total_seconds()
    if elapsed <= 0:
        return "(estimating)"
    remaining = elapsed * (total_steps - current_step) / max(current_step, 1)
    return _human_seconds(int(remaining))


def _human_seconds(s: int) -> str:
    if s < 90:
        return f"{s}s"
    if s < 60 * 60:
        return f"{s // 60}m {s % 60}s"
    return f"{s // 3600}h {(s % 3600) // 60}m"


def render_progress_text(js: JobState) -> str:
    """One-line "step X / Y — preset Z" status string for the header."""
    if js.preflight is None:
        return js.status_msg
    return (
        f"step {js.progress.current_step:,} / {js.preflight.iterations:,} "
        f"— {js.preflight.quality_preset} preset on {js.preflight.gpu_name}"
    )


def metrics_csv_rows(metrics_csv_path: Optional[str]) -> list[dict[str, float]]:
    """Parse the trainer's metrics.csv into a list of dicts. Empty on missing."""
    if not metrics_csv_path:
        return []
    p = Path(metrics_csv_path)
    if not p.is_file():
        return []
    rows: list[dict[str, float]] = []
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "step": int(row["step"]),
                    "loss": float(row["loss"]),
                    "holdout_psnr": float(row["holdout_psnr"]),
                    "holdout_ssim": float(row["holdout_ssim"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def intermediate_ply_paths(work_dir: Path) -> list[Path]:
    """Discover ``scene_step_*.ply`` files written by the trainer's checkpoint hook."""
    work_dir = Path(work_dir)
    if not work_dir.is_dir():
        return []
    return sorted(work_dir.glob("scene_step_*.ply"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


# ---------------------------------------------------------------------------
# Streamlit-only rendering
# ---------------------------------------------------------------------------

def render_live_dashboard(
    *,
    job_id: str,
    work_root: Path,
    refresh_seconds: int = DEFAULT_REFRESH_S,
) -> Optional[JobState]:
    """Render the live training dashboard for ``job_id``.

    Returns the JobState that was rendered (or None if the state.json hasn't
    been written yet). Caller is responsible for the page's auto-refresh loop.
    """
    import streamlit as st  # local import: avoid Streamlit at module load

    state_path = state_path_for(work_root, job_id)
    js = safe_read_state(state_path)
    if js is None:
        st.info("Waiting for trainer to start…")
        return None

    # --- Header line ------------------------------------------------------
    st.markdown(f"**Job:** `{js.job_id}` — **state:** `{js.state.value}`")
    if js.preflight is not None:
        st.markdown(render_progress_text(js))

    # --- Progress bar -----------------------------------------------------
    total = js.preflight.iterations if js.preflight else 0
    if total:
        progress = min(js.progress.current_step / total, 1.0)
        st.progress(progress, text=f"{js.progress.current_step:,} / {total:,}")
        st.markdown(f"**ETA:** {format_eta(js.progress.current_step, total, js.started_at)}")

    # --- Failure path -----------------------------------------------------
    if js.state is State.FAILED:
        st.error(f"Training failed: {js.error_msg or '(no detail)'}")

    # --- Metrics charts ---------------------------------------------------
    rows = metrics_csv_rows(js.outputs.metrics_csv)
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows)
        col_loss, col_psnr = st.columns(2)
        with col_loss:
            st.markdown("**Loss**")
            st.line_chart(df.set_index("step")[["loss"]])
        with col_psnr:
            st.markdown("**Holdout PSNR / SSIM**")
            st.line_chart(df.set_index("step")[["holdout_psnr", "holdout_ssim"]])

    # --- Live preview -----------------------------------------------
    strip = getattr(js.outputs, 'preview_strip_png', None)
    single = js.outputs.preview_png

    if strip and Path(strip).is_file():
        st.markdown("**Live render — 3 holdout views**")
        st.image(strip, use_container_width=True)
    elif single and Path(single).is_file():
        st.markdown("**Live render (one holdout view)**")
        st.image(single, use_container_width=True)

    # --- Downloads --------------------------------------------------------
    work_dir = work_root / job_id
    mid_plys = intermediate_ply_paths(work_dir)
    if mid_plys:
        latest = mid_plys[-1]
        st.markdown(f"**Latest intermediate ply** ({latest.name}):")
        with latest.open("rb") as f:
            st.download_button("Download intermediate .ply",
                                data=f.read(),
                                file_name=latest.name,
                                mime="application/octet-stream",
                                key=f"dl_mid_{latest.name}")

    # Timelapse video
    timelapse = getattr(js.outputs, 'timelapse_mp4', None)
    if timelapse and Path(timelapse).is_file():
        st.markdown("**Training timelapse video**")
        with Path(timelapse).open("rb") as f:
            st.download_button(
                "Download training timelapse.mp4",
                data=f.read(),
                file_name="training_timelapse.mp4",
                mime="video/mp4",
                key="dl_timelapse",
            )

    if js.state is State.DONE:
        # ── Result metrics card ───────────────────────────────────────────
        final_psnr = getattr(js.outputs, "final_psnr", None)
        final_ssim = getattr(js.outputs, "final_ssim", None)
        final_count = getattr(js.outputs, "final_splat_count", None)
        if any(v is not None for v in (final_psnr, final_ssim, final_count)):
            mc = st.columns(3)
            if final_psnr is not None:
                mc[0].metric("Final PSNR", f"{final_psnr:.2f} dB")
            if final_ssim is not None:
                mc[1].metric("Final SSIM", f"{final_ssim:.3f}")
            if final_count is not None:
                mc[2].metric("Splats (filtered)", f"{final_count / 1e6:.2f} M")

        # ── Filter breakdown (from report.json) ───────────────────────────
        report_json_path = getattr(js.outputs, "report_json", None)
        if report_json_path and Path(report_json_path).is_file():
            import json as _json
            try:
                _rpt = _json.loads(Path(report_json_path).read_text(encoding="utf-8"))
                _filt = _rpt.get("filter")
                if _filt and _filt.get("n_input"):
                    n_in = _filt["n_input"]
                    n_out = _filt["n_output"]
                    pct = 100.0 * (1 - n_out / max(n_in, 1))
                    with st.expander(f"Filter stats — {pct:.1f}% of floaters removed"):
                        fc = st.columns(4)
                        fc[0].metric("Before filter", f"{n_in / 1e6:.2f} M")
                        fc[1].metric("Opacity pass", f"{_filt.get('n_after_opacity', n_out) / 1e6:.2f} M")
                        fc[2].metric("Scale pass", f"{_filt.get('n_after_scale', n_out) / 1e6:.2f} M")
                        fc[3].metric("After SOR", f"{n_out / 1e6:.2f} M")
            except Exception:
                pass

        # ── Download buttons ──────────────────────────────────────────────
        if js.outputs.final_ply and Path(js.outputs.final_ply).is_file():
            st.success("Training complete — scene is ready.")
            with Path(js.outputs.final_ply).open("rb") as f:
                st.download_button(
                    "Download scene.ply (full quality, SH degree 3)",
                    data=f.read(), file_name="scene.ply",
                    mime="application/octet-stream", key="dl_final",
                )

        final_splat_p = getattr(js.outputs, "final_splat", None)
        if final_splat_p and Path(final_splat_p).is_file():
            size_mb = Path(final_splat_p).stat().st_size / 1e6
            with Path(final_splat_p).open("rb") as f:
                st.download_button(
                    f"Download scene.splat (web viewer, {size_mb:.0f} MB)",
                    data=f.read(), file_name="scene.splat",
                    mime="application/octet-stream", key="dl_splat",
                )

        final_unf = getattr(js.outputs, "final_ply_unfiltered", None)
        if final_unf and Path(final_unf).is_file():
            with Path(final_unf).open("rb") as f:
                st.download_button(
                    "Download scene_unfiltered.ply (pre-filter backup)",
                    data=f.read(), file_name="scene_unfiltered.ply",
                    mime="application/octet-stream", key="dl_unfiltered",
                )

    # --- Notes / warnings -------------------------------------------------
    if js.preflight and js.preflight.notes:
        with st.expander("Preflight notes"):
            for n in js.preflight.notes:
                st.markdown(f"- {n}")

    # --- Auto-refresh until terminal --------------------------------------
    if js.state not in TERMINAL_STATES:
        # streamlit-extras autorefresh isn't in our deps; use the built-in
        # st.experimental_rerun via a JS-based timer if available.
        try:
            from streamlit_autorefresh import st_autorefresh  # type: ignore
            st_autorefresh(interval=refresh_seconds * 1000, key=f"refresh_{job_id}")
        except Exception:
            st.caption(f"(Page auto-refreshes every {refresh_seconds}s. "
                       f"If it stops, click anywhere or reload.)")

    return js
