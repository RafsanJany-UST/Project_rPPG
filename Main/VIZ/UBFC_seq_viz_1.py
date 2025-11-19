# Main/VIZ/UBFC_sequence_viz_plotly.py

from pathlib import Path
import argparse

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
WIN_LEN = 8.0
PADDING = 1.0
ROI_FRAC = 0.5


def _normalize(sig: np.ndarray) -> np.ndarray:
    """We normalize to zero mean and unit variance when std > 0."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 0:
        sig = sig / std
    return sig


def estimate_global_lag(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
):
    """
    We estimate one global lag using sign-invariant cross-correlation.
    We return lag in seconds, lag in frames, and correlation curve.
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 2:
        return 0.0, 0.0, np.array([0.0]), np.array([0.0])

    corr = np.correlate(s1, s2, mode="full")
    lags = np.arange(-T + 1, T)

    max_lag_samples = int(min(max_lag_seconds / dt, T - 1))
    mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)

    lags_win = lags[mask]
    corr_win = corr[mask]

    best_idx = int(np.argmax(np.abs(corr_win)))
    best_samples = int(lags_win[best_idx])

    lag_sec = float(best_samples * dt)
    fps_est = 1.0 / dt
    lag_frames = lag_sec * fps_est

    return lag_sec, lag_frames, lags_win, corr_win


def estimate_local_lag_curve(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    local_win_seconds: float = 4.0,
    max_lag_seconds: float = 2.0,
):
    """
    We estimate a local lag for each frame using a sliding cross-correlation.

    For each center index, we take a small window around it, compute
    sign-invariant cross-correlation, and store the lag (in samples).

    We return an array lag_samples(t) of length T, and lag_frames(t).
    """
    s1 = _normalize(s_rppg)
    s2 = _normalize(s_ppg)

    T = len(s1)
    if T < 4:
        return np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64)

    fps_est = 1.0 / dt
    win_samples = int(round(local_win_seconds * fps_est))
    if win_samples < 5:
        win_samples = 5
    if win_samples % 2 == 0:
        win_samples += 1
    half = win_samples // 2

    max_lag_samples = int(min(max_lag_seconds * fps_est, half))

    lag_samples_curve = np.full(T, np.nan, dtype=np.float64)

    for c in range(half, T - half):
        seg1 = s1[c - half : c + half + 1]
        seg2 = s2[c - half : c + half + 1]

        seg1 = _normalize(seg1)
        seg2 = _normalize(seg2)

        corr = np.correlate(seg1, seg2, mode="full")
        lags = np.arange(-len(seg1) + 1, len(seg1))

        mask = (lags >= -max_lag_samples) & (lags <= max_lag_samples)
        lags_loc = lags[mask]
        corr_loc = corr[mask]

        best_idx = int(np.argmax(np.abs(corr_loc)))
        best_lag = int(lags_loc[best_idx])
        lag_samples_curve[c] = best_lag

    # We fill NaNs at edges using nearest valid value
    valid = np.where(~np.isnan(lag_samples_curve))[0]
    if len(valid) == 0:
        lag_samples_curve[:] = 0.0
    else:
        first = valid[0]
        last = valid[-1]
        lag_samples_curve[:first] = lag_samples_curve[first]
        lag_samples_curve[last + 1 :] = lag_samples_curve[last]

    lag_frames_curve = lag_samples_curve.astype(np.float64)
    return lag_samples_curve, lag_frames_curve


def auto_detect_gt_and_video(seq_dir: Path):
    """We detect the single .txt and .avi files in a UBFC sequence folder."""
    txt_files = list(seq_dir.glob("*.txt"))
    avi_files = list(seq_dir.glob("*.avi"))

    if len(txt_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .txt, found {len(txt_files)}: {txt_files}"
        )
    if len(avi_files) != 1:
        raise RuntimeError(
            f"[{seq_dir.name}] Expected exactly 1 .avi, found {len(avi_files)}: {avi_files}"
        )

    return txt_files[0], avi_files[0]


def prepare_full_signals(seq_id: str, root: Path):
    """
    We load GT and video, build overlap, and compute frame-synchronous
    GT and CHROM signals inside the overlap.
    """
    seq_dir = root / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    gt_file, vid_file = auto_detect_gt_and_video(seq_dir)

    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(gt_file)

    source = UBFCFrameSource(vid_file)
    t_all = source.t_frame_s
    fps_nominal = float(source.fps)

    overlap_start = float(max(t_ppg_s[0], t_all[0]))
    overlap_end = float(min(t_ppg_s[-1], t_all[-1]))

    if overlap_end <= overlap_start:
        raise RuntimeError("No overlap between GT and video.")

    roi = CentralRoiExtractor(frac=ROI_FRAC)
    t_frame_full, rgb_full = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi,
        t_start=overlap_start,
        t_end=overlap_end,
    )

    if len(t_frame_full) < 20:
        raise RuntimeError("Too few frames in overlap region.")

    dt_full = float(np.mean(np.diff(t_frame_full)))
    fps_est = 1.0 / dt_full

    ppg_full = ubfc_align_ppg_to_frame_times(
        t_ppg_s=t_ppg_s,
        ppg_wave=ppg_wave,
        t_frame_s=t_frame_full,
    )

    ppg_filt = bandpass_zero_phase(ppg_full, fs=fps_est)
    rppg_raw = rppg_chrom(rgb_full)
    rppg_filt = bandpass_zero_phase(rppg_raw, fs=fps_est)

    info = {
        "seq_id": seq_id,
        "t_ppg_s": t_ppg_s,
        "ppg_wave": ppg_wave,
        "t_frame_full": t_frame_full,
        "ppg_full": ppg_filt,
        "rppg_full": rppg_filt,
        "fps_nominal": fps_nominal,
        "fps_est": fps_est,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
    }
    return info


def make_window_plot(
    info,
    t_start: float,
    t_end: float,
    label: str,
    out_dir: Path,
    local_win_seconds: float = 4.0,
):
    """
    We build an interactive Plotly figure for one window:
    - Top: GT vs CHROM with CHROM points colored by local lag (frames).
    - Bottom: local lag (frames) vs time.
    """
    seq_id = info["seq_id"]
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]

    mask = (t_full >= t_start) & (t_full <= t_end)
    t_win = t_full[mask]
    ppg_win = ppg_full[mask]
    rppg_win = rppg_full[mask]

    if len(t_win) < 20:
        print(f"[{seq_id}] {label} window has too few frames ({len(t_win)}).")
        return

    dt = float(np.mean(np.diff(t_win)))

    # We estimate local lag curve for this window
    lag_samples_curve, lag_frames_curve = estimate_local_lag_curve(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        local_win_seconds=local_win_seconds,
        max_lag_seconds=2.0,
    )

    # We also compute one global lag for reference
    lag_sec_global, lag_frames_global, _, _ = estimate_global_lag(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        max_lag_seconds=2.0,
    )

    # We normalize signals for visualization
    ppg_n = _normalize(ppg_win)
    rppg_n = _normalize(rppg_win)

    out_dir.mkdir(parents=True, exist_ok=True)

    # We build Plotly figure with two rows
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=(
            f"{seq_id} – {label} window signals",
            f"{seq_id} – {label} window local lag (frames)",
        ),
    )

    # Row 1: GT vs CHROM, CHROM colored by local lag
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=ppg_n,
            mode="lines",
            name="GT PPG (normalized)",
            line=dict(width=1.5),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=rppg_n,
            mode="lines+markers",
            name="CHROM rPPG (normalized)",
            marker=dict(
                size=6,
                color=lag_frames_curve,
                colorscale="RdBu",
                colorbar=dict(
                    title="Local lag (frames)",
                ),
            ),
            line=dict(width=1.0),
        ),
        row=1,
        col=1,
    )

    # Row 2: local lag curve
    fig.add_trace(
        go.Scatter(
            x=t_win,
            y=lag_frames_curve,
            mode="lines+markers",
            name="Local lag (frames)",
        ),
        row=2,
        col=1,
    )

    # Zero-lag reference line
    fig.add_hline(
        y=0.0,
        line=dict(color="black", dash="dash", width=1),
        row=2,
        col=1,
    )

    # Global lag reference line
    fig.add_hline(
        y=lag_frames_global,
        line=dict(color="green", dash="dot", width=1),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude (normalized)", row=1, col=1)
    fig.update_yaxes(title_text="Lag (frames)", row=2, col=1)

    fig.update_layout(
        title=(
            f"{seq_id} – {label} window "
            f"[{t_start:.2f}, {t_end:.2f}] s "
            f"(global lag ≈ {lag_frames_global:.2f} frames)"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        template="plotly_white",
    )

    html_path = out_dir / f"{seq_id}_{label}_window_lag_viz.html"
    fig.write_html(html_path)
    print(f"Saved interactive {label} window visualization to: {html_path}")

    fig.show()

    print(
        f"[{seq_id}] {label} window: "
        f"global lag_frames≈{lag_frames_global:.2f}, "
        f"frames_in_window={len(t_win)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Interactive UBFC visualization with local lag per frame."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        help="Sequence ID, for example vid_1, vid_15, vid_20.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(UBFC_ROOT),
        help="UBFC root folder.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Figures/UBFC_VIZ_plotly",
        help="Output directory for HTML figures.",
    )
    parser.add_argument(
        "--win_len",
        type=float,
        default=WIN_LEN,
        help="Window length in seconds (default 8.0).",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=PADDING,
        help="Padding from start and end of overlap in seconds (default 1.0).",
    )
    parser.add_argument(
        "--local_win",
        type=float,
        default=4.0,
        help="Local window length (seconds) for per-frame lag estimation.",
    )

    args = parser.parse_args()

    seq_id = args.seq
    root = Path(args.root)
    out_dir = Path(args.out) / seq_id

    print(f"\n=== UBFC Plotly visualization for {seq_id} ===")
    print(f"UBFC root: {root}")

    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]
    win_len = float(args.win_len)
    pad = float(args.padding)

    t_start_start = overlap_start + pad
    t_end_start = t_start_start + win_len

    t_start_end = overlap_end - pad - win_len
    t_end_end = t_start_end + win_len

    print(
        f"Overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s\n"
        f"Start window: [{t_start_start:.3f}, {t_end_start:.3f}] s\n"
        f"End window:   [{t_start_end:.3f}, {t_end_end:.3f}] s"
    )

    # We build start-window visualization
    make_window_plot(
        info,
        t_start=t_start_start,
        t_end=t_end_start,
        label="start",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    # We build end-window visualization
    make_window_plot(
        info,
        t_start=t_start_end,
        t_end=t_end_end,
        label="end",
        out_dir=out_dir,
        local_win_seconds=float(args.local_win),
    )

    print("\nInteractive visualization complete.")


if __name__ == "__main__":
    main()
