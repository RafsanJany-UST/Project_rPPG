# Main/VIZ/UBFC_sequence_viz.py

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

from Main.Data_Read_Engine.ubfc_alignment import (
    load_ubfc_gt_and_fix,
    ubfc_align_ppg_to_frame_times,
)
from Main.Signal_Processing_Engine.ubfc_dataset import UBFCFrameSource
from Main.Signal_Processing_Engine.roi_central import CentralRoiExtractor
from Main.Signal_Processing_Engine.rgb_extractor import extract_rgb_timeseries
from Main.rPPG_Algorithm_Cell import rppg_chrom, bandpass_zero_phase


# We set our default UBFC root and basic window parameters
UBFC_ROOT = Path(r"D:\Data\UBFC\Dataset_3")
WIN_LEN = 8.0      # window length in seconds
PADDING = 1.0      # padding at start and end
ROI_FRAC = 0.5     # central ROI fraction
SEQUENCE_NO = "vid_1"


def _normalize(sig: np.ndarray) -> np.ndarray:
    """We normalize to zero mean and unit variance when std > 0."""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - np.mean(sig)
    std = np.std(sig)
    if std > 0:
        sig = sig / std
    return sig


def estimate_phase_lag_sign_invariant(
    s_rppg: np.ndarray,
    s_ppg: np.ndarray,
    dt: float,
    max_lag_seconds: float = 2.0,
):
    """
    We estimate lag using sign-invariant cross-correlation.

    We return:
      lag_sec, lag_frames, lags_frames, corr_win
    so that we can also visualize the correlation curve.
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


def auto_detect_gt_and_video(seq_dir: Path):
    """We detect the single .txt (GT) and .avi (video) file in a UBFC sequence folder."""
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
    We load GT and video, build overlap, and compute full-length
    frame-synchronous GT and CHROM rPPG signals inside the overlap.
    """
    seq_dir = root / seq_id
    if not seq_dir.is_dir():
        raise RuntimeError(f"Sequence folder not found: {seq_dir}")

    gt_file, vid_file = auto_detect_gt_and_video(seq_dir)

    # GT
    t_ppg_s, ppg_wave, corrected_idx = load_ubfc_gt_and_fix(gt_file)

    # Video
    source = UBFCFrameSource(vid_file)
    t_all = source.t_frame_s
    fps_nominal = float(source.fps)

    # Overlap
    overlap_start = float(max(t_ppg_s[0], t_all[0]))
    overlap_end = float(min(t_ppg_s[-1], t_all[-1]))

    if overlap_end <= overlap_start:
        raise RuntimeError("No overlap between GT and video.")

    # We extract RGB in the overlap region
    roi = CentralRoiExtractor(frac=ROI_FRAC)
    t_frame_full, rgb_full = extract_rgb_timeseries(
        source=source,
        roi_extractor=roi,
        t_start=overlap_start,
        t_end=overlap_end,
    )

    if len(t_frame_full) < 20:
        raise RuntimeError("Too few frames in overlap region.")

    # We estimate FPS from frame times
    dt_full = float(np.mean(np.diff(t_frame_full)))
    fps_est = 1.0 / dt_full

    # We align GT to frame times and compute CHROM
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


def get_window_indices(t_frame: np.ndarray, t_start: float, t_end: float):
    """We build a boolean mask for frames inside [t_start, t_end]."""
    mask = (t_frame >= t_start) & (t_frame <= t_end)
    return mask


def plot_full_signals(info, save_dir: Path):
    """We visualize full-length GT and CHROM signals inside the overlap."""
    seq_id = info["seq_id"]
    t = info["t_frame_full"]
    ppg = _normalize(info["ppg_full"])
    rppg = _normalize(info["rppg_full"])

    plt.figure(figsize=(12, 4))
    plt.plot(t, ppg, label="GT PPG (aligned)", alpha=0.8)
    plt.plot(t, rppg, label="CHROM rPPG", alpha=0.8)
    plt.title(f"{seq_id}: Full overlap region – normalized GT vs CHROM")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude")
    plt.grid(True, linestyle=":")
    plt.legend(loc="best")
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{seq_id}_full_signals.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved full-signal plot to: {out_path}")
    plt.show()


def plot_window_overlay_and_corr(
    info,
    t_start: float,
    t_end: float,
    label: str,
    save_dir: Path,
):
    """
    We visualize GT vs CHROM in one window and show the correlation curve.
    """
    seq_id = info["seq_id"]
    t_full = info["t_frame_full"]
    ppg_full = info["ppg_full"]
    rppg_full = info["rppg_full"]

    mask = get_window_indices(t_full, t_start, t_end)
    t_win = t_full[mask]
    ppg_win = ppg_full[mask]
    rppg_win = rppg_full[mask]

    if len(t_win) < 20:
        print(f"[{seq_id}] Window {label} has too few frames ({len(t_win)}).")
        return

    dt = float(np.mean(np.diff(t_win)))
    # We estimate lag and correlation
    lag_sec, lag_frames, lags_samples, corr_win = estimate_phase_lag_sign_invariant(
        s_rppg=rppg_win,
        s_ppg=ppg_win,
        dt=dt,
        max_lag_seconds=2.0,
    )

    # We plot overlay in time domain
    plt.figure(figsize=(12, 4))
    plt.plot(t_win, _normalize(ppg_win), label="GT PPG (window)")
    plt.plot(t_win, _normalize(rppg_win), label="CHROM rPPG (window)")
    plt.title(
        f"{seq_id} – {label} window [{t_start:.2f}, {t_end:.2f}] s\n"
        f"Estimated lag ≈ {lag_frames:.2f} frames"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude")
    plt.grid(True, linestyle=":")
    plt.legend(loc="best")
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path_time = save_dir / f"{seq_id}_{label}_window_signals.png"
    plt.savefig(out_path_time, dpi=200)
    print(f"Saved {label} window signal plot to: {out_path_time}")
    plt.show()

    # We plot correlation vs lag (in samples)
    plt.figure(figsize=(10, 4))
    plt.plot(lags_samples, corr_win)
    plt.axvline(
        x=lag_sec / dt,
        color="red",
        linestyle="--",
        label=f"peak lag ≈ {lag_frames:.2f} frames",
    )
    plt.title(f"{seq_id} – {label} window cross-correlation")
    plt.xlabel("Lag (samples ≈ frames)")
    plt.ylabel("Correlation")
    plt.grid(True, linestyle=":")
    plt.legend(loc="best")
    plt.tight_layout()

    out_path_corr = save_dir / f"{seq_id}_{label}_window_corr.png"
    plt.savefig(out_path_corr, dpi=200)
    print(f"Saved {label} window correlation plot to: {out_path_corr}")
    plt.show()

    print(
        f"[{seq_id}] {label} window: lag_sec={lag_sec:.4f}, "
        f"lag_frames≈{lag_frames:.2f}, frames_in_window={len(t_win)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize UBFC GT vs CHROM for start/end windows."
    )
    parser.add_argument(
        "--seq",
        type=str,
        required=True,
        default = SEQUENCE_NO,
        help="Sequence ID, for example vid_1, vid_15, vid_20.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(UBFC_ROOT),
        help="UBFC root folder (default is D:\\Data\\UBFC\\Dataset_3).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Figures/UBFC_VIZ",
        help="Output directory for figures.",
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

    args = parser.parse_args()

    seq_id = args.seq
    root = Path(args.root)
    base_out_dir = Path(args.out) / seq_id

    print(f"\n=== UBFC visualization for {seq_id} ===")
    print(f"UBFC root: {root}")

    info = prepare_full_signals(seq_id, root)

    overlap_start = info["overlap_start"]
    overlap_end = info["overlap_end"]
    win_len = float(args.win_len)
    pad = float(args.padding)

    # We define start and end windows inside the overlap
    t_start_start = overlap_start + pad
    t_end_start = t_start_start + win_len

    t_start_end = overlap_end - pad - win_len
    t_end_end = t_start_end + win_len

    print(
        f"Overlap: [{overlap_start:.3f}, {overlap_end:.3f}] s\n"
        f"Start window: [{t_start_start:.3f}, {t_end_start:.3f}] s\n"
        f"End window:   [{t_start_end:.3f}, {t_end_end:.3f}] s"
    )

    # Full signals plot
    plot_full_signals(info, base_out_dir)

    # Start and end windows with correlation
    plot_window_overlay_and_corr(
        info,
        t_start=t_start_start,
        t_end=t_end_start,
        label="start",
        save_dir=base_out_dir,
    )

    plot_window_overlay_and_corr(
        info,
        t_start=t_start_end,
        t_end=t_end_end,
        label="end",
        save_dir=base_out_dir,
    )

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
