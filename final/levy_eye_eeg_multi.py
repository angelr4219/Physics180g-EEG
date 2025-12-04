#!/usr/bin/env python3
"""
Multi-dataset Levy-flight analysis for eye-tracking (+ EEG alignment).

Per run (60 s video, 6×10 s still images):
  - Load t, x, y (deg) from Tobii-style semicolon CSV.
  - Filter to valid samples.
  - Compute step lengths and fit power-law tails p(l) ~ l^(-alpha)
    for lmin in [0.5, 1.0, 2.0] deg.
  - Compute MSD vs time lag and fit MSD ~ tau^beta.
  - Plot:
      * x–y trajectory (whole run)
      * x–y–t 3D trajectory (whole run)
      * step-length CCDF (log-log, x = step length, y = P(L ≥ ℓ))
          - log-binned with 10 bins per decade
          - overlay power-law with slope 1 - alpha
      * MSD (log-log, 10 bins per decade on both axes)
      * 6 separate x–y trajectories, one per 10 s image
  - Split each run into exactly 6×10 s segments:
      * compute segment-level alpha, beta, etc.
      * compute per-run averages over the 6 segments.

Outputs go into:
  levy_outputs/YYYYMMDD_HHMMSS/
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.ticker as mticker

# ------------- CONFIG -------------

EYE_FILES = [
    "vid1_A/Angel1-levy-flight-eyeTr-2025-12-02-13-13-12.csv",
    "vid1_B/Iago-levy-flight-eyeTr-2025-12-02-13-18-19.csv",
    "vid1_C/Ravi-levy-flight-eyeTr-2025-12-02-13-22-45.csv",
    "vid1_D/Seth-levy-flight-eyeTr-2025-12-02-13-29-08.csv",
    "vid2_A/SETH2-levy-flight-eyeTr-2025-12-02-13-35-06.csv",
    "vid2_C/Iago2-levy-flight-eyeTr-2025-12-02-13-42-21.csv",
    "vid2_D/Ravi2-levy-flight-eyeTr-2025-12-02-13-46-15.csv",
]

EEG_FILES = [
    "vid1_A/brainbit_20251202_130931.csv",
    "vid1_B/brainbit_20251202_131634.csv",
    "vid1_C/brainbit_20251202_132059.csv",
    "vid1_D/brainbit_20251202_132751.csv",
    "vid2_A/brainbit_20251202_133227.csv",
    "vid2_C/brainbit_20251202_134127.csv",
    "vid2_D/brainbit_20251202_134419.csv",
]

OUTPUT_ROOT = Path("levy_outputs")
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_STAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEG_LEN = 10.0            # seconds per still image / segment
N_SEGMENTS = 6            # exactly 6 still images
L_MINS = [0.5, 1.0, 2.0]  # tail cutoffs (deg)


# ------------- HELPERS: TIME SEGMENTS -------------

def iter_time_segments_fixed6(t, seg_len=SEG_LEN, n_segments=N_SEGMENTS):
    """
    Yield up to n_segments segments of length seg_len:
      [0,10), [10,20), ..., [50,60) in *relative* time.

    Returns (seg_idx, t_start, t_end, mask).
    """
    t = np.asarray(t)
    if t.size == 0:
        return

    t0 = t[0]
    rel_t = t - t0

    for seg_idx in range(n_segments):
        start = seg_idx * seg_len
        end = (seg_idx + 1) * seg_len
        mask = (rel_t >= start) & (rel_t < end)
        if mask.sum() > 5:
            yield seg_idx, float(start), float(end), mask


# ------------- HELPERS: LOADING -------------

def load_eye_csv(path: Path):
    """
    Load Tobii-style semicolon-separated eye-tracking CSV.

    Returns:
      t  : time in seconds (start at 0)
      x  : horizontal position (deg)
      y  : vertical position (deg)
      df : filtered DataFrame
    """
    df = pd.read_csv(path, sep=";")

    # filter valid samples if present
    if "Valid" in df.columns:
        val = df["Valid"]
        if val.dtype == bool:
            df = df[val]
        else:
            try:
                df = df[val.astype(float) > 0.5]
            except Exception:
                df = df[val.astype(str).str.upper() == "VALID"]

    if "CalcXEccDeg" not in df.columns or "CalcYEccDeg" not in df.columns:
        raise KeyError(
            f"{path} is missing CalcXEccDeg/CalcYEccDeg. Columns: {list(df.columns)}"
        )

    x = df["CalcXEccDeg"].to_numpy(dtype=float)
    y = df["CalcYEccDeg"].to_numpy(dtype=float)

    if "CaptureTimeUnixMs" in df.columns:
        t_raw = df["CaptureTimeUnixMs"].to_numpy(dtype=float)
        t = (t_raw - t_raw[0]) / 1000.0
    else:
        fs = 200.0
        t = np.arange(len(x)) / fs

    return t, x, y, df


# ------------- HELPERS: LEVY / MSD -------------

def compute_step_lengths(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx * dx + dy * dy)


def fit_power_law_tail(steps, lmin):
    """
    Continuous power-law MLE for tail: p(l) ~ l^(-alpha), l >= lmin.
    Returns (alpha_hat, n_tail).
    """
    steps = np.asarray(steps)
    tail = steps[steps >= lmin]
    tail = tail[tail > 0]
    n = tail.size
    if n < 10:
        return np.nan, n
    alpha_hat = 1.0 + n / np.sum(np.log(tail / lmin))
    return float(alpha_hat), int(n)


def log_bin_xy(x, y, nbins_per_decade=10):
    """
    Bin (x, y) onto a log-spaced x-grid with nbins_per_decade bins per decade.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return np.array([]), np.array([])

    log_min = np.log10(x.min())
    log_max = np.log10(x.max())
    n_decades = log_max - log_min
    nbins = max(1, int(np.ceil(n_decades * nbins_per_decade)))

    edges = np.logspace(log_min, log_max, nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    y_binned = np.empty_like(centers)
    for i in range(centers.size):
        if i < centers.size - 1:
            m = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        if m.sum() == 0:
            y_binned[i] = np.nan
        else:
            y_binned[i] = np.mean(y[m])

    mask2 = np.isfinite(y_binned)
    return centers[mask2], y_binned[mask2]


def compute_binned_ccdf(steps, nbins_per_decade=10, lmin=None, lmax=None):
    """
    Smooth CCDF on log-spaced step-length bins with nbins_per_decade bins/decade.
    x: step length; y: P(L ≥ ℓ).
    """
    steps = np.asarray(steps)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if steps.size == 0:
        return np.array([]), np.array([])

    if lmin is None:
        lmin = steps.min()
    if lmax is None:
        lmax = steps.max()

    if lmin <= 0 or lmax <= lmin:
        return np.array([]), np.array([])

    log_min = np.log10(lmin)
    log_max = np.log10(lmax)
    n_decades = log_max - log_min
    nbins = max(1, int(np.ceil(n_decades * nbins_per_decade)))

    edges = np.logspace(log_min, log_max, nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])

    n = steps.size
    ccdf = np.empty_like(centers)
    for i, L in enumerate(centers):
        ccdf[i] = np.count_nonzero(steps >= L) / n

    return centers, ccdf


def compute_msd(x, y, max_lag=None):
    """
    Mean-squared displacement for lags tau = 1..max_lag (in samples).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    if n < 2:
        return np.array([]), np.array([])

    if max_lag is None:
        max_lag = n // 4
    max_lag = max(1, int(max_lag))

    lags = np.arange(1, max_lag + 1, dtype=int)
    msd = np.empty_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[i] = np.mean(dx * dx + dy * dy)

    return lags, msd


def fit_loglog_slope(x, y, xmin=None, xmax=None):
    """
    Fit slope in log-log space: y ~ x^beta.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    if xmin is not None:
        m = x_fit >= xmin
        x_fit = x_fit[m]
        y_fit = y_fit[m]
    if xmax is not None:
        m = x_fit <= xmax
        x_fit = x_fit[m]
        y_fit = y_fit[m]

    if x_fit.size < 5:
        return np.nan

    logx = np.log(x_fit)
    logy = np.log(y_fit)
    beta, _ = np.polyfit(logx, logy, 1)
    return float(beta)


# ------------- PLOTTING -------------

def plot_xy_trajectory(t, x, y, title, outdir):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, linewidth=0.5)
    ax.set_xlabel("X (deg)")
    ax.set_ylabel("Y (deg)")
    ax.set_title(f"Eye trajectory (x–y)\n{title}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{title}_xy_traj.png", dpi=300)
    plt.close(fig)


def plot_xy_t_trajectory(t, x, y, title, outdir):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, t, linewidth=0.5)
    ax.set_xlabel("X (deg)")
    ax.set_ylabel("Y (deg)")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"Eye trajectory (x–y–t)\n{title}")
    fig.tight_layout()
    fig.savefig(outdir / f"{title}_xy_t_traj.png", dpi=300)
    plt.close(fig)


def plot_segment_xy_trajectories(t, x, y, title, outdir):
    """
    Plot 6 per-image flight paths (x–y) for 6×10 s segments.
    """
    t = np.asarray(t)
    x = np.asarray(x)
    y = np.asarray(y)

    for seg_idx, t_start, t_end, mask in iter_time_segments_fixed6(t):
        x_seg = x[mask]
        y_seg = y[mask]
        if x_seg.size < 2:
            continue

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(x_seg, y_seg, linewidth=0.7)
        ax.set_xlabel("X (deg)")
        ax.set_ylabel("Y (deg)")
        ax.set_title(f"{title}: segment {seg_idx+1}\n{t_start:.0f}–{t_end:.0f} s")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_name = f"{title}_seg{seg_idx+1}_xy_traj.png"
        fig.savefig(outdir / out_name, dpi=300)
        plt.close(fig)


def plot_step_ccdf(steps, alpha_hat, title, outdir):
    """
    Plot step-length CCDF with log-binning and a power-law reference of slope 1 - alpha.
    x-axis: step length (deg)
    y-axis: P(L ≥ ℓ)
    """
    L, ccdf = compute_binned_ccdf(steps, nbins_per_decade=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    if L.size > 0:
        ax.loglog(L, ccdf, marker="o", linestyle="-", alpha=0.8, label="Data")

        if np.isfinite(alpha_hat):
            i0 = L.size // 2
            L0 = L[i0]
            C0 = ccdf[i0]
            ref = C0 * (L / L0) ** (1.0 - alpha_hat)
            ax.loglog(L, ref, linestyle="--", alpha=0.7,
                      label=f"Power-law: 1-α ≈ {1.0 - alpha_hat:.2f}")
    else:
        ax.text(0.5, 0.5, "No valid steps", ha="center", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Step length (deg)")
    ax.set_ylabel("P(L ≥ ℓ)")
    ax.set_title(f"Step-length CCDF\n{title}")

    locmaj = mticker.LogLocator(base=10.0)
    locmin = mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / f"{title}_step_ccdf.png", dpi=300)
    plt.close(fig)


def plot_msd(lags, msd, dt, title, outdir):
    tau = lags * dt
    tau_b, msd_b = log_bin_xy(tau, msd, nbins_per_decade=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    if tau_b.size > 0:
        ax.loglog(tau_b, msd_b, marker="o", linestyle="-", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No MSD data", ha="center", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Time lag τ (s)")
    ax.set_ylabel("MSD(τ) (deg²)")
    ax.set_title(f"MSD\n{title}")

    locmaj = mticker.LogLocator(base=10.0)
    locmin = mticker.LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)

    fig.tight_layout()
    fig.savefig(outdir / f"{title}_msd.png", dpi=300)
    plt.close(fig)


# ------------- EEG LOADING (minimal stub) -------------

def load_eeg_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


# ------------- MAIN -------------

def main():
    summaries = []
    segment_summaries = []

    for idx, eye_file in enumerate(EYE_FILES):
        eye_path = Path(eye_file)
        if not eye_path.exists():
            print(f"[WARN] Eye file not found: {eye_path}")
            continue

        run_name = eye_path.stem
        print(f"\n=== Analyzing {run_name} ===")

        t, x, y, df_eye = load_eye_csv(eye_path)
        n_samples = x.size
        if n_samples < 2:
            print("  Not enough samples, skipping.")
            continue

        dt = float(np.median(np.diff(t)))
        duration = float(t[-1] - t[0]) if n_samples > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else np.nan

        print(f"  Samples: {n_samples}, duration ~ {duration:.2f} s, fs ~ {fs:.1f} Hz")

        steps = compute_step_lengths(x, y)
        mean_step = float(np.mean(steps))
        median_step = float(np.median(steps))
        max_step = float(np.max(steps))

        alphas = []
        counts = []
        for lmin in L_MINS:
            alpha_hat, n_tail = fit_power_law_tail(steps, lmin=lmin)
            alphas.append(alpha_hat)
            counts.append(n_tail)

        max_lag = min(200, n_samples // 4)
        lags, msd = compute_msd(x, y, max_lag=max_lag)
        beta = fit_loglog_slope(lags * dt, msd,
                                xmin=2 * dt,
                                xmax=duration / 5 if duration > 0 else None)

        run_outdir = RUN_DIR / run_name
        run_outdir.mkdir(parents=True, exist_ok=True)

        # Whole-run plots
        plot_xy_trajectory(t, x, y, run_name, run_outdir)
        plot_xy_t_trajectory(t, x, y, run_name, run_outdir)

        # 6 per-image flight paths
        plot_segment_xy_trajectories(t, x, y, run_name, run_outdir)

        # Step-length CCDF: use alpha from lmin=1.0 (middle L_MINS)
        alpha_for_plot = alphas[1] if len(alphas) > 1 else alphas[0]
        plot_step_ccdf(steps, alpha_for_plot, run_name, run_outdir)

        # MSD
        plot_msd(lags, msd, dt, run_name, run_outdir)

        # EEG info
        eeg_info = {}
        if idx < len(EEG_FILES):
            eeg_path = Path(EEG_FILES[idx])
            df_eeg = load_eeg_csv(eeg_path)
            if df_eeg is not None:
                eeg_info["eeg_file"] = str(eeg_path)
                eeg_info["n_eeg_samples"] = int(df_eeg.shape[0])
            else:
                eeg_info["eeg_file"] = None
                eeg_info["n_eeg_samples"] = np.nan
        else:
            eeg_info["eeg_file"] = None
            eeg_info["n_eeg_samples"] = np.nan

        # Segment-level stats and averages per run
        seg_alpha_list = []
        seg_beta_list = []

        for seg_idx, t_start, t_end, mask in iter_time_segments_fixed6(t):
            t_seg = t[mask]
            x_seg = x[mask]
            y_seg = y[mask]
            n_seg = x_seg.size
            if n_seg < 10:
                continue

            steps_seg = compute_step_lengths(x_seg, y_seg)
            mean_step_seg = float(np.mean(steps_seg))
            median_step_seg = float(np.median(steps_seg))
            max_step_seg = float(np.max(steps_seg))

            alphas_seg = []
            counts_seg = []
            for lmin in L_MINS:
                alpha_hat_seg, n_tail_seg = fit_power_law_tail(steps_seg, lmin=lmin)
                alphas_seg.append(alpha_hat_seg)
                counts_seg.append(n_tail_seg)

            max_lag_seg = min(100, n_seg // 4)
            lags_seg, msd_seg = compute_msd(x_seg, y_seg, max_lag=max_lag_seg)
            beta_seg = fit_loglog_slope(lags_seg * dt, msd_seg,
                                        xmin=2 * dt,
                                        xmax=(t_end - t_start) / 2)

            # store segment stats (for global CSV)
            segment_summaries.append({
                "run": run_name,
                "segment_idx": seg_idx + 1,  # 1..6
                "t_start_s": t_start,
                "t_end_s": t_end,
                "n_samples": n_seg,
                "mean_step_deg": mean_step_seg,
                "median_step_deg": median_step_seg,
                "max_step_deg": max_step_seg,
                "alpha_lmin_0.5": alphas_seg[0],
                "n_tail_0.5": counts_seg[0],
                "alpha_lmin_1.0": alphas_seg[1],
                "n_tail_1.0": counts_seg[1],
                "alpha_lmin_2.0": alphas_seg[2],
                "n_tail_2.0": counts_seg[2],
                "beta_msd": beta_seg,
            })

            # collect for per-run averages
            if np.isfinite(alphas_seg[1]):
                seg_alpha_list.append(alphas_seg[1])  # lmin=1.0
            if np.isfinite(beta_seg):
                seg_beta_list.append(beta_seg)

        alpha_seg_mean = float(np.mean(seg_alpha_list)) if seg_alpha_list else np.nan
        beta_seg_mean = float(np.mean(seg_beta_list)) if seg_beta_list else np.nan

        summaries.append({
            "run": run_name,
            "n_samples": n_samples,
            "duration_s": duration,
            "fs_Hz": fs,
            "mean_step_deg": mean_step,
            "median_step_deg": median_step,
            "max_step_deg": max_step,
            "alpha_lmin_0.5": alphas[0],
            "n_tail_0.5": counts[0],
            "alpha_lmin_1.0": alphas[1],
            "n_tail_1.0": counts[1],
            "alpha_lmin_2.0": alphas[2],
            "n_tail_2.0": counts[2],
            "beta_msd": beta,
            "alpha_seg_mean_lmin1": alpha_seg_mean,
            "beta_seg_mean": beta_seg_mean,
            **eeg_info,
        })

    # Save CSVs
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(RUN_DIR / "levy_summary_all_runs.csv", index=False)
        print("\nSaved summary to:", RUN_DIR / "levy_summary_all_runs.csv")
        print(df_sum)
    else:
        print("No runs analyzed (check file paths).")

    if segment_summaries:
        df_seg = pd.DataFrame(segment_summaries)
        df_seg.to_csv(RUN_DIR / "levy_summary_segments_10s.csv", index=False)
        print("Saved 10 s segment summary to:",
              RUN_DIR / "levy_summary_segments_10s.csv")


if __name__ == "__main__":
    main()
