#!/usr/bin/env python3
"""
Multi-dataset Levy-flight analysis for eye-tracking (+ EEG alignment).

What this does:
  - Loops over a list of eye-tracking CSV files (8+ runs).
  - For each:
      * Loads x,y in degrees (CalcXEccDeg, CalcYEccDeg).
      * Filters to Valid samples.
      * Computes step lengths and a power-law tail fit.
      * Computes MSD vs time lag and fits MSD ~ tau^beta.
      * Plots:
          - x vs y trajectory
          - x,y,t trajectory (3D)
          - step-length CCDF (log-log)
          - MSD (log-log)
  - Optionally: loads matching EEG files for future joint analysis.

Edit the FILE LISTS section to point to your actual filenames.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)


# ------------------ CONFIG: file lists ------------------

# Put all your eye-tracking files here in the order you like
EYE_FILES = [
    # Example (you already have this one):
    "Angel1-levy-flight-eyeTr-2025-12-02-13-13-12.csv",
    # Add the other 8 runs:
    # "Angel2-....csv",
    # "Angel3-....csv",
    # ...
]

# Optionally, matching EEG files (same length or same time base)
# Keep as [] for now if you just want gaze analysis
EEG_FILES = [
    # Example:
    # "brainbit_20251202_130931.csv",
    # ...
]

OUTPUT_DIR = Path("levy_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------ Helper functions ------------------

def load_eye_csv(path: Path):
    """
    Load an eye-tracking CSV.

    Assumes columns:
      - 'CalcXEccDeg' : horizontal position (deg)
      - 'CalcYEccDeg' : vertical position (deg)
      - 'Valid'       : "VALID" for good samples
      - 'Timestamp'   : (optional) time in seconds or ms

    Edit column names here if your CSV uses slightly different labels.
    """
    df = pd.read_csv(path)

    # Filter valid samples
    if "Valid" in df.columns:
        df = df[df["Valid"] == "VALID"].copy()

    # Replace these if needed
    x = df["CalcXEccDeg"].to_numpy(dtype=float)
    y = df["CalcYEccDeg"].to_numpy(dtype=float)

    # Time handling: if there's an explicit time column, use it; otherwise make an index-based time
    if "Timestamp" in df.columns:
        t = df["Timestamp"].to_numpy(dtype=float)
        # If timestamps look like ms, convert to seconds
        # (Simple heuristic: if max(t) > 1e3 and median delta ~5, assume ms)
        if t.max() > 1e3:
            t = t / 1000.0
    else:
        # Assume ~200 Hz like before; adjust if needed
        fs = 200.0
        t = np.arange(len(x)) / fs

    return t, x, y, df


def compute_step_lengths(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    steps = np.sqrt(dx*dx + dy*dy)
    return steps


def fit_power_law_tail(steps, lmin):
    """
    Continuous power-law MLE for tail: p(l) ~ l^(-alpha), l >= lmin.

    Returns (alpha_hat, n_tail).
    """
    tail = steps[steps >= lmin]
    n = len(tail)
    if n < 10:
        return np.nan, n

    # Avoid division by zero or log of non-positive
    tail = tail[tail > 0]
    n = len(tail)
    if n == 0:
        return np.nan, 0

    alpha_hat = 1.0 + n / np.sum(np.log(tail / lmin))
    return alpha_hat, n


def compute_ccdf(data):
    """Return (sorted_values, CCDF) for CCDF plot."""
    data = np.sort(data)
    n = len(data)
    ccdf = 1.0 - np.arange(1, n+1) / n
    return data, ccdf


def compute_msd(x, y, max_lag=None):
    """
    Mean-squared displacement for lags tau = 1..max_lag (in samples).
    Returns (lags, msd).
    """
    n = len(x)
    if max_lag is None:
        max_lag = n // 4  # Up to 1/4 of the record length

    lags = np.arange(1, max_lag+1)
    msd = np.empty_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        msd[i] = np.mean(dx*dx + dy*dy)

    return lags, msd


def fit_loglog_slope(x, y, xmin=None, xmax=None):
    """
    Fit slope in log-log space: y ~ x^beta.

    x, y: positive arrays
    xmin, xmax: optional range in x over which we fit.
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    if xmin is not None:
        x_fit = x_fit[x_fit >= xmin]
        y_fit = y_fit[:len(x_fit)]
    if xmax is not None:
        x_fit = x_fit[x_fit <= xmax]
        y_fit = y_fit[:len(x_fit)]

    if len(x_fit) < 5:
        return np.nan

    logx = np.log(x_fit)
    logy = np.log(y_fit)

    beta, _ = np.polyfit(logx, logy, 1)
    return beta


# ------------------ Plot functions ------------------

def plot_xy_trajectory(t, x, y, title, outdir):
    plt.figure(figsize=(5, 5))
    plt.plot(x, y, linewidth=0.5)
    plt.xlabel("X (deg)")
    plt.ylabel("Y (deg)")
    plt.title(f"Eye trajectory (x–y)\n{title}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{title}_xy_traj.png", dpi=300)
    plt.close()


def plot_xy_t_trajectory(t, x, y, title, outdir):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, t, linewidth=0.5)
    ax.set_xlabel("X (deg)")
    ax.set_ylabel("Y (deg)")
    ax.set_zlabel("Time (s)")
    ax.set_title(f"Eye trajectory (x–y–t)\n{title}")
    plt.tight_layout()
    plt.savefig(outdir / f"{title}_xy_t_traj.png", dpi=300)
    plt.close()


def plot_step_ccdf(steps, title, outdir):
    vals, ccdf = compute_ccdf(steps)
    plt.figure(figsize=(5, 4))
    plt.loglog(vals, ccdf, marker=".", linestyle="none", alpha=0.6)
    plt.xlabel("Step length (deg)")
    plt.ylabel("P(L >= ℓ)")
    plt.title(f"Step-length CCDF\n{title}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{title}_step_ccdf.png", dpi=300)
    plt.close()


def plot_msd(lags, msd, dt, title, outdir):
    tau = lags * dt  # convert sample lags to seconds
    plt.figure(figsize=(5, 4))
    plt.loglog(tau, msd, marker="o", linestyle="-", alpha=0.7)
    plt.xlabel("Time lag τ (s)")
    plt.ylabel("MSD(τ) (deg²)")
    plt.title(f"MSD\n{title}")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{title}_msd.png", dpi=300)
    plt.close()


# ------------------ EEG loading (optional for later) ------------------

def load_eeg_csv(path: Path):
    """
    Minimal EEG loader; adapt to your actual columns.

    Previous pipelines used:
      - O1_V, O2_V          : raw voltages
      - SamplingRate or dt  : sometimes implicit
      - timestamp column    : to align with eye-tracking

    For now, this just loads the CSV and returns the DataFrame.
    You can later add:
      - band-pass 8–12 Hz
      - Hilbert transform for alpha envelope
      - alignment with eye-tracking via time stamps
    """
    df = pd.read_csv(path)
    return df


# ------------------ Main loop ------------------

def main():
    summaries = []

    for idx, eye_file in enumerate(EYE_FILES):
        eye_path = Path(eye_file)
        if not eye_path.exists():
            print(f"[WARN] Eye file not found: {eye_path}")
            continue

        run_name = eye_path.stem
        print(f"\n=== Analyzing {run_name} ===")

        # 1. Load eye data
        t, x, y, df_eye = load_eye_csv(eye_path)
        n_samples = len(x)

        # 2. Basic timing stats
        dt = np.median(np.diff(t))
        duration = t[-1] - t[0] if n_samples > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else np.nan

        print(f"  Samples: {n_samples}, duration ~ {duration:.2f} s, fs ~ {fs:.1f} Hz")

        # 3. Step lengths + Levy tail fits
        steps = compute_step_lengths(x, y)
        mean_step = np.mean(steps)
        median_step = np.median(steps)
        max_step = np.max(steps)

        # choose a lower cutoff for Levy tail; tweak as needed
        lmins = [0.5, 1.0, 2.0]  # in deg
        alphas = []
        counts = []

        for lmin in lmins:
            alpha_hat, n_tail = fit_power_law_tail(steps, lmin=lmin)
            alphas.append(alpha_hat)
            counts.append(n_tail)

        # 4. MSD and MSD exponent
        lags, msd = compute_msd(x, y, max_lag=min(200, n_samples//4))
        # Fit beta on an intermediate range of lags
        beta = fit_loglog_slope(lags * dt, msd, xmin=2*dt, xmax=duration/5)

        # 5. Plots
        run_outdir = OUTPUT_DIR / run_name
        run_outdir.mkdir(exist_ok=True)

        plot_xy_trajectory(t, x, y, run_name, run_outdir)
        plot_xy_t_trajectory(t, x, y, run_name, run_outdir)
        plot_step_ccdf(steps, run_name, run_outdir)
        plot_msd(lags, msd, dt, run_name, run_outdir)

        # 6. EEG loading (for alignment later)
        eeg_info = {}
        if idx < len(EEG_FILES):
            eeg_path = Path(EEG_FILES[idx])
            if eeg_path.exists():
                df_eeg = load_eeg_csv(eeg_path)
                eeg_info["eeg_file"] = str(eeg_path)
                eeg_info["n_eeg_samples"] = len(df_eeg)
                # You can later add alpha-envelope computation here
            else:
                eeg_info["eeg_file"] = None
        else:
            eeg_info["eeg_file"] = None

        # 7. Collect summary for this run
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
            **eeg_info,
        })

    # Save summary table
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(OUTPUT_DIR / "levy_summary_all_runs.csv", index=False)
        print("\nSaved summary to:", OUTPUT_DIR / "levy_summary_all_runs.csv")
        print(df_sum)
    else:
        print("No runs analyzed (check file names).")


if __name__ == "__main__":
    main()
