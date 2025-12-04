#!/usr/bin/env python3
"""
Plot 2D gaze trajectories (x vs y) for multiple Levy-flight eye-tracking runs,
and do per-image + overall Levy / power-law analysis.

For each CSV file:
  - Load time, x, y (deg of visual angle), handling semicolon-separated files.
  - Optionally filter to valid samples if a validity column exists.
  - Plot x vs y trajectory (traj_<run>.png).
  - Assume one 60 s video = 6 still images of 10 s each:
        [0-10), [10-20), ..., [50-60) seconds.
  - For each 10 s segment:
        * compute step lengths
        * fit a power-law tail p(l) ~ l^(-alpha) above l_min
        * plot CCDF on log-log axes with
              x = P(L >= ℓ), y = ℓ (step length)
        * all segments overlaid (ccdf_segments_<run>.png)
  - For the full 60 s:
        * compute step lengths
        * fit a power-law tail (overall alpha)
        * plot CCDF with x = P(L >= ℓ), y = ℓ (ccdf_overall_<run>.png)
  - Append per-run summary to levy_summary_per_run.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)
from datetime import datetime
import matplotlib.ticker as mticker

# ---------------- CONFIG ----------------

eye_dir = Path("./")  # base folder

eye_files = [
    "vid1_A/Angel1-levy-flight-eyeTr-2025-12-02-13-13-12.csv",
    "vid1_B/Iago-levy-flight-eyeTr-2025-12-02-13-18-19.csv",
    "vid1_C/Ravi-levy-flight-eyeTr-2025-12-02-13-22-45.csv",
    "vid1_D/Seth-levy-flight-eyeTr-2025-12-02-13-29-08.csv",
    "vid2_A/SETH2-levy-flight-eyeTr-2025-12-02-13-35-06.csv",
    "vid2_C/Iago2-levy-flight-eyeTr-2025-12-02-13-42-21.csv",
    "vid2_D/Ravi2-levy-flight-eyeTr-2025-12-02-13-46-15.csv",
    
]

# Optionally, matching EEG files (same length or same time base)
# Keep as [] for now if you just want gaze analysis
EEG_FILES = [
    "vid1_A/brainbit_20251202_130931.csv",
    "vid1_B/brainbit_20251202_131634.csv",
    "vid1_C/brainbit_20251202_132059.csv",
    "vid1_D/brainbit_20251202_132751.csv",
    "vid2_A/brainbit_20251202_133227.csv",
    "vid2_C/brainbit_20251202_134127.csv",
    "final/vid2_D/brainbit_20251202_134419.csv",
]
OUTPUT_ROOT = Path("levy_outputs")

# Unique folder for this script run, e.g. levy_outputs/20251203_011530
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = OUTPUT_ROOT / RUN_STAMP
RUN_DIR.mkdir(parents=True, exist_ok=True)



SEG_LEN = 10.0  # seconds per still image / segment


def iter_time_segments(t, seg_len=SEG_LEN):
    """
    Yield (seg_idx, t_start, t_end, mask) for consecutive time windows.

    Uses t in seconds, assumed monotonic and starting at 0.
    """
    if len(t) == 0:
        return

    t0 = t[0]
    t_end_all = t[-1] - t0
    seg_idx = 0
    start = 0.0

    while start < t_end_all:
        end = start + seg_len
        # last segment: include everything up to final sample
        if end >= t_end_all:
            mask = (t - t0 >= start)
        else:
            mask = (t - t0 >= start) & (t - t0 < end)

        if mask.sum() > 5:  # skip tiny segments
            yield seg_idx, start, min(end, t_end_all), mask

        seg_idx += 1
        start += seg_len


# ------------------ Helper functions ------------------
def load_eye_csv(path: Path):
    "vid2_D/Ravi2-levy-flight-eyeTr-2025-12-02-13-46-15.csv"
]

X_LABEL = "Horizontal position (deg)"
Y_LABEL = "Vertical position (deg)"

# Tail cutoff for power-law fits (deg)
L_MIN = 0.5


# ------------- HELPERS: LOADING / FILTERING -------------

def maybe_filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there is a 'valid' column (or similar), restrict to valid samples.
    Handles numeric/bool or string ('VALID', 'INVALID', etc).
    """
    valid_candidates = ["Valid", "valid", "Validity", "EyeValid", "GazeValid"]
    for col in valid_candidates:
        if col in df.columns:
            s = df[col]
            # numeric or bool
            if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
                return df[s.astype(bool)]
            # string-like: keep rows that contain 'VALID'
            return df[s.astype(str).str.upper().str.contains("VALID")]
    return df


def load_txy(path: Path):
    """
    Load an eye-tracking CSV (semicolon-separated) and return
    t [s from 0], x [deg], y [deg], df_filtered.
    """
    # Your Tobii file is semicolon-separated
    df = pd.read_csv(path, sep=";")

    # Filter valid samples if we have a Valid column
    if "Valid" in df.columns:
        val = df["Valid"]
        if val.dtype == bool:
            df = df[val]
        else:
            try:
                df = df[val.astype(float) > 0.5]
            except Exception:
                df = df[val.astype(str).str.upper() == "VALID"]
    Load a semicolon-separated eye-tracking CSV and return (t, x, y, df).

    - t in seconds (relative to start time, using heuristics on units)
    - x, y in degrees (if such columns exist)
    """
    # semicolon-separated export
    df = pd.read_csv(path, sep=";")

    # keep only valid samples if we have a validity column
    df = maybe_filter_valid(df)

    x_candidates = ["CalcXEccDeg", "CalcXEccentricity", "XDeg", "x_deg", "X", "x", "GazeX"]
    y_candidates = ["CalcYEccDeg", "CalcYEccentricity", "YDeg", "y_deg", "Y", "y", "GazeY"]
    time_candidates = ["Timestamp", "Time", "time", "TIME", "TimeStamp", "CaptureTime"]

    def pick_column(df, candidates, what="x"):
        for name in candidates:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").to_numpy()
        raise ValueError(
            f"Could not find a {what}-column in {list(df.columns)}. "
            f"Tried: {candidates}"
        )

    # Positions
    x = df["CalcXEccDeg"].to_numpy(dtype=float)
    y = df["CalcYEccDeg"].to_numpy(dtype=float)
    # x, y in deg
    x = pick_column(df, x_candidates, "x")
    y = pick_column(df, y_candidates, "y")

    # Time from CaptureTimeUnixMs → seconds, start at 0
    if "CaptureTimeUnixMs" in df.columns:
        t_ms = df["CaptureTimeUnixMs"].to_numpy(dtype=float)
        t = (t_ms - t_ms[0]) / 1000.0
    else:
        # Fallback: synthetic time axis
        fs = 200.0
        t = np.arange(len(x)) / fs
    # time handling: subtract start, then guess units
    t = None
    for c in time_candidates:
        if c in df.columns:
            raw = pd.to_numeric(df[c], errors="coerce").to_numpy()
            raw = raw - raw[0]  # make relative
            span = np.nanmax(raw) - np.nanmin(raw)
            if span <= 0:
                # fallback: synthesize ~120 Hz
                t = np.arange(len(raw)) * (1.0 / 120.0)
            else:
                # Heuristics: treat raw as µs, ms, or s
                if span > 1e6:
                    t = raw / 1e6  # microseconds -> seconds
                elif span > 1e3:
                    t = raw / 1e3  # milliseconds -> seconds
                else:
                    t = raw        # already seconds
            break

    if t is None:
        # If no explicit time column, assume ~120 Hz
        t = np.arange(len(x)) * (1.0 / 120.0)

    return t, x, y, df
def log_bin_xy(x, y, nbins_per_decade=10):
    """
    Bin (x, y) onto a log-spaced x-grid with nbins_per_decade bins per decade.
    Returns (x_centers, y_mean).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])

    log_min = np.log10(x.min())
    log_max = np.log10(x.max())
    n_decades = log_max - log_min
    nbins = max(1, int(np.ceil(n_decades * nbins_per_decade)))

    edges = np.logspace(log_min, log_max, nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

    y_binned = np.empty_like(centers)
    for i in range(len(centers)):
        if i < len(centers) - 1:
            m = (x >= edges[i]) & (x < edges[i + 1])
        else:
            m = (x >= edges[i]) & (x <= edges[i + 1])
        if m.sum() == 0:
            y_binned[i] = np.nan
        else:
            y_binned[i] = np.mean(y[m])

    mask2 = np.isfinite(y_binned)
    return centers[mask2], y_binned[mask2]


# ------------- HELPERS: LEVY / POWER-LAW -------------

def compute_step_lengths(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx * dx + dy * dy)


def compute_ccdf(data):
    data = np.sort(data)
    n = len(data)
    ccdf = 1.0 - np.arange(1, n + 1) / n
    return data, ccdf


def fit_power_law_tail(steps, lmin):
    """
    Continuous power-law MLE for tail: p(l) ~ l^(-alpha), l >= lmin.

    Returns (alpha_hat, n_tail). If not enough data, alpha_hat = NaN.
    """
    steps = np.asarray(steps)
    tail = steps[steps >= lmin]
    tail = tail[tail > 0]
    n = len(tail)
    if n < 10:
        return np.nan, n
    alpha_hat = 1.0 + n / np.sum(np.log(tail / lmin))
    return alpha_hat, n


def compute_ccdf(data):
    """Return (sorted_values, CCDF) for CCDF plot."""
    data = np.sort(data)
    n = len(data)
    ccdf = 1.0 - np.arange(1, n+1) / n
    return data, ccdf

def compute_binned_ccdf(steps, nbins_per_decade=10, lmin=None, lmax=None):
    """
    Compute a smooth CCDF on log-spaced step-length bins with
    nbins_per_decade bins per decade.
    """
    steps = np.asarray(steps)
    steps = steps[np.isfinite(steps) & (steps > 0)]
    if len(steps) == 0:
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

    n = len(steps)
    ccdf = np.empty_like(centers)
    for i, L in enumerate(centers):
        ccdf[i] = np.count_nonzero(steps >= L) / n

    return centers, ccdf


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


def plot_step_ccdf(steps, alpha_hat, title, outdir):
    # binned CCDF with 10 bins per decade
    L, ccdf = compute_binned_ccdf(steps, nbins_per_decade=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(L) > 0:
        ax.loglog(L, ccdf, marker="o", linestyle="-", alpha=0.8, label="Data")

        # --- Power-law reference line with slope 1 - alpha ---
        if np.isfinite(alpha_hat):
            # anchor at middle of L range
            i0 = len(L) // 2
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

    # log-grid: 10 ticks per decade on both axes
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
    # sample lags (in seconds)
    tau = lags * dt

    # log-bin so that between 10^-2 and 10^-1 etc. we have 10 equal log steps
    tau_b, msd_b = log_bin_xy(tau, msd, nbins_per_decade=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    if len(tau_b) > 0:
        ax.loglog(tau_b, msd_b, marker="o", linestyle="-", alpha=0.8)
    else:
        ax.text(0.5, 0.5, "No MSD data", ha="center", va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Time lag τ (s)")
    ax.set_ylabel("MSD(τ) (deg²)")
    ax.set_title(f"MSD\n{title}")

    # --- log-grid: 10 ticks per decade on BOTH axes ---
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
# ------------- MAIN PLOTTING / ANALYSIS -------------

def analyze_overall_levy(t, x, y, eye_path: Path, out_dir: Path, lmin=L_MIN):
    """
    Compute step-lengths for the whole 60 s run, fit a power-law tail,
    and plot CCDF with step length on the Y axis and probability on X.
    """
    steps = compute_step_lengths(x, y)
    if len(steps) < 10:
        print("  [overall] Not enough steps for analysis.")
        return np.nan, 0, len(steps), float(np.mean(steps)) if len(steps) else np.nan

    alpha, n_tail = fit_power_law_tail(steps, lmin=lmin)
    vals, ccdf = compute_ccdf(steps)

    # Drop ccdf == 0 to avoid log(0)
    mask = ccdf > 0
    vals = vals[mask]
    ccdf = ccdf[mask]

    fig, ax = plt.subplots(figsize=(6, 5))
    # NOTE: x = probability, y = step length (flipped vs usual)
    ax.loglog(ccdf, vals, marker=".", linestyle="none", alpha=0.6)
    ax.set_xlabel("P(L ≥ ℓ)")
    ax.set_ylabel("Step length ℓ (deg)")
    ax.set_title(f"Overall step-length CCDF\n{eye_path.stem}\nα≈{alpha:.2f}, l_min={lmin}")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"ccdf_overall_{eye_path.stem}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

    mean_step = float(np.mean(steps))
    n_steps = int(len(steps))
    return alpha, n_tail, n_steps, mean_step


def analyze_levy_segments(
    t, x, y, eye_path: Path, out_dir: Path,
    n_segments=6, segment_length=10.0, lmin=L_MIN
):
    """
    Split a 60 s run into 6x10 s segments and do power-law analysis
    of step lengths for each segment. Save CCDF plot with all segments
    overlaid (with step length on Y and probability on X).
    """
    t = np.asarray(t)
    x = np.asarray(x)
    y = np.asarray(y)
    rel_t = t - t[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    alphas = []

    for k in range(n_segments):
        start = k * segment_length
        end = (k + 1) * segment_length
        mask = (rel_t >= start) & (rel_t < end)
        if mask.sum() < 3:
            continue

        x_seg = x[mask]
        y_seg = y[mask]
        steps = compute_step_lengths(x_seg, y_seg)
        if len(steps) < 10:
            continue

        alpha, n_tail = fit_power_law_tail(steps, lmin=lmin)
        alphas.append((k + 1, alpha, n_tail))

        vals, ccdf = compute_ccdf(steps)
        # Drop ccdf == 0
        m = ccdf > 0
        vals = vals[m]
        ccdf = ccdf[m]

        # x = probability, y = step length
        ax.loglog(
            ccdf, vals,
            marker=".", linestyle="none", alpha=0.5,
            label=f"Img {k+1}, α≈{alpha:.2f}, n={n_tail}"
        )

    ax.set_xlabel("P(L ≥ ℓ)")
    ax.set_ylabel("Step length ℓ (deg)")
    ax.set_title(f"Step-length CCDF by image\n{eye_path.stem}")
    ax.grid(True, which="both", alpha=0.3)
    if alphas:
        ax.legend(fontsize=7)
    fig.tight_layout()

    out_path = out_dir / f"ccdf_segments_{eye_path.stem}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

    if alphas:
        print("  Segment exponents (image, alpha, n_tail):")
        for img_idx, alpha, n_tail in alphas:
            print(f"    Img {img_idx}: alpha={alpha:.3f}, n_tail={n_tail}")
    else:
        print("  Not enough data per segment for power-law fits.")


def plot_xy_for_file(eye_path: Path, out_dir: Path):
    """
    Make a simple x–y plot for one eye-tracking CSV file,
    then run per-image + overall Levy analysis.
    """
    print(f"=== Plotting {eye_path.name} ===")
    t, x, y, df = load_txy(eye_path)

    # 2D trajectory
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, linewidth=0.7)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Gaze trajectory: {eye_path.stem}")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_name = f"traj_{eye_path.stem}.png"
    out_path = out_dir / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Per-image Levy analysis
    analyze_levy_segments(t, x, y, eye_path, out_dir)

    # Overall Levy analysis
    alpha_all, n_tail_all, n_steps, mean_step = analyze_overall_levy(
        t, x, y, eye_path, out_dir
    )

    return {
        "run": eye_path.stem,
        "n_samples": len(x),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "alpha_all": alpha_all,
        "n_tail_all": n_tail_all,
        "n_steps": n_steps,
        "mean_step_deg": mean_step,
    }


def main():
    summaries = []          # whole-run stats
    segment_summaries = []  # 10 s window stats

    # tail cutoffs (in deg) for Levy fits
    lmins = [0.5, 1.0, 2.0]
    out_dir = eye_dir / "plots_xy"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for fname in eye_files:
        eye_path = eye_dir / fname
        if not eye_path.exists():
            print(f"WARNING: {eye_path} does not exist, skipping.")
            continue

        run_name = eye_path.stem
        print(f"\n=== Analyzing {run_name} ===")

        # 1. Load eye data
        t, x, y, df_eye = load_eye_csv(eye_path)
        n_samples = len(x)
        if n_samples < 2:
            print("  Not enough samples, skipping.")
            continue

        # 2. Basic timing stats
        dt = np.median(np.diff(t))
        duration = t[-1] - t[0] if n_samples > 1 else 0.0
        fs = 1.0 / dt if dt > 0 else np.nan

        print(f"  Samples: {n_samples}, duration ~ {duration:.2f} s, fs ~ {fs:.1f} Hz")

        # 3. Step lengths + Levy tail fits (whole run)
        steps = compute_step_lengths(x, y)
        mean_step = np.mean(steps)
        median_step = np.median(steps)
        max_step = np.max(steps)

        alphas = []
        counts = []
        for lmin in lmins:
            alpha_hat, n_tail = fit_power_law_tail(steps, lmin=lmin)
            alphas.append(alpha_hat)
            counts.append(n_tail)

        # 4. MSD and MSD exponent (whole run)
        max_lag = min(200, n_samples // 4)
        lags, msd = compute_msd(x, y, max_lag=max_lag)
        tau = lags * dt
        beta = fit_loglog_slope(tau, msd,
                                xmin=2 * dt,
                                xmax=duration / 5 if duration > 0 else None)

        # 5. Plots for this run
        run_outdir = RUN_DIR / run_name
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
            else:
                eeg_info["eeg_file"] = None
        else:
            eeg_info["eeg_file"] = None

        # 7. Collect summary for this run (whole 60 s trial)
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

        # 8. --- Per 10 s segment metrics (one per still image) ---
        for seg_idx, t_start, t_end, mask in iter_time_segments(t, SEG_LEN):
            t_seg = t[mask]
            x_seg = x[mask]
            y_seg = y[mask]
            n_seg = len(x_seg)

            if n_seg < 10:
                continue

            steps_seg = compute_step_lengths(x_seg, y_seg)
            mean_step_seg = np.mean(steps_seg)
            median_step_seg = np.median(steps_seg)
            max_step_seg = np.max(steps_seg)

            alphas_seg = []
            counts_seg = []
            for lmin in lmins:
                alpha_hat_seg, n_tail_seg = fit_power_law_tail(steps_seg, lmin=lmin)
                alphas_seg.append(alpha_hat_seg)
                counts_seg.append(n_tail_seg)

            max_lag_seg = min(100, n_seg // 4)
            lags_seg, msd_seg = compute_msd(x_seg, y_seg, max_lag=max_lag_seg)
            beta_seg = fit_loglog_slope(lags_seg * dt, msd_seg,
                                        xmin=2 * dt,
                                        xmax=(t_end - t_start) / 2)

            segment_summaries.append({
                "run": run_name,
                "segment_idx": seg_idx,
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

    # 9. Save summary tables
        summary_row = plot_xy_for_file(eye_path, out_dir)
        summaries.append(summary_row)

    # Save per-run summary CSV
    if summaries:
        df_sum = pd.DataFrame(summaries)
        df_sum.to_csv(RUN_DIR / "levy_summary_all_runs.csv", index=False)
        print("\nSaved summary to:", RUN_DIR / "levy_summary_all_runs.csv")
        out_csv = out_dir / "levy_summary_per_run.csv"
        df_sum.to_csv(out_csv, index=False)
        print(f"\nSaved per-run summary to {out_csv}")
        print(df_sum)

    if segment_summaries:
        df_seg = pd.DataFrame(segment_summaries)
        df_seg.to_csv(RUN_DIR / "levy_summary_segments_10s.csv", index=False)
        print("Saved 10 s segment summary to:",
              RUN_DIR / "levy_summary_segments_10s.csv")



if __name__ == "__main__":
    main()
