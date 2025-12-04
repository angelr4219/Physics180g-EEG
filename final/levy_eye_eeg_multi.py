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

# ---------------- CONFIG ----------------

eye_dir = Path("./")  # base folder

eye_files = [
    "vid1_A/Angel1-levy-flight-eyeTr-2025-12-02-13-13-12.csv",
    "vid1_B/Iago-levy-flight-eyeTr-2025-12-02-13-18-19.csv",
    "vid1_C/Ravi-levy-flight-eyeTr-2025-12-02-13-22-45.csv",
    "vid1_D/Seth-levy-flight-eyeTr-2025-12-02-13-29-08.csv",
    "vid2_A/SETH2-levy-flight-eyeTr-2025-12-02-13-35-06.csv",
    "vid2_C/Iago2-levy-flight-eyeTr-2025-12-02-13-42-21.csv",
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

    # x, y in deg
    x = pick_column(df, x_candidates, "x")
    y = pick_column(df, y_candidates, "y")

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
    out_dir = eye_dir / "plots_xy"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    for fname in eye_files:
        eye_path = eye_dir / fname
        if not eye_path.exists():
            print(f"WARNING: {eye_path} does not exist, skipping.")
            continue
        summary_row = plot_xy_for_file(eye_path, out_dir)
        summaries.append(summary_row)

    # Save per-run summary CSV
    if summaries:
        df_sum = pd.DataFrame(summaries)
        out_csv = out_dir / "levy_summary_per_run.csv"
        df_sum.to_csv(out_csv, index=False)
        print(f"\nSaved per-run summary to {out_csv}")
        print(df_sum)


if __name__ == "__main__":
    main()
