#!/usr/bin/env python3
"""
Lévy-flight style gaze analysis for fg-overtrtData / fg-overteyeTr files.

This script:
  * loads reaction-time and eye-tracking CSVs
  * extracts per-trial gaze trajectories
  * computes step-lengths and fits a power-law tail (μ)
  * computes mean-squared displacement scaling (γ)
  * saves a per-trial summary CSV with RT and simple gaze stats

You can adapt paths, s_min, and max_lag at the top.
"""

import numpy as np
import pandas as pd
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

# ---------- Config ----------

RT_PATH  = Path("fg-overtrtData-2025-11-03-14-09-03.csv")
EYE_PATH = Path("fg-overteyeTr-2025-11-03-14-09-03.csv")

S_MIN = 0.4      # step-length cutoff for power-law tail (ecc units)
MAX_LAG = 20     # number of time steps for MSD
DT_MS_DEFAULT = 5.0  # fallback sampling period in ms



# ---------- Helpers ----------
def load_rt(path: Path) -> pd.DataFrame:
    rt = pd.read_csv(path)
    rt["RT_ms"] = rt["ReactionTime"] - rt["ObjShowTime"]
    return rt


def load_eye(path: Path):
    """Load eye data, keep VALID samples, return (df, dt_ms)."""
    eye = pd.read_csv(path, sep=";", header=None, low_memory=False)
    eye.columns = ["CaptureTime", "CalcXEccentricity", "CalcYEccentricity",
                   "Valid", "CombinedGazeForward"]

    # Drop header row if present
    eye = eye[eye["CaptureTime"] != "CaptureTime"].copy()

    # Convert to numeric
    eye["CaptureTime"]       = pd.to_numeric(eye["CaptureTime"], errors="coerce")
    eye["CalcXEccentricity"] = pd.to_numeric(eye["CalcXEccentricity"], errors="coerce")
    eye["CalcYEccentricity"] = pd.to_numeric(eye["CalcYEccentricity"], errors="coerce")

    # Keep only valid rows with non-NaN values
    eye = eye[eye["Valid"] == "VALID"].dropna(
        subset=["CaptureTime", "CalcXEccentricity", "CalcYEccentricity"]
    )

    # Sort by time
    eye = eye.sort_values("CaptureTime").reset_index(drop=True)

    # Estimate sampling interval
    if len(eye) > 2:
        dt_ms = float(np.median(np.diff(eye["CaptureTime"].values)))
    else:
        dt_ms = DT_MS_DEFAULT

    return eye, dt_ms


def extract_trials(rt: pd.DataFrame, eye: pd.DataFrame):
    """Return list of per-trial 2D trajectories and per-trial step arrays."""
    trajs = []
    steps_per_trial = []

    for _, trial in rt.iterrows():
        t0 = trial["ObjShowTime"]
        t1 = trial["ReactionTime"]

        seg = eye[(eye["CaptureTime"] >= t0) & (eye["CaptureTime"] <= t1)]
        if len(seg) < 3:
            trajs.append(None)
            steps_per_trial.append(None)
            continue

        xy = seg[["CalcXEccentricity", "CalcYEccentricity"]].to_numpy()
        diffs = np.diff(xy, axis=0)
        steps = np.sqrt((diffs ** 2).sum(axis=1))

        trajs.append(xy)
        steps_per_trial.append(steps)

    return trajs, steps_per_trial


def fit_mu_mle(steps_all: np.ndarray, s_min: float):
    """Continuous power-law MLE for P(s) ~ s^{-μ} for s >= s_min."""
    steps_all = np.asarray(steps_all)
    tail = steps_all[steps_all >= s_min]
    if len(tail) == 0:
        return math.nan, 0

    mu = 1.0 + len(tail) / np.sum(np.log(tail / s_min))
    return mu, len(tail)


def compute_msd(trajs, max_lag: int, dt_s: float):
    """Compute ensemble-averaged MSD(τ) across all trajectories."""
    msd = np.zeros(max_lag, dtype=float)
    counts = np.zeros(max_lag, dtype=int)

    for xy in trajs:
        if xy is None:
            continue
        n = len(xy)
        for lag in range(1, max_lag + 1):
            if n <= lag:
                continue
            disp = xy[lag:] - xy[:-lag]
            sq = np.sum(disp ** 2, axis=1)
            msd[lag - 1] += sq.sum()
            counts[lag - 1] += len(sq)

    valid = counts > 0
    msd[valid] = msd[valid] / counts[valid]
    taus = np.arange(1, max_lag + 1) * dt_s
    return taus[valid], msd[valid]


def fit_loglog_slope(x: np.ndarray, y: np.ndarray):
    """Fit log y ~ a + γ log x and return γ."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x_log = np.log(x[mask])
    y_log = np.log(y[mask])
    if len(x_log) < 2:
        return math.nan, (math.nan, math.nan)
    coeffs = np.polyfit(x_log, y_log, 1)
    slope, intercept = coeffs[0], coeffs[1]
    return slope, (slope, intercept)


def build_trial_summary(rt: pd.DataFrame, trajs, steps_per_trial):
    rows = []
    for i, (idx, trial) in enumerate(rt.iterrows()):
        steps = steps_per_trial[i]
        if steps is None or len(steps) == 0:
            mean_step = np.nan
            path_len = np.nan
            n_steps = 0
        else:
            mean_step = float(np.mean(steps))
            path_len = float(np.sum(steps))
            n_steps = int(len(steps))

        rows.append({
            "trial_index": int(idx),
            "CueShowTime": int(trial["CueShowTime"]),
            "ObjShowTime": int(trial["ObjShowTime"]),
            "ReactionTime": int(trial["ReactionTime"]),
            "RT_ms": float(trial["RT_ms"]),
            "Eccentricity": int(trial["Eccentricity"]),
            "VerticalEccentricity": int(trial["VerticalEccentricity"]),
            "StimType": str(trial["StimType"]),
            "Correct": bool(trial["Correct"]),
            "mean_step": mean_step,
            "path_len": path_len,
            "n_steps": n_steps,
        })
    return pd.DataFrame(rows)


# ---------- Main pipeline ----------
def main():
    print("Loading data...")
    rt = load_rt(RT_PATH)
    eye, dt_ms = load_eye(EYE_PATH)
    dt_s = dt_ms / 1000.0
    print(f"  RT trials: {len(rt)}")
    print(f"  Eye samples: {len(eye)}  (dt ≈ {dt_ms:.2f} ms)")

    print("Extracting per-trial trajectories and step lengths...")
    trajs, steps_per_trial = extract_trials(rt, eye)

    # Concatenate all steps across trials
    all_steps = np.concatenate(
        [s for s in steps_per_trial if s is not None and len(s) > 0]
    )
    print(f"  Total steps: {len(all_steps)}")

    print("Fitting power-law tail for step-length distribution...")
    mu, n_tail = fit_mu_mle(all_steps, S_MIN)
    print(f"  Using s_min = {S_MIN:.3f}")
    print(f"  Tail samples: {n_tail}")
    print(f"  Estimated Lévy exponent μ ≈ {mu:.3f}")

    print("Computing MSD and scaling exponent γ...")
    taus, msd = compute_msd(trajs, MAX_LAG, dt_s)
    gamma, _ = fit_loglog_slope(taus, msd)
    print(f"  Estimated MSD scaling exponent γ ≈ {gamma:.3f}")

    # Per-trial summary (for correlations with RT, alpha, etc.)
    summary = build_trial_summary(rt, trajs, steps_per_trial)
    out_csv = "trial_summary_levy_gaze.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved per-trial summary to {out_csv}")

    # Optional plots for slides
    if HAVE_MPL:
        # Step-length CCDF
        sorted_steps = np.sort(all_steps)
        ccdf = 1.0 - np.arange(len(sorted_steps)) / len(sorted_steps)
        plt.figure()
        plt.loglog(sorted_steps, ccdf, marker=".", linestyle="none")
        plt.xlabel("Step length s (ecc units)")
        plt.ylabel("P(S > s)")
        plt.title(f"Step-length CCDF (μ ≈ {mu:.2f}, s_min={S_MIN})")
        plt.tight_layout()
        plt.savefig("step_length_ccdf.png", dpi=200)

        # MSD
        plt.figure()
        plt.loglog(taus, msd, marker="o")
        plt.xlabel("Lag τ (s)")
        plt.ylabel("MSD(τ)")
        plt.title(f"MSD vs lag (γ ≈ {gamma:.2f})")
        plt.tight_layout()
        plt.savefig("msd_loglog.png", dpi=200)

        print("Saved plots: step_length_ccdf.png, msd_loglog.png")
    else:
        print("matplotlib not installed; skipping plots.")

    # Quick example correlation: RT vs path length
    valid = summary[summary["n_steps"] > 0].dropna(subset=["RT_ms", "path_len"])
    if len(valid) > 2:
        r = np.corrcoef(valid["RT_ms"], valid["path_len"])[0, 1]
        print(f"Correlation RT_ms vs path_len per trial: r ≈ {r:.3f}")


if __name__ == "__main__":
    main()
