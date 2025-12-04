#!/usr/bin/env python3
"""
Plot 2D gaze trajectories (x vs y) for multiple Levy-flight eye-tracking runs.

For each CSV file:
  - Load data
  - Try to find x and y gaze columns (deg of visual angle)
  - Optionally filter to valid samples if a validity column exists
  - Plot x vs y and save as PNG

Edit the CONFIG section to point to your actual filenames.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

# Folder containing your eye-tracking CSVs
eye_dir = Path("./")  # change this if needed

# List your 7 files here (relative to eye_dir)
eye_files = [
    # Example (you already have this one):ƒ
    "vid1_A/Angel1-levy-flight-eyeTr-2025-12-02-13-13-12.csv",
    "vid1_B/Iago-levy-flight-eyeTr-2025-12-02-13-18-19.csv",
    "vid1_C/Ravi-levy-flight-eyeTr-2025-12-02-13-22-45.csv",
    "vid1_D/Seth-levy-flight-eyeTr-2025-12-02-13-29-08.csv",
    "vid2_A/SETH2-levy-flight-eyeTr-2025-12-02-13-35-06.csv",
    "vid2_C/Iago2-levy-flight-eyeTr-2025-12-02-13-42-21.csv",
    "vid2_D/Ravi2-levy-flight-eyeTr-2025-12-02-13-46-15.csv"
    # Add more files as needed
]

# If your positions are in degrees, set labels accordingly
X_LABEL = "Horizontal position (deg)"
Y_LABEL = "Vertical position (deg)"

# ------------- HELPER FUNCTIONS -------------

def pick_column(df, candidates, what="x"):
    """
    Try to pick a column from df given a list of candidate names.
    Raises a ValueError if none of the candidates is found.
    """
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise ValueError(
        f"Could not find a {what}-column in {df.columns}. "
        f"Tried: {candidates}"
    )


def maybe_filter_valid(df):
    """
    If there is a 'valid' column (or similar), restrict to valid samples.
    Otherwise, return df unchanged.
    """
    valid_candidates = ["Valid", "valid", "Validity", "EyeValid", "GazeValid"]
    for col in valid_candidates:
        if col in df.columns:
            # assume 1 or True means valid
            return df[df[col].astype(bool)]
    return df


def load_xy(path: Path):
    """
    Load a CSV and return (x, y, df_filtered).
    Tries several likely column names for x and y.
    """
    # Your files are semicolon-separated
    df = pd.read_csv(path, sep=";")

    # Restrict to valid samples if possible
    df = maybe_filter_valid(df)

    x_candidates = ["CalcXEccDeg", "XDeg", "x_deg", "X", "x", "GazeX"]
    y_candidates = ["CalcYEccDeg", "YDeg", "y_deg", "Y", "y", "GazeY"]

    x = pick_column(df, x_candidates, "x")
    y = pick_column(df, y_candidates, "y")

    return x, y, df



# ------------- MAIN PLOTTING CODE -------------

def plot_xy_for_file(eye_path: Path, out_dir: Path):
    """
    Make a simple x–y plot for one eye-tracking CSV file.
    """
    print(f"=== Plotting {eye_path.name} ===")
    x, y, df = load_xy(eye_path)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, linewidth=0.7)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Gaze trajectory: {eye_path.stem}")
    ax.set_aspect("equal")  # preserve aspect ratio
    ax.grid(True, alpha=0.3)

    out_name = f"traj_{eye_path.stem}.png"
    out_path = out_dir / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved {out_path}")


def main():
    out_dir = eye_dir / "plots_xy"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in eye_files:
        eye_path = eye_dir / fname
        if not eye_path.exists():
            print(f"WARNING: {eye_path} does not exist, skipping.")
            continue
        plot_xy_for_file(eye_path, out_dir)


if __name__ == "__main__":
    main()
