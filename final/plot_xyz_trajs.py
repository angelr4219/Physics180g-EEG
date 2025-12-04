#!/usr/bin/env python3
"""
3D gaze trajectories: (x, y, t) for multiple Levy-flight eye-tracking runs.

For each CSV file:
  - Load time and gaze position (CalcXEccDeg, CalcYEccDeg)
  - Filter to valid samples (if 'Valid' column exists)
  - Convert CaptureTimeUnixMs to seconds, starting at t=0
  - Make a 3D plot: x vs y vs t
  - Optionally write an XDMF line file for ParaView

XDMF output (if meshio is installed via `pip install meshio`):
  - traj3d_<stem>.xdmf
  - Each is a polyline (line cells) in 3D with coordinates (x_deg, y_deg, t_sec)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Try to import meshio (for XDMF). If unavailable, we’ll just skip XDMF.
try:
    import meshio
    HAVE_MESHIO = True
except ImportError:
    HAVE_MESHIO = False

# ---------------- CONFIG ----------------

eye_dir = Path("./")  # folder with eye-tracking CSVs

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


X_LABEL = "Horizontal position (deg)"
Y_LABEL = "Vertical position (deg)"
T_LABEL = "Time (s)"

# ------------- HELPERS -------------


def pick_column(df, candidates, what="x"):
    """
    Try to pick a column from df given a list of candidate names.
    """
    for name in candidates:
        if name in df.columns:
            return df[name]
    raise ValueError(
        f"Could not find a {what}-column in {list(df.columns)}. "
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
            return df[df[col].astype(bool)]
    return df


def load_txy(path: Path):
    """
    Load a CSV and return (t_sec, x, y, df_filtered).

    - Assumes semicolon-separated files.
    - Time: CaptureTimeUnixMs (converted to seconds, t=0 at first sample).
    - Position: CalcXEccDeg, CalcYEccDeg (or fallbacks).
    """
    df = pd.read_csv(path, sep=";")

    df = maybe_filter_valid(df)

    # Time column
    t_candidates = ["CaptureTimeUnixMs", "TimeMs", "t_ms"]
    t_ms = pick_column(df, t_candidates, "time (ms)").astype(float)
    t_ms = t_ms.to_numpy()
    t_sec = (t_ms - t_ms[0]) / 1000.0

    x_candidates = ["CalcXEccDeg", "XDeg", "x_deg", "X", "x", "GazeX"]
    y_candidates = ["CalcYEccDeg", "YDeg", "y_deg", "Y", "y", "GazeY"]

    x = pick_column(df, x_candidates, "x").to_numpy()
    y = pick_column(df, y_candidates, "y").to_numpy()

    return t_sec, x, y, df


# ------------- PLOTTING & XDMF -------------


def plot_3d_txy(eye_path: Path, out_dir: Path):
    """
    Make a 3D plot (x, y, t) and save as PNG.
    """
    print(f"=== 3D plotting {eye_path.name} ===")
    t, x, y, df = load_txy(eye_path)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot as a single continuous line
    ax.plot(x, y, t, linewidth=0.7)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_zlabel(T_LABEL)
    ax.set_title(f"3D gaze trajectory: {eye_path.stem}")

    # Optional: set aspect ratio roughly equal in x,y, but time can be stretched
    # This is a heuristic; you can tweak later if you want.
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    t_range = t.max() - t.min()
    max_xy = max(x_range, y_range, 1e-9)
    ax.set_box_aspect((x_range / max_xy, y_range / max_xy, t_range / max_xy))

    fig.tight_layout()
    out_path = out_dir / f"traj3d_{eye_path.stem}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved 3D PNG: {out_path}")


def write_xdmf_txy(eye_path: Path, out_dir: Path):
    """
    Write an XDMF line mesh for ParaView with coordinates (x, y, t).

    Requires meshio: `pip install meshio`.

    - Points: (x_deg, y_deg, t_sec)
    - Cells: line segments connecting consecutive samples
    """
    if not HAVE_MESHIO:
        print("  [XDMF] meshio not installed; skipping XDMF export.")
        return

    print(f"  [XDMF] Writing XDMF for {eye_path.name}")
    t, x, y, df = load_txy(eye_path)

    # Points in 3D
    n = len(t)
    points = np.column_stack([x, y, t])  # shape (n, 3)

    # Line connectivity: (0-1), (1-2), ..., (n-2, n-1)
    if n < 2:
        print("  [XDMF] Not enough points for a polyline, skipping.")
        return

    cells = [("line", np.column_stack([np.arange(n - 1), np.arange(1, n)]))]

    # Optionally attach time as a point-data field (duplicate of z)
    point_data = {
        "t_sec": t,
        "x_deg": x,
        "y_deg": y,
    }

    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)

    out_path = out_dir / f"traj3d_{eye_path.stem}.xdmf"
    meshio.write(out_path, mesh)
    print(f"  [XDMF] Saved: {out_path}")


def main():
    out_dir = eye_dir / "plots_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in eye_files:
        eye_path = eye_dir / fname
        if not eye_path.exists():
            print(f"WARNING: {eye_path} does not exist, skipping.")
            continue

        plot_3d_txy(eye_path, out_dir)
        write_xdmf_txy(eye_path, out_dir)


if __name__ == "__main__":
    main()
