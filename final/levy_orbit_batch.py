#!/usr/bin/env pvpython
"""
Batch-render spinning 3D gaze trajectories from ParaView.

For each .vtp file:
  - open traj3d_*.vtp
  - white background + grid box with axis labels
  - Tube filter on the trajectory
  - color by t_sec
  - orbit the camera 360° and save PNG frames

Later you can turn the PNG frames into mp4 with ffmpeg.
"""

import os
import glob
from paraview.simple import *

# --------- CONFIG ---------
# Folder where your .vtp files live (the ones we wrote from Python)
INPUT_DIR = "/Users/angelramirez/physics180g/Physics180g-EEG/final/plots_3d"

# Pattern for trajectories (adjust if needed)
FILE_PATTERN = "traj3d_*.vtp"   # or "traj3d_*.xdmf" if you used XDMF

# Where to save frame sequences
OUTPUT_DIR = os.path.join(INPUT_DIR, "orbits")

# Number of frames per full 360° orbit
N_FRAMES = 180

# Image resolution
IMG_RES = [1600, 1200]
# --------------------------


def setup_view():
    """Create/return a RenderView with white background and grid axes."""
    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = IMG_RES

    # White background
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 1

    # Turn on 3D axes grid (box with ticks + grid)
    view.AxesGrid = "GridAxes3DActor"
    grid = view.AxesGrid
    grid.Visibility = 1

    grid.XTitle = "Horizontal position (deg)"
    grid.YTitle = "Vertical position (deg)"
    grid.ZTitle = "Time (s)"

    # Light gray grid lines
    grid.GridColor = [0.8, 0.8, 0.8]

    # Show grid on all faces
    grid.ShowGrid = [1, 1, 1]

    return view


def add_tube(data, view):
    """Tube filter on the line, color by t_sec."""
    display = Show(data, view)
    display.SetRepresentationType("Surface")

    tube = Tube(Input=data)
    tube.Radius = 0.3  # tweak if your units are different

    Hide(data, view)
    tube_disp = Show(tube, view)
    tube_disp.SetRepresentationType("Surface")

    # Color by t_sec if present
    try:
        ColorBy(tube_disp, ("POINTS", "t_sec"))
        tube_disp.RescaleTransferFunctionToDataRange(True, False)
        GetColorTransferFunction("t_sec").ColorSpace = "Diverging"
    except Exception:
        pass

    return tube, tube_disp


def orbit_and_save(view, basename, out_dir):
    """Orbit camera 360° and save N_FRAMES PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    cam = view.GetActiveCamera()

    # Optional tilt: look a bit down
    cam.Elevation(20)
    Render()

    dphi = 360.0 / N_FRAMES
    for i in range(N_FRAMES):
        cam.Azimuth(dphi)
        Render()
        fname = os.path.join(out_dir, f"{basename}_{i:03d}.png")
        SaveScreenshot(fname, view, ImageResolution=IMG_RES)


def process_file(fpath):
    print(f"\n=== Processing {os.path.basename(fpath)} ===")
    ResetSession()

    # Read data
    src = OpenDataFile(fpath)
    RenameSource(os.path.basename(fpath), src)

    view = setup_view()

    # Show data & fit camera
    Show(src, view)
    view.ResetCamera()

    # Tube + coloring
    tube, tube_disp = add_tube(src, view)
    view.ResetCamera()

    # Output folder for frames
    base = os.path.splitext(os.path.basename(fpath))[0]
    frames_dir = os.path.join(OUTPUT_DIR, base + "_frames")

    orbit_and_save(view, base, frames_dir)

    print(f"Frames saved in: {frames_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(INPUT_DIR, FILE_PATTERN)))
    if not files:
        print("No files found. Check INPUT_DIR and FILE_PATTERN.")
        return

    print("Found files:")
    for f in files:
        print("  ", os.path.basename(f))

    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
