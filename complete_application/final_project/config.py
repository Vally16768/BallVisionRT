"""
config.py

Centralized configuration for the DotLumen 2D/3D ball tracking and
camera stabilization pipeline.
"""

import os

# ==========================
# Paths
# ==========================

# Input video path (RealSense D435i RGB stream recorded as rgb.avi)
VIDEO_PATH = "rgb.avi"

# Output directory for all generated artifacts
OUTPUT_DIR = "output_dotlumen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Individual output files
OUTPUT_FIRST_FRAME = os.path.join(OUTPUT_DIR, "frame0.png")
OUTPUT_FLOOR_MASK = os.path.join(OUTPUT_DIR, "floor_mask.png")
OUTPUT_FLOOR_OVERLAY = os.path.join(OUTPUT_DIR, "floor_overlay.png")
OUTPUT_FLOOR_TOPVIEW = os.path.join(OUTPUT_DIR, "floor_topview.png")

OUTPUT_DET_VIDEO = os.path.join(OUTPUT_DIR, "ball_detection.avi")
OUTPUT_TRAJ_VIDEO = os.path.join(OUTPUT_DIR, "ball_trajectory.avi")
OUTPUT_TOPVIEW_VIDEO = os.path.join(OUTPUT_DIR, "topview_map.avi")
OUTPUT_TOPVIEW_IMAGE = os.path.join(OUTPUT_DIR, "topview_final.png")

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ball_3d_positions_and_velocities.csv")

# ==========================
# Camera and scene constants
# ==========================

# Approximate horizontal FOV (in degrees) for RealSense D435i RGB camera.
# Used to estimate focal length in pixels.
D435I_RGB_HFOV_DEG = 69.0

# Physical soccer ball diameter (in meters). Used for 3D distance estimation.
BALL_DIAMETER_M = 0.22
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0

# ==========================
# Top-view (bird's-eye) map size
# ==========================

TOP_W = 800  # width of the top-view map in pixels
TOP_H = 600  # height of the top-view map in pixels
