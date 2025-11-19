import numpy as np

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

# Input video (same folder as the scripts)
INPUT_VIDEO_PATH = "./rgb.avi"

# Output videos
OUTPUT_VIDEO_PATH = "output_3d_trajectory.avi"        # RGB with 3D info
OUTPUT_TOPVIEW_VIDEO_PATH = "output_topview_2d_map.avi"  # Stabilized top-view

# Output CSV
OUTPUT_CSV_PATH = "ball_3d_trajectory.csv"

# Ball diameter (meters) - standard football ~22 cm
BALL_DIAMETER_M = 0.22

# Camera intrinsics (approx.)
FX_NATIVE = 600.0
FY_NATIVE = 600.0
CX_NATIVE = None   # if None -> w/2
CY_NATIVE = None   # if None -> h/2

# Resize target width for processing (None = original)
TARGET_WIDTH = 960

# Trajectory visualization (length of the trail on ground plane)
MAX_TRAIL_LENGTH = 64

# Use / skip floor mask for detection and motion estimation
USE_FLOOR_MASK = True

# ------------------------------------------------------------------
# Fallback normalized floor polygon (values in [0,1])
# Order: top-left, top-right, bottom-right, bottom-left
# Used ONLY if automatic floor estimation from the ball fails.
# ------------------------------------------------------------------
FLOOR_POLY_NORM = np.array([
    [0.05, 0.60],
    [0.95, 0.60],
    [0.95, 0.98],
    [0.05, 0.98],
], dtype=np.float32)

# Ground rectangle in meters (our "map" size)
GROUND_WIDTH_M = 5.0
GROUND_HEIGHT_M = 3.0

# Top-view output image size (pixels)
TOPVIEW_WIDTH = 600
TOPVIEW_HEIGHT = 600

# ------------------------------------------------------------------
# Automatic floor estimation parameters (heat propagation from ball)
# ------------------------------------------------------------------
AUTO_FLOOR_COLOR_THRESH = 12.0      # LAB distance threshold for floor similarity
AUTO_FLOOR_MIN_AREA_RATIO = 0.02    # min floor area as fraction of image area
AUTO_FLOOR_MORPH_KERNEL = 7         # morphological kernel size (pixels)
