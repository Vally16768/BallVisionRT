import numpy as np

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

INPUT_VIDEO_PATH = "./rgb.avi"

# Output videos
OUTPUT_VIDEO_PATH = "output_3d_trajectory.avi"
OUTPUT_TOPVIEW_VIDEO_PATH = "output_topview_2d_map.avi"

# Output CSV
OUTPUT_CSV_PATH = "ball_3d_trajectory.csv"

# Ball diameter (meters)
BALL_DIAMETER_M = 0.22

# Camera intrinsics (approx.)
FX_NATIVE = 600.0
FY_NATIVE = 600.0
CX_NATIVE = None
CY_NATIVE = None

# Resize width
TARGET_WIDTH = 960

# Visualization
MAX_TRAIL_LENGTH = 64

# Enable/disable floor mask
USE_FLOOR_MASK = True

# Normalized floor polygon (values in [0,1])
FLOOR_POLY_NORM = np.array([
    [0.05, 0.60],
    [0.95, 0.60],
    [0.95, 0.98],
    [0.05, 0.98],
], dtype=np.float32)

# Ground rectangle in meters
GROUND_WIDTH_M = 5.0
GROUND_HEIGHT_M = 3.0

# Top-view output image size
TOPVIEW_WIDTH = 600
TOPVIEW_HEIGHT = 600
