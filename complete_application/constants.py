import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_VIDEO_PATH = "./rgb.avi"

# Output videos
OUTPUT_VIDEO_PATH = "output_3d_trajectory.avi"          # RGB cu 3D + trail
OUTPUT_TOPVIEW_VIDEO_PATH = "output_topview_2d_map.avi"  # Top-view stabilizat (ground plane)

# Output CSV
OUTPUT_CSV_PATH = "ball_3d_trajectory.csv"

# Ball real-world diameter (meters) - standard football ~22 cm
BALL_DIAMETER_M = 0.22

# --- Camera intrinsics (these will be refined from video metadata) ---
FX_NATIVE = 600.0
FY_NATIVE = 600.0
CX_NATIVE = None
CY_NATIVE = None

# Resize target width for processing (None = original)
TARGET_WIDTH = 960

# Trajectory visualization
MAX_TRAIL_LENGTH = 64  # number of points kept in the trail

# ============================================================
# FLOOR ROI / GROUND PLANE
# ============================================================

USE_FLOOR_MASK = True

# Normalized polygon (x, y in [0,1]) that covers only the terrace (gray rectangle).
# Points are in order: bottom-left, bottom-right, top-right, top-left.
FLOOR_POLY_NORM = np.array([
    [0.10, 1.00],   # bottom-left
    [1.00, 1.00],   # bottom-right
    [1.00, 0.20],   # top-right
    [0.10, 0.20],   # top-left
], dtype=np.float32)

# Approximate physical dimensions of the ground patch (terrace) in meters.
GROUND_WIDTH_M = 3.0   # left-right (X axis)
GROUND_LENGTH_M = 8.0  # forward-away-from-camera (Z axis)

# ============================================================
# TOP-VIEW MAP CONFIG (GROUND PLANE, STABILIZED)
# ============================================================

TOPVIEW_WIDTH = 800
TOPVIEW_HEIGHT = 600
TOPVIEW_BG_COLOR = (30, 30, 30)  # dark gray background

# Camera location on the top-view image (in pixels).
# We place the camera at the middle of the bottom edge.
CAMERA_PX = (TOPVIEW_WIDTH // 2, TOPVIEW_HEIGHT - 40)

# Margins (for drawing the ground rectangle inside the top-view image)
TOPVIEW_MARGIN_X = 80
TOPVIEW_MARGIN_Y = 60

# ============================================================
# STATIC BALL DETECTION (GROUND PLANE)
# ============================================================

# Time window (seconds) over which we decide if the ball is static.
STATIC_WINDOW_SEC = 1.0  # last 1 second

# Maximum total displacement in ground plane within the window (meters)
# to still consider the ball static.
STATIC_DIST_THRESHOLD_M = 0.10  # 10 cm

# Maximum average speed in ground plane within the window (meters/second)
# to still consider the ball static.
STATIC_SPEED_THRESHOLD_MS = 0.05  # 5 cm/s
