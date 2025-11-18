import cv2
import csv
import math
import numpy as np
from collections import deque

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
# If you know the exact intrinsics of your D435i RGB stream, put them here.
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
# These values do not need to be exact; they control only the scale of the top-view.
GROUND_WIDTH_M = 3.0   # left-right (X axis)
GROUND_LENGTH_M = 8.0  # forward-away-from-camera (Z axis)

# ============================================================
# TOP-VIEW MAP CONFIG (GROUND PLANE, STABILIZED)
# ============================================================

TOPVIEW_WIDTH = 800
TOPVIEW_HEIGHT = 600
TOPVIEW_BG_COLOR = (30, 30, 30)  # dark gray background

# Camera location on the top-view image (in pixels).
# We will place the camera at the middle of the bottom edge.
CAMERA_PX = (TOPVIEW_WIDTH // 2, TOPVIEW_HEIGHT - 40)

# Margins (for drawing the ground rectangle inside the top-view image)
TOPVIEW_MARGIN_X = 80
TOPVIEW_MARGIN_Y = 60

# ============================================================
# HELPERS: FLOOR ROI
# ============================================================

def create_floor_mask_and_polygon(frame_shape):
    """
    Create a binary mask (255 on floor, 0 elsewhere) based on FLOOR_POLY_NORM.
    Also return the polygon points in pixel coordinates (Nx1x2 int32).
    """
    h, w = frame_shape[:2]
    pts = np.zeros_like(FLOOR_POLY_NORM, dtype=np.int32)
    pts[:, 0] = (FLOOR_POLY_NORM[:, 0] * w).astype(np.int32)
    pts[:, 1] = (FLOOR_POLY_NORM[:, 1] * h).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts  # also return polygon points for drawing & homography


# ============================================================
# HELPERS: Video I/O
# ============================================================

def create_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {path}")
    return cap


def preprocess_frame(frame, target_width=TARGET_WIDTH):
    """
    Optionally resize the frame to a given width while keeping aspect ratio.
    """
    if target_width is None:
        return frame
    h, w = frame.shape[:2]
    if w != target_width:
        scale = target_width / float(w)
        frame = cv2.resize(frame, (target_width, int(h * scale)))
    return frame


# ============================================================
# 2D DETECTION (red ball)
# ============================================================

def get_red_mask(frame):
    """
    Binary mask for red ball in HSV.
    Combines two red ranges (0-10 and 170-180 degrees in Hue).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower reds
    lower_red1 = np.array([0, 70, 50], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    # upper reds
    lower_red2 = np.array([170, 70, 50], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def find_red_ball(
    mask,
    frame_shape,
    min_area=150,        # ignore very small blobs
    max_area_ratio=0.05,
    min_circularity=0.5
):
    """
    Find the most likely red ball contour:
    - restricted to the lower part of the frame (ball on the floor/terrace)
    - filtered by area and circularity.

    Returns (x, y, radius) in pixel units or None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    h, w = frame_shape[:2]
    max_area = h * w * max_area_ratio
    horizon_y = int(h * 0.20)  # ignore everything too high in the image

    best_circle = None
    best_score = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4.0 * np.pi * (area / (perimeter * perimeter))
        if circularity < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        x, y, radius = int(x), int(y), int(radius)

        # the ball must be below the "horizon" line
        if y < horizon_y:
            continue

        # simple score: circularity * area
        score = circularity * area
        if score > best_score:
            best_score = score
            best_circle = (x, y, radius)

    return best_circle


def draw_ball_2d_overlay(frame, circle, text=None):
    """
    Draw the detected / tracked ball and optionally some text (e.g. distance).
    """
    if circle is None:
        return frame

    x, y, r = circle
    cv2.circle(frame, (x, y), r, (0, 0, 255), 3)     # outer circle
    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)    # center

    label = "Ball"
    if text is not None:
        label += f" | {text}"

    cv2.putText(
        frame,
        label,
        (x - 40, max(0, y - r - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )
    return frame


# ============================================================
# 3D ESTIMATION (camera coordinates)
# ============================================================

def compute_3d_position_from_circle(
    circle,
    fx,
    fy,
    cx,
    cy,
    ball_diameter_m=BALL_DIAMETER_M
):
    """
    Given:
      - circle: (x_px, y_px, r_px) in image (pixels)
      - fx, fy: focal lengths in pixels
      - cx, cy: principal point in pixels
      - ball_diameter_m: real-world diameter of the ball (meters)

    Returns:
      X, Y, Z, dist (meters) in camera coordinates
      where:
        - Z is depth along the camera's forward axis
        - X is horizontal offset
        - Y is vertical offset (upwards if camera optical axis is level)
        - dist is Euclidean distance sqrt(X^2 + Y^2 + Z^2)
    """
    x_px, y_px, r_px = circle
    d_px = 2.0 * r_px  # apparent diameter in pixels

    if d_px <= 0:
        return None

    # Pinhole model: Z = (f * D) / d
    f = (fx + fy) * 0.5
    Z = (f * ball_diameter_m) / d_px  # meters

    # Back-project to 3D camera coordinates
    X = (x_px - cx) * Z / fx
    Y = (y_px - cy) * Z / fy

    dist = math.sqrt(X * X + Y * Y + Z * Z)

    return X, Y, Z, dist


# ============================================================
# TRAJECTORY TRACKING (Kalman in image space)
# ============================================================

def create_kalman_filter(dt):
    """
    Create a 2D constant-velocity Kalman filter in image space.

    State vector: [x, y, vx, vy]^T
    Measurement: [x, y]^T
    """
    kf = cv2.KalmanFilter(4, 2, 0)

    # State transition matrix A
    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]], dtype=np.float32
    )

    # Measurement matrix H (we directly observe x, y)
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )

    # Process noise covariance Q
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

    # Measurement noise covariance R
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

    # Error covariance matrix P
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    return kf


def draw_trajectory(frame, trail_points, color=(0, 0, 255)):
    """
    Draw the trajectory trail on top of the frame.

    trail_points: deque of (x, y) tuples (ints or None).
    """
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue

        thickness = int(np.sqrt(MAX_TRAIL_LENGTH / float(i + 1)) * 2.0)
        cv2.line(
            frame,
            trail_points[i - 1],
            trail_points[i],
            color,
            thickness
        )
    return frame


# ============================================================
# CAMERA MOTION ESTIMATION (LK + HOMOGRAPHY)
# ============================================================

def detect_features(gray, mask=None, max_corners=400):
    """
    Detect good features (Shi-Tomasi) for tracking on the background.
    """
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask
    )
    return corners


def update_camera_motion(prev_gray, prev_pts, gray, H_global):
    """
    Estimate camera motion between consecutive frames via optical flow
    and update global homography H_global (reference -> current).

    Returns:
      new_prev_gray, new_prev_pts, new_H_global
    """
    if prev_gray is None or prev_pts is None or len(prev_pts) < 10:
        # Not enough points, cannot update camera motion.
        return gray, prev_pts, H_global

    next_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, prev_pts, None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    if next_pts is None:
        return gray, prev_pts, H_global

    status = status.reshape(-1)
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    if len(good_prev) < 10:
        # too few valid points
        return gray, good_next.reshape(-1, 1, 2) if len(good_next) > 0 else None, H_global

    # Homography from previous frame -> current frame
    H, inliers = cv2.findHomography(good_prev, good_next, cv2.RANSAC, 4.0)
    if H is not None:
        # Update cumulative homography (reference -> current)
        H_global = H @ H_global

        # keep only inliers as points for next step
        inliers = inliers.ravel().astype(bool)
        good_next = good_next[inliers]

    # Update state for next frame
    if good_next is not None and len(good_next) > 0:
        prev_pts_new = good_next.reshape(-1, 1, 2).astype(np.float32)
    else:
        prev_pts_new = None

    return gray, prev_pts_new, H_global


def stabilize_point(x, y, H_global):
    """
    Apply inverse of H_global to bring point (x, y) from current frame
    back into reference frame coordinates (frame 0).
    """
    if H_global is None:
        return None

    try:
        H_inv = np.linalg.inv(H_global)
    except np.linalg.LinAlgError:
        return None

    pt = np.array([[[x, y]]], dtype=np.float32)  # shape (1,1,2)
    pt_stab = cv2.perspectiveTransform(pt, H_inv.astype(np.float32))
    xs, ys = float(pt_stab[0, 0, 0]), float(pt_stab[0, 0, 1])
    return xs, ys


# ============================================================
# GROUND HOMOGRAPHY (reference image -> metric ground plane)
# ============================================================

def compute_ground_homography(floor_poly_pts):
    """
    Build a homography that maps the terrace quadrilateral in the *reference*
    frame (frame 0) to a metric ground coordinate system (X,Z) in meters.

    floor_poly_pts: Nx1x2 int32, order: bottom-left, bottom-right, top-right, top-left.

    Returns:
      H_img2ground: 3x3 float64 matrix such that:
        [X, Z, 1]^T ~ H_img2ground * [u, v, 1]^T
    """
    # Source points in image (u,v) in reference frame
    src = floor_poly_pts.reshape(-1, 2).astype(np.float32)

    # Destination points in ground coords (X,Z), in meters.
    # We map the terrace rectangle to [0,GROUND_WIDTH_M] x [0,GROUND_LENGTH_M].
    dst = np.array([
        [0.0,              0.0],               # bottom-left
        [GROUND_WIDTH_M,   0.0],               # bottom-right
        [GROUND_WIDTH_M,   GROUND_LENGTH_M],   # top-right (far)
        [0.0,              GROUND_LENGTH_M],   # top-left  (far)
    ], dtype=np.float32)

    H_img2ground, _ = cv2.findHomography(src, dst, method=0)
    return H_img2ground


def image_to_ground(u, v, H_img2ground):
    """
    Map a point (u,v) in the reference image to ground coordinates (X,Z) in meters,
    using the homography H_img2ground computed from the terrace corners.
    """
    if H_img2ground is None:
        return None

    pt = np.array([[[u, v]]], dtype=np.float32)
    pt_g = cv2.perspectiveTransform(pt, H_img2ground.astype(np.float32))
    Xg, Zg = float(pt_g[0, 0, 0]), float(pt_g[0, 0, 1])
    return Xg, Zg


# ============================================================
# TOP-VIEW HELPERS
# ============================================================

def create_empty_topview_frame():
    """
    Create a blank top-view frame with:
      - camera marker
      - ground rectangle (terrace)
      - axes consistent with the sequence:
          X axis = left/right on terrace
          Z axis = distance away from camera along terrace
    """
    frame = np.full(
        (TOPVIEW_HEIGHT, TOPVIEW_WIDTH, 3),
        TOPVIEW_BG_COLOR,
        dtype=np.uint8
    )

    # Draw ground rectangle (terrace) in top-view
    gx0 = TOPVIEW_MARGIN_X
    gx1 = TOPVIEW_WIDTH - TOPVIEW_MARGIN_X
    gz1 = TOPVIEW_MARGIN_Y                 # far end of terrace
    gz0 = TOPVIEW_HEIGHT - TOPVIEW_MARGIN_Y  # near end (where camera is)

    pts = np.array([
        [gx0, gz0],
        [gx1, gz0],
        [gx1, gz1],
        [gx0, gz1]
    ], dtype=np.int32)

    cv2.polylines(frame, [pts], isClosed=True, color=(80, 80, 80), thickness=2)

    # Camera marker at the center of the near edge
    cx, cy = CAMERA_PX
    cam_pts = np.array([
        [cx, cy - 10],
        [cx - 8, cy + 10],
        [cx + 8, cy + 10]
    ], dtype=np.int32)
    cv2.fillConvexPoly(frame, cam_pts, (0, 255, 255))
    cv2.putText(frame, "Camera", (cx - 40, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # Axes:
    #   Z axis: away from camera (upwards in the image)
    cv2.arrowedLine(frame, (cx, cy), (cx, gz1 - 20),
                    (200, 200, 200), 1, tipLength=0.03)
    cv2.putText(frame, "+Z (away)", (cx + 10, gz1 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    #   X axis: left/right along terrace
    cv2.arrowedLine(frame, (cx, cy), (gx1 + 20, cy),
                    (200, 200, 200), 1, tipLength=0.03)
    cv2.putText(frame, "+X (right)", (gx1 - 40, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def world_to_topview(Xg, Zg):
    """
    Convert ground coordinates (Xg, Zg) in meters into pixel coordinates
    in the top-view image. We assume the terrace rectangle spans
    [0,GROUND_WIDTH_M] x [0,GROUND_LENGTH_M] -> [gx0,gx1] x [gz0,gz1].
    """
    gx0 = TOPVIEW_MARGIN_X
    gx1 = TOPVIEW_WIDTH - TOPVIEW_MARGIN_X
    gz1 = TOPVIEW_MARGIN_Y                 # far
    gz0 = TOPVIEW_HEIGHT - TOPVIEW_MARGIN_Y  # near (camera side)

    # clamp to valid ground range
    Xg_clamped = max(0.0, min(GROUND_WIDTH_M, Xg))
    Zg_clamped = max(0.0, min(GROUND_LENGTH_M, Zg))

    # normalized coords in [0,1]
    nx = Xg_clamped / GROUND_WIDTH_M
    nz = Zg_clamped / GROUND_LENGTH_M

    # map to pixel coords (note Z increases away from camera -> upward in image)
    u = int(round(gx0 + nx * (gx1 - gx0)))
    v = int(round(gz0 - nz * (gz0 - gz1)))
    return u, v


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    cap = create_video_capture(INPUT_VIDEO_PATH)

    out_video_rgb = None
    out_video_topview = None
    csv_rows = []

    frame_idx = 0

    # Intrinsics for working resolution
    fx_scaled = None
    fy_scaled = None
    cx_used = None
    cy_used = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    dt = 1.0 / fps

    # Kalman filter and trajectory trail in image space
    kalman = None
    trail_image = deque(maxlen=MAX_TRAIL_LENGTH)
    last_radius_px = None  # last known radius, used when detection is missing

    # Trajectory trail in top-view (ground plane, camera-motion compensated)
    trail_topview = deque(maxlen=MAX_TRAIL_LENGTH)

    floor_mask = None
    floor_poly_pts = None
    H_img2ground = None

    # Camera motion estimation (reference frame 0 -> current frame)
    prev_gray = None
    prev_pts = None
    H_global = np.eye(3, dtype=np.float64)

    # For logging stabilized image coordinates (optional)
    xs_stab = ys_stab = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break

        frame = preprocess_frame(frame, target_width=TARGET_WIDTH)
        h, w = frame.shape[:2]

        # Initialize video writers AFTER resize (use actual processing resolution)
        if out_video_rgb is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video_rgb = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

        if out_video_topview is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video_topview = cv2.VideoWriter(
                OUTPUT_TOPVIEW_VIDEO_PATH, fourcc, fps, (TOPVIEW_WIDTH, TOPVIEW_HEIGHT)
            )

        # Initialize intrinsics once we know working resolution
        if fx_scaled is None:
            # Use provided FX_NATIVE/FY_NATIVE as approximate focal lengths in pixels.
            # If the video was not recorded at the same resolution, scale them accordingly.
            scale_factor = w / float(TARGET_WIDTH) if TARGET_WIDTH is not None else 1.0
            fx_scaled = FX_NATIVE * scale_factor
            fy_scaled = FY_NATIVE * scale_factor

            if CX_NATIVE is not None and CY_NATIVE is not None:
                cx_used = CX_NATIVE * scale_factor
                cy_used = CY_NATIVE * scale_factor
            else:
                # Fallback: principal point at image center (from video metadata).
                cx_used = w / 2.0
                cy_used = h / 2.0

            print(f"[INFO] Using intrinsics (approx): fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, "
                  f"cx={cx_used:.2f}, cy={cy_used:.2f}")
            print(f"[INFO] FPS from video metadata = {fps:.2f}, dt={dt:.3f}s")

        # Floor mask (ROI) – compute once, in reference frame coordinates
        if USE_FLOOR_MASK and floor_mask is None:
            floor_mask, floor_poly_pts = create_floor_mask_and_polygon(frame.shape)
            # Build homography from image (reference frame) to metric ground plane
            H_img2ground = compute_ground_homography(floor_poly_pts)
            if H_img2ground is None:
                print("[WARN] Could not compute ground homography. Top-view will be disabled.")
            else:
                print("[INFO] Ground homography (image -> ground) initialized.")

        # Initialize Kalman filter once (we need dt)
        if kalman is None:
            kalman = create_kalman_filter(dt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------- Camera Motion Estimation -------------------
        if frame_idx == 0:
            # First frame: detect background features, restricted to floor to
            # stay on the dominant plane (ground).
            feat_mask = floor_mask if USE_FLOOR_MASK else None
            prev_gray = gray.copy()
            prev_pts = detect_features(prev_gray, mask=feat_mask)
        else:
            prev_gray, prev_pts, H_global = update_camera_motion(prev_gray, prev_pts, gray, H_global)
            # If we ran out of points, re-detect on the current frame.
            if prev_pts is None or len(prev_pts) < 50:
                feat_mask = floor_mask if USE_FLOOR_MASK else None
                prev_pts = detect_features(gray, mask=feat_mask)
                prev_gray = gray.copy()

        # ------------------- 2D Detection -------------------
        red_mask = get_red_mask(frame)

        # Limit search to floor if enabled
        if USE_FLOOR_MASK and floor_mask is not None:
            red_mask = cv2.bitwise_and(red_mask, floor_mask)

        detected_circle = find_red_ball(red_mask, frame.shape)

        t_sec = frame_idx / fps

        # ------------------- Kalman Tracking -------------------
        # Predict step
        predicted_state = kalman.predict()
        pred_x, pred_y = float(predicted_state[0]), float(predicted_state[1])

        has_measurement = detected_circle is not None

        if has_measurement:
            meas_x, meas_y, meas_r = detected_circle
            measurement = np.array([[np.float32(meas_x)],
                                    [np.float32(meas_y)]], dtype=np.float32)
            corrected_state = kalman.correct(measurement)
            x_track = float(corrected_state[0])
            y_track = float(corrected_state[1])
            last_radius_px = meas_r
        else:
            # No detection: rely purely on prediction
            x_track = pred_x
            y_track = pred_y

        # Visualization: tracked center + last valid radius
        if last_radius_px is None:
            vis_radius = 10
        else:
            vis_radius = last_radius_px

        tracked_circle = (int(round(x_track)), int(round(y_track)), int(round(vis_radius)))

        # ------------------- 3D Estimation (for CSV and overlay only) -------------------
        if last_radius_px is not None:
            circle_for_3d = (x_track, y_track, last_radius_px)
            X_cam, Y_cam, Z_cam, dist_cam = compute_3d_position_from_circle(
                circle_for_3d,
                fx_scaled,
                fy_scaled,
                cx_used,
                cy_used,
                BALL_DIAMETER_M
            )
            overlay_text = f"Z={Z_cam:.2f}m, |P|={dist_cam:.2f}m"
        else:
            X_cam = Y_cam = Z_cam = dist_cam = None
            overlay_text = None

        # ------------------- Trajectory trail in image space -------------------
        if 0 <= x_track < w and 0 <= y_track < h:
            trail_image.append((int(round(x_track)), int(round(y_track))))
        else:
            trail_image.append(None)

        # ------------------- Camera-motion compensation & Ground Projection -------------------
        current_topview_point = None
        xs_stab = ys_stab = None
        Xg = Zg = None

        if H_global is not None and H_img2ground is not None:
            # Stabilize tracked point to reference frame (remove camera motion)
            stab_result = stabilize_point(x_track, y_track, H_global)
            if stab_result is not None:
                xs_stab, ys_stab = stab_result

                # Project stabilized point to ground plane (metric coordinates)
                ground_result = image_to_ground(xs_stab, ys_stab, H_img2ground)
                if ground_result is not None:
                    Xg, Zg = ground_result

                    # Map ground coordinates to top-view pixels
                    u, v = world_to_topview(Xg, Zg)
                    if 0 <= u < TOPVIEW_WIDTH and 0 <= v < TOPVIEW_HEIGHT:
                        current_topview_point = (u, v)
                        trail_topview.append(current_topview_point)
                    else:
                        trail_topview.append(None)
                else:
                    trail_topview.append(None)
            else:
                trail_topview.append(None)
        else:
            trail_topview.append(None)

        # ------------------- Visualization: RGB + Trajectory -------------------
        vis_frame = frame.copy()

        # ROI contour (terrace)
        if USE_FLOOR_MASK and floor_poly_pts is not None:
            cv2.polylines(vis_frame, [floor_poly_pts], isClosed=True,
                          color=(255, 0, 0), thickness=2)

        # Draw ball (tracked) with text
        vis_frame = draw_ball_2d_overlay(vis_frame, tracked_circle, text=overlay_text)

        # Draw image-space trajectory (2D in pixel coords)
        vis_frame = draw_trajectory(vis_frame, trail_image, color=(0, 0, 255))

        # If we had an actual detection this frame, mark measurement in green
        if has_measurement and detected_circle is not None:
            mx, my, _ = detected_circle
            cv2.circle(vis_frame, (mx, my), 4, (0, 255, 0), -1)

        status_txt = "Tracking: M+P" if has_measurement else "Tracking: PRED ONLY"
        cv2.putText(
            vis_frame,
            status_txt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ------------------- Visualization: Top-View (Ground, Camera-Motion Compensated) -------------------
        topview_frame = create_empty_topview_frame()

        # Draw ground-plane trajectory (stabilized)
        topview_frame = draw_trajectory(topview_frame, trail_topview, color=(0, 0, 255))

        # Draw current ball position on top-view
        if current_topview_point is not None:
            cv2.circle(topview_frame, current_topview_point, 6, (0, 0, 255), -1)
            cv2.putText(
                topview_frame,
                "Ball",
                (current_topview_point[0] + 8, current_topview_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ------------------- Output videos -------------------
        out_video_rgb.write(vis_frame)
        out_video_topview.write(topview_frame)

        cv2.imshow("3D Ball + Trajectory Tracking (RGB)", vis_frame)
        cv2.imshow("2D Top-View Ground Map (Stabilized)", topview_frame)

        # ------------------- CSV logging -------------------
        row = {
            "frame_idx": frame_idx,
            "time_s": t_sec,
            "tracked_x_px": x_track,
            "tracked_y_px": y_track,
            "radius_px": last_radius_px if last_radius_px is not None else "",
            "X_cam_m": X_cam if X_cam is not None else "",
            "Y_cam_m": Y_cam if Y_cam is not None else "",
            "Z_cam_m": Z_cam if Z_cam is not None else "",
            "distance_cam_m": dist_cam if dist_cam is not None else "",
            "has_measurement": int(has_measurement),
            "stab_x_ref_px": xs_stab if xs_stab is not None else "",
            "stab_y_ref_px": ys_stab if ys_stab is not None else "",
            "X_ground_m": Xg if Xg is not None else "",
            "Z_ground_m": Zg if Zg is not None else "",
        }
        csv_rows.append(row)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Interrupted by user.")
            break

        frame_idx += 1

    # Cleanup video
    cap.release()
    if out_video_rgb is not None:
        out_video_rgb.release()
    if out_video_topview is not None:
        out_video_topview.release()
    cv2.destroyAllWindows()

    # ------------------- Save CSV (Excel-friendly) -------------------
    fieldnames = [
        "frame_idx",
        "time_s",
        "tracked_x_px",
        "tracked_y_px",
        "radius_px",
        "X_cam_m",
        "Y_cam_m",
        "Z_cam_m",
        "distance_cam_m",
        "has_measurement",
        "stab_x_ref_px",
        "stab_y_ref_px",
        "X_ground_m",
        "Z_ground_m",
    ]

    with open(OUTPUT_CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"[INFO] Saved trajectory to: {OUTPUT_CSV_PATH}")
    print(f"[INFO] Output RGB video with trajectory saved to: {OUTPUT_VIDEO_PATH}")
    print(f"[INFO] Output top-view video (stabilized ground map) saved to: {OUTPUT_TOPVIEW_VIDEO_PATH}")


if __name__ == "__main__":
    main()
