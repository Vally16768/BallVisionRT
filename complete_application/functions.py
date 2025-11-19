import cv2
import math
import numpy as np

import constants as cfg


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

        circularity = 4.0 * math.pi * (area / (perimeter * perimeter))
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


# ============================================================
# 3D ESTIMATION (camera coordinates)
# ============================================================

def compute_3d_position_from_circle(
    circle,
    fx,
    fy,
    cx,
    cy,
    ball_diameter_m=cfg.BALL_DIAMETER_M
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
        - Y is vertical offset
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
        new_pts = good_next.reshape(-1, 1, 2).astype(np.float32) if len(good_next) > 0 else None
        return gray, new_pts, H_global

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
    dst = np.array([
        [0.0,                0.0],                    # bottom-left
        [cfg.GROUND_WIDTH_M, 0.0],                    # bottom-right
        [cfg.GROUND_WIDTH_M, cfg.GROUND_LENGTH_M],    # top-right (far)
        [0.0,                cfg.GROUND_LENGTH_M],    # top-left  (far)
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
