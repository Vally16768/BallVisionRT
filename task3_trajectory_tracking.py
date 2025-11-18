import cv2
import csv
import math
import numpy as np
from collections import deque

# ============================================================
# CONFIG
# ============================================================

INPUT_VIDEO_PATH = "./rgb.avi"
OUTPUT_VIDEO_PATH = "output_3d_trajectory.avi"
OUTPUT_CSV_PATH = "ball_3d_trajectory.csv"

# Ball real-world diameter (meters) - standard football ~22 cm
BALL_DIAMETER_M = 0.22

# --- Camera intrinsics (you SHOULD tune these to your camera) ---
FX_NATIVE = 600.0
FY_NATIVE = 600.0
CX_NATIVE = None
CY_NATIVE = None

# Resize target width for processing (None = original)
TARGET_WIDTH = 960

# Trajectory visualization
MAX_TRAIL_LENGTH = 64  # number of points kept in the trail

# ============================================================
# FLOOR ROI
# ============================================================
USE_FLOOR_MASK = True

# Normalized polygon (x, y in [0,1]) that covers only the terrace (gray rectangle).
# Points are in order: bottom-left, bottom-right, top-right, top-left.
# Adjust the values until the ROI matches your terrace.
FLOOR_POLY_NORM = np.array([
    [0.1, 1.0],   # bottom-left
    [1.0, 1.0],   # bottom-right
    [1.0, 0.2],   # top-right
    [0.1, 0.2],   # top-left
], dtype=np.float32)


def create_floor_mask(frame_shape):
    """
    Create a binary mask (255 on floor, 0 elsewhere) based on FLOOR_POLY_NORM.
    """
    h, w = frame_shape[:2]
    pts = np.zeros_like(FLOOR_POLY_NORM, dtype=np.int32)
    pts[:, 0] = (FLOOR_POLY_NORM[:, 0] * w).astype(np.int32)
    pts[:, 1] = (FLOOR_POLY_NORM[:, 1] * h).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts  # also return polygon points for drawing if needed


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
# 2D DETECTION
# ============================================================

def get_red_mask(frame):
    """
    Create a binary mask for a red ball in HSV color space.
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
    - only in the lower part of the frame (ball on the floor/terrace)
    - filtering by area and circularity.
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

        # simple score: circularity * area (tunable)
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
# 3D ESTIMATION
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
      X, Y, Z, dist (meters)
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
# TRAJECTORY TRACKING (Kalman)
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


def draw_trajectory(frame, trail_points):
    """
    Draw the trajectory trail on top of the frame.

    trail_points: deque of (x, y) tuples (ints).
    """
    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue

        thickness = int(np.sqrt(MAX_TRAIL_LENGTH / float(i + 1)) * 2.0)
        cv2.line(
            frame,
            trail_points[i - 1],
            trail_points[i],
            (0, 0, 255),
            thickness
        )
    return frame


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    cap = create_video_capture(INPUT_VIDEO_PATH)

    out_video = None
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

    # Kalman filter and trajectory trail
    kalman = None
    trail = deque(maxlen=MAX_TRAIL_LENGTH)
    last_radius_px = None  # last known radius, used when detection is missing

    floor_mask = None
    floor_poly_pts = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break

        frame = preprocess_frame(frame, target_width=TARGET_WIDTH)
        h, w = frame.shape[:2]

        # Initialize video writer AFTER resize
        if out_video is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

        # Initialize intrinsics after we know the working resolution
        if fx_scaled is None:
            fx_scaled = FX_NATIVE
            fy_scaled = FY_NATIVE

            if CX_NATIVE is not None and CY_NATIVE is not None:
                cx_used = CX_NATIVE
                cy_used = CY_NATIVE
            else:
                cx_used = w / 2.0
                cy_used = h / 2.0

            print(f"[INFO] Using intrinsics: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, "
                  f"cx={cx_used:.2f}, cy={cy_used:.2f}")
            print(f"[INFO] FPS={fps:.2f}, dt={dt:.3f}s")

        # Floor mask (ROI) – compute once
        if USE_FLOOR_MASK and floor_mask is None:
            floor_mask, floor_poly_pts = create_floor_mask(frame.shape)

        # Initialize Kalman filter once (we need dt)
        if kalman is None:
            kalman = create_kalman_filter(dt)

        # ------------------- 2D Detection -------------------
        red_mask = get_red_mask(frame)

        # **** LIMITATION: search the ball only on the floor ****
        if USE_FLOOR_MASK:
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
            # No detection: rely on prediction
            x_track = pred_x
            y_track = pred_y

        # Visualization: tracked center + last valid radius
        if last_radius_px is None:
            vis_radius = 10
        else:
            vis_radius = last_radius_px

        tracked_circle = (int(round(x_track)), int(round(y_track)), int(round(vis_radius)))

        # ------------------- 3D Estimation -------------------
        if last_radius_px is not None:
            circle_for_3d = (x_track, y_track, last_radius_px)
            X, Y, Z, dist = compute_3d_position_from_circle(
                circle_for_3d,
                fx_scaled,
                fy_scaled,
                cx_used,
                cy_used,
                BALL_DIAMETER_M
            )
            overlay_text = f"Z={Z:.2f}m, |P|={dist:.2f}m"
        else:
            X = Y = Z = dist = None
            overlay_text = None

        # ------------------- Trajectory Trail -------------------
        if 0 <= x_track < w and 0 <= y_track < h:
            trail.append((int(round(x_track)), int(round(y_track))))
        else:
            trail.append(None)

        # ------------------- Visualization -------------------
        vis_frame = frame.copy()

        # Optional: draw ROI contour to clearly see the search area
        if USE_FLOOR_MASK and floor_poly_pts is not None:
            cv2.polylines(vis_frame, [floor_poly_pts], isClosed=True,
                          color=(255, 0, 0), thickness=2)

        # Draw ball (tracked) with text
        vis_frame = draw_ball_2d_overlay(vis_frame, tracked_circle, text=overlay_text)

        # Draw trajectory
        vis_frame = draw_trajectory(vis_frame, trail)

        # If we had an actual detection this frame, mark it in green
        if has_measurement:
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

        # ------------------- Output video -------------------
        out_video.write(vis_frame)
        cv2.imshow("3D Ball + Trajectory Tracking (floor limited)", vis_frame)

        # ------------------- CSV logging -------------------
        row = {
            "frame_idx": frame_idx,
            "time_s": t_sec,
            "tracked_x_px": x_track,
            "tracked_y_px": y_track,
            "radius_px": last_radius_px if last_radius_px is not None else "",
            "X_m": X if X is not None else "",
            "Y_m": Y if Y is not None else "",
            "Z_m": Z if Z is not None else "",
            "distance_m": dist if dist is not None else "",
            "has_measurement": int(has_measurement),
        }
        csv_rows.append(row)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Interrupted by user.")
            break

        frame_idx += 1

    # Cleanup video
    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()

    # ------------------- Save CSV (Excel-friendly) -------------------
    fieldnames = [
        "frame_idx",
        "time_s",
        "tracked_x_px",
        "tracked_y_px",
        "radius_px",
        "X_m",
        "Y_m",
        "Z_m",
        "distance_m",
        "has_measurement",
    ]

    with open(OUTPUT_CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"[INFO] Saved trajectory to: {OUTPUT_CSV_PATH}")
    print(f"[INFO] Output video with trajectory saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
