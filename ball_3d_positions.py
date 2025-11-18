import cv2
import csv
import math
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT_VIDEO_PATH = "./rgb.avi"
OUTPUT_VIDEO_PATH = "output_3d_detection.avi"
OUTPUT_CSV_PATH = "ball_3d_positions.csv"

# Ball real-world diameter (meters) - standard football ~22 cm
BALL_DIAMETER_M = 0.22

# --- Camera intrinsics (you SHOULD tune these to your camera) ---
# For a RealSense D435i @ 1280x720, fx, fy are typically around 600–620 px.
# If you don't know them:
#   1) Use RealSense SDK to query intrinsics, OR
#   2) Do a quick one-point calibration with the ball at known distance.
#
# In this script:
#  - FX and FY will be scaled if we resize the frame.
#  - CX and CY are set to the image center by default (good approximation).
FX_NATIVE = 600.0
FY_NATIVE = 600.0

# If you know principal point, put it here for the NATIVE resolution.
# If not, we will override with center-of-frame after we know frame size.
CX_NATIVE = None
CY_NATIVE = None

# Resize target width for processing (None = original)
TARGET_WIDTH = 960


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
# 2D DETECTION (same spirit as Task 1)
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
    min_area=150,        # ignore tiny blobs
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
    Draw the detected ball and optionally some text (e.g. distance).
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
        - Z is depth along camera's forward axis
        - X is horizontal offset
        - Y is vertical offset
        - dist is Euclidean distance sqrt(X^2 + Y^2 + Z^2)
    """
    x_px, y_px, r_px = circle
    d_px = 2.0 * r_px  # apparent diameter in pixels

    if d_px <= 0:
        return None

    # Pinhole model: Z = (f * D) / d
    # Here, use fx for f (or average of fx, fy)
    f = (fx + fy) * 0.5
    Z = (f * ball_diameter_m) / d_px  # meters

    # Back-project to 3D camera coordinates
    X = (x_px - cx) * Z / fx
    Y = (y_px - cy) * Z / fy

    dist = math.sqrt(X * X + Y * Y + Z * Z)

    return X, Y, Z, dist


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    cap = create_video_capture(INPUT_VIDEO_PATH)

    out_video = None
    csv_rows = []

    frame_idx = 0

    # We will scale fx, fy and set cx, cy after we know the frame size
    fx_scaled = None
    fy_scaled = None
    cx_used = None
    cy_used = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

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
            # We assume FX_NATIVE, FY_NATIVE are for this resolution.
            # If they are for another resolution W_native, you should scale:
            # fx_scaled = FX_NATIVE * (w / W_native), etc.
            fx_scaled = FX_NATIVE
            fy_scaled = FY_NATIVE

            if CX_NATIVE is not None and CY_NATIVE is not None:
                # Use supplied principal point (scaled if needed)
                cx_used = CX_NATIVE
                cy_used = CY_NATIVE
            else:
                # Fallback: assume principal point at image center
                cx_used = w / 2.0
                cy_used = h / 2.0

            print(f"[INFO] Using intrinsics: fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, "
                  f"cx={cx_used:.2f}, cy={cy_used:.2f}")

        # ------------------- 2D Detection -------------------
        red_mask = get_red_mask(frame)
        circle = find_red_ball(red_mask, frame.shape)

        # ------------------- 3D Estimation -------------------
        t_sec = frame_idx / fps

        if circle is not None:
            X, Y, Z, dist = compute_3d_position_from_circle(
                circle,
                fx_scaled,
                fy_scaled,
                cx_used,
                cy_used,
                BALL_DIAMETER_M
            )

            # For overlay: show depth (forward distance) and optionally Euclidean distance
            overlay_text = f"Z={Z:.2f}m, |P|={dist:.2f}m"
            vis_frame = draw_ball_2d_overlay(frame.copy(), circle, text=overlay_text)

            x_px, y_px, r_px = circle
            csv_rows.append({
                "frame_idx": frame_idx,
                "time_s": t_sec,
                "x_px": x_px,
                "y_px": y_px,
                "radius_px": r_px,
                "X_m": X,
                "Y_m": Y,
                "Z_m": Z,
                "distance_m": dist
            })

        else:
            # No ball detected in this frame
            vis_frame = frame.copy()
            cv2.putText(
                vis_frame,
                "Ball not detected",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            # Optionally log a row with NaNs for missing data
            csv_rows.append({
                "frame_idx": frame_idx,
                "time_s": t_sec,
                "x_px": "",
                "y_px": "",
                "radius_px": "",
                "X_m": "",
                "Y_m": "",
                "Z_m": "",
                "distance_m": ""
            })

        # ------------------- Output video -------------------
        out_video.write(vis_frame)
        cv2.imshow("3D Ball Detection (Position & Distance)", vis_frame)

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
        "x_px",
        "y_px",
        "radius_px",
        "X_m",
        "Y_m",
        "Z_m",
        "distance_m"
    ]

    with open(OUTPUT_CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"[INFO] Saved 3D positions to: {OUTPUT_CSV_PATH}")
    print(f"[INFO] Output video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
