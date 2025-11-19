import cv2
import numpy as np

import constants as cfg


# ============================================================
# FLOOR ROI
# ============================================================

def create_floor_mask_and_polygon(frame_shape, floor_poly_norm=None):
    """
    Create a binary mask (255 on floor, 0 elsewhere) based on FLOOR_POLY_NORM.
    Also return the polygon points in pixel coordinates (Nx1x2 int32).
    """
    if floor_poly_norm is None:
        floor_poly_norm = cfg.FLOOR_POLY_NORM

    h, w = frame_shape[:2]
    pts = np.zeros_like(floor_poly_norm, dtype=np.int32)
    pts[:, 0] = (floor_poly_norm[:, 0] * w).astype(np.int32)
    pts[:, 1] = (floor_poly_norm[:, 1] * h).astype(np.int32)
    pts = pts.reshape((-1, 1, 2))

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, pts  # polygon points for drawing & homography


# ============================================================
# Video I/O
# ============================================================

def create_video_capture(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {path}")
    return cap


def preprocess_frame(frame, target_width=None):
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
# Drawing helpers
# ============================================================

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


def draw_trajectory(frame, trail_points, color=(0, 0, 255), max_trail_length=None):
    """
    Draw the trajectory trail on top of the frame.

    trail_points: deque of (x, y) tuples (ints or None).
    """
    if max_trail_length is None:
        max_trail_length = cfg.MAX_TRAIL_LENGTH

    for i in range(1, len(trail_points)):
        if trail_points[i - 1] is None or trail_points[i] is None:
            continue

        thickness = int(np.sqrt(max_trail_length / float(i + 1)) * 2.0)
        cv2.line(
            frame,
            trail_points[i - 1],
            trail_points[i],
            color,
            thickness
        )
    return frame


# ============================================================
# TOP-VIEW HELPERS
# ============================================================

def create_empty_topview_frame():
    """
    Create a blank top-view frame with:
      - camera marker
      - ground rectangle (terrace)
      - axes consistent with:
          X axis = left/right on terrace
          Z axis = distance away from camera along terrace
    """
    frame = np.full(
        (cfg.TOPVIEW_HEIGHT, cfg.TOPVIEW_WIDTH, 3),
        cfg.TOPVIEW_BG_COLOR,
        dtype=np.uint8
    )

    # Draw ground rectangle (terrace) in top-view
    gx0 = cfg.TOPVIEW_MARGIN_X
    gx1 = cfg.TOPVIEW_WIDTH - cfg.TOPVIEW_MARGIN_X
    gz1 = cfg.TOPVIEW_MARGIN_Y                 # far end of terrace
    gz0 = cfg.TOPVIEW_HEIGHT - cfg.TOPVIEW_MARGIN_Y  # near end (where camera is)

    pts = np.array([
        [gx0, gz0],
        [gx1, gz0],
        [gx1, gz1],
        [gx0, gz1]
    ], dtype=np.int32)

    cv2.polylines(frame, [pts], isClosed=True, color=(80, 80, 80), thickness=2)

    # Camera marker at the center of the near edge
    cx, cy = cfg.CAMERA_PX
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
    gx0 = cfg.TOPVIEW_MARGIN_X
    gx1 = cfg.TOPVIEW_WIDTH - cfg.TOPVIEW_MARGIN_X
    gz1 = cfg.TOPVIEW_MARGIN_Y                 # far
    gz0 = cfg.TOPVIEW_HEIGHT - cfg.TOPVIEW_MARGIN_Y  # near (camera side)

    # clamp to valid ground range
    Xg_clamped = max(0.0, min(cfg.GROUND_WIDTH_M, Xg))
    Zg_clamped = max(0.0, min(cfg.GROUND_LENGTH_M, Zg))

    # normalized coords in [0,1]
    nx = Xg_clamped / cfg.GROUND_WIDTH_M
    nz = Zg_clamped / cfg.GROUND_LENGTH_M

    # map to pixel coords (note Z increases away from camera -> upward in image)
    u = int(round(gx0 + nx * (gx1 - gx0)))
    v = int(round(gz0 - nz * (gz0 - gz1)))
    return u, v
