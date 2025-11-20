"""
ball_tracking.py

Utilities related to ball detection and 3D position / velocity estimation.

Responsibilities:
- Estimate camera focal length in pixels from HFOV and image width.
- Detect a red ball constrained to the floor region.
- Estimate the ball's 3D position in camera coordinates and its speed.
"""

import cv2
import numpy as np
import math

from config import BALL_RADIUS_M


def estimate_focal_length_pixels(width, hfov_deg):
    """
    WHAT:
        Estimate the camera focal length (in pixels) from the image width
        and horizontal field-of-view (HFOV) in degrees.

    WHY:
        For a simple pinhole camera model, we need the focal length in pixels
        to convert a measured object size in pixels into distance in meters.

    Model:
        f = (width / 2) / tan(HFOV / 2)
    """
    hfov_rad = math.radians(hfov_deg)
    f = (width / 2.0) / math.tan(hfov_rad / 2.0)
    return f


def detect_red_ball_on_floor(frame, floor_mask, prev_center=None, max_shift=150):
    """
    WHAT:
        Detect a red ball in HSV color space, restricted ONLY to the floor region
        defined by the given mask. Returns (x, y, r) in pixels or None.

    WHY:
        - We use color-based detection for a red ball.
        - We restrict detection to the floor to reduce false positives (walls,
          clothes, etc.).
        - We use contours and minimum enclosing circles for robust localization
          and radius estimation.

    Args:
        frame (np.ndarray): BGR frame from the video.
        floor_mask (np.ndarray): Binary floor mask in the same resolution.
        prev_center (tuple or None): Previous (x, y, r) center to enforce
            temporal consistency. Only the (x, y) are used.
        max_shift (int): Maximum allowed shift between consecutive frames to
            consider a detection as the same ball (in pixels).

    Returns:
        (x, y, r) or None
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red in HSV is split into two ranges due to the circular Hue.
    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask1, mask2)

    # Restrict red detection to the floor mask
    mask_red = cv2.bitwise_and(mask_red, mask_red, mask=floor_mask)

    # Morphological operations to clean up small noise blobs
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    candidate_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            # Ignore very small blobs.
            continue

        (x, y), r = cv2.minEnclosingCircle(cnt)
        x = int(x)
        y = int(y)
        r = int(r)

        if r < 5:
            # Ignore extremely small circles.
            continue

        candidate_circles.append((x, y, r, area))

    if not candidate_circles:
        return None

    # If we do not have a previous center, pick the largest red region.
    if prev_center is None:
        candidate_circles.sort(key=lambda c: -c[3])  # sort by area descending
        x, y, r, _ = candidate_circles[0]
        return (x, y, r)

    # If we have a previous center, choose the closest candidate to it,
    # as long as the displacement is not too large.
    px, py = prev_center
    best = None
    best_dist = float("inf")

    for x, y, r, area in candidate_circles:
        d = math.hypot(x - px, y - py)
        if d < best_dist and d < max_shift:
            best_dist = d
            best = (x, y, r)

    # If no candidate is close enough, fall back to the largest area.
    if best is None:
        candidate_circles.sort(key=lambda c: -c[3])
        x, y, r, _ = candidate_circles[0]
        return (x, y, r)

    return best


def estimate_ball_3d(u, v, r_px, f_px, cx, cy):
    """
    WHAT:
        Estimate the 3D position (X, Y, Z) and distance of the ball in the
        camera coordinate system, using a simple pinhole model.

    WHY:
        The radius of the ball in pixels is inversely proportional to the
        distance from the camera. With a known real-world ball radius and
        focal length (in pixels), we can estimate the Z coordinate.

    Model:
        Given:
            - R: real ball radius in meters,
            - r_px: ball radius in pixels,
            - f_px: focal length in pixels,
            - (u, v): pixel coordinates (ball center),
            - (cx, cy): principal point (image center).

        Then:
            Z = f_px * R / r_px
            X = (u - cx) * Z / f_px
            Y = (v - cy) * Z / f_px

        distance = sqrt(X^2 + Y^2 + Z^2)

    Returns:
        (X, Y, Z, distance) in meters, or None if r_px <= 0.
    """
    if r_px <= 0:
        return None

    Z = f_px * BALL_RADIUS_M / float(r_px)
    X = (u - cx) * Z / f_px
    Y = (v - cy) * Z / f_px
    dist = math.sqrt(X * X + Y * Y + Z * Z)
    return X, Y, Z, dist
