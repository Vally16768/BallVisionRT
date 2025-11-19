import cv2
import math
import numpy as np


def detect_ball(frame, floor_mask=None):
    """Detect red ball using HSV + circularity filter."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Two ranges for red
    lower1 = np.array([0, 120, 70], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 120, 70], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # Optionally restrict to floor region
    if floor_mask is not None:
        mask = cv2.bitwise_and(mask, floor_mask)

    # Clean small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, None, None, None

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30:   # reject tiny blobs
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 3:  # reject very small circles
            continue

        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue

        circularity = 4.0 * math.pi * (area / (peri * peri))
        score = circularity * area  # favor large & round blobs

        if score > best_score:
            best_score = score
            best = (x, y, radius)

    if best is None:
        return False, None, None, None

    x, y, r = best
    return True, float(x), float(y), float(r)
