"""
camera_motion.py

Simple camera motion tracking using template matching on the floor region.

Responsibilities:
- Extract a floor template from the reference frame using the floor mask.
- Track the template in each subsequent frame using normalized cross-correlation.
- From the best match position, estimate the 2D translation (dx, dy) of the camera.
- Warp the initial floor mask into the current frame using the estimated shift.
"""

import cv2
import numpy as np


def build_floor_template(frame0, floor_mask0):
    """
    WHAT:
        Build a floor template from the reference frame and floor mask.
        Returns:
            - floor_template: grayscale patch of the floor,
            - (x_min, y_min, x_max, y_max): bounding box of the floor mask.

    WHY:
        We use this template as a "fingerprint" of the floor region. By matching
        it in subsequent frames, we can estimate the camera's translation with
        respect to the reference frame.
    """
    ys, xs = np.where(floor_mask0 > 0)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError("Floor mask is empty, cannot build floor template.")

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    floor_template = gray0[y_min:y_max + 1, x_min:x_max + 1]
    return floor_template, (x_min, y_min, x_max, y_max)


def track_camera(gray_frame, floor_template, floor_bbox):
    """
    WHAT:
        Estimate camera translation (dx, dy) between the current frame and the
        reference frame using template matching.

    WHY:
        The floor is assumed to be fixed in world coordinates. If the camera
        moves, the best-matching location of the reference floor template in
        the current frame will shift. The shift (dx, dy) approximates the
        camera motion.

    Args:
        gray_frame (np.ndarray): Grayscale current frame.
        floor_template (np.ndarray): Grayscale floor patch from frame0.
        floor_bbox (tuple): (x_min, y_min, x_max, y_max) from frame0.

    Returns:
        (dx, dy, max_val, match_loc):
            dx, dy: estimated translation of camera (pixels, frame0 -> current)
            max_val: best match score (for debugging / confidence)
            match_loc: (x, y) location of best match in current frame.
    """
    x_min, y_min, _, _ = floor_bbox

    # Template matching using normalized cross-correlation
    res = cv2.matchTemplate(gray_frame, floor_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    match_x, match_y = max_loc

    # Note:
    # The floor template was originally at (x_min, y_min) in frame0.
    # If we find it now at (match_x, match_y), the camera must have moved
    # by approximately:
    #   dx = x_min - match_x
    #   dy = y_min - match_y
    dx = x_min - match_x
    dy = y_min - match_y

    return dx, dy, max_val, (match_x, match_y)


def warp_floor_mask(floor_mask0, dx, dy, width, height):
    """
    WHAT:
        Warp (translate) the reference floor mask into the current frame using
        the estimated camera translation (dx, dy).

    WHY:
        We want an approximate floor mask for the current frame in order to:
        - restrict ball detection only to the floor region,
        - stay robust to small camera motions.

    Args:
        floor_mask0 (np.ndarray): Floor mask from frame0.
        dx, dy (float): Estimated camera translation.
        width, height (int): Size of the current frame.

    Returns:
        floor_mask_curr (np.ndarray): Floor mask aligned to the current frame.
    """
    # We shift the floor mask in the opposite direction of camera movement.
    # If the camera moves right, the floor in the image appears to move left,
    # so we translate the mask by (-dx, -dy).
    M = np.float32([
        [1, 0, -dx],
        [0, 1, -dy],
    ])
    floor_mask_curr = cv2.warpAffine(
        floor_mask0,
        M,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )
    return floor_mask_curr
