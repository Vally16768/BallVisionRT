import cv2
import numpy as np

def create_floor_mask(frame_shape, poly_norm):
    if len(frame_shape) == 3:
        h, w = frame_shape[:2]
    else:
        h, w = frame_shape
    pts = poly_norm.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts_int = pts.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int], 255)
    return mask


def init_ground_homography_from_floor_poly(
    frame_shape, floor_poly_norm, ground_width_m, ground_height_m
):
    """
    Automatically map normalized floor polygon -> ground rectangle in meters.
    """
    if len(frame_shape) == 3:
        h, w = frame_shape[:2]
    else:
        h, w = frame_shape

    # Convert polygon from normalized -> pixel coordinates
    img_pts = floor_poly_norm.copy()
    img_pts[:, 0] *= w
    img_pts[:, 1] *= h
    img_pts = img_pts.astype(np.float32)

    # Target rectangle in real-world space
    ground_pts = np.array([
        [0.0, 0.0],
        [ground_width_m, 0.0],
        [ground_width_m, ground_height_m],
        [0.0, ground_height_m]
    ], dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, ground_pts, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("Could not compute ground homography from floor polygon")

    return H.astype(np.float32), 0.0, ground_width_m, 0.0, ground_height_m


def world_to_topview(X, Y, img_w, img_h, min_x, max_x, min_y, max_y, padding=20):
    Xc = max(min_x, min(X, max_x))
    Yc = max(min_y, min(Y, max_y))

    u = padding + (Xc - min_x) / (max_x - min_x) * (img_w - 2 * padding)
    v = img_h - padding - (Yc - min_y) / (max_y - min_y) * (img_h - 2 * padding)

    return int(round(u)), int(round(v))
