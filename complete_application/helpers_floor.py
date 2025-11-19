import cv2
import numpy as np


def create_floor_mask(frame_shape, poly_norm):
    """Create a binary floor mask from a normalized polygon in [0,1].

    The polygon is given in normalized coordinates (x / width, y / height).
    This function scales it to the current frame size and fills it.
    """
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


# -------------------------------------------------------------------------
# Automatic floor estimation starting from a detected ball (heat propagation)
# -------------------------------------------------------------------------

def _order_quad_points(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype=np.float32)
    if pts.shape[0] != 4:
        raise ValueError("_order_quad_points expects exactly 4 points")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl], axis=0)


def estimate_floor_from_ball(
    frame,
    ball_cx,
    ball_cy,
    ball_r,
    color_thresh=12.0,
    min_area_ratio=0.02,
    morph_kernel_size=7,
):
    """Estimate floor mask + polygon using color similarity around the ball.

    This implements the "heat propagation" idea:
    - we sample a ring/band of pixels just below the ball (expected floor)
    - compute the average color in LAB space
    - mark as floor all pixels with color close to this average
    - keep the largest connected component and approximate it with a quad

    Parameters
    ----------
    frame : np.ndarray (H,W,3), BGR
    ball_cx, ball_cy : float
        Ball center in pixels (on the reference frame).
    ball_r : float
        Ball radius in pixels.
    color_thresh : float
        Maximum Euclidean distance in LAB space to accept a pixel as floor.
    min_area_ratio : float
        Minimum area of the floor region (fraction of image area) to accept.
    morph_kernel_size : int
        Kernel size for morphological open/close operations.

    Returns
    -------
    floor_mask : np.ndarray uint8, shape (H,W)
        Binary mask with 255 on floor, 0 elsewhere.
    floor_poly_norm : np.ndarray float32, shape (4,2) or None
        Normalized polygon (x/w, y/h) with points ordered TL,TR,BR,BL.
        If something fails, floor_poly_norm is None.
    """
    h, w = frame.shape[:2]
    cx = int(round(ball_cx))
    cy = int(round(ball_cy))
    r = int(max(ball_r, 5))

    # Define a vertical band just below the ball where we are confident it's floor
    y1 = min(h - 1, cy + int(1.2 * r))
    y2 = min(h - 1, cy + int(2.5 * r))
    x1 = max(0, cx - int(1.5 * r))
    x2 = min(w - 1, cx + int(1.5 * r))

    if y1 >= y2 or x1 >= x2:
        # Degenerate ROI
        return None, None

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    roi_lab = lab[y1 : y2 + 1, x1 : x2 + 1]

    if roi_lab.size == 0:
        return None, None

    mean_color = roi_lab.reshape(-1, 3).mean(axis=0)
    L0, a0, b0 = mean_color.astype(np.float32)

    L = lab[:, :, 0].astype(np.float32)
    A = lab[:, :, 1].astype(np.float32)
    B = lab[:, :, 2].astype(np.float32)

    dist = np.sqrt((L - L0) ** 2 + (A - a0) ** 2 + (B - b0) ** 2)
    floor_mask = np.zeros((h, w), dtype=np.uint8)
    floor_mask[dist < color_thresh] = 255

    # Morphological clean-up
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask)
    if num_labels <= 1:
        return None, None

    # Skip background (index 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    floor_mask_clean = np.zeros_like(floor_mask)
    floor_mask_clean[labels == largest_label] = 255

    # Check area size
    area = areas.max()
    if area < min_area_ratio * (h * w):
        # Too small to be a meaningful floor
        return floor_mask_clean, None

    # Approximate with a quadrilateral using minimum-area rectangle
    contours, _ = cv2.findContours(
        floor_mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return floor_mask_clean, None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)  # 4 points
    box = _order_quad_points(box)

    # Normalize to [0,1]
    poly_norm = box.copy()
    poly_norm[:, 0] /= float(w)
    poly_norm[:, 1] /= float(h)

    return floor_mask_clean, poly_norm.astype(np.float32)


# -------------------------------------------------------------------------
# Ground homography and top-view utilities
# -------------------------------------------------------------------------

def init_ground_homography_from_floor_poly(
    frame_shape, floor_poly_norm, ground_width_m, ground_height_m
):
    """Compute homography from image floor quadrilateral to ground rectangle.

    The image polygon is provided in normalized coordinates (x/w, y/h) and is
    assumed to be ordered as TL, TR, BR, BL. The ground rectangle is
    [0,0] - [W,0] - [W,H] - [0,H] in meters.
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

    # Target rectangle in real-world space (meters)
    ground_pts = np.array(
        [
            [0.0, 0.0],
            [ground_width_m, 0.0],
            [ground_width_m, ground_height_m],
            [0.0, ground_height_m],
        ],
        dtype=np.float32,
    )

    H, _ = cv2.findHomography(img_pts, ground_pts, cv2.RANSAC, 2.0)
    if H is None:
        raise RuntimeError("Could not compute ground homography from floor polygon")

    # We return the homography + ground extents (in meters)
    return H.astype(np.float32), 0.0, ground_width_m, 0.0, ground_height_m


def world_to_topview(X, Y, img_w, img_h, min_x, max_x, min_y, max_y, padding=20):
    """Project world (X,Y) coordinates into a top-view image.

    Parameters
    ----------
    X, Y : float
        World coordinates in meters.
    img_w, img_h : int
        Top-view image size.
    min_x, max_x, min_y, max_y : float
        Extents of the ground plane (in meters).
    padding : int
        Border padding in pixels.

    Returns
    -------
    u, v : int
        Pixel coordinates in the top-view image.
    """
    Xc = max(min_x, min(X, max_x))
    Yc = max(min_y, min(Y, max_y))

    u = padding + (Xc - min_x) / (max_x - min_x) * (img_w - 2 * padding)
    v = img_h - padding - (Yc - min_y) / (max_y - min_y) * (img_h - 2 * padding)

    return int(round(u)), int(round(v))

def estimate_floor_fullframe(frame, color_thresh=10.0, morph_kernel_size=9, min_area_ratio=0.05):
    """
    Fully automatic floor estimation using LAB histogram peak detection.
    No sklearn, no ball required.

    Steps:
    1. Convert to LAB.
    2. Blur to reduce noise.
    3. Detect dominant LAB color by histogram peak.
    4. Segment all pixels close to this peak.
    5. Keep largest connected component.
    6. Fit quadrilateral via minAreaRect.
    """

    import cv2
    import numpy as np

    h, w = frame.shape[:2]

    # Convert to LAB and blur
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab_blur = cv2.GaussianBlur(lab, (9, 9), 0)

    # Histogram peaks for L, A, B
    hist_L = cv2.calcHist([lab_blur], [0], None, [256], [0, 256]).flatten()
    hist_A = cv2.calcHist([lab_blur], [1], None, [256], [0, 256]).flatten()
    hist_B = cv2.calcHist([lab_blur], [2], None, [256], [0, 256]).flatten()

    L0 = np.argmax(hist_L)
    A0 = np.argmax(hist_A)
    B0 = np.argmax(hist_B)

    # Segment floor by color distance to dominant LAB peak
    LL = lab_blur[:, :, 0].astype(np.float32)
    AA = lab_blur[:, :, 1].astype(np.float32)
    BB = lab_blur[:, :, 2].astype(np.float32)

    dist = np.sqrt((LL - L0) ** 2 + (AA - A0) ** 2 + (BB - B0) ** 2)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[dist < color_thresh] = 255

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Largest connected component
    num_labels, lbls, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None, None

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + np.argmax(areas)

    if areas.max() < min_area_ratio * (h * w):
        return None, None

    floor_mask = np.zeros_like(mask)
    floor_mask[lbls == largest_idx] = 255

    # Contour & minAreaRect
    contours, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return floor_mask, None

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)

    # Order TL, TR, BR, BL
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).reshape(-1)

    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]

    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Normalize in [0,1]
    quad_norm = quad.copy()
    quad_norm[:, 0] /= w
    quad_norm[:, 1] /= h

    return floor_mask, quad_norm
