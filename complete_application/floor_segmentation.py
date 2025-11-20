"""
floor_segmentation.py

Floor segmentation and homography estimation utilities.

Responsibilities:
- Read the first frame from the RGB video.
- Segment the floor region using KMeans clustering and simple heuristics.
- Create an overlay for debugging the segmentation.
- Compute a homography from the original camera frame to a 2D top-view
  (bird's-eye) coordinate system.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

from config import TOP_W, TOP_H


def read_first_frame(video_path):
    """
    WHAT:
        Read the very first frame from the input video.

    WHY:
        The first frame is used as a static reference for:
        - floor segmentation (to get a stable ground plane mask),
        - defining the homography from camera view to top-view,
        - building a floor template for camera motion tracking.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("Failed to read the first frame from the video.")

    return frame


def segment_floor_kmeans(bgr_img, k=4):
    """
    WHAT:
        Segment the floor in the reference frame using KMeans in BGR color space
        combined with simple heuristics. Returns a binary floor mask.

    WHY:
        We need a floor mask to:
        - restrict ball detection only to the relevant ground plane region,
        - define a bounding box used for camera motion tracking,
        - later map both the ball and the camera onto a stable top-view plane.
    """
    h, w = bgr_img.shape[:2]

    # Downscale for faster clustering (KMeans on full HD can be slow).
    scale = 0.25
    small = cv2.resize(bgr_img, (int(w * scale), int(h * scale)))
    sh, sw = small.shape[:2]

    data = small.reshape(-1, 3).astype(np.float32)

    # KMeans clustering in BGR space
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_.reshape(sh, sw)

    # Convert to LAB to get luminance (L channel) for floor heuristics
    lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    L = lab_small[:, :, 0]

    best_score = -math.inf
    best_idx = 0

    # Heuristic: the floor is assumed to be:
    # - mostly in the lower part of the image (large mean y),
    # - taking up a significant area (large area fraction),
    # - not extremely bright (medium luminance).
    for i in range(k):
        ys, xs = np.where(labels == i)
        if ys.size == 0:
            continue

        frac_bottom = ys.mean() / sh           # 0 at top, 1 at bottom
        area_frac = ys.size / float(sh * sw)   # area fraction of the cluster
        mean_L = L[ys, xs].mean() / 255.0      # normalized luminance

        # Simple scoring function: prefer bottom, large, medium-bright region.
        score = frac_bottom + 0.5 * area_frac - 0.3 * mean_L

        if score > best_score:
            best_score = score
            best_idx = i

    # Binary mask for the selected cluster in the downscaled image
    floor_mask_small = (labels == best_idx).astype(np.uint8) * 255

    # Upsample mask to the original resolution
    floor_mask = cv2.resize(
        floor_mask_small, (w, h), interpolation=cv2.INTER_NEAREST
    )

    # Morphological operations to remove small holes and noise
    kernel = np.ones((25, 25), np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected component (assumed to be the full floor)
    contours, _ = cv2.findContours(
        floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Could not find any floor contour.")

    largest = max(contours, key=cv2.contourArea)
    mask_clean = np.zeros_like(floor_mask)
    cv2.drawContours(mask_clean, [largest], -1, 255, thickness=-1)

    return mask_clean


def create_overlay(frame, floor_mask):
    """
    WHAT:
        Create a green semi-transparent overlay on the detected floor region.

    WHY:
        This is purely for visual debugging:
        - quickly verify that the floor segmentation is reasonable,
        - spot obvious segmentation errors early.
    """
    overlay = frame.copy()
    green = np.zeros_like(frame)
    green[:, :, 1] = 255  # green channel

    alpha = 0.3
    # Broadcast the mask to three channels and normalize to [0,1]
    mask_3ch = cv2.merge([floor_mask, floor_mask, floor_mask]) / 255.0
    overlay = (frame * (1 - alpha * mask_3ch) + green * (alpha * mask_3ch)).astype(
        np.uint8
    )
    return overlay


def compute_topview_and_H(frame, floor_mask):
    """
    WHAT:
        Define a homography from the reference frame to a top-view map using
        four manually chosen points on the floor, and warp the floor mask to
        that top-view.

    WHY:
        We need a 2D "world" coordinate system (bird's-eye view) in which:
        - the floor is stable and camera motion is factored out,
        - we can draw both the ball trajectory and camera path consistently.
    """
    h, w = frame.shape[:2]

    # NOTE:
    # These 4 points should be adapted to your specific scene and calibration.
    # They should lie on the floor and ideally form a rectangle in the real world.
    src = np.float32([
        [100, 300],        # top-left corner on the floor (in image coordinates)
        [w - 100, 300],    # top-right floor point
        [w - 50, h - 50],  # bottom-right floor point
        [50, h - 50],      # bottom-left floor point
    ])

    top_w = TOP_W
    top_h = TOP_H
    dst = np.float32([
        [0, 0],
        [top_w - 1, 0],
        [top_w - 1, top_h - 1],
        [0, top_h - 1],
    ])

    # Homography from reference frame to top-view
    H = cv2.getPerspectiveTransform(src, dst)

    # Warp floor mask into top-view (top-down floor map)
    topview_mask = cv2.warpPerspective(
        floor_mask, H, (top_w, top_h), flags=cv2.INTER_NEAREST
    )

    return H, topview_mask
