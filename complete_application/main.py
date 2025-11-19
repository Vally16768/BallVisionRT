import csv
from collections import deque

import cv2
import numpy as np

from constants import (
    INPUT_VIDEO_PATH,
    OUTPUT_VIDEO_PATH,
    OUTPUT_TOPVIEW_VIDEO_PATH,
    OUTPUT_CSV_PATH,
    TARGET_WIDTH,
    MAX_TRAIL_LENGTH,
    USE_FLOOR_MASK,
    GROUND_WIDTH_M,
    GROUND_HEIGHT_M,
    TOPVIEW_WIDTH,
    TOPVIEW_HEIGHT,
)

from helpers_video import create_video_capture, preprocess_frame, create_video_writer
from helpers_floor import (
    create_floor_mask,
    estimate_floor_fullframe,
    init_ground_homography_from_floor_poly,
    world_to_topview,
)
from helpers_geometry import compute_intrinsics, estimate_distance_from_radius
from detection_ball import detect_ball
from compensator_camera import CameraMotionCompensator


def main():
    # ----------------------------------------------------------
    # Open input video and read reference frame
    # ----------------------------------------------------------
    cap = create_video_capture(INPUT_VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    dt = 1.0 / fps
    print(f"[INFO] FPS from video metadata = {fps:.2f}, dt={dt:.3f}s")

    ret, ref_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame from video")

    ref_frame = preprocess_frame(ref_frame, TARGET_WIDTH)
    h0, w0 = ref_frame.shape[:2]

    # ----------------------------------------------------------
    # 1) Global ball detection on the reference frame
    #    (no floor mask yet, search the whole image)
    # ----------------------------------------------------------
    found0, cx0, cy0, r0 = detect_ball(ref_frame, floor_mask=None)
    if found0:
        print(
            f"[INFO] Ball found in reference frame: "
            f"cx={cx0:.1f}, cy={cy0:.1f}, r={r0:.1f}"
        )
    else:
        print(
            "[WARN] Ball was NOT found in the reference frame. "
            "Trajectory will start only when the ball is first detected."
        )

    # ----------------------------------------------------------
    # 2) Fully automatic floor estimation (no manual poly, no ball dependency)
    # ----------------------------------------------------------
    floor_mask_ref, floor_poly_norm = estimate_floor_fullframe(ref_frame)
    if floor_poly_norm is None:
        raise RuntimeError("[FATAL] Automatic full-frame floor detection failed.")

    print("[INFO] Floor polygon automatically estimated from full frame.")

    # ----------------------------------------------------------
    # Intrinsics (approximate)
    # ----------------------------------------------------------
    fx, fy, cx, cy = compute_intrinsics(ref_frame.shape)
    print(
        f"[INFO] Using intrinsics (approx): "
        f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}"
    )

    # ----------------------------------------------------------
    # 3) Homography from image -> ground plane (2D floor map)
    #    Uses ONLY the automatically estimated floor polygon.
    # ----------------------------------------------------------
    H_img2ground, min_x, max_x, min_y, max_y = init_ground_homography_from_floor_poly(
        ref_frame.shape, floor_poly_norm, GROUND_WIDTH_M, GROUND_HEIGHT_M
    )
    print("[INFO] Ground homography (image -> ground) initialized from auto floor.")

    # ----------------------------------------------------------
    # 4) Camera motion compensator
    #    Uses ORB features mainly on the floor region to stabilize
    # ----------------------------------------------------------
    cam_motion = CameraMotionCompensator(
        ref_frame, floor_mask_ref if USE_FLOOR_MASK else None
    )

    # ----------------------------------------------------------
    # 5) Output writers and trajectory buffer
    # ----------------------------------------------------------
    rgb_writer = create_video_writer(OUTPUT_VIDEO_PATH, fps, (w0, h0))
    top_writer = create_video_writer(
        OUTPUT_TOPVIEW_VIDEO_PATH, fps, (TOPVIEW_WIDTH, TOPVIEW_HEIGHT)
    )

    trail = deque(maxlen=MAX_TRAIL_LENGTH)

    csv_file = open(OUTPUT_CSV_PATH, "w", newline="")
    logger = csv.writer(csv_file)
    logger.writerow(
        ["frame", "time", "X", "Y", "Z", "u_cur", "v_cur", "u_ref", "v_ref"]
    )

    frame_idx = 0

    # ----------------------------------------------------------
    # Main processing loop
    # ----------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame, TARGET_WIDTH)

        # Floor mask for the current frame (same normalized polygon, but in current size)
        floor_mask = create_floor_mask(frame.shape, floor_poly_norm)

        # 4.a) Update camera motion (current frame -> reference frame)
        motion_mask = floor_mask if USE_FLOOR_MASK else None
        cam_motion.update(frame, motion_mask)

        # 4.b) Detect the ball (restricted to floor region if enabled)
        detect_mask = floor_mask if USE_FLOOR_MASK else None
        found, cx_cur, cy_cur, r_cur = detect_ball(frame, detect_mask)

        X = Y = Z = None
        cx_ref = cy_ref = None

        if found:
            # Map ball center from current frame into reference frame
            cx_ref, cy_ref = cam_motion.warp_point(cx_cur, cy_cur)

            # Map from reference image coordinates to ground plane
            pt_ref = np.array([[[cx_ref, cy_ref]]], dtype=np.float32)
            pt_ground = cv2.perspectiveTransform(pt_ref, H_img2ground)
            X = float(pt_ground[0, 0, 0])
            Y = float(pt_ground[0, 0, 1])

            # Estimate ball distance from radius (simple pinhole model)
            Z = float(estimate_distance_from_radius(r_cur, fx))

            # Update ground-plane 2D trajectory (X,Y)
            trail.append((X, Y))

            # Draw ball on RGB frame
            cv2.circle(
                frame,
                (int(round(cx_cur)), int(round(cy_cur))),
                int(round(r_cur)),
                (0, 0, 255),
                2,
            )
            cv2.circle(
                frame,
                (int(round(cx_cur)), int(round(cy_cur))),
                3,
                (0, 255, 0),
                -1,
            )

        # ------------------------------------------------------
        # Build top-view "map" of the floor and trajectory (2D)
        # ------------------------------------------------------
        topview = np.zeros(
            (TOPVIEW_HEIGHT, TOPVIEW_WIDTH, 3), dtype=np.uint8
        )

        # Draw outer rectangle (ground plane limits)
        cv2.rectangle(
            topview,
            (0, 0),
            (TOPVIEW_WIDTH - 1, TOPVIEW_HEIGHT - 1),
            (80, 80, 80),
            1,
        )

        # Draw trajectory in top-view (in meters -> pixels)
        for i in range(1, len(trail)):
            X1, Y1 = trail[i - 1]
            X2, Y2 = trail[i]
            u1, v1 = world_to_topview(
                X1, Y1, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x, max_x, min_y, max_y
            )
            u2, v2 = world_to_topview(
                X2, Y2, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x, max_x, min_y, max_y
            )
            cv2.line(topview, (u1, v1), (u2, v2), (0, 255, 255), 2)

        # Draw current ball position in top-view
        if found and (X is not None) and (Y is not None):
            u, v = world_to_topview(
                X, Y, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x, max_x, min_y, max_y
            )
            cv2.circle(topview, (u, v), 6, (0, 0, 255), -1)

        # ------------------------------------------------------
        # Log to CSV (only if ball is visible)
        # ------------------------------------------------------
        if found:
            logger.writerow(
                [
                    frame_idx,
                    frame_idx * dt,
                    X,
                    Y,
                    Z,
                    cx_cur,
                    cy_cur,
                    cx_ref,
                    cy_ref,
                ]
            )

        # Write frames to disk
        rgb_writer.write(frame)
        top_writer.write(topview)

        # Show visualization
        stacked = np.hstack((frame, cv2.resize(topview, (w0, h0))))
        cv2.imshow("RGB (left) + TopView (right)", stacked)
        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord("q")]:
            break

        frame_idx += 1

    # ----------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------
    cap.release()
    rgb_writer.release()
    top_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
