import cv2
import csv
import numpy as np
from collections import deque

from constants import *
from helpers_video import *
from helpers_floor import *
from helpers_geometry import *
from detection_ball import detect_ball
from compensator_camera import CameraMotionCompensator


def main():

    cap = create_video_capture(INPUT_VIDEO_PATH)

    # FPS
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / fps
    print(f"[INFO] FPS={fps:.2f}")

    # First frame = reference
    ret, ref_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read first frame")

    ref_frame = preprocess_frame(ref_frame, TARGET_WIDTH)
    h0, w0 = ref_frame.shape[:2]

    # Floor mask in reference
    floor_mask_ref = create_floor_mask(ref_frame.shape, FLOOR_POLY_NORM)

    # Intrinsics
    fx, fy, cx, cy = compute_intrinsics(ref_frame.shape)

    # Ground homography from normalized floor polygon
    H_img2ground, min_x, max_x, min_y, max_y = init_ground_homography_from_floor_poly(
        ref_frame.shape, FLOOR_POLY_NORM, GROUND_WIDTH_M, GROUND_HEIGHT_M
    )

    # Camera motion compensation
    cam_motion = CameraMotionCompensator(ref_frame, floor_mask_ref)

    # Video writers
    rgb_writer = create_video_writer(OUTPUT_VIDEO_PATH, fps, (w0, h0))
    top_writer = create_video_writer(OUTPUT_TOPVIEW_VIDEO_PATH, fps, (TOPVIEW_WIDTH, TOPVIEW_HEIGHT))

    # Trajectory
    trail = deque(maxlen=MAX_TRAIL_LENGTH)

    # CSV
    csv_file = open(OUTPUT_CSV_PATH, "w", newline="")
    logger = csv.writer(csv_file)
    logger.writerow([
        "frame","time","X","Y","Z","u_cur","v_cur","u_ref","v_ref"
    ])

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = preprocess_frame(frame, TARGET_WIDTH)
        floor_mask = create_floor_mask(frame.shape, FLOOR_POLY_NORM)

        # Update camera motion
        H = cam_motion.update(frame, floor_mask)

        # Detect ball
        found, cx, cy, r = detect_ball(frame, floor_mask)

        X = Y = Z = None
        cx_ref = cy_ref = None

        if found:
            # Map ball position to reference frame
            cx_ref, cy_ref = cam_motion.warp_point(cx, cy)

            # Map to ground plane
            pt = np.array([[[cx_ref, cy_ref]]], dtype=np.float32)
            pt_g = cv2.perspectiveTransform(pt, H_img2ground)
            X, Y = pt_g[0,0]

            # Depth
            Z = estimate_distance_from_radius(r, fx)

            # Add to trajectory
            trail.append((X, Y))

            # Draw in RGB
            cv2.circle(frame, (int(cx), int(cy)), int(r), (0,0,255), 2)
            cv2.circle(frame, (int(cx), int(cy)), 3, (0,255,0), -1)

        # Build top-view
        topview = np.zeros((TOPVIEW_HEIGHT, TOPVIEW_WIDTH, 3), dtype=np.uint8)

        # Draw rect boundary
        cv2.rectangle(topview, (0,0),(TOPVIEW_WIDTH-1, TOPVIEW_HEIGHT-1),(80,80,80),1)

        # Draw trajectory
        for i in range(1, len(trail)):
            X1,Y1 = trail[i-1]
            X2,Y2 = trail[i]
            u1,v1 = world_to_topview(X1,Y1, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x,max_x,min_y,max_y)
            u2,v2 = world_to_topview(X2,Y2, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x,max_x,min_y,max_y)
            cv2.line(topview,(u1,v1),(u2,v2),(0,255,255),2)

        # Draw ball point
        if found and X is not None and Y is not None:
            u,v = world_to_topview(X,Y, TOPVIEW_WIDTH, TOPVIEW_HEIGHT, min_x,max_x,min_y,max_y)
            cv2.circle(topview,(u,v),6,(0,0,255),-1)

        # Log CSV
        if found:
            logger.writerow([
                frame_idx, frame_idx*dt,
                X, Y, Z,
                cx, cy,
                cx_ref, cy_ref
            ])

        rgb_writer.write(frame)
        top_writer.write(topview)

        cv2.imshow("RGB (left) + TopView (right)", np.hstack((frame, cv2.resize(topview,(w0,h0)))))
        if cv2.waitKey(1)&0xFF in [27, ord('q')]:
            break

        frame_idx += 1

    cap.release()
    rgb_writer.release()
    top_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
