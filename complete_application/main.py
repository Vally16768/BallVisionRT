import csv
from collections import deque

import cv2
import numpy as np

import constants as cfg
from helpers import (
    create_floor_mask_and_polygon,
    create_video_capture,
    preprocess_frame,
    draw_ball_2d_overlay,
    draw_trajectory,
    create_empty_topview_frame,
    world_to_topview,
)
from functions import (
    get_red_mask,
    find_red_ball,
    compute_3d_position_from_circle,
    create_kalman_filter,
    detect_features,
    update_camera_motion,
    stabilize_point,
    compute_ground_homography,
    image_to_ground,
)


def main():
    cap = create_video_capture(cfg.INPUT_VIDEO_PATH)

    out_video_rgb = None
    out_video_topview = None
    csv_rows = []

    frame_idx = 0

    # Intrinsics for working resolution
    fx_scaled = None
    fy_scaled = None
    cx_used = None
    cy_used = None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    dt = 1.0 / fps

    # Kalman filter and trajectory trail in image space
    kalman = None
    trail_image = deque(maxlen=cfg.MAX_TRAIL_LENGTH)
    last_radius_px = None  # last known radius, used when detection is missing

    # Trajectory trail in top-view (ground plane, camera-motion compensated)
    trail_topview = deque(maxlen=cfg.MAX_TRAIL_LENGTH)

    floor_mask = None
    floor_poly_pts = None
    H_img2ground = None

    # Camera motion estimation (reference frame 0 -> current frame)
    prev_gray = None
    prev_pts = None
    H_global = np.eye(3, dtype=np.float64)

    # For logging stabilized image coordinates (optional)
    xs_stab = ys_stab = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot fetch frame.")
            break

        frame = preprocess_frame(frame, target_width=cfg.TARGET_WIDTH)
        h, w = frame.shape[:2]

        # Initialize video writers AFTER resize (use actual processing resolution)
        if out_video_rgb is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video_rgb = cv2.VideoWriter(cfg.OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))

        if out_video_topview is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_video_topview = cv2.VideoWriter(
                cfg.OUTPUT_TOPVIEW_VIDEO_PATH, fourcc, fps,
                (cfg.TOPVIEW_WIDTH, cfg.TOPVIEW_HEIGHT)
            )

        # Initialize intrinsics once we know working resolution
        if fx_scaled is None:
            # Use provided FX_NATIVE/FY_NATIVE as approximate focal lengths in pixels.
            # If the video was not recorded at the same resolution, scale them accordingly.
            if cfg.TARGET_WIDTH is not None:
                scale_factor = w / float(cfg.TARGET_WIDTH)
            else:
                scale_factor = 1.0

            fx_scaled = cfg.FX_NATIVE * scale_factor
            fy_scaled = cfg.FY_NATIVE * scale_factor

            if cfg.CX_NATIVE is not None and cfg.CY_NATIVE is not None:
                cx_used = cfg.CX_NATIVE * scale_factor
                cy_used = cfg.CY_NATIVE * scale_factor
            else:
                # Fallback: principal point at image center (from video metadata).
                cx_used = w / 2.0
                cy_used = h / 2.0

            print(f"[INFO] Using intrinsics (approx): fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, "
                  f"cx={cx_used:.2f}, cy={cy_used:.2f}")
            print(f"[INFO] FPS from video metadata = {fps:.2f}, dt={dt:.3f}s")

        # Floor mask (ROI) – compute once, in reference frame coordinates
        if cfg.USE_FLOOR_MASK and floor_mask is None:
            floor_mask, floor_poly_pts = create_floor_mask_and_polygon(frame.shape)
            # Build homography from image (reference frame) to metric ground plane
            H_img2ground = compute_ground_homography(floor_poly_pts)
            if H_img2ground is None:
                print("[WARN] Could not compute ground homography. Top-view will be disabled.")
            else:
                print("[INFO] Ground homography (image -> ground) initialized.")

        # Initialize Kalman filter once (we need dt)
        if kalman is None:
            kalman = create_kalman_filter(dt)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------- Camera Motion Estimation -------------------
        if frame_idx == 0:
            # First frame: detect background features, restricted to floor to
            # stay on the dominant plane (ground).
            feat_mask = floor_mask if cfg.USE_FLOOR_MASK else None
            prev_gray = gray.copy()
            prev_pts = detect_features(prev_gray, mask=feat_mask)
        else:
            prev_gray, prev_pts, H_global = update_camera_motion(prev_gray, prev_pts, gray, H_global)
            # If we ran out of points, re-detect on the current frame.
            if prev_pts is None or len(prev_pts) < 50:
                feat_mask = floor_mask if cfg.USE_FLOOR_MASK else None
                prev_pts = detect_features(gray, mask=feat_mask)
                prev_gray = gray.copy()

        # ------------------- 2D Detection -------------------
        red_mask = get_red_mask(frame)

        # Limit search to floor if enabled
        if cfg.USE_FLOOR_MASK and floor_mask is not None:
            red_mask = cv2.bitwise_and(red_mask, floor_mask)

        detected_circle = find_red_ball(red_mask, frame.shape)

        t_sec = frame_idx / fps

        # ------------------- Kalman Tracking -------------------
        # Predict step
        predicted_state = kalman.predict()
        pred_x, pred_y = float(predicted_state[0]), float(predicted_state[1])

        has_measurement = detected_circle is not None

        if has_measurement:
            meas_x, meas_y, meas_r = detected_circle
            measurement = np.array([[np.float32(meas_x)],
                                    [np.float32(meas_y)]], dtype=np.float32)
            corrected_state = kalman.correct(measurement)
            x_track = float(corrected_state[0])
            y_track = float(corrected_state[1])
            last_radius_px = meas_r
        else:
            # No detection: rely purely on prediction
            x_track = pred_x
            y_track = pred_y

        # Visualization: tracked center + last valid radius
        if last_radius_px is None:
            vis_radius = 10
        else:
            vis_radius = last_radius_px

        tracked_circle = (int(round(x_track)), int(round(y_track)), int(round(vis_radius)))

        # ------------------- 3D Estimation (for CSV and overlay only) -------------------
        if last_radius_px is not None:
            circle_for_3d = (x_track, y_track, last_radius_px)
            result_3d = compute_3d_position_from_circle(
                circle_for_3d,
                fx_scaled,
                fy_scaled,
                cx_used,
                cy_used,
                cfg.BALL_DIAMETER_M
            )
            if result_3d is not None:
                X_cam, Y_cam, Z_cam, dist_cam = result_3d
                overlay_text = f"Z={Z_cam:.2f}m, |P|={dist_cam:.2f}m"
            else:
                X_cam = Y_cam = Z_cam = dist_cam = None
                overlay_text = None
        else:
            X_cam = Y_cam = Z_cam = dist_cam = None
            overlay_text = None

        # ------------------- Trajectory trail in image space -------------------
        if 0 <= x_track < w and 0 <= y_track < h:
            trail_image.append((int(round(x_track)), int(round(y_track))))
        else:
            trail_image.append(None)

        # ------------------- Camera-motion compensation & Ground Projection -------------------
        current_topview_point = None
        xs_stab = ys_stab = None
        Xg = Zg = None

        if H_global is not None and H_img2ground is not None:
            # Stabilize tracked point to reference frame (remove camera motion)
            stab_result = stabilize_point(x_track, y_track, H_global)
            if stab_result is not None:
                xs_stab, ys_stab = stab_result

                # Project stabilized point to ground plane (metric coordinates)
                ground_result = image_to_ground(xs_stab, ys_stab, H_img2ground)
                if ground_result is not None:
                    Xg, Zg = ground_result

                    # Map ground coordinates to top-view pixels
                    u, v = world_to_topview(Xg, Zg)
                    if 0 <= u < cfg.TOPVIEW_WIDTH and 0 <= v < cfg.TOPVIEW_HEIGHT:
                        current_topview_point = (u, v)
                        trail_topview.append(current_topview_point)
                    else:
                        trail_topview.append(None)
                else:
                    trail_topview.append(None)
            else:
                trail_topview.append(None)
        else:
            trail_topview.append(None)

        # ------------------- Visualization: RGB + Trajectory -------------------
        vis_frame = frame.copy()

        # ROI contour (terrace)
        if cfg.USE_FLOOR_MASK and floor_poly_pts is not None:
            cv2.polylines(vis_frame, [floor_poly_pts], isClosed=True,
                          color=(255, 0, 0), thickness=2)

        # Draw ball (tracked) with text
        vis_frame = draw_ball_2d_overlay(vis_frame, tracked_circle, text=overlay_text)

        # Draw image-space trajectory (2D in pixel coords)
        vis_frame = draw_trajectory(
            vis_frame,
            trail_image,
            color=(0, 0, 255),
            max_trail_length=cfg.MAX_TRAIL_LENGTH
        )

        # If we had an actual detection this frame, mark measurement in green
        if has_measurement and detected_circle is not None:
            mx, my, _ = detected_circle
            cv2.circle(vis_frame, (mx, my), 4, (0, 255, 0), -1)

        status_txt = "Tracking: M+P" if has_measurement else "Tracking: PRED ONLY"
        cv2.putText(
            vis_frame,
            status_txt,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ------------------- Visualization: Top-View (Ground, Camera-Motion Compensated) -------------------
        topview_frame = create_empty_topview_frame()

        # Draw ground-plane trajectory (stabilized)
        topview_frame = draw_trajectory(
            topview_frame,
            trail_topview,
            color=(0, 0, 255),
            max_trail_length=cfg.MAX_TRAIL_LENGTH
        )

        # Draw current ball position on top-view
        if current_topview_point is not None:
            cv2.circle(topview_frame, current_topview_point, 6, (0, 0, 255), -1)
            cv2.putText(
                topview_frame,
                "Ball",
                (current_topview_point[0] + 8, current_topview_point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        # ------------------- Output videos -------------------
        out_video_rgb.write(vis_frame)
        out_video_topview.write(topview_frame)

        cv2.imshow("3D Ball + Trajectory Tracking (RGB)", vis_frame)
        cv2.imshow("2D Top-View Ground Map (Stabilized)", topview_frame)

        # ------------------- CSV logging -------------------
        row = {
            "frame_idx": frame_idx,
            "time_s": t_sec,
            "tracked_x_px": x_track,
            "tracked_y_px": y_track,
            "radius_px": last_radius_px if last_radius_px is not None else "",
            "X_cam_m": X_cam if X_cam is not None else "",
            "Y_cam_m": Y_cam if Y_cam is not None else "",
            "Z_cam_m": Z_cam if Z_cam is not None else "",
            "distance_cam_m": dist_cam if dist_cam is not None else "",
            "has_measurement": int(has_measurement),
            "stab_x_ref_px": xs_stab if xs_stab is not None else "",
            "stab_y_ref_px": ys_stab if ys_stab is not None else "",
            "X_ground_m": Xg if Xg is not None else "",
            "Z_ground_m": Zg if Zg is not None else "",
        }
        csv_rows.append(row)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("Interrupted by user.")
            break

        frame_idx += 1

    # Cleanup video
    cap.release()
    if out_video_rgb is not None:
        out_video_rgb.release()
    if out_video_topview is not None:
        out_video_topview.release()
    cv2.destroyAllWindows()

    # ------------------- Save CSV (Excel-friendly) -------------------
    fieldnames = [
        "frame_idx",
        "time_s",
        "tracked_x_px",
        "tracked_y_px",
        "radius_px",
        "X_cam_m",
        "Y_cam_m",
        "Z_cam_m",
        "distance_cam_m",
        "has_measurement",
        "stab_x_ref_px",
        "stab_y_ref_px",
        "X_ground_m",
        "Z_ground_m",
    ]

    with open(cfg.OUTPUT_CSV_PATH, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    print(f"[INFO] Saved trajectory to: {cfg.OUTPUT_CSV_PATH}")
    print(f"[INFO] Output RGB video with trajectory saved to: {cfg.OUTPUT_VIDEO_PATH}")
    print(f"[INFO] Output top-view video (stabilized ground map) saved to: {cfg.OUTPUT_TOPVIEW_VIDEO_PATH}")


if __name__ == "__main__":
    main()
