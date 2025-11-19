import csv
import math
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
    last_radius_px = None  # last known radius, used for drawing
    last_center = None     # last measured center (x, y)

    # Trajectory trail in top-view (ground plane, camera-motion compensated)
    trail_topview = deque(maxlen=cfg.MAX_TRAIL_LENGTH)

    # Ground-plane positions for static/moving classification
    ground_history = deque()  # stores (t_sec, Xg, Zg)
    missing_ground_frames = 0
    ball_is_static = False

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
                cx_used = w / 2.0
                cy_used = h / 2.0

            print(f"[INFO] Using intrinsics (approx): fx={fx_scaled:.2f}, fy={fy_scaled:.2f}, "
                  f"cx={cx_used:.2f}, cy={cy_used:.2f}")
            print(f"[INFO] FPS from video metadata = {fps:.2f}, dt={dt:.3f}s")

        # Floor mask (ROI) – compute once, in reference frame coordinates
        if cfg.USE_FLOOR_MASK and floor_mask is None:
            floor_mask, floor_poly_pts = create_floor_mask_and_polygon(frame.shape)
            H_img2ground = compute_ground_homography(floor_poly_pts)
            if H_img2ground is None:
                print("[WARN] Could not compute ground homography. Top-view will be disabled.")
            else:
                print("[INFO] Ground homography (image -> ground) initialized.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ------------------- Camera Motion Estimation -------------------
        if frame_idx == 0:
            feat_mask = floor_mask if cfg.USE_FLOOR_MASK else None
            prev_gray = gray.copy()
            prev_pts = detect_features(prev_gray, mask=feat_mask)
        else:
            prev_gray, prev_pts, H_global = update_camera_motion(prev_gray, prev_pts, gray, H_global)
            if prev_pts is None or len(prev_pts) < 50:
                feat_mask = floor_mask if cfg.USE_FLOOR_MASK else None
                prev_pts = detect_features(gray, mask=feat_mask)
                prev_gray = gray.copy()

        # ------------------- 2D Detection -------------------
        red_mask = get_red_mask(frame)
        if cfg.USE_FLOOR_MASK and floor_mask is not None:
            red_mask = cv2.bitwise_and(red_mask, floor_mask)

        detected_circle = find_red_ball(red_mask, frame.shape)
        t_sec = frame_idx / fps
        has_measurement = detected_circle is not None

        # ------------------- Kalman Tracking (ONLY with measurement) -------------------
        if has_measurement:
            meas_x, meas_y, meas_r = detected_circle

            if kalman is None:
                kalman = create_kalman_filter(dt)
                # initialize state with first measurement
                kalman.statePost = np.array([[meas_x],
                                             [meas_y],
                                             [0.0],
                                             [0.0]], dtype=np.float32)
                x_track = float(meas_x)
                y_track = float(meas_y)
            else:
                kalman.predict()
                measurement = np.array([[np.float32(meas_x)],
                                        [np.float32(meas_y)]], dtype=np.float32)
                corrected_state = kalman.correct(measurement)
                x_track = float(corrected_state[0, 0])
                y_track = float(corrected_state[1, 0])

            last_radius_px = meas_r
            last_center = (x_track, y_track)
        else:
            # No detection: do NOT run Kalman; just keep last center (or None)
            if last_center is not None:
                x_track, y_track = last_center
            else:
                x_track, y_track = None, None

        # Visualization: tracked center + last valid radius
        if last_radius_px is None or x_track is None or y_track is None:
            tracked_circle = None
        else:
            tracked_circle = (
                int(round(x_track)),
                int(round(y_track)),
                int(round(last_radius_px))
            )

        # ------------------- 3D Estimation (for CSV and overlay only) -------------------
        if tracked_circle is not None:
            circle_for_3d = (tracked_circle[0], tracked_circle[1], last_radius_px)
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
        if tracked_circle is not None:
            tx, ty, _ = tracked_circle
            if 0 <= tx < w and 0 <= ty < h:
                trail_image.append((tx, ty))
            else:
                trail_image.append(None)
        else:
            trail_image.append(None)

        # ------------------- Camera-motion compensation & Ground Projection -------------------
        current_topview_point = None
        xs_stab = ys_stab = None
        Xg = Zg = None

        if tracked_circle is not None and H_global is not None and H_img2ground is not None:
            tx, ty, _ = tracked_circle

            stab_result = stabilize_point(tx, ty, H_global)
            if stab_result is not None:
                xs_stab, ys_stab = stab_result

                ground_result = image_to_ground(xs_stab, ys_stab, H_img2ground)
                if ground_result is not None:
                    Xg, Zg = ground_result

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

        # ------------------- Static / Dynamic classification in ground plane -------------------
        if Xg is not None and Zg is not None and has_measurement:
            ground_history.append((t_sec, Xg, Zg))
            missing_ground_frames = 0

            # drop old points outside the time window
            while ground_history and (t_sec - ground_history[0][0]) > cfg.STATIC_WINDOW_SEC:
                ground_history.popleft()
        else:
            missing_ground_frames += 1
            if missing_ground_frames > int(fps * cfg.STATIC_WINDOW_SEC):
                ground_history.clear()

        if len(ground_history) >= 2:
            t0, X0, Z0 = ground_history[0]
            tn, Xn, Zn = ground_history[-1]
            dt_hist = max(tn - t0, 1e-6)
            dist_hist = math.hypot(Xn - X0, Zn - Z0)
            speed_hist = dist_hist / dt_hist
            ball_is_static = (
                dist_hist < cfg.STATIC_DIST_THRESHOLD_M
                and speed_hist < cfg.STATIC_SPEED_THRESHOLD_MS
            )
        else:
            ball_is_static = False

        # ------------------- Visualization: RGB + Trajectory -------------------
        vis_frame = frame.copy()

        if cfg.USE_FLOOR_MASK and floor_poly_pts is not None:
            cv2.polylines(vis_frame, [floor_poly_pts], isClosed=True,
                          color=(255, 0, 0), thickness=2)

        vis_frame = draw_ball_2d_overlay(vis_frame, tracked_circle, text=overlay_text)

        vis_frame = draw_trajectory(
            vis_frame,
            trail_image,
            color=(0, 0, 255),
            max_trail_length=cfg.MAX_TRAIL_LENGTH
        )

        # If we had a measurement this frame, mark the measurement center in green
        if has_measurement and detected_circle is not None:
            mx, my, _ = detected_circle
            cv2.circle(vis_frame, (mx, my), 4, (0, 255, 0), -1)

        # ------------------- Visualization: Top-View -------------------
        topview_frame = create_empty_topview_frame()

        topview_frame = draw_trajectory(
            topview_frame,
            trail_topview,
            color=(0, 0, 255),
            max_trail_length=cfg.MAX_TRAIL_LENGTH
        )

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
            "tracked_x_px": tracked_circle[0] if tracked_circle is not None else "",
            "tracked_y_px": tracked_circle[1] if tracked_circle is not None else "",
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
            "ball_static": int(ball_is_static),
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

    # ------------------- Save CSV -------------------
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
        "ball_static",
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
