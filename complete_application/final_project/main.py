"""
main.py

Top-level pipeline orchestrating the DotLumen 2D/3D ball tracking + top-view map:

Steps:
  1) Read the first frame and segment the floor.
  2) Compute a homography from the camera frame to a top-view plane.
  3) Build a floor template for camera motion tracking.
  4) For each video frame:
      - Track camera motion via template matching on the floor.
      - Warp the floor mask into the current frame.
      - Detect the red ball constrained to the floor.
      - Estimate its 3D position (X, Y, Z) and distance.
      - Compute 3D velocities by finite differences over time.
      - Map both ball and camera into the top-view "world" coordinates.
      - Save debug videos and a CSV with positions and velocities.

Outputs:
  - Ball detection video (per-frame detections).
  - Ball trajectory video (2D trajectory in image space).
  - Top-view video (bird's-eye world map, ball path + camera path).
  - CSV file with 3D positions and velocities.
"""

import csv
import math

import cv2
import numpy as np

from config import (
    VIDEO_PATH,
    OUTPUT_FIRST_FRAME,
    OUTPUT_FLOOR_MASK,
    OUTPUT_FLOOR_OVERLAY,
    OUTPUT_FLOOR_TOPVIEW,
    OUTPUT_DET_VIDEO,
    OUTPUT_TRAJ_VIDEO,
    OUTPUT_TOPVIEW_VIDEO,
    OUTPUT_TOPVIEW_IMAGE,
    OUTPUT_CSV,
    D435I_RGB_HFOV_DEG,
    TOP_W,
    TOP_H,
)

from floor_segmentation import (
    read_first_frame,
    segment_floor_kmeans,
    create_overlay,
    compute_topview_and_H,
)

from ball_tracking import (
    estimate_focal_length_pixels,
    detect_red_ball_on_floor,
    estimate_ball_3d,
)

from camera_motion import (
    build_floor_template,
    track_camera,
    warp_floor_mask,
)


def main():
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback if FPS is missing in metadata.
        fps = 30.0

    # ----------------------------------------------------------------------
    # 1) REFERENCE FRAME (frame0) for floor, homography, and camera tracking
    # ----------------------------------------------------------------------
    frame0 = read_first_frame(VIDEO_PATH)
    h, w = frame0.shape[:2]
    print(f"[INFO] Video resolution: {w}x{h}, FPS = {fps:.2f}")

    cv2.imwrite(OUTPUT_FIRST_FRAME, frame0)
    print(f"[OK] First frame saved to: {OUTPUT_FIRST_FRAME}")

    # Floor segmentation in frame0
    floor_mask0 = segment_floor_kmeans(frame0)
    cv2.imwrite(OUTPUT_FLOOR_MASK, floor_mask0)
    print(f"[OK] Floor mask saved to: {OUTPUT_FLOOR_MASK}")

    # Overlay for visual debug
    overlay0 = create_overlay(frame0, floor_mask0)
    cv2.imwrite(OUTPUT_FLOOR_OVERLAY, overlay0)
    print(f"[OK] Floor overlay saved to: {OUTPUT_FLOOR_OVERLAY}")

    # Homography frame0 -> top-view + top-view floor map
    H0_to_top, floor_topview0 = compute_topview_and_H(frame0, floor_mask0)
    cv2.imwrite(OUTPUT_FLOOR_TOPVIEW, floor_topview0)
    print(f"[OK] Top-view floor map saved to: {OUTPUT_FLOOR_TOPVIEW}")

    # Build floor template for camera tracking
    floor_template, floor_bbox = build_floor_template(frame0, floor_mask0)
    th, tw = floor_template.shape[:2]
    print(f"[INFO] Floor template size: {tw}x{th} px")

    # Top-view: start from the floor as static background (world map).
    topview_accum = cv2.cvtColor(floor_topview0, cv2.COLOR_GRAY2BGR)

    # Estimate focal length and optical center (for ball 3D estimation)
    f_px = estimate_focal_length_pixels(w, D435I_RGB_HFOV_DEG)
    cx, cy = w / 2.0, h / 2.0
    print(f"[INFO] Estimated focal length: {f_px:.2f} px")

    # ----------------------------------------------------------------------
    # Video writers
    # ----------------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    det_writer = cv2.VideoWriter(OUTPUT_DET_VIDEO, fourcc, fps, (w, h), True)
    traj_writer = cv2.VideoWriter(OUTPUT_TRAJ_VIDEO, fourcc, fps, (w, h), True)
    topview_writer = cv2.VideoWriter(
        OUTPUT_TOPVIEW_VIDEO, fourcc, fps, (TOP_W, TOP_H), True
    )
    print("[OK] Video writers initialized.")

    # ----------------------------------------------------------------------
    # CSV writer for 3D positions and velocities
    # ----------------------------------------------------------------------
    csv_file = open(OUTPUT_CSV, mode="w", newline="")
    csv_writer = csv.writer(csv_file)

    # Column description:
    # - frame_idx: frame index (integer).
    # - time_s: absolute timestamp of the frame in seconds.
    # - u_px, v_px, r_px: 2D image position and radius of the detected ball.
    # - X_m, Y_m, Z_m: estimated 3D coordinates in camera space (meters).
    # - distance_m: Euclidean distance from the camera to the ball (meters).
    # - Vx_m_s, Vy_m_s, Vz_m_s: 3D velocity components (meters/second).
    # - V_m_s: magnitude of the velocity (meters/second).
    csv_writer.writerow([
        "frame_idx", "time_s",
        "u_px", "v_px", "r_px",
        "X_m", "Y_m", "Z_m", "distance_m",
        "Vx_m_s", "Vy_m_s", "Vz_m_s", "V_m_s"
    ])

    # ----------------------------------------------------------------------
    # Tracking state
    # ----------------------------------------------------------------------
    trajectory_points = []  # ball trajectory in image space for visualization
    prev_ball = None        # previous (x, y, r) for temporal ball tracking
    prev_3d = None          # previous (X, Y, Z) for velocity estimation
    prev_t = None           # previous time in seconds

    frame_idx = 0

    # Reset video to the beginning (we have already read frame0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ----------------------------------------------------------------------
    # MAIN PROCESSING LOOP
    # ----------------------------------------------------------------------
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_s = frame_idx / fps  # absolute time in seconds for this frame

        # --------------------------------------------------------------
        # 2) CAMERA TRACKING: compute camera translation using floor template
        # --------------------------------------------------------------
        dx, dy, match_score, match_loc = track_camera(
            gray, floor_template, floor_bbox
        )

        # --------------------------------------------------------------
        # 3) CURRENT FLOOR MASK: warp floor_mask0 using (dx, dy)
        # --------------------------------------------------------------
        floor_mask_curr = warp_floor_mask(floor_mask0, dx, dy, w, h)

        # Prepare frames for drawing:
        det_frame = frame.copy()
        traj_frame = frame.copy()
        topview_frame = topview_accum.copy()

        # --------------------------------------------------------------
        # 4) RED BALL DETECTION CONSTRAINED TO FLOOR
        # --------------------------------------------------------------
        prev_center = prev_ball[:2] if prev_ball is not None else None
        ball = detect_red_ball_on_floor(
            frame,
            floor_mask_curr,
            prev_center=prev_center
        )

        # Variables for 3D and velocities
        X_m = Y_m = Z_m = dist_m = None
        Vx = Vy = Vz = V = None
        est_3d = None

        if ball is not None:
            bx, by, br = ball
            prev_ball = ball

            # Draw detection on RGB frame
            cv2.circle(det_frame, (bx, by), br, (0, 255, 0), 2)
            cv2.circle(det_frame, (bx, by), 2, (0, 0, 255), -1)

            # ----------------------------------------------------------
            # 5) 3D ESTIMATION (X, Y, Z, distance)
            # ----------------------------------------------------------
            est_3d = estimate_ball_3d(bx, by, br, f_px, cx, cy)
            if est_3d is not None:
                X_m, Y_m, Z_m, dist_m = est_3d
                txt = f"Dist = {dist_m:.2f} m, Z = {Z_m:.2f} m"
                cv2.putText(
                    det_frame,
                    txt,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # ------------------------------------------------------
                # 6) 3D VELOCITIES (finite differences over frames)
                # ------------------------------------------------------
                if prev_3d is not None and prev_t is not None:
                    dT = t_s - prev_t
                    if dT > 0:
                        X_prev, Y_prev, Z_prev = prev_3d
                        Vx = (X_m - X_prev) / dT
                        Vy = (Y_m - Y_prev) / dT
                        Vz = (Z_m - Z_prev) / dT
                        V = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)

                prev_3d = (X_m, Y_m, Z_m)
                prev_t = t_s

            # ----------------------------------------------------------
            # Image-space trajectory for visualization
            # ----------------------------------------------------------
            trajectory_points.append((bx, by))
            if len(trajectory_points) >= 2:
                for i in range(1, len(trajectory_points)):
                    cv2.line(
                        traj_frame,
                        trajectory_points[i - 1],
                        trajectory_points[i],
                        (0, 0, 255),
                        2
                    )

            # ----------------------------------------------------------
            # 7) MAP BALL INTO FRAME0 COORDINATES AND THEN TOP-VIEW
            # ----------------------------------------------------------
            # The camera moved by (dx, dy), so to map the current ball center
            # back to frame0 coordinates we shift by +dx, +dy.
            x_ref = int(round(bx + dx))
            y_ref = int(round(by + dy))

            if (0 <= x_ref < w and 0 <= y_ref < h and
                    floor_mask0[y_ref, x_ref] > 0):
                p0 = np.array([x_ref, y_ref, 1.0]).reshape(3, 1)
                p_top = H0_to_top @ p0
                if p_top[2, 0] != 0:
                    p_top /= p_top[2, 0]
                tx, ty = int(p_top[0, 0]), int(p_top[1, 0])

                if 0 <= tx < TOP_W and 0 <= ty < TOP_H:
                    # Draw the ball trajectory in the 2D world map (top-view).
                    cv2.circle(topview_accum, (tx, ty), 3, (0, 0, 255), -1)
                    cv2.circle(topview_frame, (tx, ty), 3, (0, 0, 255), -1)

        # --------------------------------------------------------------
        # 8) CAMERA POSITION IN TOP-VIEW
        # --------------------------------------------------------------
        # We consider the image center (cx, cy) plus the shift (dx, dy) as the
        # camera position relative to frame0, then project it to the top-view.
        cam_ref = np.array([cx + dx, cy + dy, 1.0]).reshape(3, 1)
        p_cam_top = H0_to_top @ cam_ref
        if p_cam_top[2, 0] != 0:
            p_cam_top /= p_cam_top[2, 0]
        cam_tx, cam_ty = int(p_cam_top[0, 0]), int(p_cam_top[1, 0])

        if 0 <= cam_tx < TOP_W and 0 <= cam_ty < TOP_H:
            cv2.circle(topview_frame, (cam_tx, cam_ty), 5, (255, 0, 0), 2)

        # --------------------------------------------------------------
        # 9) CSV: 3D POSITION + 3D VELOCITY VS TIME
        # --------------------------------------------------------------
        def fmt(x):
            """Format floats as strings, leave empty if None."""
            return "" if x is None else f"{x:.6f}"

        if ball is not None and est_3d is not None:
            bx, by, br = ball
            csv_writer.writerow([
                frame_idx, f"{t_s:.4f}",
                bx, by, br,
                fmt(X_m), fmt(Y_m), fmt(Z_m), fmt(dist_m),
                fmt(Vx), fmt(Vy), fmt(Vz), fmt(V),
            ])
        else:
            # No valid ball detection or 3D estimate for this frame
            csv_writer.writerow([
                frame_idx, f"{t_s:.4f}",
                "", "", "",
                "", "", "", "",    # X, Y, Z, distance
                "", "", "", ""     # Vx, Vy, Vz, V
            ])

        # --------------------------------------------------------------
        # 10) WRITE OUTPUT FRAMES TO VIDEOS
        # --------------------------------------------------------------
        det_writer.write(det_frame)
        traj_writer.write(traj_frame)
        topview_writer.write(topview_frame)

        frame_idx += 1

    # ----------------------------------------------------------------------
    # CLEANUP
    # ----------------------------------------------------------------------
    cap.release()
    det_writer.release()
    traj_writer.release()
    topview_writer.release()
    csv_file.close()

    cv2.imwrite(OUTPUT_TOPVIEW_IMAGE, topview_accum)
    print(f"[OK] Final top-view image saved to: {OUTPUT_TOPVIEW_IMAGE}")
    print(f"[OK] Detection video saved to: {OUTPUT_DET_VIDEO}")
    print(f"[OK] Trajectory video saved to: {OUTPUT_TRAJ_VIDEO}")
    print(f"[OK] Top-view video saved to: {OUTPUT_TOPVIEW_VIDEO}")
    print(f"[OK] CSV with 3D positions and velocities saved to: {OUTPUT_CSV}")
    print("[DONE] DotLumen pipeline completed successfully.")


if __name__ == "__main__":
    main()
