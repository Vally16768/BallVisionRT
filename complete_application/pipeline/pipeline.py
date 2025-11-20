"""
pipeline.py

Main orchestration logic for the DotLumen 2D/3D ball tracking and
top-view mapping pipeline.

It glues together:
- config.py
- floor_segmentation.py
- camera_motion.py
- ball_tracking.py
- and the pipeline submodules:
    * IOManager
    * CameraManager
    * BallManager
    * RawTopViewManager
    * TopViewStabilizer
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    VIDEO_PATH,
    OUTPUT_FIRST_FRAME,
    OUTPUT_FLOOR_MASK,
    OUTPUT_FLOOR_OVERLAY,
    OUTPUT_FLOOR_TOPVIEW,
    OUTPUT_TOPVIEW_IMAGE,
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
from ball_tracking import estimate_focal_length_pixels
from pipeline.io_manager import IOManager
from pipeline.camera_manager import CameraManager
from pipeline.ball_manager import BallManager
from pipeline.topview_raw import RawTopViewManager
from pipeline.topview_stabilizer import TopViewStabilizer
from pipeline.utils import fmt_float


class DotLumenPipeline:
    """
    Orchestrates the entire DotLumen processing pipeline:

    - Camera tracking and floor stabilization.
    - Ball detection and 3D estimation.
    - Real-time RAW top-view accumulation.
    - PCA-based trajectory straightening.
    - Stabilized top-view image and video generation.
    """

    def run(self) -> None:
        # --------------------------------------------------------------
        # 1) OPEN VIDEO AND READ METADATA
        # --------------------------------------------------------------
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # fallback

        # --------------------------------------------------------------
        # 2) REFERENCE FRAME + FLOOR SEGMENTATION
        # --------------------------------------------------------------
        frame0 = read_first_frame(VIDEO_PATH)
        h, w = frame0.shape[:2]
        print(f"[INFO] Video resolution: {w}x{h}, FPS = {fps:.2f}")

        cv2.imwrite(OUTPUT_FIRST_FRAME, frame0)
        print(f"[OK] First frame saved to: {OUTPUT_FIRST_FRAME}")

        floor_mask0 = segment_floor_kmeans(frame0)
        cv2.imwrite(OUTPUT_FLOOR_MASK, floor_mask0)
        print(f"[OK] Floor mask saved to: {OUTPUT_FLOOR_MASK}")

        overlay0 = create_overlay(frame0, floor_mask0)
        cv2.imwrite(OUTPUT_FLOOR_OVERLAY, overlay0)
        print(f"[OK] Floor overlay saved to: {OUTPUT_FLOOR_OVERLAY}")

        # Homography + warped floor in top-view
        H0_to_top, floor_topview0 = compute_topview_and_H(frame0, floor_mask0)
        cv2.imwrite(OUTPUT_FLOOR_TOPVIEW, floor_topview0)
        print(f"[OK] Top-view floor map saved to: {OUTPUT_FLOOR_TOPVIEW}")

        topview_base = cv2.cvtColor(floor_topview0, cv2.COLOR_GRAY2BGR)

        # --------------------------------------------------------------
        # 3) CAMERA AND BALL MANAGERS
        # --------------------------------------------------------------
        f_px = estimate_focal_length_pixels(w, D435I_RGB_HFOV_DEG)
        cx, cy = w / 2.0, h / 2.0
        print(f"[INFO] Estimated focal length: {f_px:.2f} px")

        cam_mgr = CameraManager(
            frame0=frame0,
            floor_mask0=floor_mask0,
            H0_to_top=H0_to_top,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
            alpha=0.9,
        )

        ball_mgr = BallManager(f_px=f_px, cx=cx, cy=cy)

        # --------------------------------------------------------------
        # 4) IO MANAGER & RAW TOPVIEW MANAGER
        # --------------------------------------------------------------
        io_mgr = IOManager(fps=fps, frame_size=(w, h))
        raw_topview_mgr = RawTopViewManager(topview_base=topview_base)

        # --------------------------------------------------------------
        # 5) MAIN LOOP STATE
        # --------------------------------------------------------------
        trajectory_points: List[Tuple[int, int]] = []
        frame_idx = 0

        # Runs for stabilization: each is a list of (frame_idx, tx, ty)
        topview_runs: List[List[Tuple[int, int, int]]] = []
        current_run: Optional[List[Tuple[int, int, int]]] = None
        current_run_idx: Optional[int] = None
        gap_frames = 0
        GAP_THRESHOLD = 15  # frames without ball to end a run

        # Meta for stabilized video
        cam_positions: List[Optional[Tuple[int, int]]] = []
        run_for_frame: List[Optional[int]] = []

        # Reset video to start processing from frame 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # --------------------------------------------------------------
        # 6) MAIN PROCESSING LOOP
        # --------------------------------------------------------------
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t_s = frame_idx / fps

            # ---- 6.1 Camera processing ----
            dx_eff, dy_eff, floor_mask_curr, cam_topview_pos = cam_mgr.process_frame(
                gray_frame=gray,
                frame_idx=frame_idx,
            )

            # Clamp camera pos to top-view bounds or mark None
            cam_pos_clamped: Optional[Tuple[int, int]] = None
            if cam_topview_pos is not None:
                cam_tx, cam_ty = cam_topview_pos
                if 0 <= cam_tx < TOP_W and 0 <= cam_ty < TOP_H:
                    cam_pos_clamped = (cam_tx, cam_ty)

            cam_positions.append(cam_pos_clamped)

            # ---- 6.2 Ball processing ----
            ball_info = ball_mgr.process_frame(
                frame=frame,
                floor_mask_curr=floor_mask_curr,
                t_s=t_s,
            )
            ball = ball_info["ball"]

            det_frame = frame.copy()
            traj_frame = frame.copy()
            topview_ball_point: Optional[Tuple[int, int]] = None
            run_idx_for_this_frame: Optional[int] = None

            # Draw ball on detection frame + trajectory in image-space
            if ball is not None:
                bx, by, br = ball
                cv2.circle(det_frame, (bx, by), br, (0, 255, 0), 2)
                cv2.circle(det_frame, (bx, by), 2, (0, 0, 255), -1)

                # 3D info overlay
                X_m = ball_info["X_m"]
                Y_m = ball_info["Y_m"]
                Z_m = ball_info["Z_m"]
                dist_m = ball_info["dist_m"]
                if dist_m is not None and Z_m is not None:
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

                # Image-space trajectory
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

                # ---- Map ball to top-view ----
                x_ref = int(round(bx + dx_eff))
                y_ref = int(round(by + dy_eff))

                if (0 <= x_ref < w and 0 <= y_ref < h and
                        floor_mask0[y_ref, x_ref] > 0):

                    p0 = np.array(
                        [x_ref, y_ref, 1.0],
                        dtype=np.float32
                    ).reshape(3, 1)
                    p_top = H0_to_top @ p0
                    if p_top[2, 0] != 0:
                        p_top /= p_top[2, 0]
                    tx = int(p_top[0, 0])
                    ty = int(p_top[1, 0])

                    if 0 <= tx < TOP_W and 0 <= ty < TOP_H:
                        topview_ball_point = (tx, ty)

                        # Run management for stabilization
                        if current_run is None or gap_frames > GAP_THRESHOLD:
                            current_run = []
                            current_run_idx = len(topview_runs)
                            gap_frames = 0

                        current_run.append((frame_idx, tx, ty))
                        run_idx_for_this_frame = current_run_idx
                        gap_frames = 0

            # If no ball was detected this frame, manage gaps and possibly close run
            if ball is None:
                gap_frames += 1
                if current_run is not None and gap_frames > GAP_THRESHOLD:
                    topview_runs.append(current_run)
                    current_run = None
                    current_run_idx = None

            run_for_frame.append(run_idx_for_this_frame)

            # ---- 6.3 RAW top-view frame (real-time style) ----
            raw_topview_frame = raw_topview_mgr.make_frame(
                ball_topview_point=topview_ball_point,
                cam_topview_point=cam_pos_clamped,
            )

            # ---- 6.4 CSV logging ----
            X_m = ball_info["X_m"]
            Y_m = ball_info["Y_m"]
            Z_m = ball_info["Z_m"]
            dist_m = ball_info["dist_m"]
            Vx = ball_info["Vx"]
            Vy = ball_info["Vy"]
            Vz = ball_info["Vz"]
            V = ball_info["V"]

            if ball is not None and X_m is not None and Y_m is not None and Z_m is not None:
                bx, by, br = ball
                io_mgr.write_csv_row(
                    frame_idx=frame_idx,
                    t_s_str=f"{t_s:.4f}",
                    u_px=str(bx),
                    v_px=str(by),
                    r_px=str(br),
                    X_m=fmt_float(X_m),
                    Y_m=fmt_float(Y_m),
                    Z_m=fmt_float(Z_m),
                    dist_m=fmt_float(dist_m),
                    Vx=fmt_float(Vx),
                    Vy=fmt_float(Vy),
                    Vz=fmt_float(Vz),
                    V=fmt_float(V),
                )
            else:
                io_mgr.write_csv_row(
                    frame_idx=frame_idx,
                    t_s_str=f"{t_s:.4f}",
                    u_px="",
                    v_px="",
                    r_px="",
                    X_m="",
                    Y_m="",
                    Z_m="",
                    dist_m="",
                    Vx="",
                    Vy="",
                    Vz="",
                    V="",
                )

            # ---- 6.5 Write videos ----
            io_mgr.write_detection_frame(det_frame)
            io_mgr.write_trajectory_frame(traj_frame)
            io_mgr.write_raw_topview_frame(raw_topview_frame)

            frame_idx += 1

        # Close last open run if needed
        if current_run is not None and len(current_run) > 0:
            topview_runs.append(current_run)

        total_frames = frame_idx
        print(f"[INFO] Processed frames: {total_frames}")
        print(f"[INFO] Number of ball runs detected: {len(topview_runs)}")

        # --------------------------------------------------------------
        # 7) STABILIZATION (PCA STRAIGHTENING) + FINAL IMAGE
        # --------------------------------------------------------------
        stabilizer = TopViewStabilizer(
            topview_base=topview_base,
            topview_runs=topview_runs,
            cam_positions=cam_positions,
            run_for_frame=run_for_frame,
        )
        stabilizer.stabilize()  # computes straightened runs and saves PNG

        # --------------------------------------------------------------
        # 8) STABILIZED TOP-VIEW VIDEO (MATCH FINAL IMAGE)
        # --------------------------------------------------------------
        io_mgr.open_stabilized_topview_writer()
        for frame_img in stabilizer.build_stabilized_video_frames(total_frames):
            io_mgr.write_stabilized_topview_frame(frame_img)

        print(f"[OK] Stabilized top-view video saved to: {OUTPUT_TOPVIEW_IMAGE}")

        # --------------------------------------------------------------
        # 9) CLEANUP
        # --------------------------------------------------------------
        cap.release()
        io_mgr.close()
        print("[DONE] DotLumen pipeline completed successfully.")
