"""
io_manager.py

IOManager: handles creation and management of video writers and the CSV file.

Responsibilities:
- Open video writers for:
    * detection video (RGB frames with ball overlay),
    * trajectory video (2D trajectory in image space),
    * RAW top-view video (real-time accumulated top-view),
    * stabilized top-view video (PCA-straightened trajectories).
- Open the CSV file and write per-frame rows with 3D position and velocity.
- Provide a clean shutdown for all open IO resources.
"""

import csv
import os
from typing import Optional, Tuple

import cv2

from config import (
    OUTPUT_DIR,
    OUTPUT_DET_VIDEO,
    OUTPUT_TRAJ_VIDEO,
    OUTPUT_TOPVIEW_VIDEO,
    OUTPUT_CSV,
    TOP_W,
    TOP_H,
)


class IOManager:
    """
    Manages all videos and the CSV file used in the pipeline.

    Typical usage:
        io = IOManager(fps, frame_size=(w, h))
        io.write_detection_frame(det_frame)
        io.write_trajectory_frame(traj_frame)
        io.write_raw_topview_frame(raw_topview_frame)
        io.write_csv_row(...)
        ...
        io.open_stabilized_topview_writer()
        io.write_stabilized_topview_frame(stab_frame)
        ...
        io.close()
    """

    def __init__(self, fps: float, frame_size: Tuple[int, int]) -> None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.fps = fps
        self.frame_size = frame_size  # (width, height)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        # Detection video (RGB frames with detection overlay)
        self.det_writer = cv2.VideoWriter(
            OUTPUT_DET_VIDEO, fourcc, fps, frame_size, True
        )

        # Trajectory video (RGB frames with 2D trajectory in image space)
        self.traj_writer = cv2.VideoWriter(
            OUTPUT_TRAJ_VIDEO, fourcc, fps, frame_size, True
        )

        # RAW top-view video: we derive a path by suffixing '_raw' before '.avi'
        if OUTPUT_TOPVIEW_VIDEO.endswith(".avi"):
            self.output_topview_raw = OUTPUT_TOPVIEW_VIDEO.replace(
                ".avi", "_raw.avi"
            )
        else:
            self.output_topview_raw = OUTPUT_TOPVIEW_VIDEO + "_raw.avi"

        self.raw_topview_writer = cv2.VideoWriter(
            self.output_topview_raw, fourcc, fps, (TOP_W, TOP_H), True
        )

        # Stabilized top-view video writer will be created later, after
        # the PCA-based straightening step, because we need the full
        # run information before we can reconstruct frames.
        self.stab_topview_writer: Optional[cv2.VideoWriter] = None

        # CSV file for 3D positions and velocities
        self.csv_file = open(OUTPUT_CSV, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_idx", "time_s",
            "u_px", "v_px", "r_px",
            "X_m", "Y_m", "Z_m", "distance_m",
            "Vx_m_s", "Vy_m_s", "Vz_m_s", "V_m_s"
        ])

    # ------------------------------------------------------------------ #
    # Video writing methods
    # ------------------------------------------------------------------ #
    def write_detection_frame(self, frame) -> None:
        self.det_writer.write(frame)

    def write_trajectory_frame(self, frame) -> None:
        self.traj_writer.write(frame)

    def write_raw_topview_frame(self, frame) -> None:
        self.raw_topview_writer.write(frame)

    def open_stabilized_topview_writer(self) -> None:
        """
        WHAT:
            Instantiate the stabilized top-view video writer.

        WHY:
            We only need this after all frames have been processed and
            the PCA-straightened trajectories are available.
        """
        if self.stab_topview_writer is not None:
            return  # already opened

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.stab_topview_writer = cv2.VideoWriter(
            OUTPUT_TOPVIEW_VIDEO, fourcc, self.fps, (TOP_W, TOP_H), True
        )

    def write_stabilized_topview_frame(self, frame) -> None:
        if self.stab_topview_writer is None:
            raise RuntimeError("Stabilized top-view writer is not opened yet.")
        self.stab_topview_writer.write(frame)

    # ------------------------------------------------------------------ #
    # CSV writing
    # ------------------------------------------------------------------ #
    def write_csv_row(
        self,
        frame_idx: int,
        t_s_str: str,
        u_px: str,
        v_px: str,
        r_px: str,
        X_m: str,
        Y_m: str,
        Z_m: str,
        dist_m: str,
        Vx: str,
        Vy: str,
        Vz: str,
        V: str,
    ) -> None:
        self.csv_writer.writerow([
            frame_idx, t_s_str,
            u_px, v_px, r_px,
            X_m, Y_m, Z_m, dist_m,
            Vx, Vy, Vz, V,
        ])

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self.det_writer is not None:
            self.det_writer.release()
            self.det_writer = None

        if self.traj_writer is not None:
            self.traj_writer.release()
            self.traj_writer = None

        if self.raw_topview_writer is not None:
            self.raw_topview_writer.release()
            self.raw_topview_writer = None

        if self.stab_topview_writer is not None:
            self.stab_topview_writer.release()
            self.stab_topview_writer = None

        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        print(f"[OK] Detection video saved to: {OUTPUT_DET_VIDEO}")
        print(f"[OK] Trajectory video saved to: {OUTPUT_TRAJ_VIDEO}")
        print(f"[OK] RAW top-view video saved to: {self.output_topview_raw}")
        print(f"[OK] CSV with 3D positions and velocities saved to: {OUTPUT_CSV}")
