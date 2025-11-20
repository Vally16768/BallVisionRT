"""
topview_stabilizer.py

TopViewStabilizer: takes raw top-view trajectories and camera positions
and produces both:

- a final stabilized top-view image (PNG), and
- a stabilized top-view video with straightened trajectories and
  camera positions projected onto the PCA line of the corresponding run.

Responsibilities:
- Group ball points into runs (already done in pipeline, passed in).
- For each run:
    - Fit a straight line using PCA.
    - Project all points of the run onto the fitted line.
- Draw straightened points onto a final top-view image.
- Reconstruct a video where for each frame:
    - all straightened points up to that frame are drawn, and
    - the camera is projected onto the run's line (if frame belongs to a run).
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    OUTPUT_TOPVIEW_IMAGE,
    TOP_W,
    TOP_H,
)
from pipeline.utils import fmt_float  # not strictly needed here but ok to reuse


class TopViewStabilizer:
    """
    Encapsulates PCA-based straightening logic and stabilized-video generation.
    """

    def __init__(
        self,
        topview_base,
        topview_runs: List[List[Tuple[int, int, int]]],
        cam_positions: List[Optional[Tuple[int, int]]],
        run_for_frame: List[Optional[int]],
    ) -> None:
        """
        Args:
            topview_base: static floor in top-view (BGR).
            topview_runs: list of runs, each run is a list of
                (frame_idx, tx, ty) in top-view coordinates.
            cam_positions: list indexed by frame_idx with camera positions
                in top-view coordinates or None.
            run_for_frame: for each frame_idx, the run index it belongs to,
                or None if that frame has no detected ball.
        """
        self.topview_base = topview_base
        self.topview_runs = topview_runs
        self.cam_positions = cam_positions
        self.run_for_frame = run_for_frame

        # Straightened runs and PCA models will be computed in stabilize()
        self.straightened_runs: List[List[Tuple[int, int, int]]] = []
        self.run_models = []  # each element: {"mean": mean, "dir": principal_dir}

    # ------------------------------------------------------------------ #
    # PCA-based straightening
    # ------------------------------------------------------------------ #
    def stabilize(self) -> None:
        """
        WHAT:
            For each run, fit a straight line using PCA and project all points.
            Also build the final top-view image with straightened runs.
        """
        topview_final = self.topview_base.copy()
        straightened_runs: List[List[Tuple[int, int, int]]] = []
        run_models = []

        for run_idx, run_pts in enumerate(self.topview_runs):
            if len(run_pts) == 0:
                straightened_runs.append([])
                run_models.append({"mean": None, "dir": None})
                continue

            frames_arr = np.array([p[0] for p in run_pts], dtype=np.int32)
            pts = np.array([[p[1], p[2]] for p in run_pts], dtype=np.float32)

            # Short run: no reliable PCA; use points as they are.
            if pts.shape[0] < 3:
                straightened_run: List[Tuple[int, int, int]] = []
                for fidx, (tx, ty) in zip(frames_arr, pts):
                    tx_i = int(round(tx))
                    ty_i = int(round(ty))
                    if 0 <= tx_i < TOP_W and 0 <= ty_i < TOP_H:
                        cv2.circle(topview_final, (tx_i, ty_i), 3, (0, 0, 255), -1)
                    straightened_run.append((int(fidx), tx_i, ty_i))

                straightened_runs.append(straightened_run)
                run_models.append({"mean": None, "dir": None})
                continue

            # PCA line fit
            mean = pts.mean(axis=0)
            pts_centered = pts - mean
            cov = (pts_centered.T @ pts_centered) / float(pts_centered.shape[0])
            eigvals, eigvecs = np.linalg.eig(cov)
            principal_dir = eigvecs[:, np.argmax(eigvals)].real
            norm = np.linalg.norm(principal_dir)
            if norm == 0:
                principal_dir = np.array([1.0, 0.0], dtype=np.float32)
            else:
                principal_dir = principal_dir / norm

            # Project points onto the fitted line
            straightened_run: List[Tuple[int, int, int]] = []
            for fidx, p in zip(frames_arr, pts):
                proj_len = float(np.dot(p - mean, principal_dir))
                p_proj = mean + proj_len * principal_dir
                tx_s = int(round(p_proj[0]))
                ty_s = int(round(p_proj[1]))

                if 0 <= tx_s < TOP_W and 0 <= ty_s < TOP_H:
                    cv2.circle(topview_final, (tx_s, ty_s), 3, (0, 0, 255), -1)

                straightened_run.append((int(fidx), tx_s, ty_s))

            straightened_runs.append(straightened_run)
            run_models.append({"mean": mean, "dir": principal_dir})

        # Save state
        self.straightened_runs = straightened_runs
        self.run_models = run_models

        # Save final image
        cv2.imwrite(OUTPUT_TOPVIEW_IMAGE, topview_final)
        print(f"[OK] Final stabilized top-view image saved to: {OUTPUT_TOPVIEW_IMAGE}")

    # ------------------------------------------------------------------ #
    # Stabilized video reconstruction
    # ------------------------------------------------------------------ #
    def build_stabilized_video_frames(self, total_frames: int):
        """
        Generator that yields stabilized top-view frames for each frame index.

        For each frame:
            - Draw all straightened ball points with frame_idx <= current frame.
            - Draw camera position, projected on PCA line of the corresponding
              run (if frame belongs to a run and model is available).
        """
        for fidx in range(total_frames):
            frame_img = self.topview_base.copy()

            # Draw all straightened ball points up to this frame
            for straight_run in self.straightened_runs:
                for (p_fidx, tx_s, ty_s) in straight_run:
                    if p_fidx <= fidx:
                        if 0 <= tx_s < TOP_W and 0 <= ty_s < TOP_H:
                            cv2.circle(frame_img, (tx_s, ty_s), 3, (0, 0, 255), -1)

            # Camera position for this frame
            cam_raw = (
                self.cam_positions[fidx]
                if fidx < len(self.cam_positions)
                else None
            )
            run_idx_here = (
                self.run_for_frame[fidx]
                if fidx < len(self.run_for_frame)
                else None
            )

            if cam_raw is not None:
                cam_tx, cam_ty = cam_raw
                cam_tx_draw, cam_ty_draw = cam_tx, cam_ty

                # Option B: project camera onto the PCA line of the current run.
                if (run_idx_here is not None and
                        0 <= run_idx_here < len(self.run_models)):
                    model = self.run_models[run_idx_here]
                    mean = model["mean"]
                    direction = model["dir"]

                    if mean is not None and direction is not None:
                        p_cam = np.array([cam_tx, cam_ty], dtype=np.float32)
                        proj_len = float(np.dot(p_cam - mean, direction))
                        p_proj = mean + proj_len * direction
                        cam_tx_draw = int(round(p_proj[0]))
                        cam_ty_draw = int(round(p_proj[1]))

                if 0 <= cam_tx_draw < TOP_W and 0 <= cam_ty_draw < TOP_H:
                    cv2.circle(
                        frame_img,
                        (cam_tx_draw, cam_ty_draw),
                        5,
                        (255, 0, 0),
                        2
                    )

            yield frame_img
