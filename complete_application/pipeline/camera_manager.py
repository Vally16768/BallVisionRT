"""
camera_manager.py

CameraManager: handles camera motion estimation and derived quantities.

Responsibilities:
- Use template matching on the floor template to estimate camera translation (dx, dy).
- Apply exponential smoothing on (dx, dy) to reduce jitter.
- Warp the reference floor mask into the current frame using the smoothed translation.
- Map the (approximate) camera center into top-view coordinates via homography.
"""

from typing import Optional, Tuple

import numpy as np

from camera_motion import build_floor_template, track_camera, warp_floor_mask


class CameraManager:
    """
    Maintains camera tracking state and provides per-frame camera-related outputs.

    Usage pattern:
        cam_mgr = CameraManager(frame0, floor_mask0, H0_to_top, cx, cy)
        dx_eff, dy_eff, floor_mask_curr, cam_pos = cam_mgr.process_frame(
            gray_frame, frame_idx
        )
    """

    def __init__(
        self,
        frame0,
        floor_mask0,
        H0_to_top,
        cx: float,
        cy: float,
        width: int,
        height: int,
        alpha: float = 0.9,
    ) -> None:
        """
        Args:
            frame0: reference BGR frame (first frame of the video).
            floor_mask0: floor mask in the reference frame.
            H0_to_top: 3x3 homography from reference frame to top-view.
            cx, cy: principal point (image center) in pixel coordinates.
            width, height: frame dimensions.
            alpha: smoothing factor for (dx, dy).
        """
        self.floor_mask0 = floor_mask0
        self.H0_to_top = H0_to_top
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

        self.floor_template, self.floor_bbox = build_floor_template(
            frame0, floor_mask0
        )

        # Smoothed camera translation
        self.dx_smooth = 0.0
        self.dy_smooth = 0.0
        self.alpha = alpha

    def process_frame(
        self,
        gray_frame,
        frame_idx: int,
    ) -> Tuple[float, float, 'np.ndarray', Optional[Tuple[int, int]]]:
        """
        WHAT:
            Process a new frame and return:
                - smoothed camera translation (dx_eff, dy_eff),
                - current floor mask (aligned to this frame),
                - camera position in top-view coordinates.

        WHY:
            We need consistent camera motion estimation for:
            - warping the floor mask into the current frame,
            - mapping the camera center into top-view.
        """
        # 1) Template matching to estimate translation
        dx, dy, match_score, match_loc = track_camera(
            gray_frame, self.floor_template, self.floor_bbox
        )

        # 2) Exponential smoothing on dx, dy
        if frame_idx == 0:
            self.dx_smooth, self.dy_smooth = dx, dy
        else:
            self.dx_smooth = (
                self.alpha * self.dx_smooth + (1.0 - self.alpha) * dx
            )
            self.dy_smooth = (
                self.alpha * self.dy_smooth + (1.0 - self.alpha) * dy
            )

        dx_eff = self.dx_smooth
        dy_eff = self.dy_smooth

        # 3) Warp reference floor mask into current frame
        floor_mask_curr = warp_floor_mask(
            self.floor_mask0, dx_eff, dy_eff, self.width, self.height
        )

        # 4) Camera position in top-view:
        #     we consider (cx + dx_eff, cy + dy_eff) as the camera center
        #     in the reference frame coordinates, then map to top-view.
        cam_ref = np.array(
            [self.cx + dx_eff, self.cy + dy_eff, 1.0],
            dtype=np.float32
        ).reshape(3, 1)
        p_cam_top = self.H0_to_top @ cam_ref

        cam_pos: Optional[Tuple[int, int]] = None
        if p_cam_top[2, 0] != 0:
            p_cam_top /= p_cam_top[2, 0]
            cam_tx = int(p_cam_top[0, 0])
            cam_ty = int(p_cam_top[1, 0])

            if 0 <= cam_tx < self.width and 0 <= cam_ty < self.height:
                cam_pos = (cam_tx, cam_ty)

        return dx_eff, dy_eff, floor_mask_curr, cam_pos
