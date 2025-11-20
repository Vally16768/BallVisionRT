"""
topview_raw.py

RawTopViewManager: builds the real-time style top-view map (raw, unstabilized).

Responsibilities:
- Maintain a persistent "raw" top-view image with all past ball points drawn.
- For each frame:
    - start from the accumulated image,
    - draw the current ball point (if any),
    - draw the current camera position (only for this frame),
    - return the resulting frame to be written to the RAW top-view video.
"""

from typing import Optional, Tuple

import cv2

from config import TOP_W, TOP_H


class RawTopViewManager:
    """
    Maintains a raw top-view accumulation buffer and produces
    per-frame raw top-view frames.
    """

    def __init__(self, topview_base) -> None:
        """
        Args:
            topview_base: initial top-view image (e.g., warped floor from frame0).
        """
        self.topview_raw_accum = topview_base.copy()

    def make_frame(
        self,
        ball_topview_point: Optional[Tuple[int, int]],
        cam_topview_point: Optional[Tuple[int, int]],
    ):
        """
        WHAT:
            Build the raw top-view frame for the current time step.

        HOW:
            - Start from the accumulated raw map.
            - If there is a ball point, draw it permanently (accumulate).
            - If there is a camera point, draw it only in the returned frame.

        Returns:
            The raw top-view BGR frame to be written to video.
        """
        # Start with a copy of the accumulated map
        frame = self.topview_raw_accum.copy()

        # 1) Ball point: accumulate and draw
        if ball_topview_point is not None:
            tx, ty = ball_topview_point
            if 0 <= tx < TOP_W and 0 <= ty < TOP_H:
                # Draw on current frame
                cv2.circle(frame, (tx, ty), 3, (0, 0, 255), -1)
                # Also accumulate into persistent map
                cv2.circle(self.topview_raw_accum, (tx, ty), 3, (0, 0, 255), -1)

        # 2) Camera point: drawn only for this frame
        if cam_topview_point is not None:
            cam_tx, cam_ty = cam_topview_point
            if 0 <= cam_tx < TOP_W and 0 <= cam_ty < TOP_H:
                cv2.circle(frame, (cam_tx, cam_ty), 5, (255, 0, 0), 2)

        return frame
