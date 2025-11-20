"""
ball_manager.py

BallManager: detects the red ball, estimates its 3D position, and
computes 3D velocities over time.

Responsibilities:
- Use detect_red_ball_on_floor() to find the ball in the current frame.
- Estimate 3D position (X, Y, Z, distance) using a pinhole model.
- Compute 3D velocities as finite differences between consecutive frames.
"""

import math
from typing import Dict, Optional, Tuple

from ball_tracking import (
    detect_red_ball_on_floor,
    estimate_ball_3d,
)


class BallManager:
    """
    Keeps track of per-frame ball detection and 3D information.

    Usage pattern:
        ball_mgr = BallManager(f_px, cx, cy)
        result = ball_mgr.process_frame(frame, floor_mask_curr, t_s)

        where result contains:
            - "ball": (bx, by, br) or None
            - "X_m", "Y_m", "Z_m", "dist_m"
            - "Vx", "Vy", "Vz", "V"
    """

    def __init__(self, f_px: float, cx: float, cy: float) -> None:
        self.f_px = f_px
        self.cx = cx
        self.cy = cy

        # Tracking state across frames
        self.prev_ball: Optional[Tuple[int, int, int]] = None  # (x, y, r)
        self.prev_3d: Optional[Tuple[float, float, float]] = None
        self.prev_t: Optional[float] = None

    def process_frame(
        self,
        frame,
        floor_mask_curr,
        t_s: float,
    ) -> Dict[str, Optional[float]]:
        """
        Process a new frame and update ball-related information.

        Args:
            frame: current BGR frame.
            floor_mask_curr: binary floor mask aligned to the current frame.
            t_s: current time in seconds (frame_idx / fps).

        Returns:
            dict with keys:
                "ball"   : (bx, by, br) or None
                "X_m"    : X in meters or None
                "Y_m"    : Y in meters or None
                "Z_m"    : Z in meters or None
                "dist_m" : distance in meters or None
                "Vx"     : velocity component X or None
                "Vy"     : velocity component Y or None
                "Vz"     : velocity component Z or None
                "V"      : speed magnitude or None
        """
        result: Dict[str, Optional[float]] = {
            "ball": None,
            "X_m": None,
            "Y_m": None,
            "Z_m": None,
            "dist_m": None,
            "Vx": None,
            "Vy": None,
            "Vz": None,
            "V": None,
        }

        prev_center = (
            self.prev_ball[:2] if self.prev_ball is not None else None
        )

        # 1) Detect ball in current frame constrained to floor region.
        ball = detect_red_ball_on_floor(
            frame, floor_mask_curr, prev_center=prev_center
        )
        if ball is None:
            # No ball this frame.
            self.prev_ball = None
            return result

        bx, by, br = ball
        self.prev_ball = ball
        result["ball"] = ball

        # 2) Estimate 3D position.
        est_3d = estimate_ball_3d(bx, by, br, self.f_px, self.cx, self.cy)
        if est_3d is None:
            return result

        X_m, Y_m, Z_m, dist_m = est_3d
        result["X_m"] = X_m
        result["Y_m"] = Y_m
        result["Z_m"] = Z_m
        result["dist_m"] = dist_m

        # 3) Estimate 3D velocity using finite differences.
        if self.prev_3d is not None and self.prev_t is not None:
            dT = t_s - self.prev_t
            if dT > 0:
                X_prev, Y_prev, Z_prev = self.prev_3d
                Vx = (X_m - X_prev) / dT
                Vy = (Y_m - Y_prev) / dT
                Vz = (Z_m - Z_prev) / dT
                V = math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)

                result["Vx"] = Vx
                result["Vy"] = Vy
                result["Vz"] = Vz
                result["V"] = V

        # Update state
        self.prev_3d = (X_m, Y_m, Z_m)
        self.prev_t = t_s

        return result
