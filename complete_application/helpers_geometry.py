import numpy as np
from constants import FX_NATIVE, FY_NATIVE, CX_NATIVE, CY_NATIVE, BALL_DIAMETER_M

def compute_intrinsics(frame_shape):
    h, w = frame_shape[:2]
    fx = FX_NATIVE
    fy = FY_NATIVE
    cx = CX_NATIVE if CX_NATIVE is not None else w / 2
    cy = CY_NATIVE if CY_NATIVE is not None else h / 2
    return fx, fy, cx, cy


def estimate_distance_from_radius(radius_px, fx, ball_diameter_m=BALL_DIAMETER_M, eps=1e-6):
    d_px = max(2 * radius_px, eps)
    return fx * ball_diameter_m / d_px
