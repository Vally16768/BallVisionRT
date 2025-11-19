import cv2
import numpy as np


class CameraMotionCompensator:
    """ORB-based camera motion compensator using a reference frame.

    It estimates a homography H that maps current frame coordinates
    into the reference frame coordinates.
    """

    def __init__(self, ref_frame, floor_mask=None,
                 n_features=800, ratio_thresh=0.75, ransac_thresh=3.0):

        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        self.ratio = ratio_thresh
        self.ransac = ransac_thresh

        self.ref_kp, self.ref_des = self._detect(ref_frame, floor_mask)
        self.H = np.eye(3, dtype=np.float32)

    def _detect(self, frame, mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = self.orb.detect(gray, mask)
        if not kp:
            return [], None
        kp, des = self.orb.compute(gray, kp)
        return kp, des

    def update(self, frame, mask):
        """Update homography H(current -> reference) using ORB matches."""
        cur_kp, cur_des = self._detect(frame, mask)
        if cur_des is None or self.ref_des is None:
            return self.H

        matches = self.bf.knnMatch(cur_des, self.ref_des, k=2)
        good = [m for m, n in matches if m.distance < self.ratio * n.distance]

        if len(good) < 4:
            return self.H

        src = np.float32([cur_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac)
        if H is not None:
            self.H = H.astype(np.float32)

        return self.H

    def warp_point(self, u, v):
        """Warp a point (u,v) from current frame to reference frame."""
        pt = np.array([[[u, v]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
