"""
homography.py
Maintains the homography mapping between:
- pixel coordinates (camera image)
- world coordinates (your floor board in metres)

A homography is valid when your 4 corner markers are detected.
We keep the last good homography briefly to survive short dropouts.
"""

import time
import numpy as np
import cv2


class HomographyBoard:

    # function to initialize homography board
    def __init__(self, world_points_by_id: dict[int, tuple[float, float]], timeout_s: float = 1.0):
        """
        world_points_by_id:
            Dict mapping marker_id -> (x_m, y_m) world coordinate of that marker's *centre*.

        timeout_s:
            How long to keep using the last homography if corner markers vanish temporarily.
        """
        self.world_points_by_id = world_points_by_id
        self.timeout_s = timeout_s

        self.H_pix_to_world = None
        self.H_world_to_pix = None
        self.last_update_time = 0.0

    # function to update homography
    def update(self, marker_px: dict[int, tuple[float, float]]) -> bool:
        """
        Try to compute/update homography using current marker centres.
        Returns True if updated this call, else False.
        """
        required_ids = list(self.world_points_by_id.keys())

        # Check if all required corner markers are present this frame
        if not all(mid in marker_px for mid in required_ids):
            # Expire old homography if it’s too old
            if self.H_pix_to_world is not None and (time.perf_counter() - self.last_update_time) > self.timeout_s:
                self.H_pix_to_world = None
                self.H_world_to_pix = None
            return False

        # Build matching point sets:
        # image points are marker centres in pixels,
        # world points are the known positions in metres.
        img_pts = []
        world_pts = []
        for mid in required_ids:
            cx, cy = marker_px[mid]
            wx, wy = self.world_points_by_id[mid]
            img_pts.append([cx, cy])
            world_pts.append([wx, wy])

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        H, _mask = cv2.findHomography(img_pts, world_pts, method=0)
        if H is None:
            return False

        self.H_pix_to_world = H
        self.H_world_to_pix = np.linalg.inv(H)
        self.last_update_time = time.perf_counter()
        return True

    # function to check if homography is valid
    def valid(self) -> bool:
        """True if we currently have a usable homography."""
        return self.H_pix_to_world is not None and self.H_world_to_pix is not None

    # function to convert pixel to world coordinates
    def pix_to_world(self, px: float, py: float) -> tuple[float, float]:
        """
        Convert a pixel point -> world (metres) using current homography.
        """
        pts = np.array([[[px, py]]], dtype=np.float32)            # shape (1,1,2)
        out = cv2.perspectiveTransform(pts, self.H_pix_to_world)  # shape (1,1,2)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    # function to convert world to pixel coordinates
    def world_to_pix(self, x: float, y: float) -> tuple[float, float]:
        """
        Convert a world point (metres) -> pixel using inverse homography.
        """
        pts = np.array([[[x, y]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self.H_world_to_pix)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
