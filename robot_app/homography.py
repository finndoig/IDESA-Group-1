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

    # function to rotate corners of board corner markers
    @staticmethod
    def _rotate_corners(c4: np.ndarray, rot_k: int) -> np.ndarray:
        """
        Rotate the 4 detected corners by 90° steps so they match our assumed world orientation.

        c4: (4,2) corners from OpenCV
        rot_k: 0,1,2,3 -> rotate order by 0,90,180,270 degrees.
        """
        rot_k = int(rot_k) % 4
        if rot_k == 0:
            return c4
        return np.roll(c4, -rot_k, axis=0)

    # function to get World corners for a marker centred at (cx, cy), assuming marker edges align with world axes.
    @staticmethod
    def _marker_world_corners(cx: float, cy: float, size_m: float) -> np.ndarray:
        """
        World corners for a marker centred at (cx, cy), assuming marker edges align with world axes.

        Ordering MUST match OpenCV corner convention *when marker is upright*:
          0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
        """
        h = size_m / 2.0
        return np.array([
            [cx - h, cy + h],  # TL
            [cx + h, cy + h],  # TR
            [cx + h, cy - h],  # BR
            [cx - h, cy - h],  # BL
        ], dtype=np.float32)

    # function to update homography using all 4 corners of markers for more accuracy/robustness
    def update(self,
               marker_px: dict[int, tuple[float, float]],
               marker_corners_px: dict[int, np.ndarray] | None = None,
               corner_marker_size_m: float | None = None,
               corner_rot_k: dict[int, int] | None = None) -> bool:
        """
        Update homography.
        Preferred: use 16 points (4 corners × 4 markers).
        Fallback: use 4 points (marker centres) if corners aren’t supplied.

        marker_px: {id -> (cx,cy)} in pixels
        marker_corners_px: {id -> (4,2)} in pixels
        corner_marker_size_m: physical side length of the board corner markers
        corner_rot_k: {id -> rot_k} to correct corner ordering if marker rotated on floor
        """
        required_ids = list(self.world_points_by_id.keys())

        # If required markers not all visible, expire after timeout
        if not all(mid in marker_px for mid in required_ids):
            if self.H_pix_to_world is not None and (time.perf_counter() - self.last_update_time) > self.timeout_s:
                self.H_pix_to_world = None
                self.H_world_to_pix = None
            return False

        use_16pt = (
            marker_corners_px is not None
            and corner_marker_size_m is not None
            and all(mid in marker_corners_px for mid in required_ids)
        )

        img_pts = []
        world_pts = []

        if use_16pt:
            # Build 16 correspondences
            corner_rot_k = corner_rot_k or {}

            for mid in required_ids:
                # Marker centre in world coordinates (as defined in config)
                cx_w, cy_w = self.world_points_by_id[mid]

                # World corners for this marker (assuming axis-aligned placement)
                wc = self._marker_world_corners(cx_w, cy_w, float(corner_marker_size_m))

                # Pixel corners detected by OpenCV
                c4 = marker_corners_px[mid].astype(np.float32)  # (4,2)

                # Apply per-marker rotation correction (if marker rotated on the floor)
                rk = int(corner_rot_k.get(mid, 0))
                c4 = self._rotate_corners(c4, rk)

                # Append 4 corners
                for k in range(4):
                    img_pts.append(c4[k])
                    world_pts.append(wc[k])

            img_pts = np.array(img_pts, dtype=np.float32)
            world_pts = np.array(world_pts, dtype=np.float32)

            # With >4 points, use RANSAC to reject outliers (glare/bad corner)
            H, _mask = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)

        else:
            # Fallback: 4 points using marker centres (your old behaviour)
            for mid in required_ids:
                cx_px, cy_px = marker_px[mid]
                cx_w, cy_w = self.world_points_by_id[mid]
                img_pts.append([cx_px, cy_px])
                world_pts.append([cx_w, cy_w])

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
