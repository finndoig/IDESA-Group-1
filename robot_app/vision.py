"""
vision.py
All image-processing / marker detection lives here.

Design goal:
- This module returns DATA (detections), not drawings.
- That keeps it reusable in different “states” later (setup/manual/auto).
"""

import cv2
import numpy as np


def marker_pixel_centre(corners_4x2: np.ndarray) -> tuple[float, float]:
    """
    Compute the pixel centre of a marker from its 4 corner points.

    corners_4x2 is shape (4,2) holding the marker corners in pixel coordinates.
    """
    c = corners_4x2.mean(axis=0)
    return float(c[0]), float(c[1])


class VisionAruco:
    """
    ArUco detection wrapper using OpenCV 4.12+ API (ArucoDetector).
    """

    def __init__(self, aruco_dict, aruco_params):
        # Creating the detector once is faster and avoids repeated allocations each frame see
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    def detect(self, gray: np.ndarray) -> dict:
        """
        Detect markers in a grayscale image and return detection results.

        Returns a dictionary containing:
        - corners: list of detected marker corners (OpenCV format)
        - ids: detected marker IDs (N x 1 array) or None
        - rejected: rejected candidates (debug)
        - marker_px: {id -> (cx, cy)} centre points in pixels
        - marker_corners_px: {id -> (4,2)} corner points in pixels
        """
        corners, ids, rejected = self.detector.detectMarkers(gray)

        marker_px = {}
        marker_corners_px = {}

        if ids is not None and len(ids) > 0:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)

                # OpenCV returns corners as (1,4,2) for each marker
                c4 = corners[i][0]  # shape (4,2)

                marker_corners_px[mid] = c4
                marker_px[mid] = marker_pixel_centre(c4)

        return {
            "corners": corners,
            "ids": ids,
            "rejected": rejected,
            "marker_px": marker_px,
            "marker_corners_px": marker_corners_px,
        }
