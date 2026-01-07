"""
overlay.py
All drawing/visualisation goes here.

This keeps your main loop clean:
- compute -> then draw.
"""

import cv2
import numpy as np

# function to draw homography grid
def draw_homography_grid(out_bgr, H_world_to_pix, board_w_m, board_h_m,
                        grid_step_m=0.10, major_every=5):
    """
    Draw a floor grid overlay in pixel space using the world->pixel homography.

    Notes:
    - This is purely a debug overlay.
    - It lets you visually confirm that your homography is stable and correct.
    """
    if H_world_to_pix is None:
        return

    # Board outline: world rectangle -> pixel polygon
    board_world = np.array([[
        [0.0, 0.0],
        [board_w_m, 0.0],
        [board_w_m, board_h_m],
        [0.0, board_h_m]
    ]], dtype=np.float32)

    board_pix = cv2.perspectiveTransform(board_world, H_world_to_pix)[0]  # (4,2)
    cv2.polylines(out_bgr, [board_pix.astype(int)], True, (0, 255, 0), 2)

    # Vertical grid lines (x constant)
    x_vals = np.arange(0.0, board_w_m + 1e-9, grid_step_m)
    for idx, x in enumerate(x_vals):
        pts_world = np.array([[
            [x, 0.0],
            [x, board_h_m]
        ]], dtype=np.float32)

        pts_pix = cv2.perspectiveTransform(pts_world, H_world_to_pix)[0]
        thickness = 2 if (idx % major_every == 0) else 1

        cv2.line(out_bgr,
                 (int(pts_pix[0, 0]), int(pts_pix[0, 1])),
                 (int(pts_pix[1, 0]), int(pts_pix[1, 1])),
                 (0, 255, 0),
                 thickness)

    # Horizontal grid lines (y constant)
    y_vals = np.arange(0.0, board_h_m + 1e-9, grid_step_m)
    for idx, y in enumerate(y_vals):
        pts_world = np.array([[
            [0.0, y],
            [board_w_m, y]
        ]], dtype=np.float32)

        pts_pix = cv2.perspectiveTransform(pts_world, H_world_to_pix)[0]
        thickness = 2 if (idx % major_every == 0) else 1

        cv2.line(out_bgr,
                 (int(pts_pix[0, 0]), int(pts_pix[0, 1])),
                 (int(pts_pix[1, 0]), int(pts_pix[1, 1])),
                 (0, 255, 0),
                 thickness)

    # Label world origin
    origin_world = np.array([[[0.0, 0.0]]], dtype=np.float32)
    origin_pix = cv2.perspectiveTransform(origin_world, H_world_to_pix)[0, 0]
    cv2.putText(out_bgr, "(0,0)",
                (int(origin_pix[0]) + 6, int(origin_pix[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# function to draw robot position
def draw_robot_pose(out_bgr, robot_px, theta_world_rad, length_px=35):
    """
    Draw a simple arrow showing robot position (pixel) and heading (world angle).
    """
    if robot_px is None or theta_world_rad is None:
        return

    x, y = int(robot_px[0]), int(robot_px[1])
    cv2.circle(out_bgr, (x, y), 5, (0, 255, 0), -1)

    # Arrow endpoint in pixel space (for visual only)
    dx = int(length_px * np.cos(theta_world_rad))
    dy = int(-length_px * np.sin(theta_world_rad))
    cv2.arrowedLine(out_bgr, (x, y), (x + dx, y + dy), (0, 255, 0), 2, tipLength=0.3)

# function to draw targets and home
def draw_targets(out_bgr, targets_pix, home_pix=None):
    """
    Draw home + target points (pixel space).

    - Home is a house-like marker.
    - Targets are numbered circles.
    """
    # Home marker
    if home_pix is not None:
        hx, hy = int(home_pix[0]), int(home_pix[1])
        cv2.drawMarker(out_bgr, (hx, hy), (255, 255, 0),
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=18, thickness=2)
        cv2.putText(out_bgr, "HOME", (hx + 8, hy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Targets
    for idx, pt in enumerate(targets_pix, start=1):
        tx, ty = int(pt[0]), int(pt[1])
        cv2.circle(out_bgr, (tx, ty), 6, (0, 255, 255), -1)
        cv2.putText(out_bgr, str(idx), (tx + 8, ty - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# function to draw status info
def draw_status(out_bgr, lines, origin=(10, 20), line_gap=22):
    """
    Draw a stack of status lines at the top-left of the image.
    """
    x, y = origin
    for i, text in enumerate(lines):
        yy = y + i * line_gap
        cv2.putText(out_bgr, text, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

# function to draw route lines from robot to targets to home
def draw_route_lines(out_bgr, points_pix, colour=(255, 255, 255), thickness=2):
    """
    Draw a route as connected line segments through a list of pixel points.

    points_pix: list of (x,y) floats or ints, e.g. [robot, t1, t2, ..., home]
    """
    if points_pix is None or len(points_pix) < 2:
        return

    # Convert to integer tuples for OpenCV drawing
    pts = [(int(x), int(y)) for (x, y) in points_pix]

    # Draw segment-by-segment (easiest to control)
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(out_bgr, a, b, colour, thickness)
