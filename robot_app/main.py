"""
main.py
Wires all modules together.

Current features:
- ArUco detection (OpenCV 4.12+ API)
- Homography from 4 corner markers (centre-based)
- Grid overlay to visualise homography stability
- Click-to-target (left-click)
- “Set home” mode (press H then click)
- Distance/bearing to NEXT target in the list (in world metres/radians)

Keyboard controls:
- q : quit
- h : next click sets HOME (toggles click mode)
- t : next click adds TARGET (toggles click mode)
- c : clear all targets
- u : undo last target
"""

import time
import logging
import numpy as np
import cv2

import config
from vision import VisionAruco
from homography import HomographyBoard
from targets import TargetManager, ClickMode
import overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# -----------------------------
# Maths helpers (kept small and local)
# -----------------------------
# function to wrap angle to [-pi, pi]
def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle to [-pi, pi] so bearings don’t jump around at +/-180 degrees."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

# Function to compute circular mean of angles (avoid error when angles flip between -179 and +180 degrees)
def circular_mean(angles_rad) -> float:
    """
    Circular mean for angles.
    Use this for bearings if you start smoothing bearings over time.
    """
    angles = np.asarray(list(angles_rad), dtype=float)
    if angles.size == 0:
        return 0.0
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return float(np.arctan2(s, c))

# function for processing image to improve marker detection
def preprocess_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Pre-process the frame to improve marker detection under difficult lighting.

    This is where you keep your contrast/CLAHE tricks.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast/brightness tweak (helps black/white separation under glare)
    gray = cv2.convertScaleAbs(gray, alpha=config.CONTRAST_ALPHA, beta=config.CONTRAST_BETA)

    # CLAHE to boost local contrast
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID
    )
    gray = clahe.apply(gray)

    return gray


# -----------------------------
# Mouse callback wiring
# -----------------------------
def make_mouse_handler(targets: TargetManager, board: HomographyBoard):
    """
    Returns a function that will be used as OpenCV’s mouse callback.

    The callback can access the targets + homography objects.
    """

    def on_mouse(event, x, y, flags, param):
        # only left click 
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        click_px = (float(x), float(y))

        # If homography is not ready yet, store the click and resolve later
        if not board.valid():
            targets.set_pending_click(click_px, targets.click_mode)
            logging.info("Click stored (no homography yet): px=%s mode=%s", click_px, targets.click_mode.name)
            return

        # Convert click pixel to world coordinate (metres)
        click_world = board.pix_to_world(click_px[0], click_px[1])

        if targets.click_mode == ClickMode.SET_HOME:
            targets.set_home_world(click_world)
            logging.info("HOME set: world=(%.3f, %.3f)", click_world[0], click_world[1])

            # After setting home, switch back to adding targets (nice UX)
            targets.set_click_mode(ClickMode.ADD_TARGET)

        else:
            targets.add_target_world(click_world)
            logging.info("Target added: world=(%.3f, %.3f) (total=%d)",
                         click_world[0], click_world[1], len(targets.targets_world))

    return on_mouse


# -----------------------------
# Robot position estimation in world frame (2D, homography-based)
# -----------------------------
def robot_pose_from_marker(board: HomographyBoard, marker_px: dict, marker_corners_px: dict, robot_id: int):
    """
    Compute robot (x,y,theta) in world frame using:
    - marker centre for position
    - a marker edge (corner0->corner1) for heading

    This avoids 3D rvec/tvec pose estimation, which can be noisy at long distance.
    """
    if not board.valid():
        return None, None, None

    if robot_id not in marker_px or robot_id not in marker_corners_px:
        return None, None, None

    # 1) Position: centre pixel -> world
    rx_px, ry_px = marker_px[robot_id]
    rx_w, ry_w = board.pix_to_world(rx_px, ry_px)

    # 2) Heading: take one edge direction in pixel space and transform to world
    c4 = marker_corners_px[robot_id]   # (4,2) pixels
    p0 = c4[0]
    p1 = c4[1]

    w0 = board.pix_to_world(float(p0[0]), float(p0[1]))
    w1 = board.pix_to_world(float(p1[0]), float(p1[1]))

    vx = w1[0] - w0[0]
    vy = w1[1] - w0[1]
    theta = float(np.arctan2(vy, vx))  # radians, world frame

    return (rx_w, ry_w), (rx_px, ry_px), theta


def main():
    # Load calibration file 
    camera = np.load(config.CALIB_PATH)
    CM = camera["CM"]
    dist_coef = camera["dist_coef"]
    _ = (CM, dist_coef)  # not used yet; placeholder for future use

    # Create modules
    vision = VisionAruco(config.ARUCO_DICT, config.ARUCO_PARAMS)
    board = HomographyBoard(config.BOARD_WORLD_POINTS_M, timeout_s=config.H_TIMEOUT_S)
    targets = TargetManager()

    # Setup camera
    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(config.WINDOW_NAME, make_mouse_handler(targets, board))

    logging.info("Running. Press H then click to set HOME. Left-click to add targets. Q quits.")

    while True:
        t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            logging.warning("Camera frame not received.")
            continue

        gray = preprocess_gray(frame)

        # Detect markers (data only)
        det = vision.detect(gray)

        # Update homography based on corner markers
        hom_updated = board.update(det["marker_px"])

        # Resolve pending click once homography becomes available
        if board.valid() and targets.pending_click_px is not None:
            px = targets.pending_click_px
            mode = targets.pending_click_mode or ClickMode.ADD_TARGET
            world = board.pix_to_world(px[0], px[1])

            if mode == ClickMode.SET_HOME:
                targets.set_home_world(world)
                logging.info("Resolved stored HOME click: world=(%.3f, %.3f)", world[0], world[1])
            else:
                targets.add_target_world(world)
                logging.info("Resolved stored target click: world=(%.3f, %.3f)", world[0], world[1])

            targets.clear_pending_click()

        # Prepare output image for drawing (use BGR so colours work)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw detected markers (purely visual)
        if det["ids"] is not None and len(det["ids"]) > 0:
            out = cv2.aruco.drawDetectedMarkers(out, det["corners"], det["ids"])

        # Draw homography grid overlay if valid
        if board.valid():
            overlay.draw_homography_grid(
                out,
                board.H_world_to_pix,
                config.BOARD_WIDTH_M,
                config.BOARD_HEIGHT_M,
                grid_step_m=config.GRID_STEP_M,
                major_every=config.GRID_MAJOR_EVERY
            )

        # Compute robot pose in world coords
        robot_world, robot_px, robot_theta = robot_pose_from_marker(
            board, det["marker_px"], det["marker_corners_px"], config.ROBOT_ID
        )

        # Draw robot position and heading arrow
        overlay.draw_robot_pose(out, robot_px, robot_theta)

        # Convert home + targets from world->pixel for drawing
        home_pix = None
        if board.valid() and targets.home_world is not None:
            home_pix = board.world_to_pix(targets.home_world[0], targets.home_world[1])

        targets_pix = []
        if board.valid():
            for (tx, ty) in targets.targets_world:
                targets_pix.append(board.world_to_pix(tx, ty))

        overlay.draw_targets(out, targets_pix, home_pix=home_pix)

        # Compute distance/bearing to NEXT target (if possible)
        dist_txt = "Next: -"
        if robot_world is not None and robot_theta is not None and targets.targets_world:
            rx, ry = robot_world
            tx, ty = targets.targets_world[0]

            dx = tx - rx
            dy = ty - ry
            dist_m = float(np.hypot(dx, dy))
            ang_world = float(np.arctan2(dy, dx))
            bearing = float(wrap_to_pi(ang_world - robot_theta))

            dist_txt = f"Next: d={dist_m:.2f}m  b={np.degrees(bearing):.0f}deg"

        # Status text
        mode_txt = f"ClickMode: {targets.click_mode.name} (H=set HOME, T=add TARGET)"
        hom_txt = "Homography: OK" if board.valid() else "Homography: NOT AVAILABLE (need 4 corner markers)"
        tgt_txt = f"Targets: {len(targets.targets_world)}"

        # Show FPS + key info
        t1 = time.perf_counter()
        fps = 1.0 / (t1 - t0) if t1 > t0 else 0.0
        fps_txt = f"FPS: {fps:.1f}"

        overlay.draw_status(out, [hom_txt, mode_txt, tgt_txt, dist_txt, fps_txt])

        # Display
        cv2.imshow(config.WINDOW_NAME, out)

        # Keyboard handling
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("h"):
            targets.set_click_mode(ClickMode.SET_HOME)
            logging.info("Click mode -> SET_HOME (next click sets home).")
        elif k == ord("t"):
            targets.set_click_mode(ClickMode.ADD_TARGET)
            logging.info("Click mode -> ADD_TARGET (click adds target).")
        elif k == ord("c"):
            targets.clear_targets()
            logging.info("Targets cleared.")
        elif k == ord("u"):
            targets.undo_last_target()
            logging.info("Undid last target.")

        # Optional: log homography refresh events
        if hom_updated:
            logging.info("Homography updated.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
