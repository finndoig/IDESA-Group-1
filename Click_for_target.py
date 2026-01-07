# -----------------------------
# ArUco + Homography Click-to-Target (OpenCV 4.12+)
# - Uses 4 fixed floor ArUcos to compute a pixel->floor homography
# - Lets user click anywhere in the image to set a target point on the floor
# - Computes robot->target distance and bearing in floor coordinates
# - Still supports ArUco target markers as additional targets (optional)
# -----------------------------

import cv2
import numpy as np
import time
import cv2.aruco as aruco
import logging
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# ---- Load camera calibration (still useful for future, even though homography does most 2D work)
Camera = np.load('Calibration.npz')
CM = Camera['CM']
dist_coef = Camera['dist_coef']

# -----------------------------
# USER SETTINGS
# -----------------------------

# Robot + (optional) target marker IDs
ROBOT_ID = 0
TARGET_IDS = {11, 12, 13, 14, 15}  # optional physical targets as ArUcos

# Floor homography markers (FOUR ArUcos on the board corners)
# Choose IDs that are NOT used by robot/targets
BOARD_CORNER_IDS = {
    "TL": 1,  # top-left marker ID
    "TR": 2,  # top-right
    "BR": 4,  # bottom-right
    "BL": 3,  # bottom-left
}

# Real-world coordinates of those 4 corners (metres), matching your physical layout.
# Here we set origin at bottom-left (BL) and +x to the right, +y upwards.
BOARD_WIDTH_M = 0.5
BOARD_HEIGHT_M = 0.6
BOARD_WORLD_POINTS_M = {
    BOARD_CORNER_IDS["BL"]: (0.0, 0.0),
    BOARD_CORNER_IDS["BR"]: (BOARD_WIDTH_M, 0.0),
    BOARD_CORNER_IDS["TR"]: (BOARD_WIDTH_M, BOARD_HEIGHT_M),
    BOARD_CORNER_IDS["TL"]: (0.0, BOARD_HEIGHT_M),
}

# Smoothing buffers (20 frames)
buffers = {tid: {'dist': deque(maxlen=20), 'bear': deque(maxlen=20)} for tid in TARGET_IDS}

# -----------------------------
# ArUco detector setup (OpenCV 4.12+ API)
# -----------------------------
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
pa = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, pa)

# -----------------------------
# Camera setup
# -----------------------------
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

WINDOW_NAME = "aruco-image"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WINDOW_NAME, 640, 100)

# -----------------------------
# Helper functions
# -----------------------------
def wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def circular_mean(angles_rad) -> float:
    angles = np.asarray(list(angles_rad), dtype=float)
    if angles.size == 0:
        return 0.0
    s = np.mean(np.sin(angles))
    c = np.mean(np.cos(angles))
    return float(np.arctan2(s, c))

def marker_pixel_centre(corners_4x2: np.ndarray) -> tuple[float, float]:
    c = corners_4x2.mean(axis=0)
    return float(c[0]), float(c[1])

def pix_to_world(H_pix_to_world: np.ndarray, px: float, py: float) -> tuple[float, float]:
    pts = np.array([[[px, py]]], dtype=np.float32)  # shape (1,1,2)
    out = cv2.perspectiveTransform(pts, H_pix_to_world)  # (1,1,2)
    return float(out[0, 0, 0]), float(out[0, 0, 1])

def world_to_pix(H_world_to_pix: np.ndarray, x: float, y: float) -> tuple[float, float]:
    pts = np.array([[[x, y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, H_world_to_pix)
    return float(out[0, 0, 0]), float(out[0, 0, 1])

def draw_homography_grid(out_bgr, H_world_to_pix, board_w_m, board_h_m,
                        grid_step_m=0.10, major_every=5):
    """
    Draw a floor grid overlay (in pixel space) using the world->pixel homography.

    out_bgr:        image to draw onto (BGR)
    H_world_to_pix: 3x3 homography mapping world(m) -> pixel
    board_w_m:      board width in metres
    board_h_m:      board height in metres
    grid_step_m:    grid spacing in metres (e.g., 0.10 = 10cm)
    major_every:    draw a thicker 'major' line every N grid lines
    """
    if H_world_to_pix is None:
        return

    # Draw board outline (world rectangle -> pixel polygon)
    board_world = np.array([[
        [0.0, 0.0],
        [board_w_m, 0.0],
        [board_w_m, board_h_m],
        [0.0, board_h_m]
    ]], dtype=np.float32)
    board_pix = cv2.perspectiveTransform(board_world, H_world_to_pix)[0]  # (4,2)
    cv2.polylines(out_bgr, [board_pix.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Vertical grid lines (x constant)
    x_vals = np.arange(0.0, board_w_m + 1e-9, grid_step_m)
    for idx, x in enumerate(x_vals):
        pts_world = np.array([[
            [x, 0.0],
            [x, board_h_m]
        ]], dtype=np.float32)  # (1,2,2)
        pts_pix = cv2.perspectiveTransform(pts_world, H_world_to_pix)[0]  # (2,2)
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

    # Optional: label the origin (0,0)
    origin_world = np.array([[[0.0, 0.0]]], dtype=np.float32)
    origin_pix = cv2.perspectiveTransform(origin_world, H_world_to_pix)[0, 0]
    cv2.putText(out_bgr, "(0,0)",
                (int(origin_pix[0]) + 6, int(origin_pix[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# -----------------------------
# Click-to-target state
# -----------------------------
clicked_px = None          # (x_px, y_px)
clicked_world = None       # (x_m, y_m)
have_click = False

H_pix_to_world = None
H_world_to_pix = None
last_H_time = 0.0
H_TIMEOUT_S = 1.0  # if board corners disappear briefly, keep last homography for up to 1s

def on_mouse(event, x, y, flags, param):
    global clicked_px, clicked_world, have_click, H_pix_to_world
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_px = (float(x), float(y))
        have_click = True
        # If we already have a homography, immediately convert to world
        if H_pix_to_world is not None:
            clicked_world = pix_to_world(H_pix_to_world, clicked_px[0], clicked_px[1])
            logging.info("Clicked target pixel=%s -> world(m)=(%.3f, %.3f)", clicked_px, clicked_world[0], clicked_world[1])
        else:
            clicked_world = None
            logging.info("Clicked target pixel=%s (no homography yet; will convert when available)", clicked_px)

cv2.setMouseCallback(WINDOW_NAME, on_mouse)

# -----------------------------
# Main loop
# -----------------------------
while True:
    start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        logging.warning("Camera frame not received.")
        continue

    # Work in grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Your contrast tweaks (keep if they help under your lighting)
    gray = cv2.convertScaleAbs(gray, alpha=0.7, beta=-50)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(32, 32))
    gray = clahe.apply(gray)

    # Detect markers (OpenCV 4.12+)
    corners, ids, rejected = detector.detectMarkers(gray)

    # For visual overlay, use BGR image
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if ids is not None and len(ids) > 0:
        out = aruco.drawDetectedMarkers(out, corners, ids)

    # Build per-frame pixel dictionaries
    marker_px = {}  # marker_id -> (cx, cy) in pixels
    marker_corners_px = {}  # marker_id -> (4,2) corners in pixels

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            marker_id = int(marker_id)
            c4 = corners[i][0]  # (4,2)
            marker_corners_px[marker_id] = c4
            marker_px[marker_id] = marker_pixel_centre(c4)

    # -----------------------------
    # 1) Compute / refresh homography using the 4 board corner markers
    # -----------------------------
    now = time.perf_counter()

    board_ids_present = all(mid in marker_px for mid in BOARD_WORLD_POINTS_M.keys())
    if board_ids_present:
        # Image points: marker centres in pixels, ordered to match world point list
        img_pts = []
        world_pts = []
        for mid, (wx, wy) in BOARD_WORLD_POINTS_M.items():
            cx, cy = marker_px[mid]
            img_pts.append([cx, cy])
            world_pts.append([wx, wy])

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        # Homography from pixel -> world
        H, mask = cv2.findHomography(img_pts, world_pts, method=0)
        if H is not None:
            H_pix_to_world = H
            H_world_to_pix = np.linalg.inv(H_pix_to_world)
            last_H_time = now
            logging.info("Homography updated (pixel->world).")
    else:
        # If we lose board corners briefly, keep the last homography for a short time
        if H_pix_to_world is not None and (now - last_H_time) > H_TIMEOUT_S:
            H_pix_to_world = None
            H_world_to_pix = None
            logging.warning("Homography expired (board corners not visible).")

    # Draw homography grid overlay
    if H_world_to_pix is not None:
        draw_homography_grid(
            out,                 # your BGR output image
            H_world_to_pix,
            BOARD_WIDTH_M,
            BOARD_HEIGHT_M,
            grid_step_m=0.10,     # 10cm grid
            major_every=5         # thicker line every 50cm
        )



    # If user clicked earlier but we had no homography then, convert now once H becomes available
    if have_click and clicked_world is None and H_pix_to_world is not None and clicked_px is not None:
        clicked_world = pix_to_world(H_pix_to_world, clicked_px[0], clicked_px[1])
        logging.info("Converted earlier click -> world(m)=(%.3f, %.3f)", clicked_world[0], clicked_world[1])

    # -----------------------------
    # 2) If we have homography, compute robot pose in world frame (x,y,theta)
    # -----------------------------
    robot_world = None
    robot_theta = None

    if H_pix_to_world is not None and ROBOT_ID in marker_px and ROBOT_ID in marker_corners_px:
        # Robot position: transform marker centre pixel -> world
        rx_px, ry_px = marker_px[ROBOT_ID]
        rx, ry = pix_to_world(H_pix_to_world, rx_px, ry_px)
        robot_world = (rx, ry)

        # Robot orientation: use marker edge direction (corner0 -> corner1), transform to world and compute angle
        c4 = marker_corners_px[ROBOT_ID]
        p0 = c4[0]  # (x,y)
        p1 = c4[1]
        w0 = pix_to_world(H_pix_to_world, float(p0[0]), float(p0[1]))
        w1 = pix_to_world(H_pix_to_world, float(p1[0]), float(p1[1]))

        vx = w1[0] - w0[0]
        vy = w1[1] - w0[1]
        robot_theta = float(np.arctan2(vy, vx))  # radians in world frame

        # Draw robot pose
        cv2.circle(out, (int(rx_px), int(ry_px)), 5, (0, 255, 0), -1)
        cv2.putText(out, f"Robot ({rx:.2f},{ry:.2f})m", (int(rx_px) + 8, int(ry_px) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # -----------------------------
    # 3) Determine current target(s)
    #    - A) Click target (if user clicked)
    #    - B) ArUco target markers (optional)
    # -----------------------------
    # A) Click target
    if H_pix_to_world is not None and clicked_world is not None and H_world_to_pix is not None:
        tx, ty = clicked_world
        tx_px, ty_px = world_to_pix(H_world_to_pix, tx, ty)
        cv2.drawMarker(out, (int(tx_px), int(ty_px)), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        cv2.putText(out, "CLICK TARGET", (int(tx_px) + 10, int(ty_px) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # B) Physical target markers -> world positions
    target_world = {}  # tid -> (x,y)
    if H_pix_to_world is not None:
        for tid in TARGET_IDS:
            if tid in marker_px:
                cx, cy = marker_px[tid]
                target_world[tid] = pix_to_world(H_pix_to_world, cx, cy)

    # -----------------------------
    # 4) Compute distance + bearing
    #    - For click target (primary)
    #    - For each ArUco target marker (optional)
    # -----------------------------
    if robot_world is not None and robot_theta is not None:
        rx, ry = robot_world

        # Primary: click target
        if clicked_world is not None:
            tx, ty = clicked_world
            dx = tx - rx
            dy = ty - ry
            dist_click = float(np.hypot(dx, dy))
            ang_world = float(np.arctan2(dy, dx))
            bear_click = float(wrap_to_pi(ang_world - robot_theta))

            cv2.putText(out, f"Click: d={dist_click:.2f}m b={np.degrees(bear_click):.0f}deg",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Optional: each ArUco target marker (with smoothing)
        for tid, (tx, ty) in target_world.items():
            dx = tx - rx
            dy = ty - ry

            distance = float(np.hypot(dx, dy))
            angle_world = float(np.arctan2(dy, dx))
            bearing = float(wrap_to_pi(angle_world - robot_theta))

            buffers[tid]['dist'].append(distance)
            buffers[tid]['bear'].append(bearing)

            # Smoothed outputs (distance mean, bearing circular mean)
            display_distance = float(np.mean(buffers[tid]['dist'])) if len(buffers[tid]['dist']) > 0 else distance
            display_bearing = circular_mean(buffers[tid]['bear']) if len(buffers[tid]['bear']) > 0 else bearing

            logging.info("Robot->Target %d: distance=%.3f m, bearing=%.2f deg",
                         tid, display_distance, np.degrees(display_bearing))

            # Draw line in pixel space for visualisation
            if tid in marker_px and ROBOT_ID in marker_px:
                xrp, yrp = marker_px[ROBOT_ID]
                xtp, ytp = marker_px[tid]
                cv2.line(out, (int(xrp), int(yrp)), (int(xtp), int(ytp)), (255, 255, 255), 2)
                cv2.putText(out,
                            f"d={display_distance:.2f}m b={np.degrees(display_bearing):.0f}deg",
                            (int(xtp) + 8, int(ytp) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Homography status overlay
    if H_pix_to_world is None:
        cv2.putText(out, "HOMOGRAPHY: NOT AVAILABLE (need 4 board corners)", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    else:
        cv2.putText(out, "HOMOGRAPHY: OK", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Show output
    cv2.imshow(WINDOW_NAME, out)

    end = time.perf_counter()
    fps = 1.0 / (end - start) if end > start else 0.0
    logging.info("FPS: %.1f", fps)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
exit(0)
