"""
config.py
Central place for configuration constants.

Keep IDs, board dimensions, camera settings, and tuning parameters here instead of
scattered throughout the project.
"""

import cv2.aruco as aruco

# -----------------------------
# Camera / display settings
# -----------------------------
CAM_INDEX = 1
FRAME_W = 500
FRAME_H = 500
WINDOW_NAME = "aruco-image"

# Path to your camera calibration file 
CALIB_PATH = "Calibration.npz"

# -----------------------------
# ArUco settings
# -----------------------------
# Dictionary must match the type of markers printed.
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
ARUCO_PARAMS = aruco.DetectorParameters()

# -----------------------------
# IDs in your arena
# -----------------------------
# Robot marker ID
ROBOT_ID = 0

# If you still want optional physical target markers, you can keep these,
# but the click-to-target system does NOT require them.
TARGET_MARKER_IDS = {11, 12, 13, 14, 15}

# Four fixed “corner markers” for the homography board.
# IMPORTANT: choose IDs not used by robot/targets.
BOARD_CORNER_IDS = {
    "TL": 1,
    "TR": 2,
    "BR": 4,
    "BL": 3,
}

# -----------------------------
# Board geometry (metres)
# -----------------------------
# These are the real-world distances between the *marker centres*, not the paper edges.
# If you define the world points as marker-centres, measure the marker-centre spacing.
BOARD_WIDTH_M = 0.5
BOARD_HEIGHT_M = 0.5

# World coordinates for each corner marker centre (metres).
# Coordinate convention:
#   (0,0) at BL, +x to BR, +y to TL
BOARD_WORLD_POINTS_M = {
    BOARD_CORNER_IDS["BL"]: (0.0, 0.0),
    BOARD_CORNER_IDS["BR"]: (BOARD_WIDTH_M, 0.0),
    BOARD_CORNER_IDS["TR"]: (BOARD_WIDTH_M, BOARD_HEIGHT_M),
    BOARD_CORNER_IDS["TL"]: (0.0, BOARD_HEIGHT_M),
}

# How long we keep using the last homography if the corner markers temporarily disappear
H_TIMEOUT_S = 1.0

# -----------------------------
# Overlay grid settings
# -----------------------------
GRID_STEP_M = 0.10        # 10 cm grid
GRID_MAJOR_EVERY = 5      # thicker line every 5 steps (50 cm)

# -----------------------------
# Pre-processing tuning
# -----------------------------
# adjustable contrast/CLAHE values to help detection under bad lighting.
CONTRAST_ALPHA = 0.7
CONTRAST_BETA = -50
CLAHE_CLIP_LIMIT = 1.3
CLAHE_TILE_GRID = (32, 32)
