import time
import numpy as np
import cv2
from PySide6 import QtCore, QtGui

import config
import planner
from vision import VisionAruco
from homography import HomographyBoard
from targets import TargetManager, ClickMode
import overlay


class VisionWorker(QtCore.QThread):
    """
    Grabs camera frames, runs your existing pipeline, and emits:
    - a QImage for display
    - a status dict for UI labels/lists
    """
    frame_ready = QtCore.Signal(QtGui.QImage)
    status_ready = QtCore.Signal(dict)

    # function to initialize vision worker
    def __init__(self, parent=None):
        super().__init__(parent)

        self._running = True

        # Core modules (reuse your refactor)
        self.vision = VisionAruco(config.ARUCO_DICT, config.ARUCO_PARAMS)
        self.board = HomographyBoard(config.BOARD_WORLD_POINTS_M, timeout_s=config.H_TIMEOUT_S)
        self.targets = TargetManager()

        # Click requests from UI arrive here (thread-safe via signal/slot)
        self._pending_click = None  # (x_px, y_px)

        # Camera handle created in run()
        self.cap = None

    # function to handle video click
    @QtCore.Slot(int, int)
    def on_video_click(self, x, y):
        """
        Receive clicks from GUI (image pixel coords).
        Worker decides what to do with them based on click_mode + homography.
        """
        self._pending_click = (float(x), float(y))

    # function to set click mode to home
    @QtCore.Slot()
    def set_mode_home(self):
        self.targets.set_click_mode(ClickMode.SET_HOME)

    # function to set click mode to target
    @QtCore.Slot()
    def set_mode_target(self):
        self.targets.set_click_mode(ClickMode.ADD_TARGET)

    # function to clear all targets
    @QtCore.Slot()
    def clear_targets(self):
        self.targets.clear_targets()

    # function to undo last target
    @QtCore.Slot()
    def undo_target(self):
        self.targets.undo_last_target()

    # function to plan route
    @QtCore.Slot()
    def plan_route(self):
        # Guard brute-force size
        if len(self.targets.targets_world) > 9:
            return

        if not self.targets.targets_world:
            return

        # Prefer starting from home if set (common for your use case)
        start_xy = self.targets.home_world if self.targets.home_world is not None else self.targets.targets_world[0]
        end_xy = self.targets.home_world if self.targets.home_world is not None else None

        self.targets.targets_world = planner.plan_optimal_order(
            targets=list(self.targets.targets_world),
            start=start_xy,
            end=end_xy
        )


    # function to stop the worker
    def stop(self):
        self._running = False

    # function to preprocess grayscale image
    def _preprocess_gray(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=config.CONTRAST_ALPHA, beta=config.CONTRAST_BETA)
        clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID)
        return clahe.apply(gray)

    # function to get robot pose from marker
    def _robot_pose_from_marker(self, marker_px, marker_corners_px):
        if not self.board.valid():
            return None, None, None
        if config.ROBOT_ID not in marker_px or config.ROBOT_ID not in marker_corners_px:
            return None, None, None

        rx_px, ry_px = marker_px[config.ROBOT_ID]
        rx_w, ry_w = self.board.pix_to_world(rx_px, ry_px)

        c4 = marker_corners_px[config.ROBOT_ID]
        p0 = c4[0]
        p1 = c4[1]
        w0 = self.board.pix_to_world(float(p0[0]), float(p0[1]))
        w1 = self.board.pix_to_world(float(p1[0]), float(p1[1]))

        vx = w1[0] - w0[0]
        vy = w1[1] - w0[1]
        theta = float(np.arctan2(vy, vx))  # world angle

        return (rx_w, ry_w), (rx_px, ry_px), theta

    # function to wrap angle to pi
    def _wrap_to_pi(self, a):
        return (a + np.pi) % (2*np.pi) - np.pi

    # function to run the vision worker
    def run(self):
        # Open camera in this thread
        self.cap = cv2.VideoCapture(config.CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)

        last_time = time.perf_counter()

        while self._running:
            t0 = time.perf_counter()

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            gray = self._preprocess_gray(frame)

            det = self.vision.detect(gray)

            # Update homography (keeps last for timeout)
            self.board.update(det["marker_px"])

            # Apply any pending click (if any)
            if self._pending_click is not None:
                x, y = self._pending_click
                self._pending_click = None

                if not self.board.valid():
                    # store for later
                    self.targets.set_pending_click((x, y), self.targets.click_mode)
                else:
                    world = self.board.pix_to_world(x, y)
                    if self.targets.click_mode == ClickMode.SET_HOME:
                        self.targets.set_home_world(world)
                        self.targets.set_click_mode(ClickMode.ADD_TARGET)
                    else:
                        self.targets.add_target_world(world)

            # Resolve pending clicks once homography becomes valid
            if self.board.valid() and self.targets.pending_click_px is not None:
                px = self.targets.pending_click_px
                mode = self.targets.pending_click_mode
                world = self.board.pix_to_world(px[0], px[1])
                if mode == ClickMode.SET_HOME:
                    self.targets.set_home_world(world)
                    self.targets.set_click_mode(ClickMode.ADD_TARGET)
                else:
                    self.targets.add_target_world(world)
                self.targets.clear_pending_click()

            # Output image for drawing
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Draw markers (visual only)
            if det["ids"] is not None and len(det["ids"]) > 0:
                out = cv2.aruco.drawDetectedMarkers(out, det["corners"], det["ids"])

            # Draw homography grid if valid
            if self.board.valid():
                overlay.draw_homography_grid(
                    out,
                    self.board.H_world_to_pix,
                    config.BOARD_WIDTH_M,
                    config.BOARD_HEIGHT_M,
                    grid_step_m=config.GRID_STEP_M,
                    major_every=config.GRID_MAJOR_EVERY
                )

            # Robot pose
            robot_world, robot_px, robot_theta = self._robot_pose_from_marker(det["marker_px"], det["marker_corners_px"])
            
            # capture/remove targets when robot is within the ball radius ---
            removed_count = 0
            if robot_world is not None and self.targets.targets_world:
                rx, ry = robot_world
                thresh = config.ROBOT_CAPTURE_RADIUS_M + config.CAPTURE_TOL_M

                before = len(self.targets.targets_world)
                self.targets.targets_world = [
                    (tx, ty) for (tx, ty) in self.targets.targets_world
                    if np.hypot(tx - rx, ty - ry) > thresh
                ]
                removed_count = before - len(self.targets.targets_world)

            # draw robot arrow/heading
            overlay.draw_robot_pose(out, robot_px, robot_theta)

            # draw the 200mm capture ring in world space ---
            if self.board.valid() and robot_world is not None:
                overlay.draw_world_circle(
                    out,
                    self.board.H_world_to_pix,
                    centre_world=robot_world,
                    radius_m=config.ROBOT_CAPTURE_RADIUS_M,
                    colour=(0, 255, 255),
                    thickness=2
                )
            



            # Convert targets/home to pixels for drawing
            home_pix = None
            if self.board.valid() and self.targets.home_world is not None:
                home_pix = self.board.world_to_pix(self.targets.home_world[0], self.targets.home_world[1])

            targets_pix = []
            if self.board.valid():
                for (tx, ty) in self.targets.targets_world:
                    targets_pix.append(self.board.world_to_pix(tx, ty))

            overlay.draw_targets(out, targets_pix, home_pix=home_pix)

            # Route lines (robot -> targets -> home)
            route_points = []
            if robot_px is not None:
                route_points.append(robot_px)
            route_points.extend(targets_pix)
            if home_pix is not None:
                route_points.append(home_pix)

            # You added this earlier; keep it if present
            if hasattr(overlay, "draw_route_lines"):
                overlay.draw_route_lines(out, route_points, thickness=2)

            # Status
            t1 = time.perf_counter()
            fps = 1.0 / (t1 - last_time) if (t1 - last_time) > 1e-6 else 0.0
            last_time = t1

            next_info = "-"
            if robot_world is not None and robot_theta is not None and self.targets.targets_world:
                rx, ry = robot_world
                tx, ty = self.targets.targets_world[0]
                dx, dy = tx - rx, ty - ry
                dist = float(np.hypot(dx, dy))
                ang = float(np.arctan2(dy, dx))
                bear = float(self._wrap_to_pi(ang - robot_theta))
                next_info = f"d={dist:.2f}m b={np.degrees(bear):.0f}deg"

            status = {
                "homography_ok": self.board.valid(),
                "robot_ok": robot_world is not None,
                "fps": fps,
                "num_targets": len(self.targets.targets_world),
                "click_mode": self.targets.click_mode.name,
                "next_info": next_info,
                "home_set": self.targets.home_world is not None,
                "targets_world": list(self.targets.targets_world),
                "removed_count": removed_count,
            }
            self.status_ready.emit(status)

            # Convert BGR -> RGB -> QImage
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

            # IMPORTANT: copy so memory remains valid after this loop iteration
            self.frame_ready.emit(qimg.copy())

            # Light throttle to avoid pegging CPU if needed
            elapsed = time.perf_counter() - t0
            if elapsed < 0.005:
                time.sleep(0.001)

        if self.cap is not None:
            self.cap.release()
