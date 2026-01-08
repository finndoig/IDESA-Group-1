from PySide6 import QtCore, QtWidgets, QtGui
import planner  # your existing brute-force planner

from .video_widget import VideoLabel
from .worker import VisionWorker

class MainWindow(QtWidgets.QMainWindow):

    # function to initialize main window
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Robot Control (Qt) - Setup/Targets/Plan")
        self.resize(1100, 700)

        # Central widget layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        main_layout = QtWidgets.QHBoxLayout(central)

        # --- Left controls ---
        left = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left, 0)

        self.btn_home = QtWidgets.QPushButton("Set Home (H)")
        self.btn_target = QtWidgets.QPushButton("Add Target (T)")
        self.btn_undo = QtWidgets.QPushButton("Undo (U)")
        self.btn_clear = QtWidgets.QPushButton("Clear Targets (C)")
        self.btn_plan = QtWidgets.QPushButton("Plan Route (P)")

        for b in [self.btn_home, self.btn_target, self.btn_undo, self.btn_clear, self.btn_plan]:
            b.setMinimumHeight(36)
            left.addWidget(b)

        left.addSpacing(10)

        self.status_label = QtWidgets.QLabel("Status: -")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)

        left.addStretch(1)

        # --- Video in the middle ---
        self.video = VideoLabel()
        main_layout.addWidget(self.video, 1)

        # --- Right panel: targets list ---
        right = QtWidgets.QVBoxLayout()
        main_layout.addLayout(right, 0)

        right.addWidget(QtWidgets.QLabel("Targets (world metres):"))
        self.targets_list = QtWidgets.QListWidget()
        self.targets_list.setMinimumWidth(260)
        right.addWidget(self.targets_list, 1)

        # Worker thread
        self.worker = VisionWorker()
        self.worker.frame_ready.connect(self.video.set_frame)
        self.worker.status_ready.connect(self.on_status)

        # Clicks from video -> worker
        self.video.clicked.connect(self.worker.on_video_click)

        # Buttons -> worker actions (thread safe via slots)
        self.btn_home.clicked.connect(self.worker.set_mode_home)
        self.btn_target.clicked.connect(self.worker.set_mode_target)
        self.btn_undo.clicked.connect(self.worker.undo_target)
        self.btn_clear.clicked.connect(self.worker.clear_targets)

        # Plan route: easiest is to ask the worker for current targets and rewrite order.
        # For simplicity in this first version, we do it by sending a request into the worker via a queued call.
        self.btn_plan.clicked.connect(self.request_plan)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("H"), self, activated=self.worker.set_mode_home)
        QtGui.QShortcut(QtGui.QKeySequence("T"), self, activated=self.worker.set_mode_target)
        QtGui.QShortcut(QtGui.QKeySequence("U"), self, activated=self.worker.undo_target)
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=self.worker.clear_targets)
        QtGui.QShortcut(QtGui.QKeySequence("P"), self, activated=self.request_plan)

        # Start worker
        self.worker.start()

    # function to handle close event
    def closeEvent(self, event):
        # Clean shutdown: stop thread, wait for it
        self.worker.stop()
        self.worker.wait(1500)
        super().closeEvent(event)

    # function to handle status updates
    def on_status(self, s: dict):
        hom = "OK" if s["homography_ok"] else "NO"
        rob = "OK" if s["robot_ok"] else "NO"
        home = "YES" if s["home_set"] else "NO"

        self.status_label.setText(
            f"Homography: {hom}\n"
            f"Robot detected: {rob}\n"
            f"Home set: {home}\n"
            f"Click mode: {s['click_mode']}\n"
            f"Targets: {s['num_targets']}\n"
            f"Next: {s['next_info']}\n"
            f"FPS: {s['fps']:.1f}"
        )

        # Update target list view
        self.targets_list.clear()
        for i, (x, y) in enumerate(s["targets_world"], start=1):
            self.targets_list.addItem(f"{i}: ({x:.3f}, {y:.3f})")

    # function to request route planning
    def request_plan(self):
        """
        Route planning needs access to the worker's current targets and (ideally) a start point.
        Simplest v1 approach: plan order using the first target as start fallback.
        Better approach later: plan from robot pose or home.
        """
        # We can safely schedule a function to run in the worker thread using QMetaObject.invokeMethod.
        QtCore.QMetaObject.invokeMethod(self.worker, "plan_route", QtCore.Qt.QueuedConnection)

    
def run():
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
