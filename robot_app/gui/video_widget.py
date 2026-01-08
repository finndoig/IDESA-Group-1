from PySide6 import QtCore, QtGui, QtWidgets

class VideoLabel(QtWidgets.QLabel):
    """
    QLabel that displays a video frame and emits clicked pixel coordinates
    in IMAGE space (not widget space), accounting for scaling/letterboxing.
    """
    clicked = QtCore.Signal(int, int)

    # function to initialize video label
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background: #111; color: #ddd;")

        self._img_w = None
        self._img_h = None
        self._last_qpix = None

    # function to set video frame
    def set_frame(self, qimg: QtGui.QImage):
        """Display a QImage and remember its native size for click mapping."""
        self._img_w = qimg.width()
        self._img_h = qimg.height()

        qpix = QtGui.QPixmap.fromImage(qimg)
        self._last_qpix = qpix

        # Scale to fit while keeping aspect ratio (letterboxing possible)
        scaled = qpix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)

    # function to handle resize event
    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale the last frame on resize
        if self._last_qpix is not None:
            scaled = self._last_qpix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(scaled)

    # function to handle mouse press event
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() != QtCore.Qt.LeftButton:
            return

        if self._img_w is None or self._img_h is None or self.pixmap() is None:
            return

        # Map click from widget coords -> image coords, accounting for letterboxing
        widget_w = self.width()
        widget_h = self.height()

        # Displayed pixmap size (after aspect scaling)
        pix = self.pixmap()
        disp_w = pix.width()
        disp_h = pix.height()

        # Offsets where the pixmap is drawn inside the label
        offset_x = (widget_w - disp_w) / 2.0
        offset_y = (widget_h - disp_h) / 2.0

        # Click in widget coordinates
        xw = event.position().x()
        yw = event.position().y()

        # Check if click was inside the displayed image area
        if xw < offset_x or xw > offset_x + disp_w or yw < offset_y or yw > offset_y + disp_h:
            return

        # Normalised within displayed pixmap
        xn = (xw - offset_x) / disp_w
        yn = (yw - offset_y) / disp_h

        # Convert to original image pixel coordinates
        xi = int(xn * self._img_w)
        yi = int(yn * self._img_h)

        self.clicked.emit(xi, yi)
