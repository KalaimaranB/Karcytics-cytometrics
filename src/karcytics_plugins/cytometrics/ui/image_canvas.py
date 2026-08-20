"""Custom hardware-accelerated image canvas for multi-channel TIFFs."""

from karcytics_sdk.plugin import PrimaryButton
from karcytics_sdk.plugin.theme_fallback import Colors
from PyQt6.QtCore import QLineF, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_MIN_POLYGON_POINTS = 3  # a closed polygon needs at least a triangle

#: Extensions accepted for drag-and-drop, matching ChannelManagerWidget's file dialog filter.
_SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class CellPolygonItem(QGraphicsPolygonItem):
    """Custom polygon that knows its own ID and holds a text label."""

    def __init__(self, cell_id, polygon):
        super().__init__(polygon)
        self.cell_id = cell_id

        self.set_highlighted(False)  # Use our new method for default styling

        # Add Cell ID Text
        self.text_item = QGraphicsSimpleTextItem(str(cell_id), self)
        self.text_item.setBrush(QBrush(Qt.GlobalColor.white))
        font = QFont("Arial", 10, QFont.Weight.Bold)
        self.text_item.setFont(font)

        center = polygon.boundingRect().center()
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(
            center.x() - text_rect.width() / 2, center.y() - text_rect.height() / 2
        )

    def set_highlighted(self, is_highlighted: bool):
        """Swaps the styling between default and highlighted states."""
        if is_highlighted:
            pen = QPen(QColor(255, 255, 0))  # Bold Yellow
            pen.setWidth(3)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 255, 0, 100)))  # Brighter yellow fill
            self.setZValue(1)  # Pop to the front so it's not hidden by overlapping cells
        else:
            pen = QPen(QColor(255, 0, 255))  # Standard Magenta
            pen.setWidth(2)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 0, 255, 40)))  # Transparent Magenta
            self.setZValue(0)


class MultiChannelCanvas(QGraphicsView):
    calibration_line_drawn = pyqtSignal(float)
    cell_drawn = pyqtSignal(list)
    cell_deleted = pyqtSignal(int)  # <-- NEW PHASE 3 SIGNAL
    load_requested = pyqtSignal()  # user clicked the empty-state "Load Image" button
    files_dropped = pyqtSignal(list)  # user dropped one or more image files onto the canvas

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(500)

        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; border: none;")
        self.setRenderHint(self.renderHints().Antialiasing)
        self.setAcceptDrops(True)

        self._image_item = None
        self._user_zoomed = False
        self._empty_state = self._build_empty_state()

        self.mode = "PAN"
        self._calib_start = None
        self._calib_line_item = None

        self._drawing_points = []
        self._drawing_item = None
        self._cell_items = []

    def _build_empty_state(self) -> QWidget:
        """Centered overlay shown when no image is loaded: a CTA button plus a drop hint."""
        overlay = QWidget(self.viewport())
        layout = QVBoxLayout(overlay)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel("🔬")
        icon.setStyleSheet("font-size: 40px; border: none;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        self._empty_state_title = QLabel("Load an image to begin")
        self._empty_state_title.setStyleSheet(
            f"color: {Colors.FG_PRIMARY}; font-size: 15px; font-weight: bold; border: none;"
        )
        self._empty_state_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_state_title)

        self._btn_load = PrimaryButton("➕ Load Image")
        self._btn_load.clicked.connect(self.load_requested.emit)
        layout.addWidget(self._btn_load)

        self._empty_state_hint = QLabel("or drag & drop an image file here")
        self._empty_state_hint.setStyleSheet(
            f"color: {Colors.FG_SECONDARY}; font-size: 12px; border: none;"
        )
        self._empty_state_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._empty_state_hint)

        overlay.adjustSize()
        return overlay

    def _apply_theme_styles(self) -> None:
        """Re-applies theme-aware styles when the active theme changes.

        The canvas background and the empty-state labels below bake
        ``Colors.*`` into their stylesheets at construction time, so they
        need an explicit refresh — unlike ``self._btn_load`` (an SDK
        ``PrimaryButton``), which already re-styles itself.
        """
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; border: none;")
        self._empty_state_title.setStyleSheet(
            f"color: {Colors.FG_PRIMARY}; font-size: 15px; font-weight: bold; border: none;"
        )
        self._empty_state_hint.setStyleSheet(
            f"color: {Colors.FG_SECONDARY}; font-size: 12px; border: none;"
        )

    def _center_empty_state(self):
        size = self._empty_state.sizeHint()
        x = (self.viewport().width() - size.width()) // 2
        y = (self.viewport().height() - size.height()) // 2
        self._empty_state.setGeometry(max(0, x), max(0, y), size.width(), size.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._center_empty_state()
        if self._image_item is not None and not self._user_zoomed:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def load_pixmap(self, pixmap: QPixmap):
        if self._image_item:
            self._scene.removeItem(self._image_item)
        self._empty_state.hide()
        self._image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._image_item)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._user_zoomed = False
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.set_mode("PAN")

    def _has_supported_extension(self, path: str) -> bool:
        return path.lower().endswith(_SUPPORTED_EXTENSIONS)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(self._has_supported_extension(url.toLocalFile()) for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if self._has_supported_extension(url.toLocalFile())
        ]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_mode(self, mode_str: str):
        self.mode = mode_str
        if self.mode == "PAN":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)  # type: ignore[union-attr]
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)  # type: ignore[union-attr]

    def draw_cells_from_state(self, cells_data: list):
        for item in self._cell_items:
            if item.scene():
                self._scene.removeItem(item)
        self._cell_items.clear()

        for cell in cells_data:
            poly = QPolygonF([QPointF(x, y) for x, y in cell["points"]])
            item = CellPolygonItem(cell["id"], poly)  # <-- Uses our new custom object
            self._scene.addItem(item)
            self._cell_items.append(item)

    # ── MOUSE EVENTS ──
    def mousePressEvent(self, event):
        pos = self.mapToScene(event.pos())

        # --- PHASE 3: RIGHT CLICK DELETE ---
        if event.button() == Qt.MouseButton.RightButton:
            clicked_item = self.itemAt(event.pos())  # Let Qt handle the collision math

            # If they clicked the ID number, grab the parent polygon
            if isinstance(clicked_item, QGraphicsSimpleTextItem) and isinstance(
                clicked_item.parentItem(), CellPolygonItem
            ):
                self.cell_deleted.emit(clicked_item.parentItem().cell_id)
            # If they clicked the polygon directly
            elif isinstance(clicked_item, CellPolygonItem):
                self.cell_deleted.emit(clicked_item.cell_id)
            return

        if self.mode == "CALIBRATE" and event.button() == Qt.MouseButton.LeftButton:
            self._calib_start = pos
            if self._calib_line_item:
                self._scene.removeItem(self._calib_line_item)
            self._calib_line_item = QGraphicsLineItem(QLineF(pos, pos))
            self._calib_line_item.setPen(QPen(Qt.GlobalColor.yellow, 3))
            self._scene.addItem(self._calib_line_item)

        elif self.mode == "DRAW" and event.button() == Qt.MouseButton.LeftButton:
            self._drawing_points = [pos]
            self._drawing_item = QGraphicsPolygonItem()
            self._drawing_item.setPen(QPen(QColor(255, 255, 0), 2))
            self._scene.addItem(self._drawing_item)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = self.mapToScene(event.pos())
        if self.mode == "CALIBRATE" and self._calib_start is not None:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                if abs(pos.x() - self._calib_start.x()) > abs(pos.y() - self._calib_start.y()):
                    pos.setY(self._calib_start.y())
                else:
                    pos.setX(self._calib_start.x())
            self._calib_line_item.setLine(QLineF(self._calib_start, pos))

        elif self.mode == "DRAW" and self._drawing_points:
            self._drawing_points.append(pos)
            self._drawing_item.setPolygon(QPolygonF(self._drawing_points))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "CALIBRATE" and event.button() == Qt.MouseButton.LeftButton:
            if self._calib_line_item:
                self.calibration_line_drawn.emit(self._calib_line_item.line().length())
            self._calib_start = None
            self.set_mode("PAN")

        elif self.mode == "DRAW" and event.button() == Qt.MouseButton.LeftButton:
            if len(self._drawing_points) >= _MIN_POLYGON_POINTS:
                points_list = [(p.x(), p.y()) for p in self._drawing_points]
                self.cell_drawn.emit(points_list)

            if self._drawing_item and self._drawing_item.scene():
                self._scene.removeItem(self._drawing_item)
            self._drawing_points = []
            self._drawing_item = None
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self._user_zoomed = True
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
        else:
            self.scale(0.85, 0.85)

    def set_show_ids(self, show: bool):
        """Toggles the visibility of all cell ID numbers on the canvas."""
        for item in self._cell_items:
            # text_item is a child of CellPolygonItem
            if hasattr(item, "text_item"):
                item.text_item.setVisible(show)

    def highlight_cell(self, target_id):
        """Highlights a specific cell ID. Pass None to clear all highlights."""
        for item in self._cell_items:
            if isinstance(item, CellPolygonItem):
                item.set_highlighted(item.cell_id == target_id)

    def cleanup(self) -> None:
        """Release UI resources. Called when the plugin panel is closed."""
        if hasattr(self, "_scene"):
            self._scene.clear()
        self._cell_items.clear()
        self._drawing_points.clear()
        self._calib_line_item = None
        self._empty_state = None
        self._image_item = None
