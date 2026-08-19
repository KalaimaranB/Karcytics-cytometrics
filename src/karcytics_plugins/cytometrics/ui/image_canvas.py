"""Custom hardware-accelerated image canvas for multi-channel TIFFs."""

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
    QSizePolicy,
)


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(500)

        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; border: none;")
        self.setRenderHint(self.renderHints().Antialiasing)

        self._placeholder = self.scene.addText("Load an image to begin CytoMetrics.")
        self._placeholder.setDefaultTextColor(QColor(Colors.FG_SECONDARY))
        self._image_item = None

        self.mode = "PAN"
        self._calib_start = None
        self._calib_line_item = None

        self._drawing_points = []
        self._drawing_item = None
        self._cell_items = []

    def load_pixmap(self, pixmap: QPixmap):
        if self._image_item:
            self.scene.removeItem(self._image_item)
        self._placeholder.hide()
        self._image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self._image_item)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.set_mode("PAN")

    def set_mode(self, mode_str: str):
        self.mode = mode_str
        if self.mode == "PAN":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def draw_cells_from_state(self, cells_data: list):
        for item in self._cell_items:
            if item.scene():
                self.scene.removeItem(item)
        self._cell_items.clear()

        for cell in cells_data:
            poly = QPolygonF([QPointF(x, y) for x, y in cell["points"]])
            item = CellPolygonItem(cell["id"], poly)  # <-- Uses our new custom object
            self.scene.addItem(item)
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
                self.scene.removeItem(self._calib_line_item)
            self._calib_line_item = QGraphicsLineItem(QLineF(pos, pos))
            self._calib_line_item.setPen(QPen(Qt.GlobalColor.yellow, 3))
            self.scene.addItem(self._calib_line_item)

        elif self.mode == "DRAW" and event.button() == Qt.MouseButton.LeftButton:
            self._drawing_points = [pos]
            self._drawing_item = QGraphicsPolygonItem()
            self._drawing_item.setPen(QPen(QColor(255, 255, 0), 2))
            self.scene.addItem(self._drawing_item)
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
            if len(self._drawing_points) > 2:
                points_list = [(p.x(), p.y()) for p in self._drawing_points]
                self.cell_drawn.emit(points_list)

            if self._drawing_item and self._drawing_item.scene():
                self.scene.removeItem(self._drawing_item)
            self._drawing_points = []
            self._drawing_item = None
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
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
        if hasattr(self, "scene"):
            self.scene.clear()
        self._cell_items.clear()
        self._drawing_points.clear()
        self._calib_line_item = None
        self._placeholder = None
        self._image_item = None
