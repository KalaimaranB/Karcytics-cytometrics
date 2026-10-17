"""CytoMetrics canvas: the SDK ImageView plus cell outlines and drawing.

Zoom, pan, fit, the empty state, drag-and-drop, the calibration line and
the scale bar all come from `karcytics_sdk.plugin.ImageView`. This adds
cell polygons with ID labels (the "cells" layer), right-click delete, and a
freehand tool for drawing a missed cell.
"""

from karcytics_sdk.plugin import ImageView
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QGraphicsPolygonItem, QGraphicsSimpleTextItem, QSizePolicy

_MIN_POLYGON_POINTS = 3  # a closed polygon needs at least a triangle
_CELLS_LAYER = "cells"

#: Extensions accepted for drag-and-drop, matching ChannelManagerWidget's file dialog filter.
_SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class CellPolygonItem(QGraphicsPolygonItem):
    """Custom polygon that knows its own ID and holds a text label."""

    def __init__(self, cell_id, polygon):
        super().__init__(polygon)
        self.cell_id = cell_id

        self.set_highlighted(False)

        self.text_item = QGraphicsSimpleTextItem(str(cell_id), self)
        self.text_item.setBrush(QBrush(Qt.GlobalColor.white))
        self.text_item.setFont(QFont("Arial", 10, QFont.Weight.Bold))

        center = polygon.boundingRect().center()
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(
            center.x() - text_rect.width() / 2, center.y() - text_rect.height() / 2
        )

    def set_highlighted(self, is_highlighted: bool):
        """Swaps the styling between default and highlighted states."""
        if is_highlighted:
            pen = QPen(QColor(255, 255, 0))
            pen.setWidth(3)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 255, 0, 100)))
            self.setZValue(1)  # in front of overlapping neighbours
        else:
            pen = QPen(QColor(255, 0, 255))
            pen.setWidth(2)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 0, 255, 40)))
            self.setZValue(0)


class MultiChannelCanvas(ImageView):
    cell_drawn = Signal(list)
    cell_deleted = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(500)
        self.set_empty_state(
            "Load an image to begin",
            icon="🔬",
            button_text="➕ Load Image",
            hint="or drag & drop an image file here",
        )
        self.accept_dropped_files(_SUPPORTED_EXTENSIONS)
        self.calibration_line_drawn.connect(self._on_calibration_finished)

        self.mode = "PAN"
        self._drawing_points: list[QPointF] = []
        self._drawing_item: QGraphicsPolygonItem | None = None
        self._cell_items: list[CellPolygonItem] = []
        self.set_mode("PAN")

    def load_pixmap(self, pixmap: QPixmap):
        self.set_pixmap(pixmap)
        self.set_mode("PAN")

    def set_mode(self, mode_str: str):
        """PAN (left-drag pans), CALIBRATE (draw one scale line) or DRAW (outline a cell)."""
        self.mode = mode_str
        if mode_str == "CALIBRATE":
            self.start_calibration()
            return
        if self.is_calibrating:
            self.cancel_calibration()
        self.set_left_drag_pans(mode_str == "PAN")
        if mode_str == "PAN":
            self.viewport().unsetCursor()
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _on_calibration_finished(self, _length: float) -> None:
        self.set_mode("PAN")

    def draw_cells_from_state(self, cells_data: list):
        self.clear_layer(_CELLS_LAYER)
        self._cell_items.clear()
        for cell in cells_data:
            poly = QPolygonF([QPointF(x, y) for x, y in cell["points"]])
            item = CellPolygonItem(cell["id"], poly)
            self.add_to_layer(_CELLS_LAYER, item)
            self._cell_items.append(item)

    # ── Mouse: right-click delete and the DRAW tool; the rest is ImageView's ──

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            clicked = self.itemAt(event.position().toPoint())
            if isinstance(clicked, QGraphicsSimpleTextItem) and isinstance(
                clicked.parentItem(), CellPolygonItem
            ):
                clicked = clicked.parentItem()
            if isinstance(clicked, CellPolygonItem):
                self.cell_deleted.emit(clicked.cell_id)
            return

        if self.mode == "DRAW" and event.button() == Qt.MouseButton.LeftButton:
            self._drawing_points = [self.mapToScene(event.position().toPoint())]
            self._drawing_item = QGraphicsPolygonItem()
            self._drawing_item.setPen(QPen(QColor(255, 255, 0), 2))
            self.scene().addItem(self._drawing_item)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "DRAW" and self._drawing_points and self._drawing_item is not None:
            self._drawing_points.append(self.mapToScene(event.position().toPoint()))
            self._drawing_item.setPolygon(QPolygonF(self._drawing_points))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.mode == "DRAW" and event.button() == Qt.MouseButton.LeftButton:
            if len(self._drawing_points) >= _MIN_POLYGON_POINTS:
                self.cell_drawn.emit([(p.x(), p.y()) for p in self._drawing_points])
            if self._drawing_item is not None and self._drawing_item.scene():
                self.scene().removeItem(self._drawing_item)
            self._drawing_points = []
            self._drawing_item = None
            return
        super().mouseReleaseEvent(event)

    def set_show_ids(self, show: bool):
        """Toggles the visibility of all cell ID numbers on the canvas."""
        for item in self._cell_items:
            item.text_item.setVisible(show)

    def highlight_cell(self, target_id):
        """Highlights a specific cell ID. Pass None to clear all highlights."""
        for item in self._cell_items:
            item.set_highlighted(item.cell_id == target_id)

    def cleanup(self) -> None:
        """Release UI resources. Called when the plugin panel is closed."""
        self._cell_items.clear()
        self._drawing_points.clear()
        self._drawing_item = None
        super().cleanup()
