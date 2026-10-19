"""CytoMetrics canvas: the SDK ImageView plus cell outlines and drawing.

Zoom, pan, fit, the empty state, drag-and-drop, the calibration line and
the scale bar all come from `karcytics_sdk.plugin.ImageView`. This adds
cell polygons with ID labels (the "cells" layer), right-click delete, and a
freehand tool for drawing a missed cell.
"""

import numpy as np
from karcytics_sdk.plugin import ImageView
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QSizePolicy,
)

_MIN_POLYGON_POINTS = 3  # a closed polygon needs at least a triangle
_CELLS_LAYER = "cells"
_PREVIEW_LAYER = "preview"
_OVERLAY_LAYER = "assay"
_DRAG_TOOLS = ("SPLIT", "BRUSH_ADD", "BRUSH_ERASE", "BOX")
_CLICK_SLOP_PX = 4  # a press and release closer than this is a click, not a drag
_PROBABILITY_LAYER = "probability"  # below the outlines
_DEFAULT_OUTLINE = QColor(255, 0, 255)
_PROBABILITY_MAX_ALPHA = 170

#: Extensions accepted for drag-and-drop, matching ChannelManagerWidget's file dialog filter.
_SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


class CellPolygonItem(QGraphicsPolygonItem):
    """Custom polygon that knows its own ID and holds a text label."""

    def __init__(self, cell_id, polygon, color: QColor = _DEFAULT_OUTLINE, flagged: bool = False):
        super().__init__(polygon)
        self.cell_id = cell_id
        self.color = QColor(color)
        self.flagged = flagged

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
            # White, so it stands out whatever the outlines are coloured by.
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(4)
            self.setPen(pen)
            self.setBrush(QBrush(QColor(255, 255, 255, 70)))
            self.setZValue(1)  # in front of overlapping neighbours
        else:
            pen = QPen(self.color)
            pen.setWidth(2)
            if self.flagged:
                pen.setStyle(Qt.PenStyle.DashLine)
            self.setPen(pen)
            fill = QColor(self.color)
            fill.setAlpha(40)
            self.setBrush(QBrush(fill))
            self.setZValue(0)


class MultiChannelCanvas(ImageView):
    """Modes: PAN (drag pans, click selects; Shift+click adds), CALIBRATE, DRAW (outline a
    cell), SPLIT (draw a cut), BRUSH_ADD / BRUSH_ERASE (paint the selected cell) and BOX
    (drag a region).
    """

    cell_drawn = Signal(list)
    cell_deleted = Signal(int)
    selection_changed = Signal(object)  # sorted list of selected cell ids
    split_drawn = Signal(object)  # ((x0, y0), (x1, y1)) in image pixels
    brush_painted = Signal(object)  # (points, radius, add)
    box_drawn = Signal(object)  # (x0, y0, x1, y1) in image pixels

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumWidth(500)
        self.set_empty_state("Image will appear here once loaded")
        self.accept_dropped_files(_SUPPORTED_EXTENSIONS)
        self.calibration_line_drawn.connect(self._on_calibration_finished)

        self.mode = "PAN"
        self.brush_radius = 4
        self._drawing_points: list[QPointF] = []
        self._drawing_item: QGraphicsPolygonItem | None = None
        self._rubber: QGraphicsLineItem | QGraphicsRectItem | QGraphicsPathItem | None = None
        self._press_pos: QPointF | None = None
        self._cell_items: list[CellPolygonItem] = []
        self._selection: set[int] = set()
        self.set_mode("PAN")

    def load_pixmap(self, pixmap: QPixmap):
        self.set_pixmap(pixmap)
        self.set_mode("PAN")

    def set_mode(self, mode_str: str):
        """Switch tool; see the class docstring for the modes."""
        self._clear_rubber()
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

    def draw_cells_from_state(self, cells_data: list, colors: dict[int, QColor] | None = None):
        """Outline every cell; `colors` maps cell id → outline colour (default magenta)."""
        self.clear_layer(_CELLS_LAYER)
        self._cell_items.clear()
        colors = colors or {}
        for cell in cells_data:
            poly = QPolygonF([QPointF(x, y) for x, y in cell["points"]])
            item = CellPolygonItem(
                cell["id"],
                poly,
                colors.get(cell["id"], _DEFAULT_OUTLINE),
                bool(cell.get("flagged")),
            )
            self.add_to_layer(_CELLS_LAYER, item)
            self._cell_items.append(item)
        present = {c["id"] for c in cells_data}
        self._selection &= present
        for item in self._cell_items:
            item.set_highlighted(item.cell_id in self._selection)

    def set_preview(self, cells: list[dict] | None) -> None:
        """Dashed outlines of a preview run (not part of the analysis); None clears them."""
        self.clear_layer(_PREVIEW_LAYER)
        for cell in cells or []:
            item = QGraphicsPolygonItem(QPolygonF([QPointF(x, y) for x, y in cell["points"]]))
            pen = QPen(QColor(0, 230, 255), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            item.setPen(pen)
            self.add_to_layer(_PREVIEW_LAYER, item)
        self.layer(_PREVIEW_LAYER).setZValue(2)

    def set_overlays(self, overlays) -> None:
        """Draw an assay's shapes (``analysis.assays.Overlay``) over the image."""
        self.clear_layer(_OVERLAY_LAYER)
        for o in overlays:
            pen = QPen(
                QColor(o.color), 2, Qt.PenStyle.DashLine if o.dashed else Qt.PenStyle.SolidLine
            )
            pen.setCosmetic(True)
            item: QGraphicsEllipseItem | QGraphicsRectItem | QGraphicsLineItem
            if o.kind == "circle":
                x, y, r = o.coords
                item = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            elif o.kind == "rect":
                x0, y0, x1, y1 = o.coords
                item = QGraphicsRectItem(QRectF(QPointF(x0, y0), QPointF(x1, y1)).normalized())
            else:
                item = QGraphicsLineItem(*o.coords)
            item.setPen(pen)
            self.add_to_layer(_OVERLAY_LAYER, item)
            if o.label:
                text = QGraphicsSimpleTextItem(o.label)
                text.setBrush(QColor(o.color))
                text.setFlag(QGraphicsSimpleTextItem.GraphicsItemFlag.ItemIgnoresTransformations)
                text.setPos(item.boundingRect().topLeft())
                self.add_to_layer(_OVERLAY_LAYER, text)
        self.layer(_OVERLAY_LAYER).setZValue(3)

    @property
    def selection(self) -> list[int]:
        return sorted(self._selection)

    def set_selection(self, cell_ids, emit: bool = False) -> None:
        self._selection = set(cell_ids)
        for item in self._cell_items:
            item.set_highlighted(item.cell_id in self._selection)
        if emit:
            self.selection_changed.emit(self.selection)

    def set_probability_map(self, probability: np.ndarray | None) -> None:
        """Overlay a 0–1 probability map as a translucent heat map (None removes it)."""
        self.clear_layer(_PROBABILITY_LAYER)
        if probability is None:
            return
        self.layer(_PROBABILITY_LAYER, z=-0.5)
        self.add_to_layer(_PROBABILITY_LAYER, QGraphicsPixmapItem(_heatmap_pixmap(probability)))

    def center_on_cell(self, cell_id) -> None:
        for item in self._cell_items:
            if item.cell_id == cell_id:
                self.centerOn(item.sceneBoundingRect().center())
                return

    # ── Mouse: tools; panning and calibration are ImageView's ───────────────

    def _cell_at(self, event) -> CellPolygonItem | None:
        clicked = self.itemAt(event.position().toPoint())
        if isinstance(clicked, QGraphicsSimpleTextItem) and isinstance(
            clicked.parentItem(), CellPolygonItem
        ):
            clicked = clicked.parentItem()
        return clicked if isinstance(clicked, CellPolygonItem) else None

    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            cell = self._cell_at(event)
            if cell is not None:
                self.cell_deleted.emit(cell.cell_id)
            return
        left = event.button() == Qt.MouseButton.LeftButton
        if left and self.mode == "DRAW":
            self._drawing_points = [self._scene_pos(event)]
            self._drawing_item = QGraphicsPolygonItem()
            self._drawing_item.setPen(QPen(QColor(255, 255, 0), 2))
            self.scene().addItem(self._drawing_item)
            return
        if left and self.mode in _DRAG_TOOLS:
            self._start_rubber(self._scene_pos(event))
            return
        if left and self.mode == "PAN":
            self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.mode == "DRAW" and self._drawing_points and self._drawing_item is not None:
            self._drawing_points.append(self._scene_pos(event))
            self._drawing_item.setPolygon(QPolygonF(self._drawing_points))
            return
        if self.mode in _DRAG_TOOLS and self._rubber is not None:
            self._move_rubber(self._scene_pos(event))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        left = event.button() == Qt.MouseButton.LeftButton
        if left and self.mode == "DRAW":
            if len(self._drawing_points) >= _MIN_POLYGON_POINTS:
                self.cell_drawn.emit([(p.x(), p.y()) for p in self._drawing_points])
            if self._drawing_item is not None and self._drawing_item.scene():
                self.scene().removeItem(self._drawing_item)
            self._drawing_points = []
            self._drawing_item = None
            return
        if left and self.mode in _DRAG_TOOLS and self._rubber is not None:
            self._finish_rubber(self._scene_pos(event))
            return
        if left and self.mode == "PAN" and self._press_pos is not None:
            moved = (event.position() - self._press_pos).manhattanLength()
            self._press_pos = None
            if moved < _CLICK_SLOP_PX:
                self._click_select(event)
        super().mouseReleaseEvent(event)

    def _click_select(self, event) -> None:
        cell = self._cell_at(event)
        add = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if cell is None:
            selection = self._selection if add else set()
        elif add:
            selection = self._selection ^ {cell.cell_id}
        else:
            selection = {cell.cell_id}
        self.set_selection(selection, emit=True)

    # Rubber-band shapes for SPLIT (line), BOX (rectangle) and BRUSH (stroke).

    def _start_rubber(self, pos: QPointF) -> None:
        self._clear_rubber()
        self._drawing_points = [pos]
        pen = QPen(QColor(255, 220, 0), 2)
        pen.setCosmetic(True)
        if self.mode == "SPLIT":
            self._rubber = QGraphicsLineItem(QLineF(pos, pos))
        elif self.mode == "BOX":
            self._rubber = QGraphicsRectItem(QRectF(pos, pos))
            pen.setStyle(Qt.PenStyle.DashLine)
        else:
            colour = (
                QColor(0, 255, 120, 110) if self.mode == "BRUSH_ADD" else QColor(255, 60, 60, 110)
            )
            pen = QPen(colour, 2 * self.brush_radius)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            path = QPainterPath(pos)
            path.lineTo(pos)
            self._rubber = QGraphicsPathItem(path)
        self._rubber.setPen(pen)
        self._rubber.setZValue(1000)
        self.scene().addItem(self._rubber)

    def _move_rubber(self, pos: QPointF) -> None:
        start = self._drawing_points[0]
        if isinstance(self._rubber, QGraphicsLineItem):
            self._rubber.setLine(QLineF(start, pos))
        elif isinstance(self._rubber, QGraphicsRectItem):
            self._rubber.setRect(QRectF(start, pos).normalized())
        elif isinstance(self._rubber, QGraphicsPathItem):
            self._drawing_points.append(pos)
            path = self._rubber.path()
            path.lineTo(pos)
            self._rubber.setPath(path)

    def _finish_rubber(self, pos: QPointF) -> None:
        start = self._drawing_points[0]
        if self.mode == "SPLIT":
            self.split_drawn.emit(((start.x(), start.y()), (pos.x(), pos.y())))
        elif self.mode == "BOX":
            rect = QRectF(start, pos).normalized()
            if rect.width() >= _CLICK_SLOP_PX and rect.height() >= _CLICK_SLOP_PX:
                self.box_drawn.emit((rect.left(), rect.top(), rect.right(), rect.bottom()))
        else:
            points = [(p.x(), p.y()) for p in [*self._drawing_points, pos]]
            self.brush_painted.emit((points, self.brush_radius, self.mode == "BRUSH_ADD"))
        self._clear_rubber()

    def _clear_rubber(self) -> None:
        if self._rubber is not None and self._rubber.scene():
            self.scene().removeItem(self._rubber)
        self._rubber = None
        if self.mode in _DRAG_TOOLS:
            self._drawing_points = []

    def visible_region(self) -> tuple[float, float, float, float]:
        """The part of the image in view, as (x0, y0, x1, y1) in image pixels."""
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        return rect.left(), rect.top(), rect.right(), rect.bottom()

    def set_show_ids(self, show: bool):
        """Toggles the visibility of all cell ID numbers on the canvas."""
        for item in self._cell_items:
            item.text_item.setVisible(show)

    def highlight_cell(self, target_id):
        """Highlight one cell (None clears), e.g. from a table row."""
        self.set_selection(set() if target_id is None else {target_id})

    def cleanup(self) -> None:
        """Release UI resources. Called when the plugin panel is closed."""
        self._cell_items.clear()
        self._drawing_points.clear()
        self._drawing_item = None
        self._rubber = None
        super().cleanup()

    def set_outlines_visible(self, visible: bool) -> None:
        self.set_layer_visible(_CELLS_LAYER, visible)


# Viridis (van der Walt & Smith, matplotlib's default colormap; CC0), 5 stops.
_VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]


def value_color(t: float) -> QColor:
    """Viridis colour for `t` in 0–1 (dark purple → yellow), readable on dark images."""
    t = min(1.0, max(0.0, t)) * (len(_VIRIDIS) - 1)
    i = min(int(t), len(_VIRIDIS) - 2)
    f = t - i
    a, b = _VIRIDIS[i], _VIRIDIS[i + 1]
    return QColor(*(int(round(x + (y - x) * f)) for x, y in zip(a, b, strict=True)))


def confidence_color(confidence: float) -> QColor:
    """Red (≤ 0.5) through amber to green (1.0)."""
    t = min(1.0, max(0.0, (confidence - 0.5) / 0.5))
    if t < 0.5:  # noqa: PLR2004
        return QColor(255, int(80 + 340 * t), 40)
    return QColor(int(255 - 400 * (t - 0.5)), 250, 60)


def _heatmap_pixmap(probability: np.ndarray) -> QPixmap:
    """Cyan, alpha proportional to probability."""
    p = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    rgba = np.zeros((*p.shape, 4), dtype=np.uint8)
    rgba[..., 1] = 220
    rgba[..., 2] = 255
    rgba[..., 3] = (p * _PROBABILITY_MAX_ALPHA).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    h, w = p.shape
    image = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())
