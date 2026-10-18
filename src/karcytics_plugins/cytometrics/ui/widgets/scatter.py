"""Scatter plot of two measurements: pick cells, see populations, draw gates.

Click a point to select its cell (Shift adds). With a gate tool on, drag a
rectangle or click the corners of a polygon (double-click, or click the
first corner, to close it; Esc cancels). Gates come back in data units, with
the log scale they were drawn on (see ``analysis.gating``).
"""

from __future__ import annotations

import math

import numpy as np
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from ...analysis.gating import Gate
from .plot_axis import Axis

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 58, 14, 10, 38
_RADIUS, _SELECTED_RADIUS = 3.0, 5.5
_PICK_PX = 8.0
_CLOSE_PX = 10.0  # clicking this close to a polygon's first corner closes it


class ScatterPlot(QWidget):
    point_clicked = Signal(int, bool)  # cell id, Shift held (add to the selection)
    gate_drawn = Signal(str, object)  # kind ("rect" | "polygon"), params in data units

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._ids: list[int] = []
        self._colors: dict[int, str] = {}
        self._selected: set[int] = set()
        self._labels = ("", "")
        self.columns: tuple[str, ...] = ("", "")
        self.x_axis, self.y_axis = Axis(), Axis()
        self._gates: list[tuple[Gate, bool]] = []  # (Gate, highlighted)
        self.tool: str | None = None
        self._drag_start: QPointF | None = None
        self._drag_now: QPointF | None = None
        self._vertices: list[QPointF] = []
        self.empty_text = "No cells to plot yet"

    # ── Data ─────────────────────────────────────────────────────────────────

    def set_data(
        self, x, y, ids, x_label: str, y_label: str, colors=None, columns=("", "")
    ) -> None:
        """`columns` names the measurements on each axis, to place gates drawn on them."""
        self.columns = tuple(columns)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        self._x, self._y = x[keep], y[keep]
        self._ids = [int(i) for i, k in zip(ids, keep, strict=True) if k]
        self._labels = (x_label, y_label)
        self._colors = dict(colors or {})
        self._fit()
        self.update()

    def set_log(self, x_log: bool, y_log: bool) -> None:
        self.x_axis.log, self.y_axis.log = x_log, y_log
        self._fit()
        self.update()

    def set_selected(self, ids) -> None:
        self._selected = set(ids)
        self.update()

    def set_tool(self, tool: str | None) -> None:
        """None (select cells), "rect" or "polygon"."""
        self.tool = tool
        self._cancel_drawing()
        self.setCursor(Qt.CursorShape.CrossCursor if tool else Qt.CursorShape.ArrowCursor)

    @property
    def point_count(self) -> int:
        return len(self._ids)

    def _fit(self) -> None:
        self.x_axis.fit(self._x)
        self.y_axis.fit(self._y)

    # ── Geometry ─────────────────────────────────────────────────────────────

    def _plot_rect(self) -> QRectF:
        return QRectF(
            _MARGIN_LEFT,
            _MARGIN_TOP,
            max(1.0, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1.0, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM),
        )

    def _layout_axes(self) -> QRectF:
        r = self._plot_rect()
        self.x_axis.start, self.x_axis.end = r.left(), r.right()
        self.y_axis.start, self.y_axis.end = r.bottom(), r.top()
        return r

    def _to_screen(self, x: float, y: float) -> QPointF:
        self._layout_axes()
        return QPointF(self.x_axis.to_screen(x), self.y_axis.to_screen(y))

    def _to_data(self, p: QPointF) -> tuple[float, float]:
        self._layout_axes()
        return self.x_axis.from_screen(p.x()), self.y_axis.from_screen(p.y())

    def point_at(self, pos: QPointF) -> int | None:
        if not self._ids:
            return None
        self._layout_axes()
        sx = self.x_axis.to_screen_many(self._x)
        sy = self.y_axis.to_screen_many(self._y)
        d = np.hypot(sx - pos.x(), sy - pos.y())
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmin(d))
        return self._ids[i] if d[i] <= _PICK_PX else None

    # ── Mouse ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self.tool:
            self._cancel_drawing()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        pos = event.position()
        if self.tool == "rect":
            self._drag_start = self._drag_now = pos
        elif self.tool == "polygon":
            if len(self._vertices) >= 3 and _near(pos, self._vertices[0]):  # noqa: PLR2004
                self._finish_polygon()
            else:
                self._vertices.append(pos)
            self.update()

    def mouseMoveEvent(self, event) -> None:
        self._drag_now = event.position()
        if self.tool:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self.tool == "rect" and self._drag_start is not None:
            start, end = self._drag_start, event.position()
            self._drag_start = None
            if abs(start.x() - end.x()) > 3 and abs(start.y() - end.y()) > 3:  # noqa: PLR2004
                (x0, y0), (x1, y1) = self._to_data(start), self._to_data(end)
                self.gate_drawn.emit(
                    "rect",
                    {"x0": min(x0, x1), "x1": max(x0, x1), "y0": min(y0, y1), "y1": max(y0, y1)},
                )
            self.update()
            return
        if self.tool is None:
            hit = self.point_at(event.position())
            if hit is not None:
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self.point_clicked.emit(hit, shift)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool == "polygon" and len(self._vertices) >= 3:  # noqa: PLR2004
            self._finish_polygon()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self.tool:
            self._cancel_drawing()
            return
        super().keyPressEvent(event)

    def _finish_polygon(self) -> None:
        points = [list(self._to_data(v)) for v in self._vertices]
        self._vertices = []
        self.gate_drawn.emit("polygon", {"points": points})
        self.update()

    def _cancel_drawing(self) -> None:
        self._drag_start = None
        self._vertices = []
        self.update()

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        r = self._layout_axes()
        self._paint_axes(p, r)
        if not self._ids:
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty_text)
        else:
            p.setClipRect(r)
            self._paint_points(p)
        p.setClipRect(r)
        self._paint_gates(p)
        self._paint_drawing(p)

    def _paint_axes(self, p: QPainter, r: QRectF) -> None:
        p.setPen(QPen(QColor(Colors.BORDER), 1))
        p.drawRect(r)
        p.setPen(QColor(Colors.FG_SECONDARY))
        for x, label in self.x_axis.ticks():
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 3))
            p.drawText(QRectF(x - 40, r.bottom() + 3, 80, 14), Qt.AlignmentFlag.AlignCenter, label)
        for y, label in self.y_axis.ticks():
            p.drawLine(QPointF(r.left() - 3, y), QPointF(r.left(), y))
            p.drawText(
                QRectF(0, y - 7, _MARGIN_LEFT - 6, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
        x_label = self._labels[0] + ("  (log)" if self.x_axis.log else "")
        y_label = self._labels[1] + ("  (log)" if self.y_axis.log else "")
        p.drawText(
            QRectF(r.left(), r.bottom() + 18, r.width(), 16), Qt.AlignmentFlag.AlignCenter, x_label
        )
        p.save()
        p.translate(11, r.center().y())
        p.rotate(-90)
        p.drawText(
            QRectF(-r.height() / 2, -8, r.height(), 16), Qt.AlignmentFlag.AlignCenter, y_label
        )
        p.restore()

    def _paint_points(self, p: QPainter) -> None:
        sx = self.x_axis.to_screen_many(self._x)
        sy = self.y_axis.to_screen_many(self._y)
        default = QColor(Colors.ACCENT_PRIMARY)
        brushes: dict[str, QBrush] = {}
        p.setPen(Qt.PenStyle.NoPen)
        selected = []
        for cell_id, x, y in zip(self._ids, sx, sy, strict=True):
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if cell_id in self._selected:
                selected.append((x, y))
                continue
            hex_color = self._colors.get(cell_id, "")
            if hex_color not in brushes:
                color = QColor(hex_color) if hex_color else QColor(default)
                color.setAlpha(190)
                brushes[hex_color] = QBrush(color)
            p.setBrush(brushes[hex_color])
            p.drawEllipse(QPointF(x, y), _RADIUS, _RADIUS)
        p.setBrush(QBrush(QColor(255, 255, 255)))
        p.setPen(QPen(QColor(Colors.BG_DARKEST), 1))
        for x, y in selected:
            p.drawEllipse(QPointF(x, y), _SELECTED_RADIUS, _SELECTED_RADIUS)

    def set_gates(self, gates: list[tuple[Gate, bool]]) -> None:
        """Gates to draw: ``(gate, highlighted)``. Rectangles and polygons on these two
        measurements are outlined; thresholds and ranges on either one are drawn as lines.
        """
        self._gates = list(gates)
        self.update()

    def _paint_gates(self, p: QPainter) -> None:
        r = self._plot_rect()
        cx, cy = self.columns
        for gate, highlighted in self._gates:
            color = QColor(gate.color)
            fill = QColor(color)
            fill.setAlpha(40 if highlighted else 18)
            p.setPen(QPen(color, 2.5 if highlighted else 1.5))
            p.setBrush(QBrush(fill))
            if gate.kind in ("rect", "polygon") and (gate.x, gate.y) == (cx, cy):
                anchor = self._paint_shape(p, gate)
            elif gate.kind in ("threshold", "range") and gate.x in (cx, cy):
                anchor = self._paint_band(p, gate, r, vertical=gate.x == cx)
            else:
                continue
            p.setPen(color)
            p.drawText(anchor + QPointF(4, 14), gate.name)

    def _paint_shape(self, p: QPainter, gate: Gate) -> QPointF:
        params = gate.params
        if gate.kind == "rect":
            a = self._to_screen(params["x0"], params["y0"])
            b = self._to_screen(params["x1"], params["y1"])
            shape = QRectF(a, b).normalized()
            p.drawRect(shape)
            return shape.topLeft()
        poly = QPolygonF([self._to_screen(x, y) for x, y in params.get("points", [])])
        p.drawPolygon(poly)
        return poly.boundingRect().topLeft()

    def _paint_band(self, p: QPainter, gate: Gate, r: QRectF, vertical: bool) -> QPointF:
        """A threshold or range on one axis: its line(s), and the kept side shaded."""
        axis = self.x_axis if vertical else self.y_axis
        params = gate.params
        if gate.kind == "threshold":
            at = axis.to_screen(float(params.get("value", 0.0)))
            above = params.get("above", True)
            lo, hi = (at, axis.end) if above else (axis.start, at)
            edges = [at]
        else:
            lo = axis.to_screen(float(params.get("lo", 0.0)))
            hi = axis.to_screen(float(params.get("hi", 0.0)))
            edges = [lo, hi]
        if vertical:
            band = QRectF(QPointF(lo, r.top()), QPointF(hi, r.bottom())).normalized()
            p.fillRect(band, p.brush())
            for e in edges:
                p.drawLine(QPointF(e, r.top()), QPointF(e, r.bottom()))
        else:
            band = QRectF(QPointF(r.left(), lo), QPointF(r.right(), hi)).normalized()
            p.fillRect(band, p.brush())
            for e in edges:
                p.drawLine(QPointF(r.left(), e), QPointF(r.right(), e))
        return band.topLeft()

    def _paint_drawing(self, p: QPainter) -> None:
        pen = QPen(QColor(Colors.FG_PRIMARY), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.tool == "rect" and self._drag_start is not None and self._drag_now is not None:
            p.drawRect(QRectF(self._drag_start, self._drag_now).normalized())
        if self.tool == "polygon" and self._vertices:
            points = list(self._vertices)
            if self._drag_now is not None:
                points.append(self._drag_now)
            p.drawPolyline(QPolygonF(points))
            p.setBrush(QBrush(QColor(Colors.FG_PRIMARY)))
            p.drawEllipse(self._vertices[0], 3.5, 3.5)


def _near(a: QPointF, b: QPointF) -> bool:
    return math.hypot(a.x() - b.x(), a.y() - b.y()) <= _CLOSE_PX
