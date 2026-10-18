"""Histogram of one measurement, with threshold and range gates.

With the threshold tool, click where the cut-off goes (cells above it are
in the gate; flip it to "below" afterwards in the Gate step). With the range
tool, drag across the values to keep.
"""

from __future__ import annotations

import numpy as np
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ...analysis.gating import Gate
from .plot_axis import Axis

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 46, 14, 10, 38
_MAX_BINS = 64


class Histogram(QWidget):
    gate_drawn = Signal(str, object)  # kind ("threshold" | "range"), params in data units

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setMouseTracking(True)
        self.data: list[float] = []
        self.label = ""
        self.axis = Axis()
        self._counts = np.zeros(0)
        self._edges = np.zeros(0)  # in display space
        self._gates: list[tuple[Gate, bool]] = []
        self.tool: str | None = None
        self._drag_start: float | None = None
        self._hover: float | None = None
        self.empty_text = "No values to plot yet"

    def set_data(self, values, label: str = "") -> None:
        values = np.asarray(values, dtype=float)
        self.data = values[np.isfinite(values)].tolist()
        self.label = label
        self._bin()
        self.update()

    def set_log(self, log: bool) -> None:
        self.axis.log = log
        self._bin()
        self.update()

    def set_gates(self, gates: list[tuple[Gate, bool]]) -> None:
        self._gates = list(gates)
        self.update()

    def set_tool(self, tool: str | None) -> None:
        """None, "threshold" or "range"."""
        self.tool = tool
        self._drag_start = None
        self.setCursor(Qt.CursorShape.SplitHCursor if tool else Qt.CursorShape.ArrowCursor)
        self.update()

    def _bin(self) -> None:
        shown = self.axis.display(self.data)
        shown = shown[np.isfinite(shown)]
        self.axis.fit(self.data)
        if not shown.size:
            self._counts, self._edges = np.zeros(0), np.zeros(0)
            return
        bins = int(np.clip(np.sqrt(shown.size) * 2, 8, _MAX_BINS))
        lo, hi = float(shown.min()), float(shown.max())
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        self._counts, self._edges = np.histogram(shown, bins=bins, range=(lo, hi))

    def _plot_rect(self) -> QRectF:
        r = QRectF(
            _MARGIN_LEFT,
            _MARGIN_TOP,
            max(1.0, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1.0, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM),
        )
        self.axis.start, self.axis.end = r.left(), r.right()
        return r

    # ── Mouse ────────────────────────────────────────────────────────────────

    def _value_at(self, x: float) -> float:
        self._plot_rect()
        return self.axis.from_screen(x)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.tool:
            self._drag_start = event.position().x()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self._hover = event.position().x()
        if self.tool:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            super().mouseReleaseEvent(event)
            return
        start, end = self._drag_start, event.position().x()
        self._drag_start = None
        if self.tool == "threshold":
            self.gate_drawn.emit("threshold", {"value": self._value_at(end), "above": True})
        elif self.tool == "range" and abs(end - start) > 3:  # noqa: PLR2004
            a, b = sorted((self._value_at(start), self._value_at(end)))
            self.gate_drawn.emit("range", {"lo": a, "hi": b})
        self.update()

    # ── Painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        r = self._plot_rect()
        p.setPen(QPen(QColor(Colors.BORDER), 1))
        p.drawRect(r)
        if not self._counts.size:
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        top = max(1, int(self._counts.max()))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(Colors.ACCENT_PRIMARY)))
        for count, left, right in zip(self._counts, self._edges[:-1], self._edges[1:], strict=True):
            x0 = self.axis.display_to_screen(float(left))
            x1 = self.axis.display_to_screen(float(right))
            h = count / top * (r.height() - 4)
            p.drawRect(QRectF(x0 + 0.5, r.bottom() - h, max(1.0, x1 - x0 - 1), h))
        self._paint_axes(p, r, top)
        p.setClipRect(r)
        self._paint_gates(p, r)
        self._paint_drawing(p, r)

    def _paint_axes(self, p: QPainter, r: QRectF, top: int) -> None:
        p.setPen(QColor(Colors.FG_SECONDARY))
        for x, label in self.axis.ticks():
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 3))
            p.drawText(QRectF(x - 40, r.bottom() + 3, 80, 14), Qt.AlignmentFlag.AlignCenter, label)
        p.drawText(QRectF(0, r.top(), _MARGIN_LEFT - 6, 14), Qt.AlignmentFlag.AlignRight, str(top))
        p.drawText(
            QRectF(0, r.bottom() - 14, _MARGIN_LEFT - 6, 14), Qt.AlignmentFlag.AlignRight, "0"
        )
        label = self.label + ("  (log)" if self.axis.log else "")
        p.drawText(
            QRectF(r.left(), r.bottom() + 18, r.width(), 16), Qt.AlignmentFlag.AlignCenter, label
        )

    def _paint_gates(self, p: QPainter, r: QRectF) -> None:
        for gate, highlighted in self._gates:
            color = QColor(gate.color)
            fill = QColor(color)
            fill.setAlpha(45 if highlighted else 22)
            pen = QPen(color, 2.5 if highlighted else 1.5)
            if gate.kind == "threshold":
                x = self.axis.to_screen(float(gate.params.get("value", 0.0)))
                kept = (
                    QRectF(x, r.top(), r.right() - x, r.height())
                    if gate.params.get("above", True)
                    else QRectF(r.left(), r.top(), x - r.left(), r.height())
                )
            elif gate.kind == "range":
                x = self.axis.to_screen(float(gate.params.get("lo", 0.0)))
                x2 = self.axis.to_screen(float(gate.params.get("hi", 0.0)))
                kept = QRectF(x, r.top(), x2 - x, r.height())
            else:
                continue
            p.fillRect(kept.normalized(), fill)
            p.setPen(pen)
            p.drawLine(QPointF(kept.left(), r.top()), QPointF(kept.left(), r.bottom()))
            if gate.kind == "range":
                p.drawLine(QPointF(kept.right(), r.top()), QPointF(kept.right(), r.bottom()))
            p.drawText(QPointF(kept.normalized().left() + 4, r.top() + 14), gate.name)

    def _paint_drawing(self, p: QPainter, r: QRectF) -> None:
        if not self.tool or self._hover is None:
            return
        pen = QPen(QColor(Colors.FG_PRIMARY), 1.5, Qt.PenStyle.DashLine)
        p.setPen(pen)
        if self.tool == "range" and self._drag_start is not None:
            band = QRectF(self._drag_start, r.top(), self._hover - self._drag_start, r.height())
            fill = QColor(Colors.FG_PRIMARY)
            fill.setAlpha(30)
            p.fillRect(band.normalized(), fill)
        p.drawLine(QPointF(self._hover, r.top()), QPointF(self._hover, r.bottom()))
        p.drawText(QPointF(self._hover + 4, r.bottom() - 6), f"{self._value_at(self._hover):.4g}")
