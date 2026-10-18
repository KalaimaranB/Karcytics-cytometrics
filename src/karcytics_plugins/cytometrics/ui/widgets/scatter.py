"""Scatter plot of two measurements; click a point to select its cell."""

from __future__ import annotations

import math

import numpy as np
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 52, 12, 10, 34
_RADIUS, _SELECTED_RADIUS = 3.0, 5.5
_PICK_PX = 8.0


class ScatterPlot(QWidget):
    point_clicked = Signal(int, bool)  # cell id, Shift held (add to the selection)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._x = np.empty(0)
        self._y = np.empty(0)
        self._ids: list[int] = []
        self._selected: set[int] = set()
        self._labels = ("", "")
        self._ranges = ((0.0, 1.0), (0.0, 1.0))

    def set_data(self, x, y, ids, x_label: str, y_label: str) -> None:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        self._x, self._y = x[keep], y[keep]
        self._ids = [i for i, k in zip(ids, keep, strict=True) if k]
        self._labels = (x_label, y_label)
        self._ranges = (_padded(self._x), _padded(self._y))
        self.update()

    def set_selected(self, ids) -> None:
        self._selected = set(ids)
        self.update()

    @property
    def point_count(self) -> int:
        return len(self._ids)

    def _plot_rect(self) -> QRectF:
        return QRectF(
            _MARGIN_LEFT,
            _MARGIN_TOP,
            max(1.0, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1.0, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM),
        )

    def _to_screen(self, x: float, y: float) -> QPointF:
        r = self._plot_rect()
        (x0, x1), (y0, y1) = self._ranges
        return QPointF(
            r.left() + (x - x0) / (x1 - x0) * r.width(),
            r.bottom() - (y - y0) / (y1 - y0) * r.height(),
        )

    def point_at(self, pos: QPointF) -> int | None:
        best, best_d = None, _PICK_PX
        for cell_id, x, y in zip(self._ids, self._x, self._y, strict=True):
            p = self._to_screen(x, y)
            d = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if d <= best_d:
                best, best_d = cell_id, d
        return best

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self.point_at(event.position())
            if hit is not None:
                shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                self.point_clicked.emit(hit, shift)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        r = self._plot_rect()
        p.setPen(QPen(QColor(Colors.BORDER), 1))
        p.drawRect(r)
        p.setPen(QColor(Colors.FG_SECONDARY))
        (x0, x1), (y0, y1) = self._ranges
        p.drawText(
            QRectF(r.left(), r.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignLeft, f"{x0:.3g}"
        )
        p.drawText(
            QRectF(r.right() - 80, r.bottom() + 2, 80, 14), Qt.AlignmentFlag.AlignRight, f"{x1:.3g}"
        )
        p.drawText(
            QRectF(0, r.bottom() - 14, _MARGIN_LEFT - 4, 14),
            Qt.AlignmentFlag.AlignRight,
            f"{y0:.3g}",
        )
        p.drawText(
            QRectF(0, r.top(), _MARGIN_LEFT - 4, 14), Qt.AlignmentFlag.AlignRight, f"{y1:.3g}"
        )
        p.drawText(
            QRectF(r.left(), r.bottom() + 16, r.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self._labels[0],
        )
        p.save()
        p.translate(12, r.center().y())
        p.rotate(-90)
        p.drawText(
            QRectF(-r.height() / 2, -8, r.height(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self._labels[1],
        )
        p.restore()
        if not self._ids:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No cells to plot yet")
            return
        normal = QColor(Colors.ACCENT_PRIMARY)
        normal.setAlpha(170)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(normal))
        for cell_id, x, y in zip(self._ids, self._x, self._y, strict=True):
            if cell_id not in self._selected:
                p.drawEllipse(self._to_screen(x, y), _RADIUS, _RADIUS)
        p.setBrush(QBrush(QColor(255, 220, 0)))
        p.setPen(QPen(QColor(Colors.BG_DARKEST), 1))
        for cell_id, x, y in zip(self._ids, self._x, self._y, strict=True):
            if cell_id in self._selected:
                p.drawEllipse(self._to_screen(x, y), _SELECTED_RADIUS, _SELECTED_RADIUS)


def _padded(values: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return 0.0, 1.0
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * 0.05 or max(abs(lo) * 0.05, 1.0)
    return lo - pad, hi + pad
