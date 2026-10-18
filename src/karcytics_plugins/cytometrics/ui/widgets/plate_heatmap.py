"""A plate heat-map: one circle per well, coloured by a value (viridis)."""

from __future__ import annotations

import math

import numpy as np
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from ...analysis.batch import ROWS
from ..image_canvas import value_color

_MARGIN_LEFT, _MARGIN_TOP, _LEGEND_W = 26, 22, 64


class PlateHeatmap(QWidget):
    well_clicked = Signal(str)  # "A01"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self._grid = np.full((8, 12), np.nan)
        self._label = ""
        self._selected: str | None = None
        self.empty_text = "Run a batch with well names to see the plate"

    def set_data(self, grid: np.ndarray, label: str) -> None:
        self._grid = np.asarray(grid, dtype=float)
        self._label = label
        self.update()

    def set_selected(self, well: str | None) -> None:
        self._selected = well
        self.update()

    @property
    def shape(self) -> tuple[int, int]:
        return self._grid.shape  # type: ignore[return-value]

    def _cell_size(self) -> float:
        rows, cols = self.shape
        w = self.width() - _MARGIN_LEFT - _LEGEND_W - 10
        h = self.height() - _MARGIN_TOP - 10
        return max(4.0, min(w / cols, h / rows))

    def _centre(self, r: int, c: int) -> QPointF:
        size = self._cell_size()
        return QPointF(_MARGIN_LEFT + (c + 0.5) * size, _MARGIN_TOP + (r + 0.5) * size)

    def well_at(self, pos: QPointF) -> str | None:
        size = self._cell_size()
        c = math.floor((pos.x() - _MARGIN_LEFT) / size)
        r = math.floor((pos.y() - _MARGIN_TOP) / size)
        rows, cols = self.shape
        return f"{ROWS[r]}{c + 1:02d}" if 0 <= r < rows and 0 <= c < cols else None

    def _range(self) -> tuple[float, float] | None:
        finite = self._grid[np.isfinite(self._grid)]
        if not finite.size:
            return None
        return float(finite.min()), float(finite.max())

    def mouseReleaseEvent(self, event) -> None:
        well = self.well_at(event.position())
        if well and event.button() == Qt.MouseButton.LeftButton:
            self.well_clicked.emit(well)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event) -> None:
        well = self.well_at(event.position())
        if well:
            r, c = ROWS.index(well[0]), int(well[1:]) - 1
            value = self._grid[r, c]
            text = f"{well}: {value:.4g}" if np.isfinite(value) else f"{well}: no image"
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        super().mouseMoveEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        span = self._range()
        if span is None:
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        lo, hi = span
        rows, cols = self.shape
        size = self._cell_size()
        radius = size * 0.42
        p.setPen(QColor(Colors.FG_SECONDARY))
        for c in range(cols):
            p.drawText(
                QRectF(_MARGIN_LEFT + c * size, 4, size, 16),
                Qt.AlignmentFlag.AlignCenter,
                str(c + 1),
            )
        for r in range(rows):
            p.drawText(
                QRectF(4, _MARGIN_TOP + r * size, _MARGIN_LEFT - 8, size),
                Qt.AlignmentFlag.AlignCenter,
                ROWS[r],
            )
        for r in range(rows):
            for c in range(cols):
                value = self._grid[r, c]
                centre = self._centre(r, c)
                if np.isfinite(value):
                    t = (value - lo) / (hi - lo) if hi > lo else 0.5
                    p.setBrush(QBrush(value_color(t)))
                    p.setPen(Qt.PenStyle.NoPen)
                else:
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.setPen(QPen(QColor(Colors.BORDER), 1))
                if self._selected == f"{ROWS[r]}{c + 1:02d}":
                    p.setPen(QPen(QColor(255, 255, 255), 2.5))
                p.drawEllipse(centre, radius, radius)
        self._paint_legend(p, lo, hi, rows * size)

    def _paint_legend(self, p: QPainter, lo: float, hi: float, height: float) -> None:
        x = self.width() - _LEGEND_W + 8
        bar = QRectF(x, _MARGIN_TOP, 12, max(20.0, height))
        gradient = QLinearGradient(bar.bottomLeft(), bar.topLeft())
        for i in range(11):
            gradient.setColorAt(i / 10, value_color(i / 10))
        p.fillRect(bar, gradient)
        p.setPen(QColor(Colors.FG_SECONDARY))
        p.drawText(QPointF(x + 16, bar.top() + 10), f"{hi:.3g}")
        p.drawText(QPointF(x + 16, bar.bottom()), f"{lo:.3g}")
        p.save()
        p.translate(x - 4, bar.center().y())
        p.rotate(-90)
        p.drawText(
            QRectF(-bar.height() / 2, -14, bar.height(), 14),
            Qt.AlignmentFlag.AlignCenter,
            self._label,
        )
        p.restore()
