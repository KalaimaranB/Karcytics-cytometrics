"""Several distributions on one axis: each a % histogram outline, with labelled cut-off lines."""

from __future__ import annotations

import math

import numpy as np
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from .plot_axis import Axis

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 48, 14, 10, 38
_LEGEND_ROW = 16
_BINS = 60


class OverlayHistogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.x_axis = Axis()
        self.y_axis = Axis()
        self.label = ""
        self._series: list[tuple[str, np.ndarray, str]] = []  # name, % per bin, colour
        self._edges = np.zeros(0)  # in display space
        self._lines: list[tuple[float, str]] = []
        self.empty_text = "Add images to the experiment to compare them"

    def set_data(self, series, lines=(), label: str = "", log: bool = False) -> None:
        """`series`: ``experiment.Series``; `lines`: (value, label) drawn as vertical lines."""
        self.label = label
        self.x_axis.log = log
        self._lines = [(float(v), t) for v, t in lines]
        shown = [self.x_axis.display(s.values) for s in series]
        everything = np.concatenate([v[np.isfinite(v)] for v in shown]) if shown else np.zeros(0)
        self._series = []
        if not everything.size:
            self.update()
            return
        lo, hi = np.percentile(everything, [0.5, 99.5])
        for value, _text in self._lines:
            d = float(self.x_axis.display([value])[0])
            if math.isfinite(d):
                lo, hi = min(lo, d), max(hi, d)
        if hi <= lo:
            hi = lo + 1
        self._edges = np.linspace(lo, hi, _BINS + 1)
        peak = 0.0
        for s, values in zip(series, shown, strict=True):
            finite = values[np.isfinite(values)]
            counts, _ = np.histogram(np.clip(finite, lo, hi), self._edges)
            share = 100 * counts / max(1, finite.size)
            peak = max(peak, float(share.max()) if share.size else 0.0)
            self._series.append((s.name, share, s.color))
        self.x_axis.lo, self.x_axis.hi = float(lo), float(hi)
        self.y_axis.lo, self.y_axis.hi = 0.0, peak * 1.08 or 1.0
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        legend = _LEGEND_ROW * min(len(self._series), 6) if len(self._series) > 1 else 0
        r = QRectF(
            _MARGIN_LEFT,
            _MARGIN_TOP + legend,
            max(1.0, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1.0, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM - legend),
        )
        p.setPen(QPen(QColor(Colors.BORDER), 1))
        p.drawRect(r)
        if not self._series:
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.empty_text)
            return
        self.x_axis.start, self.x_axis.end = r.left(), r.right()
        self.y_axis.start, self.y_axis.end = r.bottom(), r.top()
        self._paint_axes(p, r)
        xs = [self.x_axis.display_to_screen(e) for e in self._edges]
        for i, (name, share, color) in enumerate(self._series):
            path = QPainterPath(QPointF(xs[0], r.bottom()))
            for k, value in enumerate(share):
                y = self.y_axis.to_screen(float(value))
                path.lineTo(xs[k], y)
                path.lineTo(xs[k + 1], y)
            path.lineTo(xs[-1], r.bottom())
            fill = QColor(color)
            fill.setAlpha(45)
            p.fillPath(path, fill)
            p.setPen(QPen(QColor(color), 1.6))
            p.drawPath(path)
            if legend and i < 6:  # noqa: PLR2004
                y = _MARGIN_TOP + i * _LEGEND_ROW
                p.fillRect(QRectF(r.left(), y + 4, 14, 8), QColor(color))
                p.setPen(QColor(Colors.FG_PRIMARY))
                p.drawText(
                    QRectF(r.left() + 20, y, r.width() - 20, _LEGEND_ROW),
                    Qt.AlignmentFlag.AlignVCenter,
                    name,
                )
        for value, text in self._lines:
            x = self.x_axis.to_screen(value)
            if not math.isfinite(x):
                continue
            p.setPen(QPen(QColor(Colors.FG_PRIMARY), 1.2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawText(QRectF(x + 3, r.top() + 2, 120, 14), Qt.AlignmentFlag.AlignLeft, text)

    def _paint_axes(self, p: QPainter, r: QRectF) -> None:
        p.setPen(QColor(Colors.FG_SECONDARY))
        for x, text in self.x_axis.ticks():
            p.drawLine(QPointF(x, r.bottom()), QPointF(x, r.bottom() + 3))
            p.drawText(QRectF(x - 30, r.bottom() + 4, 60, 14), Qt.AlignmentFlag.AlignCenter, text)
        for y, text in self.y_axis.ticks(4):
            p.drawLine(QPointF(r.left() - 3, y), QPointF(r.left(), y))
            p.drawText(
                QRectF(0, y - 7, _MARGIN_LEFT - 6, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{text}%",
            )
        p.drawText(
            QRectF(r.left(), r.bottom() + 20, r.width(), 16),
            Qt.AlignmentFlag.AlignCenter,
            self.label,
        )
