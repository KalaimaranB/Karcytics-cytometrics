"""Conditions side by side: each replicate (or image) as a dot, the mean as a bar with error bars."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from .plot_axis import Axis

_MARGIN_LEFT, _MARGIN_RIGHT, _MARGIN_TOP, _MARGIN_BOTTOM = 58, 14, 12, 42


class ComparisonPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._summary = pd.DataFrame()
        self._points = pd.DataFrame()
        self._by = "Condition"
        self._value = ""
        self._label = ""
        self._error = "SEM"
        self.axis = Axis()
        self.empty_text = "Run a batch with conditions to compare them"
        self.band: tuple[float, float] | None = None  # a shaded range of values (e.g. countable)
        self.band_label = ""

    def set_data(
        self,
        summary: pd.DataFrame,
        points: pd.DataFrame,
        by: str,
        value: str,
        label: str,
        error: str = "SEM",
    ) -> None:
        """`summary` from ``condition_summary``; `points` from ``condition_points``."""
        self._summary, self._points = summary, points
        self._by, self._value, self._label, self._error = by, value, label, error
        values = list(points[value]) if value in points else []
        if len(summary):
            values += list(summary["Mean"] + summary[error].fillna(0))
            values += list(summary["Mean"] - summary[error].fillna(0))
        self.axis.fit([0.0, *[v for v in values if not math.isnan(v)]])
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(Colors.BG_DARKEST))
        r = QRectF(
            _MARGIN_LEFT,
            _MARGIN_TOP,
            max(1.0, self.width() - _MARGIN_LEFT - _MARGIN_RIGHT),
            max(1.0, self.height() - _MARGIN_TOP - _MARGIN_BOTTOM),
        )
        p.setPen(QPen(QColor(Colors.BORDER), 1))
        p.drawRect(r)
        if not len(self._summary):
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.empty_text)
            return
        self.axis.start, self.axis.end = r.bottom(), r.top()
        p.setPen(QColor(Colors.FG_SECONDARY))
        for y, text in self.axis.ticks():
            p.drawLine(QPointF(r.left() - 3, y), QPointF(r.left(), y))
            p.drawText(
                QRectF(0, y - 7, _MARGIN_LEFT - 6, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        p.save()
        p.translate(11, r.center().y())
        p.rotate(-90)
        p.drawText(
            QRectF(-r.height() / 2, -8, r.height(), 16), Qt.AlignmentFlag.AlignCenter, self._label
        )
        p.restore()

        if self.band is not None:
            y0, y1 = (self.axis.to_screen(v) for v in self.band)
            top, bottom = max(min(y0, y1), r.top()), min(max(y0, y1), r.bottom())
            if bottom > top:
                shade = QColor(Colors.ACCENT_PRIMARY)
                shade.setAlpha(25)
                p.fillRect(QRectF(r.left(), top, r.width(), bottom - top), shade)
                p.setPen(QColor(Colors.FG_SECONDARY))
                p.drawText(
                    QRectF(r.left() + 4, top + 2, r.width() - 8, 14),
                    Qt.AlignmentFlag.AlignRight,
                    self.band_label,
                )
        groups = list(self._summary[self._by])
        slot = r.width() / max(1, len(groups))
        zero = self.axis.to_screen(0.0)
        accent = QColor(Colors.ACCENT_PRIMARY)
        rng = np.random.default_rng(0)  # same jitter every repaint
        for i, row in enumerate(self._summary.to_dict("records")):
            cx = r.left() + (i + 0.5) * slot
            mean = row["Mean"]
            if math.isfinite(mean):
                fill = QColor(accent)
                fill.setAlpha(70)
                top = self.axis.to_screen(mean)
                p.fillRect(
                    QRectF(cx - slot * 0.3, min(top, zero), slot * 0.6, abs(zero - top)), fill
                )
                err = row.get(self._error)
                if err is not None and math.isfinite(err):
                    p.setPen(QPen(QColor(Colors.FG_PRIMARY), 1.5))
                    y0, y1 = self.axis.to_screen(mean - err), self.axis.to_screen(mean + err)
                    p.drawLine(QPointF(cx, y0), QPointF(cx, y1))
                    for y in (y0, y1):
                        p.drawLine(QPointF(cx - 6, y), QPointF(cx + 6, y))
            mine = self._points[self._points[self._by] == row[self._by]][self._value]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(accent))
            for v in mine:
                if math.isfinite(v):
                    jitter = float(rng.uniform(-0.15, 0.15)) * slot
                    p.drawEllipse(QPointF(cx + jitter, self.axis.to_screen(v)), 3.5, 3.5)
            p.setPen(QColor(Colors.FG_SECONDARY))
            p.drawText(
                QRectF(cx - slot / 2, r.bottom() + 4, slot, 16),
                Qt.AlignmentFlag.AlignCenter,
                f"{row[self._by]}",
            )
            p.drawText(
                QRectF(cx - slot / 2, r.bottom() + 20, slot, 16),
                Qt.AlignmentFlag.AlignCenter,
                f"n = {int(row['n'])}",
            )
