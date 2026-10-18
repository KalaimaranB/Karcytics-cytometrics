"""A small native histogram with axis labels."""

from __future__ import annotations

import math

from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter
from PySide6.QtWidgets import (
    QWidget,
)


class SimpleHistogram(QWidget):
    """A lightweight, native Qt histogram with axis labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.data = []
        self.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; border: 1px solid {Colors.BORDER}; border-radius: 4px;"
        )

    def update_data(self, values):
        self.data = [v for v in values if not math.isnan(v)]
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Margins: left (for Y labels), right, top, bottom (for X labels)
        m_left, m_right, m_top, m_bottom = 35, 15, 15, 25
        w = self.width() - m_left - m_right
        h = self.height() - m_top - m_bottom

        bins = 15
        min_val, max_val = min(self.data), max(self.data)
        if min_val == max_val:
            max_val += 1

        counts = [0] * bins
        for val in self.data:
            idx = int((val - min_val) / (max_val - min_val) * (bins - 1))
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        bin_w = w / bins

        # 1. Draw the bars
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(Colors.ACCENT_PRIMARY)))
        for i, count in enumerate(counts):
            bin_h = (count / max_count) * h
            x = m_left + i * bin_w
            y = m_top + h - bin_h
            painter.drawRect(int(x), int(y), int(bin_w - 2), int(bin_h))

        # 2. Draw the Labels
        painter.setPen(QColor(Colors.FG_SECONDARY))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # Y-Axis (Max and Min counts)
        painter.drawText(5, m_top + 10, f"{max_count}")
        painter.drawText(5, m_top + h, "0")

        # X-Axis (Min and Max Areas)
        painter.drawText(m_left, m_top + h + 18, f"{min_val:.3g}")

        max_str = f"{max_val:.3g}"
        text_width = painter.fontMetrics().horizontalAdvance(max_str)
        painter.drawText(m_left + w - text_width, m_top + h + 18, max_str)
