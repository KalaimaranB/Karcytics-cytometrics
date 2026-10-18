"""Animated "scanning" bar with a percentage, shown while a run is in progress."""

from __future__ import annotations

import math

from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget,
)

_MIN_LINE_POINTS = 2  # need at least 2 points to draw a line segment
_GAUSSIAN_SPIKE_CUTOFF = 25  # exp(-25) ~= 1.4e-11, negligible past this point


class ScanningIndicator(QWidget):
    """Animated bio-themed loading indicator.
    Draws a scrolling ECG-style sine wave \u2014 looks like a live cell-signal scan.
    Fully replaces QProgressBar: call setVisible(True/False) to show / hide.
    """

    _BG = QColor(Colors.BG_DARKEST)
    _GLOW = QColor(Colors.ACCENT_PRIMARY)
    _DIM = QColor(Colors.BG_MEDIUM)
    _GRID = QColor(Colors.BORDER)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._phase = 0.0  # scrolling phase (radians)
        self._visible = False
        self._value = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ---- Public API (mirrors QProgressBar) --------------------------------
    def setValue(self, v):
        """Show `v` percent next to the animation (0 hides the number)."""
        self._value = max(0, min(100, int(v)))
        self.update()

    def value(self) -> int:
        return self._value

    def setVisible(self, visible: bool):
        super().setVisible(visible)
        if visible:
            self._timer.start(30)  # ~33 fps
        else:
            self._timer.stop()

    # ---- Internal ----------------------------------------------------------
    def _advance(self):
        self._phase = (self._phase + 0.18) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        import math as _math

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        pad = 6

        # Background
        p.fillRect(0, 0, W, H, self._BG)

        # Subtle horizontal grid line at mid-height
        p.setPen(QPen(self._GRID, 1))
        p.drawLine(pad, H // 2, W - pad, H // 2)

        # Draw two overlapping waves: a dim wide one + a bright sharp one
        cy = H / 2
        amp = (H / 2) - pad - 1
        steps = W - 2 * pad

        def _wave(x_shift, color, thick, alpha_fn):
            if steps < _MIN_LINE_POINTS:
                return
            prev_x, prev_y = None, None
            for i in range(steps + 1):
                t = i / steps  # 0 \u2192 1 across widget
                # ECG-style: gentle sine with a sharper spike near centre
                angle = t * 4 * _math.pi + self._phase + x_shift
                base = _math.sin(angle)
                # Add a narrow Gaussian spike near t=0.5 to mimic a QRS complex
                spike_t = ((t - 0.5) / 0.06) ** 2
                spike = 3.0 * _math.exp(-spike_t) if spike_t < _GAUSSIAN_SPIKE_CUTOFF else 0
                y_val = base + spike
                y_val = max(-1.6, min(1.6, y_val))  # clamp

                x = pad + i
                y = int(cy - amp * y_val / 1.6)

                # alpha fades from 40% at left to 100% at right
                a = int(80 + 175 * t)
                c = QColor(color)
                c.setAlpha(min(255, a))
                pen = QPen(c, thick)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)

                if prev_x is not None:
                    p.drawLine(prev_x, prev_y, x, y)
                prev_x, prev_y = x, y

        _wave(0, self._DIM, 4, None)  # trailing wide glow
        _wave(0, self._GLOW, 2, None)  # sharp bright line

        # Small "ANALYSING..." label on the right
        p.setPen(QColor(Colors.ACCENT_PRIMARY))
        f = QFont()
        f.setPointSize(7)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(f)
        p.drawText(
            W - 120,
            0,
            114,
            H,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"SCANNING\u2026 {self._value}%" if self._value else "SCANNING\u2026",
        )

        p.end()
