"""Live CPU / RAM / GPU sparkline shown under the Detect controls."""

from __future__ import annotations

import collections
import os
import sys

import psutil
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QWidget,
)

_MIN_LINE_POINTS = 2  # need at least 2 points to draw a line segment


class HardwareMonitor(QWidget):
    """Live system telemetry widget drawn with QPainter — zero extra dependencies."""

    _COLORS = {
        "bg": QColor(Colors.BG_DARKEST),
        "grid": QColor(Colors.BORDER),
        "sys_cpu": QColor(Colors.FG_SECONDARY),
        "app_cpu": QColor(Colors.ACCENT_PRIMARY),
        "vram": QColor(Colors.ACCENT_WARNING),
        "label": QColor(Colors.FG_SECONDARY),
    }

    LEGEND_H = 14  # px reserved for the legend strip at the very top
    PLOT_GAP = 6  # px between the two sub-plots
    ML = 10  # left margin (rotated Y-label sits here)
    MR = 8  # right margin
    MT = 4  # inner-top margin of each sub-plot
    MB = 4  # inner-bottom margin

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setMinimumWidth(200)

        self.max_ticks = 60
        self.sys_cpu_data = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)
        self.app_cpu_data = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)
        self.vram_data = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)

        self._process = psutil.Process(os.getpid())
        self._cpu_count = psutil.cpu_count() or 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000)

    def _tick(self):
        try:
            sys_cpu = psutil.cpu_percent()
            app_cpu = min(100.0, self._process.cpu_percent() / self._cpu_count)

            vram_mb = 0.0
            if "torch" in sys.modules:
                import torch

                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    try:
                        vram_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                    except Exception:
                        pass

            self.sys_cpu_data.append(sys_cpu)
            self.app_cpu_data.append(app_cpu)
            self.vram_data.append(vram_mb)
            self.update()
        except Exception:
            pass

    def stop(self):
        """Stop the monitoring timer."""
        self._timer.stop()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        LH = self.LEGEND_H
        GAP = self.PLOT_GAP
        ML = self.ML
        MR = self.MR
        MT = self.MT
        MB = self.MB

        # Each sub-plot height — fits two plots + gap below the legend strip
        ph = (H - LH - GAP) // 2
        inner_w = W - ML - MR

        p.fillRect(0, 0, W, H, self._COLORS["bg"])

        f8 = QFont()
        f8.setPointSize(7)
        p.setFont(f8)

        # ── Legend strip ──
        lx = ML + 4
        for lcolor, ltext in [
            (self._COLORS["sys_cpu"], "Sys CPU"),
            (self._COLORS["app_cpu"], "App CPU"),
            (self._COLORS["vram"], "AI VRAM"),
        ]:
            p.setPen(QPen(lcolor, 2))
            p.drawLine(lx, LH // 2, lx + 12, LH // 2)
            p.setPen(self._COLORS["label"])
            p.drawText(
                lx + 14,
                0,
                58,
                LH,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                ltext,
            )
            lx += 76

        def _draw_plot(y0, data, line_color, y_max, y_label, extra_data=None, extra_color=None):
            inner_h = ph - MT - MB

            # Grid lines
            p.setPen(QPen(self._COLORS["grid"], 1))
            for frac in (0.25, 0.5, 0.75, 1.0):
                gy = y0 + MT + inner_h - int(frac * inner_h)
                p.drawLine(ML, gy, ML + inner_w, gy)

            # Rotated Y-axis label
            p.save()
            p.setPen(self._COLORS["label"])
            cx = ML // 2  # horizontal centre of the left margin
            cy = y0 + MT + inner_h // 2
            p.translate(cx, cy)
            p.rotate(-90)
            tw = inner_h  # available width after rotation = plot height
            p.drawText(-tw // 2, -6, tw, 12, Qt.AlignmentFlag.AlignCenter, y_label)
            p.restore()

            # Live value (top-right)
            last_val = list(data)[-1] if data else 0
            unit = " MB" if "VRAM" in y_label else "%"
            p.setPen(self._COLORS["label"])
            p.drawText(
                ML,
                y0,
                inner_w - 2,
                MT + 2,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                f"{last_val:.0f}{unit}",
            )

            def _polyline(vals, color):
                n = len(vals)
                if n < _MIN_LINE_POINTS:
                    return
                pen = QPen(color, 2)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                pts = [
                    (
                        ML + int(i / (n - 1) * inner_w),
                        y0 + MT + inner_h - int(min(v, y_max) / y_max * inner_h),
                    )
                    for i, v in enumerate(vals)
                ]
                for i in range(len(pts) - 1):
                    p.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])

            _polyline(list(data), line_color)
            if extra_data is not None:
                _polyline(list(extra_data), extra_color)

        vram_max = max(500.0, max(self.vram_data) * 1.2) if self.vram_data else 500.0

        _draw_plot(
            LH,
            self.sys_cpu_data,
            self._COLORS["sys_cpu"],
            100.0,
            "CPU %",
            extra_data=self.app_cpu_data,
            extra_color=self._COLORS["app_cpu"],
        )
        _draw_plot(LH + ph + GAP, self.vram_data, self._COLORS["vram"], vram_max, "VRAM MB")

        p.end()
