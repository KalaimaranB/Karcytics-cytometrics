"""CytoMetrics Entry Point."""

import collections
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from karcytics_sdk.plugin import (
    BioComboBox,
    BioDoubleSpinBox,
    BioScrollArea,
    BioSplitter,
    BioTableWidget,
    CitationLabel,
    DangerButton,
    FunctionalTask,
    HeaderLabel,
    PluginBase,
    PluginState,
    PrimaryButton,
    SecondaryButton,
    get_double,
    get_logger,
    get_save_path,
    show_error,
    show_info,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.dialogs import SaveWorkflowDialog
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QSizePolicy,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.model import ImageSet
from ..analysis.state import CytoMetricsState
from ..analysis.units import area_unit, ascii_unit, effective_scale, is_calibrated, length_unit
from ..report import build_steps, write_methods_sidecars
from .channel_manager import ChannelManagerWidget
from .image_canvas import MultiChannelCanvas, confidence_color
from .params_form import ParamsForm
from .workers import (
    CytoPipelineWorker,
    download_model_func,
    load_libraries_func,
)

logger = get_logger(__name__, "cytometrics")

_MIN_LINE_POINTS = 2  # need at least 2 points to draw a line segment
_GAUSSIAN_SPIKE_CUTOFF = 25  # exp(-25) ~= 1.4e-11, negligible past this point


def _check_ai_model_installed() -> tuple[bool, float]:
    """Checks the on-disk Cellpose model cache. Returns (installed, size_mb)."""
    models_dir = Path.home() / ".cellpose" / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        total_size = sum(f.stat().st_size for f in models_dir.iterdir() if f.is_file())
        return True, total_size / (1024 * 1024)
    return False, 0.0


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


class ModelManagerDialog(QDialog):
    """A dedicated manager to download or delete the 1GB AI model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage AI Engine")
        self.setMinimumWidth(350)
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};")

        layout = QVBoxLayout(self)

        self.lbl_status = QLabel()
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(self.lbl_status)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()

        self.btn_download = PrimaryButton("⬇️ Download Model (~1GB)")
        self.btn_download.clicked.connect(self._start_download)

        self.btn_delete = DangerButton("🗑️ Delete to Free Space")
        self.btn_delete.clicked.connect(self._delete_model)

        btn_layout.addWidget(self.btn_download)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)

        self._check_status()

    def _check_status(self):
        installed, size_mb = _check_ai_model_installed()

        if installed:
            self.lbl_status.setText(f"✅ AI Engine Installed ({size_mb:.1f} MB)")
            self.btn_download.setEnabled(False)
            self.btn_delete.setEnabled(True)
        else:
            self.lbl_status.setText("❌ AI Engine Not Installed")
            self.btn_download.setEnabled(True)
            self.btn_delete.setEnabled(False)

    def _delete_model(self):
        models_dir = Path.home() / ".cellpose" / "models"
        try:
            if models_dir.exists():
                for f in models_dir.iterdir():
                    if f.is_file():
                        f.unlink()  # Delete every model file
            self._check_status()
            show_info(self, "Deleted", "AI model successfully removed from disk.")
        except Exception as e:
            show_warning(self, "Error", f"Could not delete file:\n{e}")

    def _start_download(self):
        self.btn_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("Downloading via TaskScheduler...")

        submit_task(
            FunctionalTask(download_model_func, "cytometrics", "Download Cellpose model"),
            on_finished=lambda results: self._on_download_finished(
                results.get("success", False),
                "Model downloaded" if results.get("success") else "Failed",
            ),
            on_error=lambda message: self._on_download_finished(False, message),
            on_progress=self.progress_bar.setValue,
        )

    def _on_download_finished(self, success, msg):
        self.progress_bar.setVisible(False)
        if success:
            show_info(self, "Success", msg)
        else:
            show_error(self, "Download Error", f"Failed to download AI model:\n{msg}")
        self._check_status()


class SimpleHistogram(QWidget):
    """A lightweight, native Qt histogram with axis labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.data = []
        self.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; border: 1px solid {Colors.BORDER}; border-radius: 4px;"
        )

    def update_data(self, areas):
        self.data = areas
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
        painter.drawText(m_left, m_top + h + 18, f"{min_val:.0f}")

        max_str = f"{max_val:.0f}"
        text_width = painter.fontMetrics().horizontalAdvance(max_str)
        painter.drawText(m_left + w - text_width, m_top + h + 18, max_str)


class NumericTableItem(QTableWidgetItem):
    """A custom table item that sorts numerically instead of alphabetically."""

    def __lt__(self, other):
        try:
            # Strip out "Cell " if it's the ID column, then sort by float value
            val1 = float(self.text().replace("Cell ", ""))
            val2 = float(other.text().replace("Cell ", ""))
            return val1 < val2
        except ValueError:
            return super().__lt__(other)


class WrappingLabel(QLabel):
    """A professional subclass to fix Qt's word-wrap layout bug.
    Calculates exact height including Mac font descenders and margins.
    """

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        # Tell layout: Can expand horizontally, but strongly resists vertical crushing
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        # Natively force Qt to leave a 10px buffer at the bottom of the widget
        self.setContentsMargins(0, 2, 0, 10)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        margins = self.contentsMargins()
        # Calculate available width minus our margins
        available_width = max(0, width - margins.left() - margins.right())

        rect = self.fontMetrics().boundingRect(
            QRect(0, 0, available_width, 10000), Qt.TextFlag.TextWordWrap, self.text()
        )

        # True Math: Bounding Box + Margins + 1 extra Line Spacing (to protect descenders)
        return rect.height() + margins.top() + margins.bottom() + self.fontMetrics().lineSpacing()

    def sizeHint(self):
        width = super().sizeHint().width()
        return QSize(width, self.heightForWidth(width))

    def minimumSizeHint(self):
        return QSize(0, self.heightForWidth(self.width()))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()


class CytoMetricsPanel(PluginBase):
    # state_changed and status_message are now provided by PluginBase

    def __init__(self, plugin_id: str = "cytometrics", parent=None):
        super().__init__(plugin_id, parent)
        self.setMinimumWidth(380)
        self.setMinimumHeight(600)

        self.state = CytoMetricsState()
        # Detect settings restored before the segmenters finished loading.
        self._pending_pipeline: str | None = None
        self._pending_values: dict[str, Any] = {}
        # Last run's label mask and (Cellpose only) cell-probability map; not saved.
        self._labels: np.ndarray | None = None
        self._probability: np.ndarray | None = None

        self.image_set = ImageSet()
        self.canvas = MultiChannelCanvas()

        # All pipelines are loaded in the background — start empty
        self.pipelines: dict[str, Any] = {}

        self._setup_ui()
        self._update_units()
        self.state_changed.connect(self._update_results_tab)

        # Kick off heavy AI imports as a background task
        self._loader_task = submit_task(
            FunctionalTask(load_libraries_func, "cytometrics", "Load segmentation pipelines"),
            self.state,
            on_finished=self._on_loader_finished,
            on_error=lambda message: self._on_ai_loaded(False, {}, message),
        )

    def _on_loader_finished(self, results: dict) -> None:
        self._on_ai_loaded(results.get("success", False), results.get("pipelines", {}), "Loaded")

    def cleanup(self) -> None:
        """Called when the CytoMetrics tab is closed."""
        logger.info("Cleaning up CytoMetrics panel...")

        # 1. Stop UI timers and child widgets
        if hasattr(self, "hw_monitor"):
            self.hw_monitor.stop()

        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)

        # 2. Cleanup key components
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.cleanup()

        if hasattr(self, "channel_manager"):
            self.channel_manager.cleanup()

        # 3. Release image data
        if hasattr(self, "image_set") and self.image_set:
            self.image_set.clear()

        # 4. Stop listening to background tasks
        if hasattr(self, "_loader_task"):
            self._loader_task.detach()

        # 5. Base cleanup (disconnects and nulls state)
        super().cleanup()

    def shutdown(self) -> None:
        """Called for global module cleanup."""
        logger.info("Shutting down CytoMetrics module...")

        # Release the heavy AI pipelines
        if hasattr(self, "pipelines"):
            for pipe in self.pipelines.values():
                if hasattr(pipe, "model"):
                    pipe.model = None
            self.pipelines.clear()

        # Clear large state objects
        if self.state:
            self.state.cells.clear()
        if self.image_set:
            self.image_set.clear()

        # Force VRAM release if torch is loaded
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have an explicit clear_cache like CUDA,
                # but setting models to None and garbage collecting helps.
                import gc

                gc.collect()

    def _setup_ui(self):
        self._section_headers: list[QLabel] = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Ensure this widget fills the core app's central container
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_splitter = BioSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.main_splitter)

        splitter = BioSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(splitter)

        controls_widget = QWidget()
        control_layout = QVBoxLayout(controls_widget)
        control_layout.setContentsMargins(15, 15, 15, 10)
        control_layout.setSpacing(10)

        header = HeaderLabel("CytoMetrics")
        control_layout.addWidget(header)

        session_layout = QHBoxLayout()

        self.btn_save_session = SecondaryButton("💾 Save Session")
        self.btn_save_session.clicked.connect(self._on_save_workflow)

        self.btn_load_session = SecondaryButton("📂 Load Session")
        self.btn_load_session.clicked.connect(self._on_load_workflow)

        session_layout.addWidget(self.btn_save_session)
        session_layout.addWidget(self.btn_load_session)
        control_layout.addLayout(session_layout)
        # -----------------------------------

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY}; border: 1px solid {Colors.BORDER}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; font-weight: bold; border-bottom-color: {Colors.BG_MEDIUM}; }}
        """)

        def create_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; padding-top: 15px; padding-bottom: 5px; border-bottom: 1px solid {Colors.BORDER};"
            )
            self._section_headers.append(lbl)
            return lbl

        # ==========================================
        # --- TAB 1: SETUP & CHANNELS ---
        # ==========================================
        tab_setup = QWidget()
        setup_layout = QVBoxLayout(tab_setup)
        setup_layout.setSpacing(15)

        setup_layout.addWidget(create_header("1. Import Images"))

        self.channel_manager = ChannelManagerWidget(self.image_set)
        self.channel_manager.channels_changed.connect(self._render_composite)
        self.channel_manager.channels_changed.connect(self._update_calibrate_button_state)
        self.channel_manager.new_image_loaded.connect(self._apply_metadata_calibration)
        setup_layout.addWidget(self.channel_manager)

        self.canvas.load_requested.connect(self.channel_manager.prompt_add_channel)
        self.canvas.files_dropped.connect(self.channel_manager.ingest_paths)

        setup_layout.addWidget(create_header("2. Calibrate Scale"))

        self.lbl_scale = QLabel("Scale: Uncalibrated")
        self.lbl_scale.setStyleSheet(
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; border: none;"
        )
        setup_layout.addWidget(self.lbl_scale)

        self.btn_calibrate = SecondaryButton("📏  Set Scale / Calibrate")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setToolTip("Add an image channel first.")
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.canvas.calibration_line_drawn.connect(self._on_calibration_drawn)
        setup_layout.addWidget(self.btn_calibrate)

        setup_layout.addStretch()

        # ==========================================
        # --- TAB 2: DETECTION RULES ---
        # ==========================================
        tab_detect = QWidget()
        tab_layout = QVBoxLayout(tab_detect)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area protects the grid from being crushed
        scroll_area = BioScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(BioScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")

        detect_widget = QWidget()
        detect_layout = QVBoxLayout(detect_widget)
        detect_layout.setContentsMargins(10, 10, 10, 10)

        pipe_layout = QGridLayout()
        pipe_layout.setSpacing(8)
        pipe_layout.setColumnStretch(1, 1)

        input_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px; min-height: 24px;
            }}
            QComboBox QAbstractItemView {{ background-color: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY}; selection-background-color: {Colors.ACCENT_PRIMARY}; }}
        """

        self.lbl_ai_status = QLabel("⏳ Loading AI engine…")
        self.lbl_ai_status.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 12px;")
        detect_layout.addWidget(self.lbl_ai_status)

        r = 0
        pipe_layout.addWidget(create_header("1. Algorithm"), r, 0, 1, 2)
        r += 1

        lbl_algo = QLabel("Algorithm:")
        lbl_algo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pipe_layout.addWidget(lbl_algo, r, 0)

        # --- NEW HORIZONTAL LAYOUT FOR DROPDOWN + BUTTON ---
        algo_row_layout = QHBoxLayout()

        self.combo_pipeline = BioComboBox()
        for key, pipe in self.pipelines.items():
            self.combo_pipeline.addItem(pipe.name, key)
        self.combo_pipeline.setStyleSheet(input_style)
        self.combo_pipeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pipeline.currentIndexChanged.connect(self._on_algorithm_changed)

        self.btn_manage_ai = SecondaryButton("⚙️ Manage AI")
        self.btn_manage_ai.clicked.connect(self._open_model_manager)

        algo_row_layout.addWidget(self.combo_pipeline)
        algo_row_layout.addWidget(self.btn_manage_ai)

        pipe_layout.addLayout(algo_row_layout, r, 1)
        r += 1

        self.lbl_algo_bio = QLabel("")
        self.lbl_algo_bio.setWordWrap(True)
        self.lbl_algo_bio.setStyleSheet(
            f"color: {Colors.FG_SECONDARY}; font-size: 11px; font-style: italic; margin-bottom: 5px;"
        )
        pipe_layout.addWidget(self.lbl_algo_bio, r, 1)
        r += 1

        # "Powered by …" credit for the selected algorithm; one CitationLabel
        # per pipeline is added once the pipelines have loaded.
        self._citation_box = QWidget()
        self._citation_layout = QVBoxLayout(self._citation_box)
        self._citation_layout.setContentsMargins(0, 2, 0, 0)
        self._citation_labels: dict[str, CitationLabel] = {}
        pipe_layout.addWidget(self._citation_box, r, 1)
        r += 1

        detect_layout.addLayout(pipe_layout)
        detect_layout.addWidget(create_header("2. Settings"))
        self.params_form = ParamsForm()
        detect_layout.addWidget(self.params_form)

        self.lbl_units_note = QLabel(
            "⚠️ Not calibrated: sizes are in pixels. Set the scale in Setup for µm."
        )
        self.lbl_units_note.setWordWrap(True)
        self.lbl_units_note.setStyleSheet(
            f"color: {Colors.ACCENT_WARNING}; font-weight: bold; margin-top: 10px;"
        )
        self.lbl_units_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detect_layout.addWidget(self.lbl_units_note)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        detect_layout.addWidget(self.progress_bar)

        self.btn_run_pipeline = PrimaryButton("✨ Run Segmentation")
        self.btn_run_pipeline.clicked.connect(self._on_run_pipeline)
        detect_layout.addWidget(self.btn_run_pipeline)

        self.hw_monitor = HardwareMonitor()
        detect_layout.addWidget(self.hw_monitor)

        detect_layout.addStretch()

        scroll_area.setWidget(detect_widget)
        tab_layout.addWidget(scroll_area)

        # ==========================================
        # --- TAB 3: RESULTS & EXPORT ---
        # ==========================================
        tab_results = QWidget()
        results_layout = QVBoxLayout(tab_results)
        results_layout.setSpacing(15)

        results_layout.addWidget(create_header("Canvas View"))
        self.chk_show_ids = QCheckBox("Show Cell ID Numbers")
        self.chk_show_ids.setChecked(True)
        self.chk_show_ids.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.chk_show_ids.toggled.connect(self.canvas.set_show_ids)
        results_layout.addWidget(self.chk_show_ids)

        self.chk_color_confidence = QCheckBox("Colour outlines by AI confidence")
        self.chk_color_confidence.setToolTip(
            "Cellpose cells only: red = uncertain, amber = borderline, green = confident."
        )
        self.chk_color_confidence.toggled.connect(self._redraw_cells)
        results_layout.addWidget(self.chk_color_confidence)

        self.chk_probability_map = QCheckBox("Show Cellpose cell-probability map")
        self.chk_probability_map.setToolTip(
            "Where the model thinks cells are, from the last Cellpose run."
        )
        self.chk_probability_map.toggled.connect(self._update_probability_map)
        results_layout.addWidget(self.chk_probability_map)

        review_row = QHBoxLayout()
        self.lbl_uncertain = QLabel("Uncertain below:")
        self.spin_uncertain = BioDoubleSpinBox()
        self.spin_uncertain.setRange(0.0, 1.0)
        self.spin_uncertain.setSingleStep(0.05)
        self.spin_uncertain.setDecimals(2)
        self.spin_uncertain.setValue(0.75)
        self.spin_uncertain.setToolTip("Cells whose AI confidence is below this need a look.")
        self.spin_uncertain.valueChanged.connect(self._update_review_controls)
        self.btn_next_uncertain = SecondaryButton("Next uncertain ▸")
        self.btn_next_uncertain.clicked.connect(self._on_next_uncertain)
        review_row.addWidget(self.lbl_uncertain)
        review_row.addWidget(self.spin_uncertain)
        review_row.addWidget(self.btn_next_uncertain)
        results_layout.addLayout(review_row)
        self._review_index = -1

        results_layout.addWidget(create_header("Area Distribution"))
        self.lbl_stats = QLabel("Total Cells: 0\nAvg Area: 0.0 µm²")
        self.lbl_stats.setStyleSheet(f"color: {Colors.FG_PRIMARY}; font-size: 13px;")
        results_layout.addWidget(self.lbl_stats)

        self.histogram = SimpleHistogram()
        results_layout.addWidget(self.histogram)

        results_layout.addWidget(create_header("Export"))
        self.btn_export = SecondaryButton("💾 Export to CSV")
        self.btn_export.clicked.connect(self._on_export_csv)
        results_layout.addWidget(self.btn_export)

        results_layout.addStretch()

        # ==========================================
        # --- FINALIZE LAYOUT ---
        # ==========================================
        self.tabs.addTab(tab_setup, "1. Setup")
        self.tabs.addTab(tab_detect, "⏳ Loading AI Engine...")
        self.tabs.addTab(tab_results, "3. Results")
        self.tabs.setTabEnabled(1, False)  # Locked until LibraryLoaderWorker finishes
        self.tabs.currentChanged.connect(self._on_tab_changed)
        control_layout.addWidget(self.tabs)

        # _on_algorithm_changed will be triggered once _on_ai_loaded populates the combo

        self.btn_draw = SecondaryButton("✏️  Draw Cells (Manual)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(self._on_draw_toggled)
        self.canvas.cell_drawn.connect(self._on_cell_drawn)

        # Make sure cell_deleted is connected if you added it to canvas!
        if hasattr(self.canvas, "cell_deleted"):
            self.canvas.cell_deleted.connect(self._on_cell_deleted)

        control_layout.addWidget(self.btn_draw)

        splitter.addWidget(controls_widget)

        self.table = BioTableWidget(0, 6)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER}; border: none;"
        )
        self.table.setSortingEnabled(True)  # <--- Enable interactive sorting!
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setCollapsible(1, False)

        # Inject the canvas into the right hand side of the main splitter
        self.main_splitter.addWidget(self.canvas)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        # setStretchFactor alone only governs how *extra* space is redistributed on
        # resize — the initial split needs an explicit ratio, or Qt sizes each pane
        # by its own sizeHint() instead (which skewed the controls panel far wider
        # than intended). Values are proportional, not literal pixels — scaled to
        # whatever the splitter's actual width is at first layout.
        self.main_splitter.setSizes([1, 3])

    def _apply_theme_styles(self) -> None:
        """Re-applies theme-aware styles when the active theme changes.

        Most widgets here are plain Qt widgets with ``Colors.*`` baked into
        an f-string stylesheet at construction time, so — unlike the SDK's
        own ``Bio*`` components (buttons, ``HeaderLabel``, etc.), which
        already re-style themselves — they need to be explicitly rebuilt
        here. ``super()`` handles the panel background plus any child that
        already knows how to restyle itself.
        """
        super()._apply_theme_styles()

        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY}; border: 1px solid {Colors.BORDER}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; font-weight: bold; border-bottom-color: {Colors.BG_MEDIUM}; }}
        """)

        header_style = (
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; "
            f"padding-top: 15px; padding-bottom: 5px; border-bottom: 1px solid {Colors.BORDER};"
        )
        for lbl in self._section_headers:
            lbl.setStyleSheet(header_style)

        self.lbl_scale.setStyleSheet(
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; border: none;"
        )

        input_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px; min-height: 24px;
            }}
            QComboBox QAbstractItemView {{ background-color: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY}; selection-background-color: {Colors.ACCENT_PRIMARY}; }}
        """
        self.combo_pipeline.setStyleSheet(input_style)
        self.params_form.apply_theme()
        self.chk_show_ids.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        for widget in (self.chk_color_confidence, self.chk_probability_map, self.lbl_uncertain):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.lbl_stats.setStyleSheet(f"color: {Colors.FG_PRIMARY}; font-size: 13px;")
        self.lbl_units_note.setStyleSheet(
            f"color: {Colors.ACCENT_WARNING}; font-weight: bold; margin-top: 10px;"
        )

        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER}; border: none;"
        )

        # lbl_ai_status's color depends on AI-load state, not just the theme —
        # reuse the state-aware helper once loaded, otherwise re-apply the
        # loading-state color with the new theme's FG_SECONDARY.
        if self.tabs.isTabEnabled(1):
            self._refresh_ai_status_label()
        else:
            self.lbl_ai_status.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 12px;")

    def _update_units(self):
        """Label every size control and readout in µm, or px when uncalibrated."""
        scale = self.state.scale
        length, area = length_unit(scale), area_unit(scale)
        if is_calibrated(scale):
            origin = (
                " (from image metadata)" if self.state.scale_source.startswith("metadata") else ""
            )
            self.lbl_scale.setText(f"Scale: {scale:.4f} µm/px{origin}")
        else:
            self.lbl_scale.setText("Scale: Uncalibrated (sizes in pixels)")
        self.canvas.set_scale(scale)
        self.lbl_units_note.setVisible(not is_calibrated(scale))
        self.params_form.set_scale(scale)
        self.table.setHorizontalHeaderLabels(
            ["ID", f"Area ({area})", f"Perim ({length})", "Circ.", f"Diam. ({length})", "Conf."]
        )
        self._update_results_tab()

    def _set_scale(self, scale: float, source: str) -> None:
        """Change the calibration; measured cells are converted to the new units."""
        ratio = effective_scale(scale) / effective_scale(self.state.scale)
        for cell in self.state.cells:
            cell["area"] = cell.get("area", 0.0) * ratio**2
            cell["perim"] = cell.get("perim", 0.0) * ratio
        self.state.scale = scale
        self.state.scale_source = source if is_calibrated(scale) else ""
        self._update_units()
        self._refresh_table()
        self.state_changed.emit()

    def _update_calibrate_button_state(self):
        has_channels = bool(self.image_set.channels)
        self.btn_calibrate.setEnabled(has_channels)
        self.btn_calibrate.setToolTip("" if has_channels else "Add an image channel first.")

    def _open_model_manager(self):
        ModelManagerDialog(self).exec()
        self._refresh_ai_status_label()

    def _refresh_ai_status_label(self):
        installed, size_mb = _check_ai_model_installed()
        if installed:
            self.lbl_ai_status.setText(f"✅ AI engine ready ({size_mb:.1f} MB)")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_SUCCESS}; font-size: 12px; font-weight: bold;"
            )
        else:
            self.lbl_ai_status.setText("⚠️ AI model not installed — click Manage AI to install")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_WARNING}; font-size: 12px; font-weight: bold;"
            )

    def _on_ai_loaded(self, success, pipelines, message):
        """Called by LibraryLoaderWorker when all libraries + pipelines are ready."""
        if success and pipelines:
            # Replace the pipeline registry with the fully constructed instances
            self.pipelines = pipelines

            # Populate the combo box with all available algorithms
            self.combo_pipeline.clear()
            for key, pipe in pipelines.items():
                self.combo_pipeline.addItem(pipe.name, key)
                label = CitationLabel(pipe.name, getattr(pipe, "citations", ()))
                label.setVisible(False)
                self._citation_layout.addWidget(label)
                self._citation_labels[key] = label

            # Unlock and rename tab 2
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabText(1, "2. Smart Detect")
            self._refresh_ai_status_label()

            # Build the settings for the current (or restored) algorithm.
            self._on_algorithm_changed()
            if self._pending_pipeline or self._pending_values:
                self._apply_ui_params(
                    {"pipeline": self._pending_pipeline, "values": self._pending_values}
                )
        else:
            # Partial load — tab stays locked with error label
            self.tabs.setTabText(1, "⚠️ Load Failed")
            self.lbl_ai_status.setText("⚠️ AI engine failed to load")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_DANGER}; font-size: 12px; font-weight: bold;"
            )

    def _on_algorithm_changed(self, _index=None):
        selected = self.combo_pipeline.currentData()
        segmenter = self.pipelines.get(selected)
        self.lbl_algo_bio.setText(getattr(segmenter, "summary", ""))
        self.btn_manage_ai.setVisible(selected == "cellpose")
        if segmenter is not None:
            self.params_form.set_params(segmenter.parameters())
            self.params_form.set_channels([c.name for c in self.image_set.channels])
            self.params_form.set_scale(self.state.scale)
        for key, label in self._citation_labels.items():
            label.setVisible(key == selected)

    def _render_composite(self):
        composite = self.image_set.composite()
        if composite is not None:
            # Keep the user's zoom when only colours or visibility changed.
            self.canvas.set_array(composite, keep_view=self.canvas.has_image)

        self._update_dropdowns()
        self.state_changed.emit()

    def _on_run_pipeline(self):
        if not self.image_set.channels:
            show_warning(self, "No Image", "Please add a channel first.")
            return

        pipeline = self.pipelines.get(self.combo_pipeline.currentData())
        if pipeline is None:
            show_warning(self, "Not Ready", "The segmentation algorithms are still loading.")
            return
        params = self.params_form.values()
        if params.get("target_channel") not in [c.name for c in self.image_set.channels]:
            show_warning(self, "No Channel", "Choose the channel to segment.")
            return

        self.btn_run_pipeline.setEnabled(False)
        self.btn_run_pipeline.setText("⏳ Initializing...")

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # --- THE FIX: Wipe old cells and reset IDs on rerun ---
        self.state.cells.clear()
        self.state.cell_counter = 0

        analyzer = CytoPipelineWorker()
        analyzer.configure(pipeline, self.image_set, params, effective_scale(self.state.scale))
        self._last_run = (pipeline, params)  # for the methods text on export

        submit_task(
            analyzer,
            self.state,
            on_finished=self._on_pipeline_result,
            on_error=self._on_pipeline_error,
            on_progress=self.progress_bar.setValue,
        )

    def _on_pipeline_result(self, results: dict) -> None:
        # CytoPipelineWorker reports a pipeline exception as {"error": ...}
        # rather than raising, so it arrives here, not in on_error.
        if "error" in results:
            self._on_pipeline_error(results["error"])
        else:
            self._labels = results.get("labels")  # label mask of the last run
            self._probability = results.get("probability")  # Cellpose only
            self._on_pipeline_finished(results.get("result_cells", []))

    def _on_pipeline_finished(self, new_cells):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)

        if not new_cells:
            show_info(self, "No Cells", "No cells found with current parameters.")
            return

        for cell_data in new_cells:
            self.state.cell_counter += 1
            cell_data["id"] = self.state.cell_counter
            self.state.cells.append(cell_data)

        self.set_state(self.state)
        self.state_changed.emit()

    def _on_pipeline_error(self, error_msg):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)
        show_error(self, "Pipeline Error", f"An error occurred:\n{error_msg}")

    # ── State Management ────────────────────────────────────────────────

    def get_state(self) -> CytoMetricsState:
        """Package the workspace state for the SDK."""
        # Sync UI params before exporting
        self.state.ui_params = {
            "pipeline": self.combo_pipeline.currentData() or self._pending_pipeline,
            "values": self.params_form.display_values() or self._pending_values,
        }

        # Sync channels metadata
        channels_meta = []
        for c in self.image_set.channels:
            channels_meta.append({"name": c.name, "path": c.path, "color": c.color})
        self.state.channels_metadata = channels_meta

        return self.state

    def set_state(self, state: PluginState) -> None:
        """Restore the workspace from an SDK state object."""
        if not state:
            return
        assert isinstance(state, CytoMetricsState)

        self.state = state

        # 1. Restore images if needed
        try:
            if state.channels_metadata:
                self.load_images_from_meta(state.channels_metadata)
        except Exception:
            logger.exception("Failed to restore images")

        # 2. Restore UI settings
        self._apply_ui_params(state.ui_params)

        # 3. Restore data & canvas
        self._update_units()

        self._redraw_cells()
        self._refresh_table()
        self._update_results_tab()

    def export_state(self) -> dict:
        """Legacy export_state for backward compatibility."""
        state_obj = self.get_state()
        return state_obj.to_dict()

    def load_state(self, state_dict: dict) -> None:
        """Legacy load_state for backward compatibility."""
        if not state_dict:
            return
        state_obj = CytoMetricsState.from_dict(state_dict)
        self.set_state(state_obj)

    # --- ALIASES FOR HUB DASHBOARD INTEGRATION ---
    def load_workflow(self, payload: dict):
        self.load_state(payload)

    # ------------------------------------------------

    def load_images_from_meta(self, channels_meta: list[dict]) -> None:
        """Reload the saved images, then reapply the saved channel names and colours."""
        current_paths = [c.path for c in self.image_set.channels]
        saved_paths = [ch.get("path", "") for ch in channels_meta]
        if current_paths == saved_paths:
            return

        self.image_set.clear()
        for path in dict.fromkeys(saved_paths):  # each file once, in order
            if not path or not Path(path).exists():
                logger.warning("Saved image not found: %s", path)
                continue
            try:
                self.image_set.add_file(path, Path(path).name)
            except (OSError, ValueError):
                logger.exception("Could not reload saved image %s", path)

        for channel, saved in zip(self.image_set.channels, channels_meta, strict=False):
            channel.name = saved.get("name", channel.name)
            channel.color = saved.get("color", channel.color)

        self.channel_manager.refresh()
        self._render_composite()
        self._update_calibrate_button_state()

    def _refresh_table(self) -> None:
        """Sync the results table with the current state."""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for cell in self.state.cells:
            row = self.table.rowCount()
            self.table.insertRow(row)
            area = cell.get("area", 0)
            diam = 2 * math.sqrt(area / math.pi) if area > 0 else 0
            self.table.setItem(row, 0, NumericTableItem(f"Cell {cell.get('id', 0)}"))
            self.table.setItem(row, 1, NumericTableItem(f"{area:.1f}"))
            self.table.setItem(row, 2, NumericTableItem(f"{cell.get('perim', 0):.1f}"))
            self.table.setItem(row, 3, NumericTableItem(f"{cell.get('circ', 0):.2f}"))
            self.table.setItem(row, 4, NumericTableItem(f"{diam:.1f}"))
            conf = cell.get("confidence")
            self.table.setItem(row, 5, NumericTableItem("" if conf is None else f"{conf:.2f}"))
        self.table.setSortingEnabled(True)
        self.table.scrollToBottom()
        self._update_results_tab()

    def _apply_ui_params(self, ui: dict) -> None:
        """Restore the Detect settings from saved `ui_params` (either format)."""
        if not ui:
            return
        pipeline, values = _ui_params_v2(ui)
        self._pending_pipeline, self._pending_values = pipeline, values
        index = self.combo_pipeline.findData(pipeline)
        if index >= 0:
            self.combo_pipeline.setCurrentIndex(index)
        self.params_form.set_display_values(values)

    def _apply_metadata_calibration(self, _path: str = "") -> None:
        """Use the pixel size stored in the image file, unless a scale is already set."""
        pixel_size = self.image_set.pixel_size
        if pixel_size is None or is_calibrated(self.state.scale):
            return
        self._set_scale(pixel_size.x_um, f"metadata ({pixel_size.source})")

    def _on_calibrate_clicked(self):
        if not self.image_set.channels:
            return
        self.lbl_scale.setText("Scale: Click and drag along the image's scale bar.")
        self.canvas.set_mode("CALIBRATE")

    def _on_calibration_drawn(self, pixels: float):
        microns = get_double(
            self, "Set Scale", f"Line is {pixels:.1f} px long.\nMicrons?", 20.0, 0.001, 10000.0, 3
        )
        if microns is not None and microns > 0:
            self._set_scale(microns / pixels, "manual")
        else:
            self._update_units()

    def _on_draw_toggled(self, checked):
        if not self.image_set.channels:
            self.btn_draw.setChecked(False)
            return
        self.canvas.set_mode("DRAW" if checked else "PAN")
        if checked:
            # Highlight as active while drawing; restored to the normal SecondaryButton
            # look via its own theme method once the user toggles it back off.
            self.btn_draw.setStyleSheet(f"""
                QPushButton {{ background-color: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST};
                    border: none; border-radius: 6px; padding: 10px 20px; font-size: 13px;
                    font-weight: bold; text-align: left; }}
            """)
        else:
            self.btn_draw._apply_theme_styles()

    def _on_cell_drawn(self, points: list):
        area_px, perim_px = 0.0, 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area_px += points[i][0] * points[j][1] - points[j][0] * points[i][1]
            perim_px += math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
        area_px = abs(area_px) / 2.0
        scale = effective_scale(self.state.scale)
        area_um2, perim_um = area_px * (scale**2), perim_px * scale
        circularity = (4 * math.pi * area_um2) / (perim_um**2) if perim_um > 0 else 0.0
        self.state.cell_counter += 1
        self.state.cells.append(
            {
                "id": self.state.cell_counter,
                "points": points,
                "area": area_um2,
                "perim": perim_um,
                "circ": circularity,
                "source": "manual",
            }
        )
        self.set_state(self.state)
        self.state_changed.emit()

    def _update_dropdowns(self):
        """Offer the current channel names in the Detect settings."""
        self.params_form.set_channels([c.name for c in self.image_set.channels])

    def _on_tab_changed(self, index):
        """Force a refresh of the dropdowns when switching to the Smart Detect tab."""
        if index == 1:  # Index 1 is the second tab (Smart Detect)
            self._update_dropdowns()

    def _on_cell_deleted(self, cell_id):
        """Removes a cell by ID and triggers a state refresh for Undo/Redo."""
        self.state.cells = [c for c in self.state.cells if c["id"] != cell_id]
        self.set_state(self.state)
        self.state_changed.emit()

    def _redraw_cells(self, *_args) -> None:
        """Outline the cells, coloured by confidence if that's switched on."""
        colors = None
        if self.chk_color_confidence.isChecked():
            colors = {
                c["id"]: confidence_color(c["confidence"])
                for c in self.state.cells
                if c.get("confidence") is not None
            }
        self.canvas.draw_cells_from_state(self.state.cells, colors)
        self.canvas.set_show_ids(self.chk_show_ids.isChecked())

    def _update_probability_map(self, *_args) -> None:
        probability = self._probability
        usable = probability is not None and probability.shape == self.image_set.shape
        self.chk_probability_map.setEnabled(usable)
        show = usable and self.chk_probability_map.isChecked()
        self.canvas.set_probability_map(probability if show else None)

    def _uncertain_cells(self) -> list[dict]:
        """Cells below the confidence threshold, least confident first."""
        limit = self.spin_uncertain.value()
        scored = [c for c in self.state.cells if c.get("confidence") is not None]
        return sorted((c for c in scored if c["confidence"] < limit), key=lambda c: c["confidence"])

    def _update_review_controls(self, *_args) -> None:
        has_confidence = any(c.get("confidence") is not None for c in self.state.cells)
        self.chk_color_confidence.setEnabled(has_confidence)
        for widget in (self.lbl_uncertain, self.spin_uncertain, self.btn_next_uncertain):
            widget.setEnabled(has_confidence)
        n = len(self._uncertain_cells())
        self.btn_next_uncertain.setText(
            f"Next uncertain ({n}) ▸" if has_confidence else "Next uncertain ▸"
        )
        self.btn_next_uncertain.setEnabled(n > 0)
        self._review_index = -1

    def _on_next_uncertain(self) -> None:
        """Select, highlight and centre the next least-confident cell."""
        uncertain = self._uncertain_cells()
        if not uncertain:
            return
        self._review_index = (self._review_index + 1) % len(uncertain)
        self.select_cell(uncertain[self._review_index]["id"])

    def select_cell(self, cell_id: int) -> None:
        """Select `cell_id` in the table and centre it on the canvas."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == f"Cell {cell_id}":
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                break
        self.canvas.highlight_cell(cell_id)
        self.canvas.center_on_cell(cell_id)

    def _update_results_tab(self):
        """Refreshes the histogram and stats when cells change."""
        cells = self.state.cells
        num = len(cells)
        avg_area = sum(c["area"] for c in cells) / num if num else 0.0
        self.lbl_stats.setText(
            f"Total Cells: {num}\nAvg Area: {avg_area:.1f} {area_unit(self.state.scale)}"
        )
        self.histogram.update_data([c["area"] for c in cells])
        confidences = [c["confidence"] for c in cells if c.get("confidence") is not None]
        if confidences:
            self.lbl_stats.setText(
                self.lbl_stats.text()
                + f"\nMean AI confidence: {sum(confidences) / len(confidences):.2f}"
            )
        self._update_review_controls()
        self._update_probability_map()

    def _on_export_csv(self):
        """Saves the data table to a CSV file."""
        if not self.state.cells:
            show_info(self, "No Data", "There are no cells to export.")
            return

        path = get_save_path(self, "Export Results", "cytometrics_results.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, "w", newline="") as f:
                    writer = csv.writer(f)
                    # Add Diameter to the header
                    length = ascii_unit(length_unit(self.state.scale))
                    area = ascii_unit(area_unit(self.state.scale))
                    writer.writerow(
                        [
                            "Cell_ID",
                            f"Area_{area}",
                            f"Perimeter_{length}",
                            "Circularity",
                            f"Diameter_{length}",
                        ]
                    )

                    for c in self.state.cells:
                        diam = 2 * math.sqrt(c["area"] / math.pi)
                        writer.writerow(
                            [
                                c["id"],
                                f"{c['area']:.2f}",
                                f"{c['perim']:.2f}",
                                f"{c['circ']:.3f}",
                                f"{diam:.2f}",
                            ]
                        )

                sidecars = self._write_methods(path)
                show_info(
                    self,
                    "Success",
                    f"Data successfully exported to:\n{path}"
                    + (
                        f"\n\nMethods and citations:\n{sidecars[0].name}\n{sidecars[1].name}"
                        if sidecars
                        else ""
                    ),
                )
            except Exception as e:
                show_error(self, "Export Error", f"Could not save file:\n{str(e)}")

    def _write_methods(self, export_path: str):
        """Write the methods paragraph and BibTeX beside an export; None if no run to describe."""
        run = getattr(self, "_last_run", None)
        if run is None:
            return None
        pipeline, params = run
        n_manual = sum(1 for c in self.state.cells if c.get("source") == "manual")
        steps = build_steps(
            pipeline,
            params,
            self.state.scale,
            len(self.state.cells),
            n_manual,
            self.state.scale_source,
        )
        return write_methods_sidecars(export_path, steps)

    def _on_table_selection(self):
        """Grabs the selected row, extracts the ID, and highlights it on the canvas."""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.canvas.highlight_cell(None)  # Clear highlight if they click off
            return

        # Get the row of whatever item they clicked
        row = selected_items[0].row()
        id_item = self.table.item(row, 0)  # Column 0 always holds the ID

        if id_item:
            try:
                # Strip out the word "Cell " to get the raw integer
                cell_id = int(id_item.text().replace("Cell ", ""))
                self.canvas.highlight_cell(cell_id)
            except ValueError:
                self.canvas.highlight_cell(None)

    def _on_save_workflow(self):
        pm = getattr(self.window(), "project_manager", None)
        if not pm:
            show_warning(self, "No Active Project", "Please open a project to save this workflow.")
            return

        dialog = SaveWorkflowDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        meta = dialog.get_metadata()
        try:
            pm.save_workflow(module_id=self.plugin_id, payload=self.export_state(), metadata=meta)
            show_info(self, "Success", f"Workflow '{meta['name']}' saved successfully!")
        except Exception as e:
            show_error(self, "Save Error", f"Failed to save workflow to project:\n{e}")

    def _on_load_workflow(self):
        """Pick one of this project's saved CytoMetrics workflows and restore it."""
        pm = getattr(self.window(), "project_manager", None)
        if not pm:
            show_warning(self, "No Active Project", "Please open a project to load a workflow.")
            return
        saved = [w for w in pm.workflows.list_all() if w.get("module") == self.plugin_id]
        if not saved:
            show_info(self, "No Workflows", "This project has no saved CytoMetrics workflows yet.")
            return

        newest_first = list(reversed(saved))
        labels = [
            f"{w.get('name', w['filename'])}  ({w.get('timestamp', '')[:16]})" for w in newest_first
        ]
        choice, ok = QInputDialog.getItem(self, "Load Workflow", "Workflow:", labels, 0, False)
        if not ok:
            return
        filename = newest_first[labels.index(choice)]["filename"]
        try:
            self.load_state(pm.load_workflow_payload(filename))
        except Exception as e:
            show_error(self, "Load Error", f"Could not load workflow:\n{e}")


# Saved v1 `ui_params` keys → (segmenter id, display values).
_V1_PIPELINES = {
    "Basic Threshold (Otsu)": "otsu",
    "Watershed (Clump Splitting)": "watershed",
    "AI Smart Detect (Cellpose)": "cellpose",
}
_V1_KEYS = {
    "target_channel": "target_channel",
    "seed_channel": "seed_channel",
    "use_dual": "use_dual_channel",
    "min_area": "min_area_px",
    "max_area": "max_area_px",
    "diameter": "diameter",
    "flow": "flow_threshold",
    "exclude_borders": "exclude_borders",
}


def _ui_params_v2(ui: dict) -> tuple[str | None, dict]:
    """``(segmenter id, display values)`` from saved `ui_params`, old or new."""
    if "values" in ui:
        return ui.get("pipeline"), dict(ui["values"])
    pipeline = str(ui.get("pipeline", ""))
    values = {new: ui[old] for old, new in _V1_KEYS.items() if old in ui}
    return _V1_PIPELINES.get(pipeline, pipeline or None), values
