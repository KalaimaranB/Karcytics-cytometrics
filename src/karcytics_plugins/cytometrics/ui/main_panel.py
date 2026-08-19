"""CytoMetrics Entry Point."""

import math
import sys
import os
import csv
import json
import collections
from pathlib import Path
from datetime import datetime
import requests
import psutil
from PIL import Image

from PyQt6.QtCore import Qt, pyqtSignal, QRect, QSize, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QBrush, QColor, QPen, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QSpinBox,
    QFormLayout, QMessageBox, QScrollArea, QSplitter, QSizePolicy,
    QCheckBox, QDoubleSpinBox, QTabWidget,
    QDialog, QLineEdit, QTextEdit, QDialogButtonBox, QFileDialog
)
from karcytics_sdk.plugin import PluginBase, HeaderLabel, PrimaryButton, SubtitleLabel, get_logger, task_scheduler
from karcytics_sdk.plugin.theme_fallback import Colors
from .workers import CytoPipelineWorker, FunctionalAnalysisTask, download_model_func, load_libraries_func
from .image_canvas import MultiChannelCanvas
from .channel_manager import ChannelManagerWidget
from ..analysis.image_stack import ImageStack
from ..analysis.state import CytoMetricsState


logger = get_logger(__name__, "cytometrics")


class HardwareMonitor(QWidget):
    """Live system telemetry widget drawn with QPainter — zero extra dependencies."""

    _COLORS = {
        "bg":       QColor("#111827"),
        "grid":     QColor("#374151"),
        "sys_cpu":  QColor("#6B7280"),
        "app_cpu":  QColor("#3B82F6"),
        "vram":     QColor("#F59E0B"),
        "label":    QColor("#9CA3AF"),
    }

    LEGEND_H = 14    # px reserved for the legend strip at the very top
    PLOT_GAP = 6     # px between the two sub-plots
    ML = 10          # left margin (rotated Y-label sits here)
    MR = 8           # right margin
    MT = 4           # inner-top margin of each sub-plot
    MB = 4           # inner-bottom margin

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setMinimumWidth(200)

        self.max_ticks = 60
        self.sys_cpu_data  = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)
        self.app_cpu_data  = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)
        self.vram_data     = collections.deque([0.0] * self.max_ticks, maxlen=self.max_ticks)

        self._process   = psutil.Process(os.getpid())
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

        W, H  = self.width(), self.height()
        LH    = self.LEGEND_H
        GAP   = self.PLOT_GAP
        ML    = self.ML
        MR    = self.MR
        MT    = self.MT
        MB    = self.MB

        # Each sub-plot height — fits two plots + gap below the legend strip
        ph = (H - LH - GAP) // 2
        inner_w = W - ML - MR

        p.fillRect(0, 0, W, H, self._COLORS["bg"])

        f8 = QFont(); f8.setPointSize(7); p.setFont(f8)

        # ── Legend strip ──
        lx = ML + 4
        for lcolor, ltext in [
            (self._COLORS["sys_cpu"], "Sys CPU"),
            (self._COLORS["app_cpu"], "App CPU"),
            (self._COLORS["vram"],   "AI VRAM"),
        ]:
            p.setPen(QPen(lcolor, 2))
            p.drawLine(lx, LH // 2, lx + 12, LH // 2)
            p.setPen(self._COLORS["label"])
            p.drawText(lx + 14, 0, 58, LH,
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, ltext)
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
            cx = ML // 2          # horizontal centre of the left margin
            cy = y0 + MT + inner_h // 2
            p.translate(cx, cy)
            p.rotate(-90)
            tw = inner_h          # available width after rotation = plot height
            p.drawText(-tw // 2, -6, tw, 12,
                       Qt.AlignmentFlag.AlignCenter, y_label)
            p.restore()

            # Live value (top-right)
            last_val = list(data)[-1] if data else 0
            unit = " MB" if "VRAM" in y_label else "%"
            p.setPen(self._COLORS["label"])
            p.drawText(ML, y0, inner_w - 2, MT + 2,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                       f"{last_val:.0f}{unit}")

            def _polyline(vals, color):
                n = len(vals)
                if n < 2:
                    return
                pen = QPen(color, 2)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                p.setPen(pen)
                pts = [(ML + int(i / (n - 1) * inner_w),
                        y0 + MT + inner_h - int(min(v, y_max) / y_max * inner_h))
                       for i, v in enumerate(vals)]
                for i in range(len(pts) - 1):
                    p.drawLine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])

            _polyline(list(data), line_color)
            if extra_data is not None:
                _polyline(list(extra_data), extra_color)

        vram_max = max(500.0, max(self.vram_data) * 1.2) if self.vram_data else 500.0

        _draw_plot(LH, self.sys_cpu_data, self._COLORS["sys_cpu"], 100.0, "CPU %",
                   extra_data=self.app_cpu_data, extra_color=self._COLORS["app_cpu"])
        _draw_plot(LH + ph + GAP, self.vram_data, self._COLORS["vram"], vram_max, "VRAM MB")

        p.end()

class ScanningIndicator(QWidget):
    """
    Animated bio-themed loading indicator.
    Draws a scrolling ECG-style sine wave \u2014 looks like a live cell-signal scan.
    Fully replaces QProgressBar: call setVisible(True/False) to show / hide.
    """
    _BG    = QColor("#0F172A")
    _GLOW  = QColor("#10B981")      # emerald green \u2014 bio / life science feel
    _DIM   = QColor("#064E3B")      # dark teal for the faded trailing wave
    _GRID  = QColor("#1E293B")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self._phase   = 0.0        # scrolling phase (radians)
        self._visible = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # ---- Public API (mirrors QProgressBar) --------------------------------
    def setValue(self, v):
        pass   # continuous animation \u2014 no discrete value needed

    def setVisible(self, visible: bool):
        super().setVisible(visible)
        if visible:
            self._timer.start(30)      # ~33 fps
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
        pad  = 6

        # Background
        p.fillRect(0, 0, W, H, self._BG)

        # Subtle horizontal grid line at mid-height
        p.setPen(QPen(self._GRID, 1))
        p.drawLine(pad, H // 2, W - pad, H // 2)

        # Draw two overlapping waves: a dim wide one + a bright sharp one
        cy    = H / 2
        amp   = (H / 2) - pad - 1
        steps = W - 2 * pad

        def _wave(x_shift, color, thick, alpha_fn):
            if steps < 2:
                return
            prev_x, prev_y = None, None
            for i in range(steps + 1):
                t   = i / steps        # 0 \u2192 1 across widget
                # ECG-style: gentle sine with a sharper spike near centre
                angle = t * 4 * _math.pi + self._phase + x_shift
                base  = _math.sin(angle)
                # Add a narrow Gaussian spike near t=0.5 to mimic a QRS complex
                spike_t = ((t - 0.5) / 0.06) ** 2
                spike   = 3.0 * _math.exp(-spike_t) if spike_t < 25 else 0
                y_val   = base + spike
                y_val   = max(-1.6, min(1.6, y_val))   # clamp

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

        _wave(0,    self._DIM,  4, None)    # trailing wide glow
        _wave(0,    self._GLOW, 2, None)    # sharp bright line

        # Small "ANALYSING..." label on the right
        p.setPen(QColor("#6EE7B7"))
        f = QFont(); f.setPointSize(7); f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        p.setFont(f)
        p.drawText(W - 90, 0, 84, H,
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   "SCANNING\u2026")

        p.end()

class ModelManagerDialog(QDialog):
    """A dedicated manager to download or delete the 1GB AI model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage AI Engine")
        self.setMinimumWidth(350)
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};")

        self.model_path = Path.home() / ".cellpose" / "models" / "cyto3"

        layout = QVBoxLayout(self)

        self.lbl_status = QLabel()
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("font-size: 14px; margin: 10px;")
        layout.addWidget(self.lbl_status)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()

        self.btn_download = QPushButton("⬇️ Download Model (~1GB)")
        self.btn_download.setStyleSheet(f"background: {Colors.BG_MEDIUM}; padding: 6px; border-radius: 4px;")
        self.btn_download.clicked.connect(self._start_download)

        self.btn_delete = QPushButton("🗑️ Delete to Free Space")
        self.btn_delete.setStyleSheet(
            f"background: #7F1D1D; color: white; padding: 6px; border-radius: 4px;")  # Dark red
        self.btn_delete.clicked.connect(self._delete_model)

        btn_layout.addWidget(self.btn_download)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)

        self._check_status()

    def _check_status(self):
        models_dir = Path.home() / ".cellpose" / "models"
        
        # Check if the folder exists and has at least one file inside it
        if models_dir.exists() and any(models_dir.iterdir()):
            # Sum up the size of all model files
            total_size = sum(f.stat().st_size for f in models_dir.iterdir() if f.is_file())
            size_mb = total_size / (1024 * 1024)

            self.lbl_status.setText(f"✅ AI Engine Installed ({size_mb:.1f} MB)")
            self.btn_download.setEnabled(False)
            self.btn_download.setStyleSheet(f"background: #374151; color: #9CA3AF; padding: 6px; border-radius: 4px;")
            self.btn_delete.setEnabled(True)
            self.btn_delete.setStyleSheet(f"background: #7F1D1D; color: white; padding: 6px; border-radius: 4px;")
        else:
            self.lbl_status.setText("❌ AI Engine Not Installed")
            self.btn_download.setEnabled(True)
            self.btn_download.setStyleSheet(f"background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; padding: 6px; border-radius: 4px;")
            self.btn_delete.setEnabled(False)
            self.btn_delete.setStyleSheet(f"background: #374151; color: #9CA3AF; padding: 6px; border-radius: 4px;")

    def _delete_model(self):
        models_dir = Path.home() / ".cellpose" / "models"
        try:
            if models_dir.exists():
                for f in models_dir.iterdir():
                    if f.is_file():
                        f.unlink()  # Delete every model file
            self._check_status()
            QMessageBox.information(self, "Deleted", "AI model successfully removed from disk.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not delete file:\n{e}")

    def _start_download(self):
        self.btn_download.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_status.setText("Downloading via TaskScheduler...")

        task = FunctionalAnalysisTask(download_model_func)
        task_id = task_scheduler.submit(task, None).task_id  # No state needed for download
        
        def _on_finished(tid, results):
            if tid != task_id: return
            task_scheduler.task_finished.disconnect(_on_finished)
            task_scheduler.task_error.disconnect(_on_error)
            
            self._on_download_finished(results.get("success", False), "Model downloaded" if results.get("success") else "Failed")

        def _on_error(tid, error_msg):
            if tid != task_id: return
            task_scheduler.task_finished.disconnect(_on_finished)
            task_scheduler.task_error.disconnect(_on_error)
            self._on_download_finished(False, error_msg)

        task_scheduler.task_finished.connect(_on_finished)
        task_scheduler.task_error.connect(_on_error)

    def _on_download_finished(self, success, msg):
        self.progress_bar.setVisible(False)
        if success:
            QMessageBox.information(self, "Success", msg)
        else:
            QMessageBox.critical(self, "Download Error", f"Failed to download AI model:\n{msg}")
        self._check_status()

class SaveWorkflowDialog(QDialog):
    """Dialog to collect metadata before handing the workflow to the Project Manager."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save CytoMetrics Workflow")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};")

        layout = QFormLayout(self)

        self.input_name = QLineEdit()
        self.input_name.setStyleSheet(f"background: {Colors.BG_DARK}; border: 1px solid {Colors.BORDER}; padding: 4px;")

        self.input_desc = QTextEdit()
        self.input_desc.setStyleSheet(f"background: {Colors.BG_DARK}; border: 1px solid {Colors.BORDER}; padding: 4px;")
        self.input_desc.setMaximumHeight(80)

        self.input_tags = QLineEdit()
        self.input_tags.setPlaceholderText("e.g. wild-type, batch_A (comma separated)")
        self.input_tags.setStyleSheet(f"background: {Colors.BG_DARK}; border: 1px solid {Colors.BORDER}; padding: 4px;")

        layout.addRow("Workflow Name:", self.input_name)
        layout.addRow("Description:", self.input_desc)
        layout.addRow("Tags:", self.input_tags)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow("", self.buttons)

    def get_data(self):
        return {
            "name": self.input_name.text().strip(),
            "description": self.input_desc.toPlainText().strip(),
            "tags": [t.strip() for t in self.input_tags.text().split(",") if t.strip()]
        }

class SimpleHistogram(QWidget):
    """A lightweight, native PyQt histogram with axis labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.data = []
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; border: 1px solid {Colors.BORDER}; border-radius: 4px;")

    def update_data(self, areas):
        self.data = areas
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.data: return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Margins: left (for Y labels), right, top, bottom (for X labels)
        m_left, m_right, m_top, m_bottom = 35, 15, 15, 25
        w = self.width() - m_left - m_right
        h = self.height() - m_top - m_bottom

        bins = 15
        min_val, max_val = min(self.data), max(self.data)
        if min_val == max_val: max_val += 1

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
    """
    A professional subclass to fix Qt's word-wrap layout bug.
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
            QRect(0, 0, available_width, 10000),
            Qt.TextFlag.TextWordWrap,
            self.text()
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

        self.image_stack = ImageStack()
        self.canvas = MultiChannelCanvas()

        # All pipelines are loaded in the background — start empty
        self.pipelines = {}

        self._setup_ui()
        self._update_run_button_state()
        self.state_changed.connect(self._update_results_tab)

        # Kick off heavy AI imports as a background task
        task = FunctionalAnalysisTask(load_libraries_func)
        self._loader_task_id = task_scheduler.submit(task, self.state).task_id

        task_scheduler.task_finished.connect(self._on_loader_finished_handler)
        task_scheduler.task_error.connect(self._on_loader_error_handler)

    def _on_loader_finished_handler(self, tid, results):
        if hasattr(self, '_loader_task_id') and tid == self._loader_task_id:
            try:
                task_scheduler.task_finished.disconnect(self._on_loader_finished_handler)
                task_scheduler.task_error.disconnect(self._on_loader_error_handler)
            except (TypeError, RuntimeError):
                pass # Already disconnected or object deleted

            # FunctionalAnalysisTask emits the function's return dict directly as results.
            self._on_ai_loaded(
                results.get("success", False),
                results.get("pipelines", {}),
                "Loaded"
            )

    def _on_loader_error_handler(self, tid, error):
        if hasattr(self, '_loader_task_id') and tid == self._loader_task_id:
            try:
                task_scheduler.task_finished.disconnect(self._on_loader_finished_handler)
                task_scheduler.task_error.disconnect(self._on_loader_error_handler)
            except (TypeError, RuntimeError):
                pass
            self._on_ai_loaded(False, {}, error)

    def cleanup(self) -> None:
        """Called when the CytoMetrics tab is closed."""
        logger.info("Cleaning up CytoMetrics panel...")

        # 1. Stop UI timers and child widgets
        if hasattr(self, 'hw_monitor'):
            self.hw_monitor.stop()

        if hasattr(self, 'progress_bar'):
            self.progress_bar.setVisible(False)

        # 2. Cleanup key components
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.cleanup()

        if hasattr(self, 'channel_manager'):
            self.channel_manager.cleanup()

        # 3. Release image data
        if hasattr(self, 'image_stack') and self.image_stack:
            self.image_stack.clear()

        # 4. Disconnect background tasks
        try:
            task_scheduler.task_finished.disconnect(self._on_loader_finished_handler)
            task_scheduler.task_error.disconnect(self._on_loader_error_handler)
        except (TypeError, RuntimeError):
            pass

        # 5. Base cleanup (disconnects and nulls state)
        super().cleanup()

    def shutdown(self) -> None:
        """Called for global module cleanup."""
        logger.info("Shutting down CytoMetrics module...")

        # Release the heavy AI pipelines
        if hasattr(self, 'pipelines'):
            for pipe in self.pipelines.values():
                if hasattr(pipe, 'model'):
                    pipe.model = None
            self.pipelines.clear()

        # Clear large state objects
        if self.state:
            self.state.cells.clear()
        if self.image_stack:
            self.image_stack.channels.clear()

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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Ensure this widget fills the core app's central container
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.main_splitter)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(splitter)

        controls_widget = QWidget()
        control_layout = QVBoxLayout(controls_widget)
        control_layout.setContentsMargins(15, 15, 15, 10)
        control_layout.setSpacing(10)

        header = HeaderLabel("CytoMetrics")
        control_layout.addWidget(header)

        session_layout = QHBoxLayout()

        self.btn_save_session = PrimaryButton("💾 Save Session")
        self.btn_save_session.clicked.connect(self._on_save_workflow)

        session_layout.addWidget(self.btn_save_session)
        control_layout.addLayout(session_layout)
        # -----------------------------------

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY}; border: 1px solid {Colors.BORDER}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; font-weight: bold; border-bottom-color: {Colors.BG_MEDIUM}; }}
        """)

        # ==========================================
        # --- TAB 1: SETUP & CHANNELS ---
        # ==========================================
        tab_setup = QWidget()
        setup_layout = QVBoxLayout(tab_setup)
        setup_layout.setSpacing(15)

        self.lbl_scale = QLabel("Scale: Uncalibrated")
        self.lbl_scale.setStyleSheet(f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; border: none;")
        setup_layout.addWidget(self.lbl_scale)

        self.btn_calibrate = QPushButton("📏  Set Scale / Calibrate")
        self.btn_calibrate.setStyleSheet(self._btn_style(Colors.BG_DARKEST, align="left"))
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.canvas.calibration_line_drawn.connect(self._on_calibration_drawn)
        setup_layout.addWidget(self.btn_calibrate)

        self.channel_manager = ChannelManagerWidget(self.image_stack)
        self.channel_manager.channels_changed.connect(self._render_composite)
        self.channel_manager.new_image_loaded.connect(self._extract_tiff_metadata)
        setup_layout.addWidget(self.channel_manager)
        setup_layout.addStretch()

        # ==========================================
        # --- TAB 2: DETECTION RULES ---
        # ==========================================
        tab_detect = QWidget()
        tab_layout = QVBoxLayout(tab_detect)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area protects the grid from being crushed
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
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

        def create_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; padding-top: 15px; padding-bottom: 5px; border-bottom: 1px solid {Colors.BORDER};")
            return lbl

        r = 0

        # --- 1. SIGNAL SELECTION ---
        pipe_layout.addWidget(create_header("1. Signal Selection"), r, 0, 1, 2)
        r += 1

        lbl_target = QLabel("Main Signal:")
        lbl_target.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pipe_layout.addWidget(lbl_target, r, 0)

        self.combo_target_channel = QComboBox()
        self.combo_target_channel.setStyleSheet(input_style)
        self.combo_target_channel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_target_channel.setMinimumWidth(180)
        self.combo_target_channel.view().setMinimumWidth(300)
        pipe_layout.addWidget(self.combo_target_channel, r, 1)
        r += 1

        lbl_target_desc = QLabel(
            "The primary boundary to outline (e.g., cytoplasm, or nuclei if using a single stain).")
        lbl_target_desc.setWordWrap(True)
        lbl_target_desc.setStyleSheet(f"color: #9CA3AF; font-size: 11px; font-style: italic; margin-bottom: 5px;")
        pipe_layout.addWidget(lbl_target_desc, r, 1)
        r += 1

        self.check_dual_channel = QCheckBox("Use a 2nd signal to split clumps")
        self.check_dual_channel.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        pipe_layout.addWidget(self.check_dual_channel, r, 1)
        r += 1

        lbl_seed = QLabel("Splitting Marker:")
        lbl_seed.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pipe_layout.addWidget(lbl_seed, r, 0)

        self.combo_seed_channel = QComboBox()
        self.combo_seed_channel.setStyleSheet(input_style)
        self.combo_seed_channel.setMinimumWidth(180)
        self.combo_seed_channel.view().setMinimumWidth(300)
        self.combo_seed_channel.setEnabled(False)
        self.check_dual_channel.toggled.connect(self.combo_seed_channel.setEnabled)
        pipe_layout.addWidget(self.combo_seed_channel, r, 1)
        r += 1

        lbl_seed_desc = QLabel("An internal marker (e.g., nuclei) to help separate touching boundaries.")
        lbl_seed_desc.setWordWrap(True)
        lbl_seed_desc.setStyleSheet(f"color: #9CA3AF; font-size: 11px; font-style: italic; margin-bottom: 5px;")
        pipe_layout.addWidget(lbl_seed_desc, r, 1)
        r += 1

        # --- 2. ALGORITHM SETTINGS ---
        pipe_layout.addWidget(create_header("2. Algorithm Settings"), r, 0, 1, 2)
        r += 1

        lbl_algo = QLabel("Algorithm:")
        lbl_algo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pipe_layout.addWidget(lbl_algo, r, 0)

        # --- NEW HORIZONTAL LAYOUT FOR DROPDOWN + BUTTON ---
        algo_row_layout = QHBoxLayout()

        self.combo_pipeline = QComboBox()
        for key, pipe in self.pipelines.items():
            self.combo_pipeline.addItem(pipe.name, key)
        self.combo_pipeline.setStyleSheet(input_style)
        self.combo_pipeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pipeline.currentTextChanged.connect(self._on_algorithm_changed)

        self.btn_manage_ai = QPushButton("⚙️ Manage AI")
        self.btn_manage_ai.setStyleSheet(self._btn_style(Colors.BG_MEDIUM, align="center"))
        self.btn_manage_ai.clicked.connect(lambda: ModelManagerDialog(self).exec())

        algo_row_layout.addWidget(self.combo_pipeline)
        algo_row_layout.addWidget(self.btn_manage_ai)

        pipe_layout.addLayout(algo_row_layout, r, 1)
        r += 1

        self.lbl_algo_bio = QLabel("")
        self.lbl_algo_bio.setWordWrap(True)
        self.lbl_algo_bio.setStyleSheet(f"color: #9CA3AF; font-size: 11px; font-style: italic; margin-bottom: 5px;")
        pipe_layout.addWidget(self.lbl_algo_bio, r, 1)
        r += 1

        self.lbl_diameter = QLabel("Cell Diameter:")
        self.lbl_diameter.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin_diameter = QDoubleSpinBox()
        self.spin_diameter.setRange(0, 500)
        self.spin_diameter.setValue(0)
        self.spin_diameter.setDecimals(1)
        self.spin_diameter.setSpecialValueText("Auto")
        self.spin_diameter.setSuffix(" µm")
        self.spin_diameter.setStyleSheet(input_style)
        pipe_layout.addWidget(self.lbl_diameter, r, 0)
        pipe_layout.addWidget(self.spin_diameter, r, 1)
        r += 1

        self.lbl_flow = QLabel("Splitting Sensitivity:")
        self.lbl_flow.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin_flow = QDoubleSpinBox()
        self.spin_flow.setRange(0.1, 1.0)
        self.spin_flow.setValue(0.4)
        self.spin_flow.setSingleStep(0.1)
        self.spin_flow.setStyleSheet(input_style)
        self.spin_flow.setToolTip("(Higher = Merges cells together, Lower = Splits clumps apart)")
        pipe_layout.addWidget(self.lbl_flow, r, 0)
        pipe_layout.addWidget(self.spin_flow, r, 1)
        r += 1

        self.check_exclude_borders = QCheckBox("Exclude cells touching image edges")
        self.check_exclude_borders.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.check_exclude_borders.setChecked(True)  # Usually a good default
        pipe_layout.addWidget(self.check_exclude_borders, r, 1)
        r += 1

        # --- 3. SIZE FILTERS ---
        pipe_layout.addWidget(create_header("3. Size Filters"), r, 0, 1, 2)
        r += 1

        self.lbl_min = QLabel("Min Area:")
        self.lbl_min.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin_min_area = QDoubleSpinBox()
        self.spin_min_area.setRange(0, 100000)
        self.spin_min_area.setValue(10.0)
        self.spin_min_area.setDecimals(1)
        self.spin_min_area.setSuffix(" µm²")
        self.spin_min_area.setStyleSheet(input_style)
        pipe_layout.addWidget(self.lbl_min, r, 0)
        pipe_layout.addWidget(self.spin_min_area, r, 1)
        r += 1

        self.lbl_max = QLabel("Max Area:")
        self.lbl_max.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.spin_max_area = QDoubleSpinBox()
        self.spin_max_area.setRange(0, 100000)
        self.spin_max_area.setValue(5000.0)
        self.spin_max_area.setDecimals(1)
        self.spin_max_area.setSuffix(" µm²")
        self.spin_max_area.setStyleSheet(input_style)
        pipe_layout.addWidget(self.lbl_max, r, 0)
        pipe_layout.addWidget(self.spin_max_area, r, 1)

        detect_layout.addLayout(pipe_layout)

        self.lbl_calibration_warning = QLabel("⚠️ Please set scale in Setup tab to run.")
        self.lbl_calibration_warning.setStyleSheet(
            "color: #F87171; font-weight: bold; text-align: center; margin-top: 10px;")
        self.lbl_calibration_warning.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detect_layout.addWidget(self.lbl_calibration_warning)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        detect_layout.addWidget(self.progress_bar)

        self.btn_run_pipeline = QPushButton("✨ Run Segmentation")
        self.btn_run_pipeline.setStyleSheet(self._btn_style(Colors.ACCENT_PRIMARY, align="center"))
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

        results_layout.addWidget(create_header("Area Distribution"))
        self.lbl_stats = QLabel("Total Cells: 0\nAvg Area: 0.0 µm²")
        self.lbl_stats.setStyleSheet(f"color: {Colors.FG_PRIMARY}; font-size: 13px;")
        results_layout.addWidget(self.lbl_stats)

        self.histogram = SimpleHistogram()
        results_layout.addWidget(self.histogram)

        results_layout.addWidget(create_header("Export"))
        self.btn_export = QPushButton("💾 Export to CSV")
        self.btn_export.setStyleSheet(self._btn_style(Colors.BG_MEDIUM, align="center"))
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

        self.btn_draw = QPushButton("✏️  Draw Cells (Manual)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.setStyleSheet(self._btn_style(Colors.BG_MEDIUM, align="left"))
        self.btn_draw.clicked.connect(self._on_draw_toggled)
        self.canvas.cell_drawn.connect(self._on_cell_drawn)

        # Make sure cell_deleted is connected if you added it to canvas!
        if hasattr(self.canvas, 'cell_deleted'):
            self.canvas.cell_deleted.connect(self._on_cell_deleted)

        control_layout.addWidget(self.btn_draw)

        splitter.addWidget(controls_widget)

        self.table = QTableWidget(0, 5)  # <--- Changed to 5 columns
        self.table.setHorizontalHeaderLabels(["ID", "Area (µm²)", "Perim (µm)", "Circ.", "Diam. (µm)"])
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
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 7)


    def _update_run_button_state(self):
        has_scale = self.state.scale > 0
        self.btn_run_pipeline.setEnabled(has_scale)
        self.lbl_calibration_warning.setVisible(not has_scale)

    def _on_ai_loaded(self, success, pipelines, message):
        """Called by LibraryLoaderWorker when all libraries + pipelines are ready."""
        if success and pipelines:
            # Replace the pipeline registry with the fully constructed instances
            self.pipelines = pipelines

            # Populate the combo box with all available algorithms
            self.combo_pipeline.clear()
            for key, pipe in pipelines.items():
                self.combo_pipeline.addItem(pipe.name, key)

            # Unlock and rename tab 2
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabText(1, "2. Smart Detect")

            # Trigger algorithm-specific controls to update for the current selection
            self._on_algorithm_changed(self.combo_pipeline.currentText())
        else:
            # Partial load — tab stays locked with error label
            self.tabs.setTabText(1, "⚠️ Load Failed")

    def _on_algorithm_changed(self, text):
        is_ai = "Cellpose" in text
        if is_ai:
            self.lbl_algo_bio.setText(
                "Best for complex clusters and faint extensions. Requires an initial ~1GB model download.")
        elif "Watershed" in text:
            self.lbl_algo_bio.setText(
                "Best for clustered, circular cells. Uses classical math to split touching boundaries.")
        else:
            self.lbl_algo_bio.setText(
                "Best for sparse, isolated cells. Very fast, but will merge touching cells together.")

        self.lbl_diameter.setVisible(is_ai)
        self.spin_diameter.setVisible(is_ai)
        self.lbl_flow.setVisible(is_ai)
        self.spin_flow.setVisible(is_ai)

        # ADD THIS LINE
        if hasattr(self, 'check_exclude_borders'):
            self.check_exclude_borders.setVisible(is_ai)

        # Add this line
        if hasattr(self, 'btn_manage_ai'):
            self.btn_manage_ai.setVisible(is_ai)



    def _render_composite(self):
        composite = self.image_stack.get_composite()
        if composite is not None:
            import cv2  # lazy — already in sys.modules once loader finishes
            rgb_image = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.canvas.load_pixmap(QPixmap.fromImage(qt_img))

        self._update_dropdowns()
        self.state_changed.emit()

    def _on_run_pipeline(self):
        if not self.image_stack.channels:
            QMessageBox.warning(self, "No Image", "Please add a channel first.")
            return

        scale = self.state.scale
        if not scale:
            QMessageBox.warning(self, "Uncalibrated", "Please set the image scale first.")
            return

        pipeline_key = self.combo_pipeline.currentData()
        pipeline = self.pipelines.get(pipeline_key)

        min_area_px = self.spin_min_area.value() / (scale ** 2)
        max_area_px = self.spin_max_area.value() / (scale ** 2)

        raw_diam_um = self.spin_diameter.value()
        diam_px = (raw_diam_um / scale) if raw_diam_um > 0 else None

        params = {
            "target_channel": self.combo_target_channel.currentText(),
            "min_area_px": min_area_px,
            "max_area_px": max_area_px,
            "use_dual_channel": self.check_dual_channel.isChecked(),
            "seed_channel": self.combo_seed_channel.currentText(),
            "diameter": diam_px,
            "flow_threshold": self.spin_flow.value(),
            "exclude_borders": self.check_exclude_borders.isChecked()
        }

        self.btn_run_pipeline.setEnabled(False)
        self.btn_run_pipeline.setText("⏳ Initializing...")
        self.btn_run_pipeline.setStyleSheet(self._btn_style(Colors.BG_DARK, align="center"))

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # --- THE FIX: Wipe old cells and reset IDs on rerun ---
        self.state.cells.clear()
        self.state.cell_counter = 0

        analyzer = CytoPipelineWorker()
        analyzer.configure(pipeline, self.image_stack, params, scale)

        task_id = task_scheduler.submit(analyzer, self.state).task_id
        
        def _on_finished(tid, results):
            if tid != task_id: return
            task_scheduler.task_finished.disconnect(_on_finished)
            task_scheduler.task_error.disconnect(_on_error)
            self._on_pipeline_finished(results.get("result_cells", []))

        def _on_error(tid, error):
            if tid != task_id: return
            task_scheduler.task_finished.disconnect(_on_finished)
            task_scheduler.task_error.disconnect(_on_error)
            self._on_pipeline_error(error)

        task_scheduler.task_finished.connect(_on_finished)
        task_scheduler.task_error.connect(_on_error)

    def _on_pipeline_finished(self, new_cells):
        self._update_run_button_state()
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.btn_run_pipeline.setStyleSheet(self._btn_style(Colors.ACCENT_PRIMARY, align="center"))
        self.progress_bar.setVisible(False)

        if not new_cells:
            QMessageBox.information(self, "No Cells", "No cells found with current parameters.")
            return

        for cell_data in new_cells:
            self.state.cell_counter += 1
            cell_data["id"] = self.state.cell_counter
            self.state.cells.append(cell_data)

        self.set_state(self.state)
        self.state_changed.emit()

    def _on_pipeline_error(self, error_msg):
        self._update_run_button_state()
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.btn_run_pipeline.setStyleSheet(self._btn_style(Colors.ACCENT_PRIMARY, align="center"))
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred:\n{error_msg}")

    # ── State Management ────────────────────────────────────────────────

    def get_state(self) -> CytoMetricsState:
        """Package the workspace state for the SDK."""
        # Sync UI params before exporting
        self.state.ui_params = {
            "target_channel": self.combo_target_channel.currentText(),
            "seed_channel": self.combo_seed_channel.currentText(),
            "pipeline": self.combo_pipeline.currentText(),
            "min_area": self.spin_min_area.value(),
            "max_area": self.spin_max_area.value(),
            "diameter": self.spin_diameter.value(),
            "flow": self.spin_flow.value(),
            "use_dual": self.check_dual_channel.isChecked(),
            "exclude_borders": getattr(self, 'check_exclude_borders', QCheckBox()).isChecked()
        }
        
        # Sync channels metadata
        channels_meta = []
        for c in self.image_stack.channels:
            img_path = getattr(c, 'path', getattr(c, 'filepath', getattr(c, 'file_path', '')))
            channels_meta.append({"name": c.name, "path": str(img_path), "color": c.color})
        self.state.channels_metadata = channels_meta
        
        return self.state

    def set_state(self, state: CytoMetricsState) -> None:
        """Restore the workspace from an SDK state object."""
        if not state:
            return
        
        self.state = state
        
        # 1. Restore images if needed
        try:
            if state.channels_metadata:
                self.load_images_from_meta(state.channels_metadata)
        except Exception as e:
            logger.exception("Failed to restore images")

        # 2. Restore UI settings
        ui = state.ui_params
        if ui:
            self.combo_target_channel.setCurrentText(ui.get("target_channel", ""))
            self.combo_seed_channel.setCurrentText(ui.get("seed_channel", ""))
            self.combo_pipeline.setCurrentText(ui.get("pipeline", ""))
            self.spin_min_area.setValue(ui.get("min_area", 10.0))
            self.spin_max_area.setValue(ui.get("max_area", 5000.0))
            self.spin_diameter.setValue(ui.get("diameter", 0.0))
            self.spin_flow.setValue(ui.get("flow", 0.4))
            self.check_dual_channel.setChecked(ui.get("use_dual", False))
            if hasattr(self, 'check_exclude_borders'):
                self.check_exclude_borders.setChecked(ui.get("exclude_borders", True))

        # 3. Restore data & canvas
        scale_val = state.scale
        self.lbl_scale.setText(f"Scale: {scale_val:.4f} µm/px" if scale_val > 0 else "Scale: Uncalibrated")
        self._update_run_button_state()

        if state.cells:
            self.canvas.draw_cells_from_state(state.cells)
            
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

    def apply_state(self, payload: dict):
        self.load_state(payload)

    # ------------------------------------------------


    def load_images_from_meta(self, channels_meta: list[dict]) -> None:
        """Helper to reload images based on metadata."""
        current_paths = [getattr(c, 'path', '') for c in self.image_stack.channels]
        saved_paths = [ch.get("path", "") for ch in channels_meta]

        if current_paths != saved_paths:
            self.image_stack.channels.clear()
            if hasattr(self.channel_manager, 'clear_ui'):
                self.channel_manager.clear_ui()

            loaded_paths = set()
            for ch in channels_meta:
                path = ch.get("path", "")
                if path and path not in loaded_paths and Path(path).exists():
                    added = self.image_stack.add_channel(path, Path(path).name, "gray")
                    if added:
                        for ch_name, ch_color in added:
                            self.channel_manager._add_row_to_ui(ch_name, ch_color)
                    loaded_paths.add(path)

            for i, ch in enumerate(channels_meta):
                if i < len(self.image_stack.channels):
                    saved_name = ch.get("name", f"Ch {i}")
                    saved_color = ch.get("color", "gray")
                    self.image_stack.channels[i].name = saved_name
                    self.image_stack.channels[i].color = saved_color
                    self.image_stack.channels[i].path = ch.get("path", "")

                    name_item = self.channel_manager.table.item(i, 0)
                    if name_item:
                        name_item.setText(saved_name)
                    color_combo = self.channel_manager.table.cellWidget(i, 1)
                    if color_combo:
                        color_combo.blockSignals(True)
                        color_combo.setCurrentText(saved_color)
                        color_combo.blockSignals(False)
            self._render_composite()

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
        self.table.setSortingEnabled(True)
        self.table.scrollToBottom()
        self._update_results_tab()

        # --- RESTORE UI SETTINGS ---
        ui_params = self.state.ui_params
        if ui_params:
            self.combo_target_channel.setCurrentText(ui_params.get("target_channel", ""))
            self.combo_seed_channel.setCurrentText(ui_params.get("seed_channel", ""))
            self.combo_pipeline.setCurrentText(ui_params.get("pipeline", ""))
            self.spin_min_area.setValue(ui_params.get("min_area", 10.0))
            self.spin_max_area.setValue(ui_params.get("max_area", 5000.0))
            self.spin_diameter.setValue(ui_params.get("diameter", 0.0))
            self.spin_flow.setValue(ui_params.get("flow", 0.4))
            self.check_dual_channel.setChecked(ui_params.get("use_dual", False))
            if hasattr(self, 'check_exclude_borders'):
                self.check_exclude_borders.setChecked(ui_params.get("exclude_borders", True))

    def _extract_tiff_metadata(self, file_path):
        try:
            with Image.open(file_path) as img:
                x_res, res_unit = img.tag_v2.get(282), img.tag_v2.get(296)
                if x_res and res_unit and x_res[1] != 0:
                    px_per_unit = x_res[0] / x_res[1]
                    if res_unit == 3 and (px_per_unit / 10000.0) != 0:
                        self.state.scale = 1.0 / (px_per_unit / 10000.0)
                        self.lbl_scale.setText(f"Scale: {self.state.scale:.4f} µm/px")
                        self._update_run_button_state()
                        self.state_changed.emit()
        except Exception:
            pass

    def _on_calibrate_clicked(self):
        if not self.image_stack.channels: return
        self.lbl_scale.setText("Scale: Click and drag over scale bar.")
        self.canvas.set_mode("CALIBRATE")

    def _on_calibration_drawn(self, pixels: float):
        microns, ok = QInputDialog.getDouble(self, "Set Scale", f"Line is {pixels:.1f} px long.\nMicrons?", 20.0, 0.001,
                                             10000.0, 3)
        if ok and microns > 0:
            self.state.scale = microns / pixels
            self.set_state(self.state)
            self._update_run_button_state()
            self.state_changed.emit()

    def _on_draw_toggled(self, checked):
        if not self.image_stack.channels:
            self.btn_draw.setChecked(False)
            return
        self.canvas.set_mode("DRAW" if checked else "PAN")
        self.btn_draw.setStyleSheet(
            self._btn_style(Colors.ACCENT_PRIMARY if checked else Colors.BG_MEDIUM, align="left"))

    def _on_cell_drawn(self, points: list):
        area_px, perim_px = 0.0, 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area_px += (points[i][0] * points[j][1] - points[j][0] * points[i][1])
            perim_px += math.hypot(points[j][0] - points[i][0], points[j][1] - points[i][1])
        area_px = abs(area_px) / 2.0
        scale = self.state.scale if self.state.scale else 1.0
        area_um2, perim_um = area_px * (scale ** 2), perim_px * scale
        circularity = (4 * math.pi * area_um2) / (perim_um ** 2) if perim_um > 0 else 0.0
        self.state.cell_counter += 1
        self.state.cells.append(
            {"id": self.state.cell_counter, "points": points, "area": area_um2, "perim": perim_um,
             "circ": circularity})
        self.set_state(self.state)
        self.state_changed.emit()

    def _btn_style(self, bg_color, align="center"):
        text_color = Colors.BG_DARKEST if bg_color == Colors.ACCENT_PRIMARY else Colors.FG_PRIMARY
        return f"""
            QPushButton {{ background-color: {bg_color}; color: {text_color}; border: 1px solid {Colors.BORDER}; padding: 10px; border-radius: 6px; font-weight: bold; text-align: {align}; }}
            QPushButton:hover {{ background-color: rgba(255, 255, 255, 0.1); }}
        """

    def _update_dropdowns(self):
        """Safely updates dropdown names without losing the current selection."""
        target_idx = self.combo_target_channel.currentIndex()
        seed_idx = self.combo_seed_channel.currentIndex()

        channel_names = [c.name for c in self.image_stack.channels]

        self.combo_target_channel.blockSignals(True)
        self.combo_seed_channel.blockSignals(True)

        self.combo_target_channel.clear()
        self.combo_target_channel.addItems(channel_names)

        self.combo_seed_channel.clear()
        self.combo_seed_channel.addItems(channel_names)

        # Restore the user's selections
        if 0 <= target_idx < len(channel_names):
            self.combo_target_channel.setCurrentIndex(target_idx)
        if 0 <= seed_idx < len(channel_names):
            self.combo_seed_channel.setCurrentIndex(seed_idx)

        self.combo_target_channel.blockSignals(False)
        self.combo_seed_channel.blockSignals(False)

    def _on_tab_changed(self, index):
        """Force a refresh of the dropdowns when switching to the Smart Detect tab."""
        if index == 1:  # Index 1 is the second tab (Smart Detect)
            self._update_dropdowns()

    def _on_cell_deleted(self, cell_id):
        """Removes a cell by ID and triggers a state refresh for Undo/Redo."""
        self.state.cells = [c for c in self.state.cells if c["id"] != cell_id]
        self.set_state(self.state)
        self.state_changed.emit()

    def _update_results_tab(self):
        """Refreshes the histogram and stats when cells change."""
        cells = self.state.cells
        num = len(cells)
        if num > 0:
            avg_area = sum(c["area"] for c in cells) / num
            self.lbl_stats.setText(f"Total Cells: {num}\nAvg Area: {avg_area:.1f} µm²")
            self.histogram.update_data([c["area"] for c in cells])
        else:
            self.lbl_stats.setText("Total Cells: 0\nAvg Area: 0.0 µm²")
            self.histogram.update_data([])

    def _on_export_csv(self):
        """Saves the data table to a CSV file."""
        if not self.state.cells:
            QMessageBox.information(self, "No Data", "There are no cells to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Results", "cytometrics_results.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Add Diameter to the header
                    writer.writerow(["Cell_ID", "Area_um2", "Perimeter_um", "Circularity", "Diameter_um"])

                    for c in self.state.cells:
                        diam = 2 * math.sqrt(c["area"] / math.pi)
                        writer.writerow(
                            [c["id"], f"{c['area']:.2f}", f"{c['perim']:.2f}", f"{c['circ']:.3f}", f"{diam:.2f}"])

                QMessageBox.information(self, "Success", f"Data successfully exported to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Could not save file:\n{str(e)}")

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
        main_win = self.window()
        pm = getattr(main_win, "project_manager", None)

        if not pm:
            QMessageBox.warning(self, "No Active Project", "Please open a project to save this workflow.")
            return

        dialog = SaveWorkflowDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            meta = dialog.get_data()
            if not meta["name"]:
                QMessageBox.warning(self, "Invalid Input", "A workflow name is required.")
                return

            try:
                # 1. Grab everything from the UI and Canvas
                state_data = self.export_state()

                # 2. Add the timestamp so the Hub Dashboard can sort it properly
                meta["timestamp"] = datetime.now().isoformat()

                # 3. Hand it off to the core Project Manager using the exact kwargs it expects
                pm.save_workflow(
                    module_id="cytometrics",
                    payload=state_data,
                    metadata=meta
                )

                QMessageBox.information(self, "Success", f"Workflow '{meta['name']}' saved successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save workflow to project:\n{str(e)}")


    def _on_load_workflow(self):
        main_win = self.window()
        pm = getattr(main_win, "project_manager", None)
        default_dir = str(pm.project_dir) if pm else ""

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load CytoMetrics Workflow",
            default_dir,
            "CytoMetrics Session (*.cyto *.json);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r') as f:
                state_data = json.load(f)

            # Restore the core state (cells, scale, etc.)
            self.load_state(state_data)

            # Restore UI Params
            ui_params = state_data.get("ui_params", {})
            if ui_params:
                self.combo_target_channel.setCurrentText(ui_params.get("target_channel", ""))
                self.combo_seed_channel.setCurrentText(ui_params.get("seed_channel", ""))
                self.combo_pipeline.setCurrentText(ui_params.get("pipeline", ""))
                self.spin_min_area.setValue(ui_params.get("min_area", 10.0))
                self.spin_max_area.setValue(ui_params.get("max_area", 5000.0))
                self.spin_diameter.setValue(ui_params.get("diameter", 0.0))
                self.spin_flow.setValue(ui_params.get("flow", 0.4))
                self.check_dual_channel.setChecked(ui_params.get("use_dual", False))
                if hasattr(self, 'check_exclude_borders'):
                    self.check_exclude_borders.setChecked(ui_params.get("exclude_borders", True))

            QMessageBox.information(self, "Success", f"Workflow loaded from:\n{Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Could not load workflow:\n{e}")