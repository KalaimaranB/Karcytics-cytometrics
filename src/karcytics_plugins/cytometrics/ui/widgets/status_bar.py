"""One line of system load under the controls: CPU (app / system), RAM and GPU memory."""

from __future__ import annotations

import os
import sys

import psutil
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel

_INTERVAL_MS = 2000
_GB = 1024**3


class SystemStatusBar(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._process = psutil.Process(os.getpid())
        self._cpus = psutil.cpu_count() or 1
        self.setToolTip(
            "CPU used by CytoMetrics / the whole system, memory, and GPU memory in use."
        )
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(_INTERVAL_MS)
        self.apply_theme()
        self.refresh()

    def refresh(self) -> None:
        try:
            app_cpu = min(100.0, self._process.cpu_percent() / self._cpus)
            memory = psutil.virtual_memory()
            text = (
                f"CPU {app_cpu:.0f}% (system {psutil.cpu_percent():.0f}%) · "
                f"RAM {memory.used / _GB:.1f}/{memory.total / _GB:.0f} GB"
            )
            gpu = _gpu_text()
            self.setText(text + (f" · {gpu}" if gpu else ""))
        except (psutil.Error, OSError):
            self.setText("")

    def stop(self) -> None:
        self._timer.stop()

    def apply_theme(self) -> None:
        self.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 11px;")


def _gpu_text() -> str:
    """GPU memory in use, only once torch is loaded (never imports it)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return ""
    try:
        if torch.cuda.is_available():
            return f"GPU (CUDA) {torch.cuda.memory_allocated() / _GB:.1f} GB"
        if torch.backends.mps.is_available():
            return f"GPU (Apple) {torch.mps.current_allocated_memory() / _GB:.1f} GB"
    except Exception:  # noqa: BLE001 - telemetry must never break the panel
        return ""
    return "CPU only"
