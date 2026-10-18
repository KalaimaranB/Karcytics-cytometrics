"""Install, check and delete the Cellpose-SAM model."""

from __future__ import annotations

from pathlib import Path

from karcytics_sdk.plugin import (
    DangerButton,
    FunctionalTask,
    PrimaryButton,
    get_logger,
    show_error,
    show_info,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

from ..widgets.scanning_indicator import ScanningIndicator
from ..workers import download_model_func

logger = get_logger(__name__, "cytometrics")


def _check_ai_model_installed() -> tuple[bool, float]:
    """Checks the on-disk Cellpose model cache. Returns (installed, size_mb)."""
    models_dir = Path.home() / ".cellpose" / "models"
    if models_dir.exists() and any(models_dir.iterdir()):
        total_size = sum(f.stat().st_size for f in models_dir.iterdir() if f.is_file())
        return True, total_size / (1024 * 1024)
    return False, 0.0


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
