"""Model Center: the AI models, their size, licence and citation; download or delete them."""

from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    CitationLabel,
    DangerButton,
    FunctionalTask,
    HeaderLabel,
    ModuleCard,
    PrimaryButton,
    SubtitleLabel,
    TaskHandle,
    ask_yes_no,
    show_error,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QVBoxLayout

from ... import citations as _registered  # noqa: F401  (the cards cite registered entries)
from ...analysis.models import EXCLUDED, MODELS, ModelInfo
from ..widgets.scanning_indicator import ScanningIndicator
from ..workers import download_model_func


class ModelCard(ModuleCard):
    def __init__(self, model: ModelInfo, parent=None):
        super().__init__(parent)
        self.model = model
        self._task: TaskHandle | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addWidget(SubtitleLabel(model.name))
        layout.addWidget(BioCaptionLabel(model.description))
        licence = BioCaptionLabel(
            f"Used by: {model.used_by} · Code: {model.code_license} · "
            f"Weights: {model.weights_license} · "
            f'<a href="{model.source_url}">source</a>'
        )
        licence.setOpenExternalLinks(True)
        layout.addWidget(licence)
        layout.addWidget(CitationLabel(model.name, model.citations, prefix="Cite"))
        self.lbl_status = QLabel()
        layout.addWidget(self.lbl_status)
        self.progress = ScanningIndicator()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        self.btn_download = PrimaryButton(f"⬇️ Download (~{model.approx_size_mb / 1000:.1f} GB)")
        self.btn_delete = DangerButton("🗑️ Delete")
        self.btn_download.clicked.connect(self._download)
        self.btn_delete.clicked.connect(self._delete)
        row.addWidget(self.btn_download)
        row.addWidget(self.btn_delete)
        row.addStretch()
        layout.addLayout(row)
        self.refresh()

    def refresh(self) -> None:
        size = self.model.installed_size_mb()
        installed = size is not None
        self.lbl_status.setText(
            f"✅ Installed ({size:.0f} MB) at {self.model.path}" if installed else "Not installed"
        )
        self.lbl_status.setStyleSheet(
            f"color: {Colors.ACCENT_SUCCESS if installed else Colors.FG_SECONDARY};"
        )
        self.btn_download.setEnabled(not installed and self._task is None)
        self.btn_delete.setEnabled(installed)

    def _download(self) -> None:
        self.btn_download.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_status.setText("Downloading… this can take a few minutes.")
        self._task = submit_task(
            FunctionalTask(download_model_func, "cytometrics", f"Download {self.model.name}"),
            on_finished=self._on_downloaded,
            on_error=self._on_failed,
            on_progress=self.progress.setValue,
        )

    def _on_downloaded(self, _results: dict) -> None:
        self._task = None
        self.progress.setVisible(False)
        self.refresh()

    def _on_failed(self, message: str) -> None:
        self._task = None
        self.progress.setVisible(False)
        self.refresh()
        show_error(
            self, "Download Failed", f"{self.model.name} could not be downloaded:\n{message}"
        )

    def _delete(self) -> None:
        if not ask_yes_no(
            self,
            "Delete Model?",
            f"Delete {self.model.name} ({self.model.path})? You can download it again later.",
        ):
            return
        try:
            self.model.path.unlink(missing_ok=True)
        except OSError as e:
            show_warning(self, "Couldn't Delete", str(e))
        self.refresh()

    def detach(self) -> None:
        if self._task is not None:
            self._task.detach()
            self._task = None


class ModelCenterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Center")
        self.setMinimumWidth(520)
        self.setStyleSheet(f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};")
        layout = QVBoxLayout(self)
        layout.addWidget(HeaderLabel("AI models"))
        layout.addWidget(
            BioCaptionLabel(
                "Models are downloaded on demand and stored on this computer. Please cite the "
                "papers behind any model you use."
            )
        )
        self.cards = [ModelCard(m) for m in MODELS]
        for card in self.cards:
            layout.addWidget(card)

        layout.addWidget(SubtitleLabel("Not included"))
        layout.addWidget(
            BioCaptionLabel(
                "Only models whose code and weights both have a licence compatible with "
                "Karcytics are offered:<br>"
                + "<br>".join(f"• <b>{m.name}</b>: {m.reason}" for m in EXCLUDED)
            )
        )
        layout.addStretch()

    def done(self, result: int) -> None:
        for card in self.cards:
            card.detach()
        super().done(result)
