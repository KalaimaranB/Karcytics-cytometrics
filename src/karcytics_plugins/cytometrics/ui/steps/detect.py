from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioComboBox,
    CitationLabel,
    PrimaryButton,
    SecondaryButton,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..params_form import ParamsForm
from ..widgets.hardware_monitor import HardwareMonitor
from ..widgets.scanning_indicator import ScanningIndicator
from .base import StepPage


class DetectStep(StepPage):
    title = "Detect"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_ai_status = QLabel("⏳ Loading segmentation algorithms…")
        self.body.addWidget(self.lbl_ai_status)

        self.section("Algorithm")
        row = QHBoxLayout()
        self.combo_pipeline = BioComboBox()
        self.combo_pipeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_manage_ai = SecondaryButton("⚙️ Manage AI")
        row.addWidget(self.combo_pipeline)
        row.addWidget(self.btn_manage_ai)
        self.body.addLayout(row)
        self.lbl_algo_bio = BioCaptionLabel("")
        self.body.addWidget(self.lbl_algo_bio)

        # "Powered by …" credit for the selected algorithm; one per segmenter,
        # added once the segmenters have loaded.
        self.citation_box = QWidget()
        self.citation_layout = QVBoxLayout(self.citation_box)
        self.citation_layout.setContentsMargins(0, 2, 0, 0)
        self.citation_labels: dict[str, CitationLabel] = {}
        self.body.addWidget(self.citation_box)

        self.section("Settings")
        self.params_form = ParamsForm()
        self.body.addWidget(self.params_form)

        self.lbl_units_note = QLabel("⚠️ Not calibrated: sizes are in pixels.")
        self.lbl_units_note.setWordWrap(True)
        self.body.addWidget(self.lbl_units_note)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        self.body.addWidget(self.progress_bar)
        self.btn_run = PrimaryButton("✨ Run Segmentation")
        self.btn_run.setEnabled(False)
        self.body.addWidget(self.btn_run)

        self.hw_monitor = HardwareMonitor()
        self.body.addWidget(self.hw_monitor)
        self.finish()
        self.apply_theme()

    def add_citation(self, key: str, name: str, citations) -> None:
        label = CitationLabel(name, citations)
        label.setVisible(False)
        self.citation_layout.addWidget(label)
        self.citation_labels[key] = label

    def show_citation(self, key: str | None) -> None:
        for k, label in self.citation_labels.items():
            label.setVisible(k == key)

    def apply_theme(self) -> None:
        self.lbl_units_note.setStyleSheet(f"color: {Colors.ACCENT_WARNING}; font-weight: bold;")
        self.params_form.apply_theme()
