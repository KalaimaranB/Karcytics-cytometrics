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
from ..widgets.scanning_indicator import ScanningIndicator
from .base import StepPage

PREVIEW_TEXT = "👁 Preview on screen area"


class DetectStep(StepPage):
    title = "Detect"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_ai_status = QLabel("⏳ Loading segmentation algorithms…")
        self.body.addWidget(self.lbl_ai_status)

        self.section(
            "Algorithm",
            "How objects are found. Cellpose-SAM is a trained AI model: the most accurate on "
            "touching or irregular cells, but slower and a large download. The classical "
            "methods (thresholding, watershed, texture) are fast and transparent, and work well "
            "on clean, well-separated objects. The assay picks a sensible default.",
        )
        row = QHBoxLayout()
        self.combo_pipeline = BioComboBox()
        self.combo_pipeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_manage_ai = SecondaryButton("🧠 Models")
        self.btn_manage_ai.setToolTip("Download, delete and cite the AI models.")
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

        self.lbl_settings = self.section(
            "Settings",
            "The algorithm's settings; each (?) explains one. Start with the defaults, use "
            "👁 Preview on a zoomed-in area, and change one thing at a time.",
        )
        self.params_form = ParamsForm()
        self.body.addWidget(self.params_form)

        self.lbl_units_note = QLabel("⚠️ Not calibrated: sizes are in pixels.")
        self.lbl_units_note.setWordWrap(True)
        self.body.addWidget(self.lbl_units_note)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        self.body.addWidget(self.progress_bar)
        row = QHBoxLayout()
        self.btn_preview = SecondaryButton(PREVIEW_TEXT)
        self.btn_preview.setToolTip(
            "Quick test of these settings before the full run.\n\n"
            "Detects cells only in the part of the image on screen (zoom in on\n"
            "a typical area first: smaller is faster) and draws them dashed.\n"
            "Nothing is kept. Adjust the settings and preview again until the\n"
            "outlines look right, then Run Segmentation on the whole image."
        )
        self.btn_preview.setEnabled(False)
        self.btn_run = PrimaryButton("✨ Run Segmentation")
        self.btn_run.setEnabled(False)
        row.addWidget(self.btn_preview)
        row.addWidget(self.btn_run)
        self.body.addLayout(row)

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
