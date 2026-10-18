from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, CitationLabel, CitationsPanel, SecondaryButton
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QTextEdit

from .base import StepPage


class ReportStep(StepPage):
    title = "Report"
    hint = "Export your cells and the methods text that describes how they were found."

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Export")
        self.btn_export = SecondaryButton("💾 Export to CSV (+ methods and BibTeX)")
        self.btn_export_rois = SecondaryButton("🔷 Export ImageJ / Fiji ROIs")
        self.btn_export_rois.setToolTip("A .zip of cell outlines for ImageJ/Fiji's ROI Manager.")
        self.btn_export_fcs = SecondaryButton("📈 Export FCS 3.1 (open in flow software)")
        self.btn_export_fcs.setToolTip(
            "One event per cell and one parameter per measurement, plus a 0/1 parameter per "
            "gate. Opens in the Karcytics flow plugin, FlowJo, FCS Express, …"
        )
        self.btn_export_stats = SecondaryButton("📊 Export population statistics (CSV)")
        self.btn_export_stats.setToolTip("Count, % of parent and % of all cells per population.")
        self.btn_export_seg = SecondaryButton("🧬 Export for Cellpose (_seg.npy)")
        self.btn_export_seg.setToolTip(
            "Masks the Cellpose GUI opens for editing, and that Cellpose can train on."
        )
        for button in (
            self.btn_export,
            self.btn_export_fcs,
            self.btn_export_stats,
            self.btn_export_rois,
            self.btn_export_seg,
        ):
            self.body.addWidget(button)
        self.body.addWidget(CitationLabel("FCS file format", ("fcs31",), prefix="Format: "))

        self.section("Methods")
        self.body.addWidget(
            BioCaptionLabel("Paste this into your paper's methods. It updates as you work.")
        )
        self.methods_text = QTextEdit()
        self.methods_text.setReadOnly(True)
        self.methods_text.setMinimumHeight(160)
        self.body.addWidget(self.methods_text)
        self.btn_copy_methods = SecondaryButton("📋 Copy methods text")
        self.body.addWidget(self.btn_copy_methods)

        self.section("References")
        self.citations = CitationsPanel()
        self.body.addWidget(self.citations)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        self.methods_text.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 4px;"
        )
