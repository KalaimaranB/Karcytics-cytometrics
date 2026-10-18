from __future__ import annotations

from karcytics_sdk.plugin import BioComboBox
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel

from ..widgets.histogram import SimpleHistogram
from .base import StepPage


class MeasureStep(StepPage):
    title = "Measure"
    hint = (
        "Every cell is measured for size, shape and the intensity of each channel. Hover over a "
        "results column for its definition."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Options")
        row = QHBoxLayout()
        self.lbl_background = QLabel("Intensity background:")
        self.combo_background = BioComboBox()
        for text, mode in (
            ("None (raw intensities)", "none"),
            ("Global (median outside cells)", "global"),
            ("Local ring around each cell", "local"),
        ):
            self.combo_background.addItem(text, mode)
        self.combo_background.setToolTip(
            "Subtract a background level from each cell's intensities. Global: the median of all "
            "pixels outside the cells. Local: the median of a 5 px ring, 2 px away from the cell, "
            "avoiding other cells."
        )
        row.addWidget(self.lbl_background)
        row.addWidget(self.combo_background, 1)
        self.body.addLayout(row)

        row = QHBoxLayout()
        self.lbl_nucleus = QLabel("Nucleus / cytoplasm from:")
        self.combo_nucleus = BioComboBox()
        self.combo_nucleus.setToolTip(
            "A nuclear stain (e.g. DAPI). Each cell is split into nucleus (above the stain's Otsu "
            "threshold) and cytoplasm, and every channel's nuclear/cytoplasmic ratio is measured."
        )
        self.combo_nucleus.addItem("Off", None)
        row.addWidget(self.lbl_nucleus)
        row.addWidget(self.combo_nucleus, 1)
        self.body.addLayout(row)

        self.chk_texture = QCheckBox("Texture (Haralick features, slower)")
        self.chk_texture.setToolTip(
            "Six grey-level co-occurrence features per channel (Haralick et al., 1973): "
            "uniformity, contrast, correlation, homogeneity, entropy and variance."
        )
        self.body.addWidget(self.chk_texture)

        self.section("Distribution")
        self.combo_hist_metric = BioComboBox()
        self.body.addWidget(self.combo_hist_metric)
        self.lbl_stats = QLabel("Total Cells: 0")
        self.body.addWidget(self.lbl_stats)
        self.histogram = SimpleHistogram()
        self.body.addWidget(self.histogram)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        for widget in (self.lbl_background, self.lbl_nucleus, self.chk_texture, self.lbl_stats):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
