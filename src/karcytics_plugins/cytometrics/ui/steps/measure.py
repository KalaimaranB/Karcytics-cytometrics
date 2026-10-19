from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, BioComboBox
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel

from .base import StepPage, help_button, with_help


class MeasureStep(StepPage):
    title = "Measure"
    hint = (
        "Every cell is measured for size, shape and the intensity of each channel. Hover over a "
        "results column for its definition."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section(
            "Options",
            "Every object is measured for size and shape, and for its intensity in each "
            "channel. These add more measurements; the extra ones are slower.",
        )
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
        row.addWidget(help_button(self.combo_background.toolTip(), "Intensity background"))
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
        row.addWidget(help_button(self.combo_nucleus.toolTip(), "Nucleus / cytoplasm"))
        self.body.addLayout(row)

        self.chk_texture = QCheckBox("Texture (Haralick features, slower)")
        self.chk_texture.setToolTip(
            "Six grey-level co-occurrence features per channel (Haralick et al., 1973): "
            "uniformity, contrast, correlation, homogeneity, entropy and variance."
        )
        self.body.addLayout(with_help(self.chk_texture, self.chk_texture.toolTip(), "Texture"))

        self.chk_rods = QCheckBox("Rod-shaped cells (length, width, curvature, division)")
        self.chk_rods.setToolTip(
            "For bacteria and other rods: length along the midline, width, curvature, and a "
            "dividing flag when the middle is pinched."
        )
        self.body.addLayout(with_help(self.chk_rods, self.chk_rods.toolTip(), "Rod-shaped cells"))

        self.section("Summary")
        self.lbl_stats = QLabel("Total Cells: 0")
        self.body.addWidget(self.lbl_stats)
        self.body.addWidget(
            BioCaptionLabel("The table, histogram and scatter plot are in the panel below.")
        )
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        for widget in (
            self.lbl_background,
            self.lbl_nucleus,
            self.chk_texture,
            self.chk_rods,
            self.lbl_stats,
        ):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
