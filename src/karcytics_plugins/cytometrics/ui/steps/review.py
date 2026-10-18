from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, BioDoubleSpinBox, SecondaryButton
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel

from .base import StepPage


class ReviewStep(StepPage):
    title = "Review"
    hint = (
        "Check the outlines on the canvas. Right-click a cell to delete it; draw the ones that "
        "were missed."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Show")
        self.chk_show_ids = QCheckBox("Cell ID numbers")
        self.chk_show_ids.setChecked(True)
        self.body.addWidget(self.chk_show_ids)
        self.chk_color_confidence = QCheckBox("Colour outlines by AI confidence")
        self.chk_color_confidence.setToolTip(
            "Cellpose cells only: red = uncertain, amber = borderline, green = confident."
        )
        self.body.addWidget(self.chk_color_confidence)
        self.chk_probability_map = QCheckBox("Cellpose cell-probability map")
        self.chk_probability_map.setToolTip("Where the model thinks cells are, from the last run.")
        self.body.addWidget(self.chk_probability_map)

        self.section("Uncertain cells")
        row = QHBoxLayout()
        self.lbl_uncertain = QLabel("Confidence below:")
        self.spin_uncertain = BioDoubleSpinBox()
        self.spin_uncertain.setRange(0.0, 1.0)
        self.spin_uncertain.setSingleStep(0.05)
        self.spin_uncertain.setDecimals(2)
        self.spin_uncertain.setValue(0.75)
        self.spin_uncertain.setToolTip("Cells whose AI confidence is below this need a look.")
        self.btn_next_uncertain = SecondaryButton("Next uncertain ▸")
        row.addWidget(self.lbl_uncertain)
        row.addWidget(self.spin_uncertain)
        row.addWidget(self.btn_next_uncertain)
        self.body.addLayout(row)
        self.body.addWidget(
            BioCaptionLabel("Steps through them least confident first and centres each one.")
        )

        self.section("Edit")
        self.btn_draw = SecondaryButton("✏️  Draw a missed cell")
        self.btn_draw.setCheckable(True)
        self.body.addWidget(self.btn_draw)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        for widget in (
            self.chk_show_ids,
            self.chk_color_confidence,
            self.chk_probability_map,
            self.lbl_uncertain,
        ):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
