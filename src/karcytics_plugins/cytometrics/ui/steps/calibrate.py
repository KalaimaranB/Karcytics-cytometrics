from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, SecondaryButton
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QLabel

from .base import StepPage


class CalibrateStep(StepPage):
    title = "Calibrate"
    hint = (
        "Sizes are reported in µm once the pixel size is known. Files that store it (OME-TIFF, "
        "ImageJ TIFF, ND2) are calibrated automatically. You can also skip this step and work in "
        "pixels."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Pixel size")
        self.lbl_scale = QLabel("Scale: Uncalibrated")
        self.body.addWidget(self.lbl_scale)
        self.btn_calibrate = SecondaryButton("📏  Calibrate from a scale bar")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setToolTip("Add an image first.")
        self.body.addWidget(self.btn_calibrate)
        self.body.addWidget(
            BioCaptionLabel(
                "Drag a line along the scale bar in the image, then type its length in µm."
            )
        )
        self.lbl_units_note = QLabel(
            "⚠️ Not calibrated: sizes are in pixels. Calibrate here for µm."
        )
        self.lbl_units_note.setWordWrap(True)
        self.body.addWidget(self.lbl_units_note)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        self.lbl_scale.setStyleSheet(f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold;")
        self.lbl_units_note.setStyleSheet(f"color: {Colors.ACCENT_WARNING}; font-weight: bold;")
