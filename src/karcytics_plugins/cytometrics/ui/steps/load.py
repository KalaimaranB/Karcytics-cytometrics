from __future__ import annotations

from ...analysis.assays import PHOTO
from ..channel_manager import ChannelManagerWidget
from .base import StepPage


class LoadStep(StepPage):
    title = "Load"
    hint = (
        "Add one image per stain, or a multi-channel file (TIFF, OME-TIFF, ND2, PNG, JPG). "
        "You can also drop files onto the canvas. Channels keep their full bit depth."
    )

    def __init__(self, image_set, parent=None):
        super().__init__(parent)
        self.lbl_section = self.section("Images and channels")
        self.channel_manager = ChannelManagerWidget(image_set)
        self.body.addWidget(self.channel_manager)
        self.finish()

    def set_assay(self, assay) -> None:
        """What to load for `assay` (``analysis.assays.Assay``): its text, and photo or stains."""
        self.lbl_hint.setText(assay.load_hint or self.hint)
        photo = assay.image_kind == PHOTO
        self.lbl_section.setText("Photo" if photo else "Images and channels")
        self.channel_manager.set_photo_mode(photo)

    def apply_theme(self) -> None:
        self.channel_manager._apply_theme_styles()
