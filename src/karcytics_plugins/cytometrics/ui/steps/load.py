from __future__ import annotations

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
        self.section("Images and channels")
        self.channel_manager = ChannelManagerWidget(image_set)
        self.body.addWidget(self.channel_manager)
        self.finish()

    def apply_theme(self) -> None:
        self.channel_manager._apply_theme_styles()
