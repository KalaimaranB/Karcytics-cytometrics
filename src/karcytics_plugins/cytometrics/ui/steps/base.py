"""Shared pieces of the step pages: a scrolling page with section headers."""

from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, BioScrollArea, SubtitleLabel
from PySide6.QtWidgets import QVBoxLayout, QWidget


class StepPage(BioScrollArea):
    """One step of the workflow. Subclasses add widgets to ``self.body``.

    Pages only build widgets; the panel connects them (``CytoMetricsPanel._wire``).
    """

    title = ""
    hint = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(BioScrollArea.Shape.NoFrame)
        content = QWidget()
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(4, 4, 8, 4)
        self.body.setSpacing(10)
        self.setWidget(content)
        if self.hint:
            self.body.addWidget(BioCaptionLabel(self.hint))

    def section(self, text: str) -> SubtitleLabel:
        label = SubtitleLabel(text)
        self.body.addWidget(label)
        return label

    def finish(self) -> None:
        self.body.addStretch()

    def apply_theme(self) -> None:
        """Re-colour plain Qt widgets; the Bio* ones restyle themselves."""
