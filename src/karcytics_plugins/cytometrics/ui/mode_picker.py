"""The assay picker ("What are you analysing?") shown before the workflow starts."""

from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioScrollArea,
    HeaderLabel,
    ModuleCard,
    SubtitleLabel,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from ..analysis.assays import Assay


class AssayCard(ModuleCard):
    clicked = Signal(str)

    def __init__(self, assay: Assay, parent=None):
        super().__init__(parent)
        self.assay_id = assay.id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        title = SubtitleLabel(f"{assay.icon}  {assay.name}")
        layout.addWidget(title)
        description = BioCaptionLabel(assay.description)
        description.setWordWrap(True)
        layout.addWidget(description)
        layout.addStretch()
        self.setToolTip(f"Start a {assay.name.lower()} analysis")

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.assay_id)
        super().mouseReleaseEvent(event)


class ModePicker(QWidget):
    assay_chosen = Signal(str)

    def __init__(self, assays: list[Assay], parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(10)
        layout.addWidget(HeaderLabel("What are you analysing?"))
        layout.addWidget(
            BioCaptionLabel(
                "This sets up the steps, the starting algorithm and the measurements. You can "
                "change any of them later."
            )
        )
        self.cards: dict[str, AssayCard] = {}
        content = QWidget()
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 0, 6, 0)
        column.setSpacing(8)
        for group in dict.fromkeys(a.group for a in assays):
            column.addWidget(SubtitleLabel(group))
            grid = QGridLayout()
            grid.setSpacing(8)
            members = [a for a in assays if a.group == group]
            for i, assay in enumerate(members):
                card = AssayCard(assay)
                card.clicked.connect(self.assay_chosen)
                grid.addWidget(card, i // 2, i % 2)
                self.cards[assay.id] = card
            column.addLayout(grid)
        column.addStretch()
        scroll = BioScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(BioScrollArea.Shape.NoFrame)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
