"""Shared pieces of the step pages: a scrolling page with section headers."""

from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, BioHelpButton, BioScrollArea, SubtitleLabel
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ..widgets.wheel_guard import guard_wheel


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
        guard_wheel(self)
        content = QWidget()
        self.body = QVBoxLayout(content)
        self.body.setContentsMargins(4, 4, 8, 4)
        self.body.setSpacing(10)
        self.setWidget(content)
        self.lbl_hint = BioCaptionLabel(self.hint)
        self.lbl_hint.setVisible(bool(self.hint))
        self.body.addWidget(self.lbl_hint)
        self.sections: dict[str, QWidget] = {}  # heading text → its row, to hide it

    def section(self, text: str, help: str = "") -> SubtitleLabel:  # noqa: A002 - Qt's word
        """A heading, with a (?) button explaining the section when `help` is given."""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        label = SubtitleLabel(text)
        layout.addWidget(label)
        if help:
            layout.addWidget(help_button(help, text))
        layout.addStretch()
        self.body.addWidget(row)
        self.sections[text] = row
        return label

    def finish(self) -> None:
        self.body.addStretch()

    def apply_theme(self) -> None:
        """Re-colour plain Qt widgets; the Bio* ones restyle themselves."""


def help_button(text: str, title: str = "") -> BioHelpButton:
    button = BioHelpButton()
    button.setHelpText(text, title)
    return button


def with_help(widget: QWidget, text: str, title: str = "") -> QHBoxLayout:
    """`widget` with a (?) button after it, as a row to add to a layout."""
    row = QHBoxLayout()
    row.setSpacing(6)
    row.addWidget(widget, 1)
    row.addWidget(help_button(text, title))
    return row
