"""A one-line step bar: Load › Calibrate › Detect › … with the current step highlighted."""

from __future__ import annotations

from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QLabel, QPushButton, QWidget


class StepBar(QWidget):
    step_clicked = Signal(int)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self.buttons: list[QPushButton] = []
        self._arrows: list[QLabel] = []
        for i, label in enumerate(labels):
            if i:
                arrow = QLabel("›")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(arrow)
                self._arrows.append(arrow)
            button = QPushButton(label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip(f"Step {i + 1}: {label}")
            self._group.addButton(button, i)
            layout.addWidget(button)
            self.buttons.append(button)
        layout.addStretch()
        self._group.idClicked.connect(self.step_clicked)
        self.apply_theme()

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.buttons):
            self.buttons[index].setChecked(True)

    def apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {Colors.FG_SECONDARY};
                border: 1px solid transparent; border-radius: 10px;
                padding: 3px 8px; font-size: 12px;
            }}
            QPushButton:hover {{ color: {Colors.FG_PRIMARY}; border-color: {Colors.BORDER}; }}
            QPushButton:checked {{
                background: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST}; font-weight: bold;
            }}
            QLabel {{ color: {Colors.FG_SECONDARY}; }}
        """)
