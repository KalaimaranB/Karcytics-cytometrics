"""Step bar + pages + Back/Next. Every step can be visited at any time."""

from __future__ import annotations

from karcytics_sdk.plugin import PrimaryButton, SecondaryButton
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QStackedWidget, QVBoxLayout, QWidget

from .steps.base import StepPage
from .widgets.step_bar import StepBar


class Stepper(QWidget):
    step_changed = Signal(str)  # the new step's id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ids: list[str] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._indicator: StepBar | None = None
        self._stack = QStackedWidget()
        self._layout.addWidget(self._stack, 1)

        buttons = QHBoxLayout()
        self.btn_back = SecondaryButton("← Back")
        self.btn_next = PrimaryButton("Next →")
        self.btn_back.clicked.connect(lambda: self.go_to(self.current_index - 1))
        self.btn_next.clicked.connect(lambda: self.go_to(self.current_index + 1))
        buttons.addWidget(self.btn_back)
        buttons.addStretch()
        buttons.addWidget(self.btn_next)
        self._layout.addLayout(buttons)

    def set_steps(self, steps: list[tuple[str, str, StepPage]]) -> None:
        """Show `steps` as ``(id, label, page)``, in order. Pages are reparented, not rebuilt."""
        while self._stack.count():
            self._stack.removeWidget(self._stack.widget(0))
        if self._indicator is not None:
            self._layout.removeWidget(self._indicator)
            self._indicator.deleteLater()
        self._ids = [sid for sid, _, _ in steps]
        self._indicator = StepBar([label for _, label, _ in steps])
        self._indicator.step_clicked.connect(self.go_to)
        self._layout.insertWidget(0, self._indicator)
        for _, _, page in steps:
            self._stack.addWidget(page)
        self.go_to(0)

    @property
    def current_index(self) -> int:
        return self._stack.currentIndex()

    @property
    def current_id(self) -> str:
        return self._ids[self.current_index] if self._ids else ""

    def step_ids(self) -> list[str]:
        return list(self._ids)

    def go_to(self, index_or_id: int | str) -> None:
        index = self._ids.index(index_or_id) if isinstance(index_or_id, str) else index_or_id
        if not 0 <= index < len(self._ids):
            return
        self._stack.setCurrentIndex(index)
        assert self._indicator is not None
        self._indicator.set_current(index)
        self.btn_back.setEnabled(index > 0)
        self.btn_next.setEnabled(index < len(self._ids) - 1)
        self.step_changed.emit(self._ids[index])

    def apply_theme(self) -> None:
        if self._indicator is not None:
            self._indicator.apply_theme()
