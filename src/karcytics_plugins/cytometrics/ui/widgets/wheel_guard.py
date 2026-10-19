"""Scrolling a step page scrolls it, instead of changing the control under the mouse.

Qt hands a wheel event to whatever spin box, drop-down or slider is under the
pointer, so scrolling down a page of settings quietly changes them. Inside a
guarded scroll area, a control only takes the wheel once it has been clicked
(has focus); otherwise the page scrolls.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)

_GUARDED = "cytometrics_wheel_guard"  # the property marking a guarded scroll area
_CONTROLS = (QAbstractSpinBox, QComboBox, QAbstractSlider)


class _WheelGuard(QObject):
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt override
        if event.type() != QEvent.Type.Wheel or not isinstance(obj, _CONTROLS):
            return False
        if obj.hasFocus() or isinstance(obj, QScrollBar):
            return False  # clicked first; or a scroll bar, which should scroll
        area = _guarded_area(obj)
        if area is None:
            return False
        QApplication.sendEvent(area.verticalScrollBar(), event)
        return True


def _guarded_area(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea) and parent.property(_GUARDED):
            return parent
        parent = parent.parentWidget()
    return None


_guard: _WheelGuard | None = None


def guard_wheel(area: QAbstractScrollArea) -> None:
    """Let the wheel scroll `area` rather than change the controls inside it."""
    global _guard  # noqa: PLW0603 - one filter for the application
    area.setProperty(_GUARDED, True)
    app = QApplication.instance()
    if _guard is None and app is not None:
        _guard = _WheelGuard(app)
        app.installEventFilter(_guard)
