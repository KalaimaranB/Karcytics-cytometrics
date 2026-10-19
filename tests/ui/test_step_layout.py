"""What the workspace shows when: the results panel, help buttons, and wheel scrolling."""

from __future__ import annotations

from karcytics_sdk.plugin import BioHelpButton
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from karcytics_plugins.cytometrics.analysis.segment import Param
from karcytics_plugins.cytometrics.ui.params_form import ParamsForm
from karcytics_plugins.cytometrics.ui.steps.base import StepPage


def _wheel(widget, dy: int = -120) -> None:
    event = QWheelEvent(
        QPointF(5, 5),
        QPointF(widget.mapToGlobal(QPoint(5, 5))),
        QPoint(0, 0),
        QPoint(0, dy),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_wheel_scrolls_the_page_not_the_setting(qtbot):
    page = StepPage()
    qtbot.addWidget(page)
    form = ParamsForm()
    form.set_params([Param("size", "Size", "float", 10.0, minimum=0, maximum=100, help="How big")])
    page.body.addWidget(form)
    for _ in range(40):
        page.section("Filler")
    page.finish()
    page.resize(300, 200)
    page.show()
    qtbot.waitExposed(page)
    spin = form.widget("size")
    _wheel(spin)
    assert form.values()["size"] == 10.0
    assert page.verticalScrollBar().value() > 0

    spin.setFocus()  # clicked: now the wheel is the control's
    qtbot.waitUntil(spin.hasFocus)
    _wheel(spin, 120)
    assert form.values()["size"] != 10.0


def test_settings_with_help_get_a_help_button(qtbot):
    form = ParamsForm()
    qtbot.addWidget(form)
    form.set_params(
        [Param("a", "A", "float", 1.0, help="Explains A"), Param("b", "B", "float", 1.0)]
    )
    assert len(form.findChildren(BioHelpButton)) == 1
    form.set_params([])
    assert form.isHidden()  # no settings: no empty form


def test_section_help(qtbot):
    page = StepPage()
    qtbot.addWidget(page)
    page.section("Plain")
    page.section("Explained", "Why this matters")
    assert len(page.sections["Plain"].findChildren(BioHelpButton)) == 0
    assert len(page.sections["Explained"].findChildren(BioHelpButton)) == 1
