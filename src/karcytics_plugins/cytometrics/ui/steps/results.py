from __future__ import annotations

import math

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    CitationLabel,
    ModuleCard,
    SecondaryButton,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ..params_form import ParamsForm
from .base import StepPage


class ResultCard(ModuleCard):
    """One headline value: its name, the number with its unit, and a warning if any."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(2)
        layout.setSizeConstraint(QVBoxLayout.SizeConstraint.SetMinimumSize)  # room for a warning
        self.lbl_name = QLabel("")
        self.lbl_value = QLabel("")
        self.lbl_warning = QLabel("")
        self.lbl_warning.setWordWrap(True)
        for label in (self.lbl_name, self.lbl_value, self.lbl_warning):
            layout.addWidget(label)
        self.apply_theme()

    def show_value(self, value) -> None:
        """`value`: an ``analysis.assays.ResultValue``."""
        self.lbl_name.setText(value.label)
        shown = format_value(value.value)
        self.lbl_value.setText(shown if shown == _MISSING else f"{shown} {value.unit}".strip())
        self.lbl_warning.setText(f"⚠️ {value.warning}" if value.warning else "")
        self.lbl_warning.setVisible(bool(value.warning))
        self.setToolTip(value.help)

    def apply_theme(self) -> None:
        plain = "background: transparent; border: none;"
        self.lbl_name.setStyleSheet(f"{plain} color: {Colors.FG_SECONDARY}; font-size: 11px;")
        self.lbl_value.setStyleSheet(
            f"{plain} color: {Colors.FG_PRIMARY}; font-size: 20px; font-weight: bold;"
        )
        self.lbl_warning.setStyleSheet(f"{plain} color: {Colors.ACCENT_WARNING}; font-size: 11px;")


class ResultsStep(StepPage):
    title = "Results"
    hint = "The assay's own results, from the measurements and the settings below."
    action_clicked = Signal(str)  # action id
    apply_gates_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Results")
        self.cards_box = QWidget()
        self.cards = QGridLayout(self.cards_box)
        self.cards.setContentsMargins(0, 0, 0, 0)
        self.cards.setSpacing(8)
        self.body.addWidget(self.cards_box)
        self.lbl_empty = BioCaptionLabel("Detect and measure cells to see the results.")
        self.body.addWidget(self.lbl_empty)
        self._cards: list[ResultCard] = []

        self.actions_box = QWidget()
        self.actions_layout = QVBoxLayout(self.actions_box)
        self.actions_layout.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(self.actions_box)
        self.btn_apply_gates = SecondaryButton("✨ Add the suggested gates")
        self.btn_apply_gates.clicked.connect(self.apply_gates_clicked)
        self.body.addWidget(self.btn_apply_gates)
        self.lbl_action = BioCaptionLabel("")
        self.body.addWidget(self.lbl_action)

        self.section("Settings")
        self.form = ParamsForm()
        self.body.addWidget(self.form)

        self.citation_box = QWidget()
        self.citation_layout = QVBoxLayout(self.citation_box)
        self.citation_layout.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(self.citation_box)
        self.finish()

    def set_assay(self, assay) -> None:
        """Rebuild the settings, actions and citations for `assay` (``analysis.assays.Assay``)."""
        analysis = assay.analysis
        self.form.set_params(analysis.parameters() if analysis else [])
        while self.actions_layout.count():
            item = self.actions_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for action, text, tip in analysis.actions if analysis else ():
            button = SecondaryButton(text)
            button.setToolTip(tip)
            button.clicked.connect(lambda _c=False, a=action: self.action_clicked.emit(a))
            self.actions_layout.addWidget(button)
        self.btn_apply_gates.setVisible(bool(analysis and analysis.suggests_gates))
        while self.citation_layout.count():
            item = self.citation_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        keys = tuple(dict.fromkeys((*assay.citations, *(analysis.citations if analysis else ()))))
        if keys:
            self.citation_layout.addWidget(CitationLabel(assay.name, keys, prefix="Method: "))

    def show_results(self, values: list) -> None:
        for card in self._cards:
            self.cards.removeWidget(card)
            card.hide()  # gone now, not at the next event loop turn
            card.deleteLater()
        self._cards = []
        for i, value in enumerate(values):
            card = ResultCard()
            card.show_value(value)
            self.cards.addWidget(card, i // 2, i % 2, Qt.AlignmentFlag.AlignTop)
            self._cards.append(card)
        self.lbl_empty.setVisible(not values)

    def apply_theme(self) -> None:
        self.form.apply_theme()
        for card in self._cards:
            card.apply_theme()


_MISSING = "–"


def format_value(value) -> str:
    """Numbers as people write them: 142, 87.3, 1.42 × 10⁸."""
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return _MISSING
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e5 or magnitude < 1e-3):  # noqa: PLR2004
        mantissa, exponent = f"{value:.3e}".split("e")
        return f"{float(mantissa):.3g} × 10{_superscript(int(exponent))}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:.3g}" if magnitude < 10 else f"{value:,.1f}"  # noqa: PLR2004


def _superscript(n: int) -> str:
    return str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
