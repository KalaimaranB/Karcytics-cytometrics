from __future__ import annotations

import math

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioListWidget,
    CitationLabel,
    ModuleCard,
    SecondaryButton,
    SubtitleLabel,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QListWidgetItem, QVBoxLayout, QWidget

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
    add_sample_clicked = Signal(str)  # role id
    remove_sample_clicked = Signal(str)  # sample id

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lbl_results = self.section("Results")
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

        # The experiment: images added with a role, and what they add up to.
        self.experiment_box = QWidget()
        experiment = QVBoxLayout(self.experiment_box)
        experiment.setContentsMargins(0, 6, 0, 0)
        experiment.setSpacing(8)
        experiment.addWidget(SubtitleLabel("Experiment"))
        self.lbl_experiment = BioCaptionLabel("")
        self.lbl_experiment.setWordWrap(True)
        experiment.addWidget(self.lbl_experiment)
        self.add_layout = QVBoxLayout()
        experiment.addLayout(self.add_layout)
        self.samples = BioListWidget()
        self.samples.setMaximumHeight(130)
        self.samples.setToolTip("The images in this experiment. Select one to remove it.")
        experiment.addWidget(self.samples)
        self.btn_remove = SecondaryButton("🗑 Remove from experiment")
        self.btn_remove.clicked.connect(self._on_remove)
        experiment.addWidget(self.btn_remove)
        self.experiment_cards_box = QWidget()
        self.experiment_cards = QGridLayout(self.experiment_cards_box)
        self.experiment_cards.setContentsMargins(0, 0, 0, 0)
        self.experiment_cards.setSpacing(8)
        experiment.addWidget(self.experiment_cards_box)
        self.lbl_note = BioCaptionLabel("")
        self.lbl_note.setWordWrap(True)
        experiment.addWidget(self.lbl_note)
        self._experiment_cards: list[ResultCard] = []
        self.body.addWidget(self.experiment_box)

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
        _clear(self.actions_layout)
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
        roles = analysis.roles if analysis else ()
        self.experiment_box.setVisible(bool(roles))
        self.lbl_results.setText("This image" if roles else "Results")
        _clear(self.add_layout)
        for role in roles:
            button = SecondaryButton(f"➕ Add this image as {role.label}")
            button.setToolTip(role.help)
            button.clicked.connect(lambda _c=False, r=role.id: self.add_sample_clicked.emit(r))
            self.add_layout.addWidget(button)
        self.lbl_experiment.setText(
            "Analyse each image, then add it here. Its measurements are kept, so load the "
            "next image with 🔄 Next image (or 📷 Load photo) and carry on."
        )
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

    def show_experiment(self, samples: list, comparison) -> None:
        """`samples`: ``experiment.Sample``s; `comparison`: their ``Comparison`` (or None)."""
        self.samples.clear()
        for s in samples:
            item = QListWidgetItem(
                f"{s.name}  ·  {len(s.table)} {'objects' if len(s.table) != 1 else 'object'}"
                + (f"  ·  {s.image}" if s.image else "")
            )
            item.setData(Qt.ItemDataRole.UserRole, s.id)
            self.samples.addItem(item)
        self.samples.setVisible(bool(samples))
        self.btn_remove.setVisible(bool(samples))
        for card in self._experiment_cards:
            self.experiment_cards.removeWidget(card)
            card.hide()
            card.deleteLater()
        self._experiment_cards = []
        for i, value in enumerate(comparison.values if comparison else []):
            card = ResultCard()
            card.show_value(value)
            self.experiment_cards.addWidget(card, i // 2, i % 2, Qt.AlignmentFlag.AlignTop)
            self._experiment_cards.append(card)
        self.lbl_note.setText(comparison.note if comparison else "")

    def _on_remove(self) -> None:
        item = self.samples.currentItem()
        if item is not None:
            self.remove_sample_clicked.emit(item.data(Qt.ItemDataRole.UserRole))

    def apply_theme(self) -> None:
        self.form.apply_theme()
        for card in (*self._cards, *self._experiment_cards):
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


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
