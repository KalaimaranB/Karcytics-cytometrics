"""The assay picker ("What are you analysing?"): a full-page grid of cards, and a quick-switch menu."""

from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioScrollArea,
    HeaderLabel,
    ModuleCard,
    SecondaryButton,
)
from karcytics_sdk.plugin.components import BioMenu
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..analysis.assays import Assay, by_group

_CARD_WIDTH = 300  # the grid gets one column per this many pixels
_MAX_COLUMNS = 3
_MAX_WIDTH = 1200  # wider than this, the page is centred rather than stretched
_SHOW_ALL = "__all__"


class AssayCard(ModuleCard):
    clicked = Signal(str)

    def __init__(self, assay: Assay, parent=None):
        super().__init__(parent)
        self.assay_id = assay.id
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.lbl_icon = QLabel(assay.icon)
        self.lbl_name = QLabel(assay.name)
        self.lbl_last = QLabel("Last used")
        self.lbl_last.setVisible(False)
        top.addWidget(self.lbl_icon)
        top.addWidget(self.lbl_name, 1)
        top.addWidget(self.lbl_last)
        layout.addLayout(top)
        self.lbl_description = QLabel(assay.description)
        self.lbl_description.setWordWrap(True)
        layout.addWidget(self.lbl_description)
        layout.addStretch()
        self.setToolTip(f"Start a {assay.name.lower()} analysis")
        self.apply_theme()

    def set_last_used(self, last: bool) -> None:
        self.lbl_last.setVisible(last)

    def apply_theme(self) -> None:
        plain = "background: transparent; border: none;"
        self.lbl_icon.setStyleSheet(f"{plain} font-size: 26px;")
        self.lbl_name.setStyleSheet(
            f"{plain} color: {Colors.FG_PRIMARY}; font-size: 15px; font-weight: 600;"
        )
        self.lbl_description.setStyleSheet(
            f"{plain} color: {Colors.FG_SECONDARY}; font-size: 12px;"
        )
        self.lbl_last.setStyleSheet(
            f"background: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST}; border: none; "
            "border-radius: 8px; padding: 2px 8px; font-size: 10px; font-weight: 600;"
        )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.assay_id)
        super().mouseReleaseEvent(event)


class ModePicker(QWidget):
    """Every assay as a card, grouped, filling the panel; reflows to the width."""

    assay_chosen = Signal(str)

    def __init__(self, assays: list[Assay], parent=None):
        super().__init__(parent)
        self._assays = assays
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 16)
        outer.setSpacing(6)

        header_box = QWidget()
        header_box.setMaximumWidth(_MAX_WIDTH)
        header = QHBoxLayout(header_box)
        header.setContentsMargins(0, 0, 8, 0)  # lines up with the cards (scroll bar gap)
        titles = QVBoxLayout()
        titles.addWidget(HeaderLabel("What are you analysing?"))
        titles.addWidget(
            BioCaptionLabel(
                "Pick the kind of experiment: it sets up the steps, the starting algorithm, the "
                "measurements and the results. You can switch any time from the button at the top."
            )
        )
        header.addLayout(titles, 1)
        self.btn_continue = SecondaryButton("")
        self.btn_continue.setVisible(False)
        self.btn_continue.clicked.connect(self._on_continue)
        header.addWidget(self.btn_continue, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(_centred(header_box))

        self.cards: dict[str, AssayCard] = {}
        self._groups: list[tuple[QLabel, list[AssayCard]]] = []
        content = QWidget()
        content.setMaximumWidth(_MAX_WIDTH)
        column = QVBoxLayout(content)
        column.setContentsMargins(0, 8, 8, 8)
        # One grid for every group, so the columns line up from group to group.
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(12)
        column.addLayout(self.grid)
        for group, members in by_group(assays):
            cards = []
            for assay in members:
                card = AssayCard(assay)
                card.clicked.connect(self.assay_chosen)
                self.cards[assay.id] = card
                cards.append(card)
            self._groups.append((QLabel(group.upper()), cards))
        column.addStretch()
        scroll = BioScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(BioScrollArea.Shape.NoFrame)
        holder = QWidget()
        holder.setLayout(_centred(content))
        scroll.setWidget(holder)
        outer.addWidget(scroll, 1)
        self._columns = 0
        self._current: str | None = None
        self._reflow()
        self.apply_theme()

    def set_current(self, assay: Assay | None, last_used: str | None = None) -> None:
        """Offer to go back to `assay` (the one open now); mark `last_used` on its card."""
        self._current = assay.id if assay else None
        if assay is not None:
            self.btn_continue.setText(f"Continue with {assay.name} →".replace("&", "&&"))
        self.btn_continue.setVisible(assay is not None)
        for assay_id, card in self.cards.items():
            card.set_last_used(assay_id == (last_used or self._current))

    def _on_continue(self) -> None:
        if self._current:
            self.assay_chosen.emit(self._current)

    def columns_for(self, width: int) -> int:
        return max(1, min(_MAX_COLUMNS, width // _CARD_WIDTH))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self) -> None:
        columns = self.columns_for(min(self.width() - 80, _MAX_WIDTH))
        if columns == self._columns:
            return
        self._columns = columns
        for label, cards in self._groups:
            for w in (label, *cards):
                self.grid.removeWidget(w)
        row = 0
        for label, cards in self._groups:
            self.grid.addWidget(label, row, 0, 1, columns)
            self.grid.setRowMinimumHeight(row, 28 if row else 0)  # space above each heading
            row += 1
            for i, card in enumerate(cards):
                self.grid.addWidget(card, row + i // columns, i % columns)
            row += (len(cards) + columns - 1) // columns
        for c in range(_MAX_COLUMNS):
            self.grid.setColumnStretch(c, 1 if c < columns else 0)

    def apply_theme(self) -> None:
        for label, _cards in self._groups:
            label.setStyleSheet(
                f"color: {Colors.FG_SECONDARY}; font-size: 11px; font-weight: 600; "
                "letter-spacing: 1px;"
            )
        for card in self.cards.values():
            card.apply_theme()


def _centred(widget: QWidget) -> QHBoxLayout:
    """A row holding `widget` centred, as wide as it may be (its maximum width)."""
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch()
    row.addWidget(widget, 100)
    row.addStretch()
    return row


def assay_menu(assays: list[Assay], current: str | None, parent=None) -> BioMenu:
    """The quick switch: every assay by group, then "Show all…" (data: assay id or "__all__")."""
    menu = BioMenu(parent)
    for group, members in by_group(assays):
        if menu.actions():
            menu.addSeparator()
        menu.addAction(group.upper().replace("&", "&&")).setEnabled(
            False
        )  # a heading (sections show no title on macOS)
        for assay in members:
            action = menu.addAction(f"{assay.icon}  {assay.name}".replace("&", "&&"))
            action.setData(assay.id)
            action.setCheckable(True)
            action.setChecked(assay.id == current)
            action.setToolTip(assay.description)
    menu.addSeparator()
    menu.addAction("▦  Show all analyses…").setData(_SHOW_ALL)
    menu.setToolTipsVisible(True)
    return menu


SHOW_ALL = _SHOW_ALL
