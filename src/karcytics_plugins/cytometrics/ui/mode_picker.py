"""The assay picker ("What are you analysing?"): a full-page grid of cards, and a quick-switch menu."""

from __future__ import annotations

from karcytics_sdk.plugin import BioScrollArea, ModuleCard, PrimaryButton
from karcytics_sdk.plugin.components import BioMenu
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..analysis.assays import PHOTO, Assay, by_group

_CARD_WIDTH = 300  # the grid gets one column per this many pixels
_MAX_COLUMNS = 3
_MAX_WIDTH = 1180  # wider than this, the page is centred rather than stretched
_SPACING = 16
_CARD_MAX = (_MAX_WIDTH - (_MAX_COLUMNS - 1) * _SPACING) // _MAX_COLUMNS
_CHIP_HEIGHT = 32  # Qt draws square corners if the radius is over half the height
_SHOW_ALL = "__all__"
_ALL = ""  # the filter showing every group
# A colour per picker group (Okabe–Ito), for the icon tile and the group label.
GROUP_COLORS = {
    "Cells": "#56B4E9",
    "Counting": "#E69F00",
    "Microbiology": "#009E73",
    "Growth & migration": "#CC79A7",
}


def _rgba(color: str, alpha: float) -> str:
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


class AssayCard(ModuleCard):
    clicked = Signal(str)

    def __init__(self, assay: Assay, parent=None):
        super().__init__(parent)
        self.assay_id = assay.id
        self.group = assay.group
        self._color = GROUP_COLORS.get(assay.group, "#56B4E9")
        self._last = False
        self.setObjectName("AssayCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(178)
        self.setMaximumWidth(_CARD_MAX)  # a filtered row of one or two stays card-sized
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(f"Start a {assay.name.lower()} analysis")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.lbl_icon = QLabel(assay.icon)
        self.lbl_icon.setFixedSize(46, 46)
        self.lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.lbl_icon)
        titles = QVBoxLayout()
        titles.setSpacing(3)
        self.lbl_name = QLabel(assay.name)
        self.lbl_group = QLabel(assay.group.upper())
        titles.addWidget(self.lbl_name)
        titles.addWidget(self.lbl_group)
        top.addLayout(titles, 1)
        self.lbl_last = QLabel("Last used")
        self.lbl_last.setFixedHeight(20)
        self.lbl_last.setVisible(False)
        top.addWidget(self.lbl_last, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(top)

        self.lbl_description = QLabel(assay.description)
        self.lbl_description.setWordWrap(True)
        self.lbl_description.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_description, 1)

        self.rule = QFrame()
        self.rule.setFixedHeight(1)
        layout.addWidget(self.rule)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        photo = assay.image_kind == PHOTO
        self.lbl_kind = QLabel("📷  Photo" if photo else "🔬  Stained images")
        self.lbl_kind.setToolTip(
            "Takes one photo (a plate, a chamber, a well)."
            if photo
            else "Takes one image per stain, or a multi-channel file."
        )
        self.lbl_outputs = QLabel(f"→  {assay.outputs}" if assay.outputs else "")
        self.lbl_outputs.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer.addWidget(self.lbl_kind)
        footer.addStretch()
        footer.addWidget(self.lbl_outputs)
        layout.addLayout(footer)
        self.apply_theme()

    def set_last_used(self, last: bool) -> None:
        self._last = last
        self.lbl_last.setVisible(last)
        self.apply_theme()

    def _apply_theme_styles(self) -> None:  # ModuleCard calls this when the theme changes
        self.apply_theme()

    def apply_theme(self) -> None:
        if not hasattr(self, "lbl_icon"):  # ModuleCard styles itself before the labels exist
            return
        border = Colors.ACCENT_PRIMARY if self._last else Colors.BORDER
        self.setStyleSheet(
            f"""
            QFrame#AssayCard {{
                background-color: {Colors.BG_DARK};
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QFrame#AssayCard:hover {{
                background-color: {Colors.BG_MEDIUM};
                border: 1px solid {Colors.ACCENT_PRIMARY};
            }}
            """
        )
        plain = "background: transparent; border: none;"
        self.lbl_icon.setStyleSheet(
            f"background: {_rgba(self._color, 0.16)}; border: 1px solid "
            f"{_rgba(self._color, 0.35)}; border-radius: 12px; font-size: 22px;"
        )
        self.lbl_name.setStyleSheet(
            f"{plain} color: {Colors.FG_PRIMARY}; font-size: 16px; font-weight: 600;"
        )
        self.lbl_group.setStyleSheet(
            f"{plain} color: {self._color}; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        self.lbl_description.setStyleSheet(
            f"{plain} color: {Colors.FG_SECONDARY}; font-size: 12px;"
        )
        self.rule.setStyleSheet(f"background: {Colors.BORDER}; border: none;")
        self.lbl_kind.setStyleSheet(f"{plain} color: {Colors.FG_SECONDARY}; font-size: 11px;")
        self.lbl_outputs.setStyleSheet(
            f"{plain} color: {Colors.ACCENT_PRIMARY}; font-size: 11px; font-weight: 600;"
        )
        self.lbl_last.setStyleSheet(
            f"background: {_rgba(Colors.ACCENT_PRIMARY, 0.15)}; color: {Colors.ACCENT_PRIMARY}; "
            f"border: 1px solid {_rgba(Colors.ACCENT_PRIMARY, 0.5)}; border-radius: 9px; "
            "padding: 0px 9px; font-size: 10px; font-weight: 600;"
        )

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.assay_id)
        super().mouseReleaseEvent(event)


class ModePicker(QWidget):
    """Every assay as a card in one grid, centred on the page, with filters by group."""

    assay_chosen = Signal(str)

    def __init__(self, assays: list[Assay], parent=None):
        super().__init__(parent)
        self._assays = assays
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 24)

        page = QWidget()
        page.setMaximumWidth(_MAX_WIDTH)
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)

        self.lbl_eyebrow = QLabel("CYTOMETRICS")
        self.lbl_title = QLabel("What are you analysing?")
        self.lbl_subtitle = QLabel(
            "Pick the kind of experiment. It sets up the steps, the starting algorithm, the "
            "measurements and the results; switch any time from the button at the top."
        )
        self.lbl_subtitle.setWordWrap(True)
        for label in (self.lbl_eyebrow, self.lbl_title, self.lbl_subtitle):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(self.lbl_eyebrow)
        column.addSpacing(8)
        column.addWidget(self.lbl_title)
        column.addSpacing(8)
        column.addWidget(self.lbl_subtitle)

        self.btn_continue = PrimaryButton("")
        self.btn_continue.setVisible(False)
        self.btn_continue.clicked.connect(self._on_continue)
        continue_row = QHBoxLayout()
        continue_row.addStretch()
        continue_row.addWidget(self.btn_continue)
        continue_row.addStretch()
        column.addSpacing(16)
        column.addLayout(continue_row)

        # Filter by group: All, then each group, as pills.
        self.filters = QButtonGroup(self)
        self.filters.setExclusive(True)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.addStretch()
        self.chips: dict[str, QPushButton] = {}
        groups = [g for g, _ in by_group(assays)]
        for key, text in [(_ALL, "All"), *((g, g) for g in groups)]:
            chip = QPushButton(text.replace("&", "&&"))
            chip.setCheckable(True)
            chip.setFixedHeight(_CHIP_HEIGHT)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setChecked(key == _ALL)
            chip.clicked.connect(lambda _c=False, k=key: self.set_filter(k))
            self.filters.addButton(chip)
            self.chips[key] = chip
            chips.addWidget(chip)
        chips.addStretch()
        column.addSpacing(22)
        column.addLayout(chips)
        column.addSpacing(22)

        self.cards: dict[str, AssayCard] = {}
        grid_box = QWidget()  # centred, so a filtered row of one or two cards sits in the middle
        self.grid = QGridLayout(grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(_SPACING)
        self.grid.setVerticalSpacing(_SPACING)
        for _group, members in by_group(assays):
            for assay in members:
                card = AssayCard(assay)
                card.clicked.connect(self.assay_chosen)
                self.cards[assay.id] = card
        centred = QHBoxLayout()
        centred.addStretch(1)
        centred.addWidget(grid_box, 100)
        centred.addStretch(1)
        column.addLayout(centred)

        # Centred both ways; the page scrolls when the window is too small for it.
        holder = QWidget()
        rows = QVBoxLayout(holder)
        rows.setContentsMargins(0, 0, 8, 0)
        rows.addStretch(1)
        across = QHBoxLayout()
        across.addStretch(1)
        across.addWidget(page, 100)
        across.addStretch(1)
        rows.addLayout(across)
        rows.addStretch(2)  # a little above the middle, where the eye rests
        scroll = BioScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(BioScrollArea.Shape.NoFrame)
        scroll.setWidget(holder)
        outer.addWidget(scroll)

        self._filter = _ALL
        self._columns = 0
        self._current: str | None = None
        self._reflow(force=True)
        self.apply_theme()

    # ── Public API ───────────────────────────────────────────────────────────

    def set_current(self, assay: Assay | None, last_used: str | None = None) -> None:
        """Offer to go back to `assay` (the one open now); mark `last_used` on its card."""
        self._current = assay.id if assay else None
        if assay is not None:
            self.btn_continue.setText(f"Continue with {assay.name} →".replace("&", "&&"))
        self.btn_continue.setVisible(assay is not None)
        for assay_id, card in self.cards.items():
            card.set_last_used(assay_id == (last_used or self._current))

    def set_filter(self, group: str) -> None:
        """Show only `group`'s assays ("" for all)."""
        self._filter = group if group in self.chips else _ALL
        self.chips[self._filter].setChecked(True)
        self._reflow(force=True)

    def visible_cards(self) -> list[AssayCard]:
        return [c for c in self.cards.values() if self._filter in (_ALL, c.group)]

    def columns_for(self, width: int) -> int:
        return max(1, min(_MAX_COLUMNS, width // _CARD_WIDTH))

    # ── Layout ───────────────────────────────────────────────────────────────

    def _on_continue(self) -> None:
        if self._current:
            self.assay_chosen.emit(self._current)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow()

    def _reflow(self, force: bool = False) -> None:
        columns = self.columns_for(min(self.width() - 80, _MAX_WIDTH))
        if columns == self._columns and not force:
            return
        self._columns = columns
        for card in self.cards.values():
            self.grid.removeWidget(card)
            card.setVisible(False)
        shown = self.visible_cards()
        used = min(columns, len(shown)) or 1
        for i, card in enumerate(shown):
            self.grid.addWidget(card, i // used, i % used)
            card.setVisible(True)
        for c in range(_MAX_COLUMNS):
            self.grid.setColumnStretch(c, 1 if c < used else 0)
        grid_box = self.grid.parentWidget()
        if grid_box is not None:
            grid_box.setMaximumWidth(used * _CARD_MAX + (used - 1) * _SPACING)

    def apply_theme(self) -> None:
        self.lbl_eyebrow.setStyleSheet(
            f"color: {Colors.ACCENT_PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: 3px;"
        )
        self.lbl_title.setStyleSheet(
            f"color: {Colors.FG_PRIMARY}; font-size: 30px; font-weight: 700;"
        )
        self.lbl_subtitle.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 13px;")
        for chip in self.chips.values():
            chip.setStyleSheet(
                f"""
                QPushButton {{
                    background: transparent; color: {Colors.FG_SECONDARY};
                    border: 1px solid {Colors.BORDER}; border-radius: 15px;
                    padding: 0px 16px; font-size: 12px;
                }}
                QPushButton:hover {{
                    color: {Colors.FG_PRIMARY}; border-color: {Colors.FG_SECONDARY};
                }}
                QPushButton:checked {{
                    background: {_rgba(Colors.ACCENT_PRIMARY, 0.15)};
                    color: {Colors.ACCENT_PRIMARY}; border-color: {Colors.ACCENT_PRIMARY};
                    font-weight: 600;
                }}
                """
            )
        for card in self.cards.values():
            card.apply_theme()


def assay_menu(assays: list[Assay], current: str | None, parent=None) -> BioMenu:
    """The quick switch: every assay by group, then "Show all…" (data: assay id or "__all__")."""
    menu = BioMenu(parent)
    for group, members in by_group(assays):
        if menu.actions():
            menu.addSeparator()
        heading = menu.addAction(group.upper().replace("&", "&&"))
        heading.setEnabled(False)  # a heading (sections show no title on macOS)
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
