from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioComboBox,
    BioDoubleSpinBox,
    DangerButton,
    SecondaryButton,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QWidget,
)

from .base import StepPage

# (tool, button text, which plot it draws on, what to do)
GATE_TOOLS = [
    (
        "threshold",
        "📏 Threshold",
        "histogram",
        "Click on the histogram where the cut-off goes. Cells above it are in the gate.",
    ),
    ("range", "↔ Range", "histogram", "Drag across the histogram over the values to keep."),
    ("rect", "▭ Rectangle", "scatter", "Drag a box around a cluster on the scatter plot."),
    (
        "polygon",
        "⬠ Polygon",
        "scatter",
        "Click around a cluster on the scatter plot; double-click (or click the first "
        "corner) to close it. Esc cancels.",
    ),
]


class GateStep(StepPage):
    title = "Gate"
    hint = (
        "Sort cells into populations by their measurements, as in flow cytometry: draw a gate "
        "on the histogram or scatter plot below. Select a population to gate inside it."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Populations")
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Population", "Cells", "% parent", "% total"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setRootIsDecorated(True)
        self.tree.setMinimumHeight(110)
        self.tree.setMaximumHeight(180)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.body.addWidget(self.tree)
        self.chk_color_cells = QCheckBox("Colour cells on the image by population")
        self.chk_color_cells.setChecked(True)
        self.body.addWidget(self.chk_color_cells)

        self.section("New gate")
        self.tool_buttons: dict[str, SecondaryButton] = {}
        grid = QGridLayout()
        for i, (tool, text, _plot, tip) in enumerate(GATE_TOOLS):
            button = SecondaryButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            grid.addWidget(button, i // 2, i % 2)
            self.tool_buttons[tool] = button
        self.body.addLayout(grid)
        self.lbl_tool_hint = BioCaptionLabel(
            "Pick a tool, then draw on the plot. New gates go inside the selected population."
        )
        self.body.addWidget(self.lbl_tool_hint)
        row = QHBoxLayout()
        self.btn_combine = SecondaryButton("∩ Combine…")
        self.btn_combine.setToolTip("A population from two others: both (AND), either (OR) or NOT.")
        row.addWidget(self.btn_combine)
        row.addStretch()
        self.body.addLayout(row)

        self.section("Selected gate")
        self.lbl_definition = BioCaptionLabel("Select a gate to see and edit it.")
        self.body.addWidget(self.lbl_definition)
        self.editor = QWidget()
        editor = QGridLayout(self.editor)
        editor.setContentsMargins(0, 0, 0, 0)
        self.lbl_value = QLabel("Cut-off:")
        self.spin_value = _value_spin()
        self.combo_side = BioComboBox()
        self.combo_side.addItem("cells above", True)
        self.combo_side.addItem("cells below", False)
        editor.addWidget(self.lbl_value, 0, 0)
        editor.addWidget(self.spin_value, 0, 1)
        editor.addWidget(self.combo_side, 0, 2)
        self.lbl_range = QLabel("From / to:")
        self.spin_lo, self.spin_hi = _value_spin(), _value_spin()
        editor.addWidget(self.lbl_range, 1, 0)
        editor.addWidget(self.spin_lo, 1, 1)
        editor.addWidget(self.spin_hi, 1, 2)
        editor.setColumnStretch(1, 1)
        editor.setColumnStretch(2, 1)
        self.body.addWidget(self.editor)
        row = QHBoxLayout()
        self.btn_rename = SecondaryButton("✏️ Rename")
        self.btn_color = SecondaryButton("🎨 Colour")
        self.btn_delete = DangerButton("🗑 Delete")
        for button in (self.btn_rename, self.btn_color, self.btn_delete):
            row.addWidget(button)
        self.body.addLayout(row)
        self.finish()
        self.show_gate(None)
        self.apply_theme()

    def show_gate(self, gate) -> None:
        """Show `gate`'s definition and the editor for its kind (None: nothing selected)."""
        for button in (self.btn_rename, self.btn_color, self.btn_delete):
            button.setEnabled(gate is not None)
        if gate is None:
            self.lbl_definition.setText("Select a gate to see and edit it.")
            self.editor.setVisible(False)
            return
        parent = gate.parent or "All cells"
        self.lbl_definition.setText(f"<b>{gate.name}</b>: {gate.describe()}  (inside {parent})")
        threshold, ranged = gate.kind == "threshold", gate.kind == "range"
        self.editor.setVisible(threshold or ranged)
        for w in (self.lbl_value, self.spin_value, self.combo_side):
            w.setVisible(threshold)
        for w in (self.lbl_range, self.spin_lo, self.spin_hi):
            w.setVisible(ranged)
        widgets = (self.spin_value, self.combo_side, self.spin_lo, self.spin_hi)
        for w in widgets:
            w.blockSignals(True)
        if threshold:
            self.spin_value.setValue(float(gate.params.get("value", 0.0)))
            self.combo_side.setCurrentIndex(0 if gate.params.get("above", True) else 1)
        if ranged:
            self.spin_lo.setValue(float(gate.params.get("lo", 0.0)))
            self.spin_hi.setValue(float(gate.params.get("hi", 0.0)))
        for w in widgets:
            w.blockSignals(False)

    def apply_theme(self) -> None:
        for widget in (self.chk_color_cells, self.lbl_value, self.lbl_range):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.tree.setStyleSheet(
            f"QTreeWidget {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
            f"border: 1px solid {Colors.BORDER}; border-radius: 4px; }}"
            f"QHeaderView::section {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_SECONDARY}; "
            f"border: none; padding: 3px; }}"
        )


def _value_spin() -> BioDoubleSpinBox:
    spin = BioDoubleSpinBox()
    spin.setRange(-1e12, 1e12)
    spin.setDecimals(4)
    spin.setKeyboardTracking(False)  # apply once the number is typed, not per keystroke
    return spin
