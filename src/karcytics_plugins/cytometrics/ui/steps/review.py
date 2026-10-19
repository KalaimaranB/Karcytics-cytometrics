from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioComboBox,
    BioDoubleSpinBox,
    BioSpinBox,
    DangerButton,
    SecondaryButton,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .base import StepPage

# (canvas mode, button text, tooltip)
TOOLS = [
    (
        "PAN",
        "✋ Select",
        "Click a cell to select it, Shift+click for several; drag to move around.",
    ),
    ("DRAW", "✏️ Draw", "Outline a missed cell by dragging around it."),
    ("SPLIT", "✂️ Split", "Draw a line across a cell to cut it in two."),
    ("BRUSH_ADD", "🖌 Brush +", "Paint pixels onto the selected cell."),
    ("BRUSH_ERASE", "🧽 Brush −", "Erase pixels from the selected cell."),
    ("BOX", "⬚ Re-detect", "Drag a box: the current algorithm runs again inside it."),
]


class ReviewStep(StepPage):
    title = "Review"
    hint = (
        "Check the outlines on the canvas. Right-click a cell to delete it; draw the ones that "
        "were missed."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section(
            "Show",
            "What's drawn over the image. Colouring the outlines by a measurement shows at a "
            "glance which objects are large, bright or round; by AI confidence, which ones the "
            "model was unsure of.",
        )
        row = QHBoxLayout()
        self.chk_show_outlines = QCheckBox("Outlines")
        self.chk_show_outlines.setChecked(True)
        self.chk_show_ids = QCheckBox("Cell ID numbers")
        self.chk_show_ids.setChecked(True)
        row.addWidget(self.chk_show_outlines)
        row.addWidget(self.chk_show_ids)
        row.addStretch()
        self.body.addLayout(row)

        row = QHBoxLayout()
        self.lbl_color_by = QLabel("Colour outlines by:")
        self.combo_color_by = BioComboBox()
        self.combo_color_by.setToolTip(
            "AI confidence: red = uncertain, amber = borderline, green = confident. "
            "A measurement: dark purple = lowest, yellow = highest (viridis)."
        )
        row.addWidget(self.lbl_color_by)
        row.addWidget(self.combo_color_by, 1)
        self.body.addLayout(row)
        self.lbl_color_legend = BioCaptionLabel("")
        self.body.addWidget(self.lbl_color_legend)
        self.chk_probability_map = QCheckBox("Cellpose cell-probability map")
        self.chk_probability_map.setToolTip("Where the model thinks cells are, from the last run.")
        self.body.addWidget(self.chk_probability_map)

        self.section(
            "Uncertain cells",
            "Cellpose gives each cell a confidence: its mean cell probability inside the outline. "
            "Low-confidence cells are where it most often merges, splits or invents a cell, so "
            "checking those first finds most mistakes for the least effort.",
        )
        self.uncertain_box = QWidget()
        uncertain = QVBoxLayout(self.uncertain_box)
        uncertain.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.lbl_uncertain = QLabel("Confidence below:")
        self.spin_uncertain = BioDoubleSpinBox()
        self.spin_uncertain.setRange(0.0, 1.0)
        self.spin_uncertain.setSingleStep(0.05)
        self.spin_uncertain.setDecimals(2)
        self.spin_uncertain.setValue(0.75)
        self.spin_uncertain.setToolTip("Cells whose AI confidence is below this need a look.")
        self.btn_next_uncertain = SecondaryButton("Next uncertain ▸")
        row.addWidget(self.lbl_uncertain)
        row.addWidget(self.spin_uncertain)
        row.addWidget(self.btn_next_uncertain)
        uncertain.addLayout(row)
        uncertain.addWidget(
            BioCaptionLabel("Steps through them least confident first and centres each one.")
        )
        self.body.addWidget(self.uncertain_box)

        self.section(
            "Edit",
            "Fix the outlines by hand. Select picks cells (Shift+click for several) to merge or "
            "delete; Draw adds a missed cell; Split cuts one in two along a line; the brushes "
            "add or erase pixels of the selected cell; Re-detect runs the algorithm again inside "
            "a box. Right-click a cell to delete it. Every edit can be undone.",
        )
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        grid = QGridLayout()
        self.tool_buttons: dict[str, SecondaryButton] = {}
        for i, (mode, text, tip) in enumerate(TOOLS):
            button = SecondaryButton(text)
            button.setCheckable(True)
            button.setToolTip(tip)
            self.tool_group.addButton(button, i)
            grid.addWidget(button, i // 3, i % 3)
            self.tool_buttons[mode] = button
        self.tool_buttons["PAN"].setChecked(True)
        self.body.addLayout(grid)

        row = QHBoxLayout()
        self.lbl_brush = QLabel("Brush size:")
        self.spin_brush = BioSpinBox()
        self.spin_brush.setRange(1, 50)
        self.spin_brush.setValue(4)
        self.spin_brush.setSuffix(" px")
        row.addWidget(self.lbl_brush)
        row.addWidget(self.spin_brush)
        row.addStretch()
        self.body.addLayout(row)

        self.lbl_selection = BioCaptionLabel("Click a cell to select it; Shift+click adds more.")
        self.body.addWidget(self.lbl_selection)
        row = QHBoxLayout()
        self.btn_merge = SecondaryButton("⧉ Merge selected")
        self.btn_merge.setToolTip("Join touching cells into one.")
        self.btn_delete = DangerButton("🗑 Delete selected")
        self.btn_merge.setEnabled(False)
        self.btn_delete.setEnabled(False)
        row.addWidget(self.btn_merge)
        row.addWidget(self.btn_delete)
        self.body.addLayout(row)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        for widget in (
            self.chk_show_outlines,
            self.chk_show_ids,
            self.lbl_color_by,
            self.chk_probability_map,
            self.lbl_uncertain,
            self.lbl_brush,
        ):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
