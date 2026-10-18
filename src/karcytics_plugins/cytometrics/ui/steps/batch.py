from __future__ import annotations

from karcytics_sdk.plugin import (
    BioCaptionLabel,
    BioLineEdit,
    DangerButton,
    PrimaryButton,
    SecondaryButton,
)
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidget,
    QWidget,
)

from ..widgets.scanning_indicator import ScanningIndicator
from .base import StepPage

PATTERN_HELP = (
    "Read conditions from file names: write how they are built, with {well}, {condition}, "
    "{replicate} or any other {field} (e.g. {well}_{condition}_r{replicate}). Wells such as "
    "A01 or B3 are found without a pattern."
)


class BatchStep(StepPage):
    title = "Batch"
    hint = (
        "Run this analysis (the same detection settings, measurements and gates) on many "
        "images, such as a whole plate, then compare conditions."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.section("Analysis to repeat")
        self.lbl_recipe = BioCaptionLabel("")
        self.lbl_recipe.setWordWrap(True)
        self.body.addWidget(self.lbl_recipe)

        self.section("Images")
        grid = QGridLayout()
        self.btn_add_folder = SecondaryButton("📁 Add folder…")
        self.btn_add_files = SecondaryButton("➕ Add files…")
        self.btn_layout = SecondaryButton("🧾 Plate layout…")
        self.btn_layout.setToolTip(
            "A CSV giving each well's condition: either a Well column (plus Condition, "
            "Replicate, Dose, …) or a plate grid with row letters and column numbers."
        )
        self.btn_clear = DangerButton("Clear list")
        for i, button in enumerate(
            (self.btn_add_folder, self.btn_add_files, self.btn_layout, self.btn_clear)
        ):
            grid.addWidget(button, i // 2, i % 2)
        self.body.addLayout(grid)

        row = QHBoxLayout()
        self.lbl_pattern = QLabel("File names:")
        self.edit_pattern = BioLineEdit()
        self.edit_pattern.setPlaceholderText("{well}_{condition}_r{replicate}")
        self.edit_pattern.setToolTip(PATTERN_HELP)
        self.btn_pattern = SecondaryButton("Apply")
        row.addWidget(self.lbl_pattern)
        row.addWidget(self.edit_pattern, 1)
        row.addWidget(self.btn_pattern)
        self.body.addLayout(row)
        self.body.addWidget(BioCaptionLabel(PATTERN_HELP))

        self.table = QTableWidget(0, 0)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(160)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setToolTip("Double-click a well, condition or replicate to change it.")
        self.body.addWidget(self.table)
        row = QHBoxLayout()
        self.btn_set_condition = SecondaryButton("🏷 Set condition…")
        self.btn_set_condition.setToolTip("Give the selected images a condition (and replicate).")
        self.btn_open = SecondaryButton("🔍 Open in editor")
        self.btn_open.setToolTip(
            "Load the selected image and its cells into the steps above to check or fix them."
        )
        row.addWidget(self.btn_set_condition)
        row.addWidget(self.btn_open)
        self.body.addLayout(row)
        self.lbl_list = BioCaptionLabel("")
        self.body.addWidget(self.lbl_list)

        self.editing = QWidget()
        editing = QHBoxLayout(self.editing)
        editing.setContentsMargins(0, 0, 0, 0)
        self.lbl_editing = BioCaptionLabel("")
        self.btn_save_back = PrimaryButton("💾 Save changes to batch")
        editing.addWidget(self.lbl_editing, 1)
        editing.addWidget(self.btn_save_back)
        self.editing.setVisible(False)
        self.body.addWidget(self.editing)

        self.section("Run")
        row = QHBoxLayout()
        self.btn_run = PrimaryButton("▶ Run batch")
        self.btn_stop = DangerButton("⏹ Stop")
        self.btn_stop.setEnabled(False)
        self.btn_rerun = SecondaryButton("↻ Re-run all")
        self.btn_rerun.setToolTip("Analyse every image again, including finished ones.")
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_rerun)
        self.body.addLayout(row)
        self.progress = ScanningIndicator()
        self.progress.setVisible(False)
        self.body.addWidget(self.progress)
        self.lbl_progress = BioCaptionLabel(
            "Results are saved after every image: stop any time and run again to continue."
        )
        self.body.addWidget(self.lbl_progress)

        self.section("Results")
        self.body.addWidget(
            BioCaptionLabel("Per image, per condition and plate views are in the Batch tab below.")
        )
        self.btn_export = SecondaryButton("💾 Export batch results…")
        self.btn_export.setToolTip(
            "A folder with every cell (CSV), one row per image, per-condition statistics, one "
            "FCS file per image, and the methods text with references."
        )
        self.body.addWidget(self.btn_export)
        self.finish()
        self.apply_theme()

    def apply_theme(self) -> None:
        self.lbl_pattern.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.table.setStyleSheet(
            f"QTableWidget {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
            f"gridline-color: {Colors.BORDER}; border: 1px solid {Colors.BORDER}; }}"
            f"QHeaderView::section {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.FG_SECONDARY}; border: none; padding: 3px; }}"
        )
