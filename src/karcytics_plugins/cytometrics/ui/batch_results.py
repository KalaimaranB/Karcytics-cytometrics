"""The results dock's Batch tab: Images · Conditions · Plate."""

from __future__ import annotations

from karcytics_sdk.plugin import BioComboBox, BioTableWidget
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets.comparison_plot import ComparisonPlot
from .widgets.plate_heatmap import PlateHeatmap


class BatchResults(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images_table = BioTableWidget(0, 0)
        self.images_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.images_table.setSortingEnabled(True)
        self.addTab(self.images_table, "Images")

        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.lbl_metric, self.lbl_by, self.lbl_error = QLabel("Show:"), QLabel("By:"), QLabel("±")
        self.combo_metric = BioComboBox()
        self.combo_by = BioComboBox()
        self.combo_error = BioComboBox()
        self.combo_error.addItem("SEM", "SEM")
        self.combo_error.addItem("SD", "SD")
        for w in (
            self.lbl_metric,
            self.combo_metric,
            self.lbl_by,
            self.combo_by,
            self.lbl_error,
            self.combo_error,
        ):
            row.addWidget(
                w, 2 if w is self.combo_metric else (1 if isinstance(w, BioComboBox) else 0)
            )
        layout.addLayout(row)
        self.comparison = ComparisonPlot()
        self.conditions_table = BioTableWidget(0, 0)
        self.conditions_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.conditions_table.verticalHeader().setVisible(False)
        side = QHBoxLayout()
        side.addWidget(self.comparison, 3)
        side.addWidget(self.conditions_table, 2)
        layout.addLayout(side, 1)
        self.lbl_n = QLabel("")
        self.lbl_n.setWordWrap(True)
        layout.addWidget(self.lbl_n)
        self.addTab(page, "Conditions")

        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.lbl_plate_metric = QLabel("Show:")
        self.combo_plate_metric = BioComboBox()
        row.addWidget(self.lbl_plate_metric)
        row.addWidget(self.combo_plate_metric, 1)
        layout.addLayout(row)
        self.plate = PlateHeatmap()
        layout.addWidget(self.plate, 1)
        self.addTab(page, "Plate")
        self.apply_theme()

    def apply_theme(self) -> None:
        for label in (self.lbl_metric, self.lbl_by, self.lbl_error, self.lbl_plate_metric):
            label.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.lbl_n.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 11px;")
        for table in (self.images_table, self.conditions_table):
            table.setStyleSheet(
                f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
                f"gridline-color: {Colors.BORDER}; border: none;"
            )
        self.comparison.update()
        self.plate.update()
