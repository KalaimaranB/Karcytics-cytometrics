"""Results under the controls: Table · Histogram · Scatter · Cell."""

from __future__ import annotations

from karcytics_sdk.plugin import BioComboBox, BioTableWidget
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets.histogram import Histogram
from .widgets.inspector import CellInspector
from .widgets.scatter import ScatterPlot


class ResultsDock(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.table = BioTableWidget(0, 0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSortingEnabled(True)
        self.addTab(self.table, "Table")

        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.combo_hist_metric = BioComboBox()
        self.chk_hist_log = QCheckBox("log")
        row.addWidget(self.combo_hist_metric, 1)
        row.addWidget(self.chk_hist_log)
        layout.addLayout(row)
        self.lbl_hist_population = QLabel("")
        layout.addWidget(self.lbl_hist_population)
        self.histogram = Histogram()
        layout.addWidget(self.histogram, 1)
        self.histogram_page = page
        self.addTab(page, "Histogram")

        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.lbl_x, self.lbl_y = QLabel("X:"), QLabel("Y:")
        self.combo_x, self.combo_y = BioComboBox(), BioComboBox()
        self.chk_x_log, self.chk_y_log = QCheckBox("log"), QCheckBox("log")
        for w in (
            self.lbl_x,
            self.combo_x,
            self.chk_x_log,
            self.lbl_y,
            self.combo_y,
            self.chk_y_log,
        ):
            row.addWidget(w, 1 if isinstance(w, BioComboBox) else 0)
        layout.addLayout(row)
        self.lbl_scatter_population = QLabel("")
        layout.addWidget(self.lbl_scatter_population)
        self.scatter = ScatterPlot()
        layout.addWidget(self.scatter, 1)
        self.scatter_page = page
        self.addTab(page, "Scatter")

        self.inspector = CellInspector()
        self.addTab(self.inspector, "Cell")
        self.apply_theme()

    def apply_theme(self) -> None:
        self.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY};
                padding: 5px 12px; border: 1px solid {Colors.BORDER}; border-bottom: none; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; }}
        """)
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
            f"gridline-color: {Colors.BORDER}; border: none;"
        )
        for label in (self.lbl_x, self.lbl_y, self.chk_hist_log, self.chk_x_log, self.chk_y_log):
            label.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        for label in (self.lbl_hist_population, self.lbl_scatter_population):
            label.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 11px;")
        self.scatter.update()
        self.histogram.update()
