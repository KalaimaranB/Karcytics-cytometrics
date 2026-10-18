"""Results under the controls: Table · Histogram · Scatter · Cell."""

from __future__ import annotations

from karcytics_sdk.plugin import BioComboBox, BioTableWidget
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QHBoxLayout, QHeaderView, QLabel, QTabWidget, QVBoxLayout, QWidget

from .widgets.histogram import SimpleHistogram
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
        self.combo_hist_metric = BioComboBox()
        layout.addWidget(self.combo_hist_metric)
        self.histogram = SimpleHistogram()
        layout.addWidget(self.histogram, 1)
        self.addTab(page, "Histogram")

        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.lbl_x, self.lbl_y = QLabel("X:"), QLabel("Y:")
        self.combo_x, self.combo_y = BioComboBox(), BioComboBox()
        for w in (self.lbl_x, self.combo_x, self.lbl_y, self.combo_y):
            row.addWidget(w, 1 if isinstance(w, BioComboBox) else 0)
        layout.addLayout(row)
        self.scatter = ScatterPlot()
        layout.addWidget(self.scatter, 1)
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
        for label in (self.lbl_x, self.lbl_y):
            label.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.scatter.update()
