"""The results dock's Compare tab: the experiment's plot and one row per image."""

from __future__ import annotations

import numpy as np
import pandas as pd
from karcytics_sdk.plugin import BioTableWidget
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtWidgets import QHeaderView, QLabel, QStackedWidget, QVBoxLayout, QWidget

from .labels import fmt
from .widgets.comparison_plot import ComparisonPlot
from .widgets.overlay_histogram import OverlayHistogram
from .widgets.table_items import NumericTableItem

GROUP, VALUE = "group", "value"


class CompareView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plots = QStackedWidget()
        self.histogram = OverlayHistogram()
        self.bars = ComparisonPlot()
        self.bars.empty_text = "Add images to the experiment to compare them"
        self.plots.addWidget(self.histogram)
        self.plots.addWidget(self.bars)
        layout.addWidget(self.plots, 3)
        self.table = BioTableWidget(0, 0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 2)
        self.lbl_note = QLabel("")
        self.lbl_note.setWordWrap(True)
        layout.addWidget(self.lbl_note)
        self.apply_theme()

    def show_comparison(self, comparison) -> None:
        """`comparison`: an ``analysis.experiment.Comparison`` (None: nothing yet)."""
        plot = comparison.plot if comparison else None
        if plot is not None and plot.kind == "bars":
            self.plots.setCurrentWidget(self.bars)
            points = plot.points
            summary = _summary(points)
            self.bars.band = plot.band
            self.bars.band_label = f"{plot.band[0]:g}–{plot.band[1]:g}" if plot.band else ""
            self.bars.set_data(summary, points, GROUP, VALUE, plot.label, "SD")
        else:
            self.plots.setCurrentWidget(self.histogram)
            if plot is None:
                self.histogram.set_data([])
            else:
                self.histogram.set_data(plot.series, plot.lines, plot.label, plot.log)
        table = comparison.table if comparison else pd.DataFrame()
        self.table.clear()
        self.table.setColumnCount(len(table.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in table.columns])
        self.table.setRowCount(len(table))
        for row, record in enumerate(table.to_dict("records")):
            for col, name in enumerate(table.columns):
                self.table.setItem(row, col, NumericTableItem(fmt(record[name])))
        self.lbl_note.setText(comparison.note if comparison else "")

    def apply_theme(self) -> None:
        self.lbl_note.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 11px;")
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; "
            f"gridline-color: {Colors.BORDER}; border: none;"
        )
        self.histogram.update()
        self.bars.update()


def _summary(points: pd.DataFrame) -> pd.DataFrame:
    """Mean, SD and n per group, in the order the groups first appear."""
    if not len(points):
        return pd.DataFrame(columns=[GROUP, "n", "Mean", "SD", "SEM"])
    grouped = points.groupby(GROUP, sort=False)[VALUE]
    out = pd.DataFrame({"n": grouped.count(), "Mean": grouped.mean(), "SD": grouped.std(ddof=1)})
    out["SEM"] = out["SD"] / np.sqrt(out["n"])
    return out.reset_index()
