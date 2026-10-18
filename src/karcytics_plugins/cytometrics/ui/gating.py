"""Gating: the Gate step, the histogram and the scatter plot working together.

Selecting a gate in the population tree shows it on the plots, highlighted,
over its parent's cells (the cells it was drawn on) next to its siblings.
Arming a drawing tool switches the plots to the selected population's own
cells: the new gate goes inside it, which is how gates are nested. Every
change goes through the session and is announced with
``gates_changed(label)``, which the panel turns into an undo step.
"""

from __future__ import annotations

import math

from karcytics_sdk.plugin import BioComboBox, ask_yes_no
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QInputDialog,
    QTreeWidgetItem,
    QWidget,
)

from ..analysis.gating import ALL, BOOLEAN_KINDS, UNGATED_COLOR, Gate, remove
from .labels import column_label, numeric_columns, short_name
from .steps.gate import GATE_TOOLS, GateStep

_HISTOGRAM_TOOLS = {t for t, _, plot, _ in GATE_TOOLS if plot == "histogram"}
_TOOL_HINTS = {t: hint for t, _, _, hint in GATE_TOOLS}


class GatingController(QObject):
    gates_changed = Signal(str)  # undo label
    population_colors_changed = Signal()

    def __init__(self, session, step: GateStep, dock, dialog_parent: QWidget) -> None:
        super().__init__(dialog_parent)
        self.session = session
        self.step = step
        self.dock = dock
        self._dialog_parent = dialog_parent
        self._population = ALL
        self._wire()

    # ── Wiring ───────────────────────────────────────────────────────────────

    def _wire(self) -> None:
        s, d = self.step, self.dock
        for tool, button in s.tool_buttons.items():
            button.clicked.connect(lambda checked, t=tool: self.set_tool(t if checked else None))
        s.tree.currentItemChanged.connect(self._on_population_selected)
        s.btn_combine.clicked.connect(self._on_combine)
        s.btn_rename.clicked.connect(self._on_rename)
        s.btn_color.clicked.connect(self._on_color)
        s.btn_delete.clicked.connect(self._on_delete)
        s.spin_value.valueChanged.connect(self._on_threshold_edited)
        s.combo_side.currentIndexChanged.connect(self._on_threshold_edited)
        s.spin_lo.valueChanged.connect(self._on_range_edited)
        s.spin_hi.valueChanged.connect(self._on_range_edited)
        s.chk_color_cells.toggled.connect(lambda _on: self.population_colors_changed.emit())
        d.histogram.gate_drawn.connect(self._on_gate_drawn)
        d.scatter.gate_drawn.connect(self._on_gate_drawn)
        for widget in (d.combo_hist_metric, d.combo_x, d.combo_y):
            widget.currentIndexChanged.connect(self.refresh_plots)
        for widget in (d.chk_hist_log, d.chk_x_log, d.chk_y_log):
            widget.toggled.connect(self.refresh_plots)

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def population(self) -> str:
        """The population the plots show and new gates go into."""
        return self._population

    @property
    def colors_cells(self) -> bool:
        return self.step.chk_color_cells.isChecked() and bool(self.session.state.gates)

    def refresh(self) -> None:
        """Rebuild the population tree, the metric pickers and the plots."""
        names = {g.name for g in self.session.gates}
        if self._population not in names:
            self._population = ALL
        self._refresh_tree()
        self._refresh_pickers()
        self.refresh_plots()
        self.step.show_gate(self._selected_gate())

    def set_tool(self, tool: str | None) -> None:
        """Arm a drawing tool on its plot (and show that plot), or disarm all."""
        for name, button in self.step.tool_buttons.items():
            button.blockSignals(True)
            button.setChecked(name == tool)
            button.blockSignals(False)
        self.dock.histogram.set_tool(tool if tool in _HISTOGRAM_TOOLS else None)
        self.dock.scatter.set_tool(tool if tool and tool not in _HISTOGRAM_TOOLS else None)
        self.refresh_plots()
        if tool:
            page = self.dock.histogram_page if tool in _HISTOGRAM_TOOLS else self.dock.scatter_page
            self.dock.setCurrentWidget(page)
        self.step.lbl_tool_hint.setText(
            _TOOL_HINTS.get(
                tool or "",
                "Pick a tool, then draw on the plot. New gates go inside the selected population.",
            )
        )

    def _shown_population(self) -> str:
        """Whose cells the plots show (see the module docstring)."""
        drawing = self.dock.histogram.tool or self.dock.scatter.tool
        selected = self._selected_gate()
        if not drawing and selected is not None and selected.kind not in BOOLEAN_KINDS:
            return selected.parent or ALL
        return self._population

    def refresh_plots(self, *_args) -> None:
        shown = self._shown_population()
        df = self.session.population(shown)
        scale = self.session.scale
        n = len(df)
        caption = f"Showing {shown}: {n} cell{'s' if n != 1 else ''}"
        if shown != self._population and self._selected_gate() is not None:
            caption += f" · '{self._population}' highlighted"
        d = self.dock
        d.lbl_hist_population.setText(caption)
        d.lbl_scatter_population.setText(caption)
        colors = {}
        if self.session.state.gates:
            gated = self.session.population_colors()
            colors = {int(i): gated.get(int(i), UNGATED_COLOR) for i in df["ObjectNumber"]}
        overlays = self._overlays(shown)

        metric = d.combo_hist_metric.currentData()
        d.histogram.set_log(d.chk_hist_log.isChecked())
        if metric and metric in df:
            d.histogram.set_data(df[metric].to_numpy(), column_label(df, metric, scale))
            d.histogram.set_gates([(g, hi) for g, hi in overlays if g.x == metric])
        else:
            d.histogram.set_data([], "")
            d.histogram.set_gates([])

        x, y = d.combo_x.currentData(), d.combo_y.currentData()
        d.scatter.set_log(d.chk_x_log.isChecked(), d.chk_y_log.isChecked())
        if n and x in df and y in df:
            d.scatter.set_data(
                df[x].to_numpy(),
                df[y].to_numpy(),
                df["ObjectNumber"].tolist(),
                column_label(df, x, scale),
                column_label(df, y, scale),
                colors,
                columns=(x, y),
            )
            d.scatter.set_gates(overlays)
        else:
            d.scatter.set_data([], [], [], "", "")
            d.scatter.set_gates([])

    # ── Tree ─────────────────────────────────────────────────────────────────

    def _refresh_tree(self) -> None:
        tree = self.step.tree
        tree.blockSignals(True)
        tree.clear()
        stats = self.session.gate_stats().set_index("Population")
        root = QTreeWidgetItem([ALL, *_stat_cells(stats.loc[ALL])])
        root.setData(0, Qt.ItemDataRole.UserRole, ALL)
        tree.addTopLevelItem(root)
        items = {ALL: root}
        pending = list(self.session.gates)
        while pending:  # parents before children, whatever order the gates are in
            placed = [g for g in pending if (g.parent or ALL) in items]
            if not placed:  # a broken parent link: show what's left under the root
                placed = pending
            for gate in placed:
                parent = items.get(gate.parent or ALL, root)
                item = QTreeWidgetItem([gate.name, *_stat_cells(stats.loc[gate.name])])
                item.setData(0, Qt.ItemDataRole.UserRole, gate.name)
                item.setIcon(0, _swatch(gate.color))
                item.setToolTip(0, gate.describe())
                parent.addChild(item)
                items[gate.name] = item
            pending = [g for g in pending if g not in placed]
        tree.expandAll()
        tree.setCurrentItem(items.get(self._population, root))
        tree.blockSignals(False)

    def _on_population_selected(self, item, _previous=None) -> None:
        self._population = item.data(0, Qt.ItemDataRole.UserRole) if item else ALL
        self.step.show_gate(self._selected_gate())
        self.refresh_plots()

    def _selected_gate(self) -> Gate | None:
        return next((g for g in self.session.gates if g.name == self._population), None)

    def _overlays(self, shown: str) -> list[tuple[Gate, bool]]:
        """The gates drawn on `shown`'s cells; the selected one highlighted."""
        return [
            (g, g.name == self._population)
            for g in self.session.gates
            if (g.parent or ALL) == shown and g.kind not in BOOLEAN_KINDS
        ]

    def _refresh_pickers(self) -> None:
        """Offer every plottable measurement; keep the user's choices when they still exist."""
        df = self.session.measurements
        numeric = numeric_columns(df)
        defaults = {
            self.dock.combo_hist_metric: "AreaShape_Area",
            self.dock.combo_x: "AreaShape_Area",
            self.dock.combo_y: next(
                (c for c in numeric if c.startswith("Intensity_MeanIntensity")), None
            ),
        }
        for combo, default in defaults.items():
            current = combo.currentData() or default
            combo.blockSignals(True)
            combo.clear()
            for name in numeric:
                combo.addItem(short_name(name), name)
            index = combo.findData(current)
            if index < 0:
                index = min(1, combo.count() - 1) if combo is self.dock.combo_y else 0
            combo.setCurrentIndex(index)
            combo.blockSignals(False)

    # ── Making and editing gates ─────────────────────────────────────────────

    def _on_gate_drawn(self, kind: str, params: dict) -> None:
        d = self.dock
        parent = self._population if self._population != ALL else None  # what the plots showed
        if kind in _HISTOGRAM_TOOLS:
            column = d.combo_hist_metric.currentData()
            gate = Gate(
                name=f"{short_name(column)}+"
                if kind == "threshold"
                else f"{short_name(column)} range",
                kind=kind,
                x=column,
                params=params,
                parent=parent,
                x_log=d.chk_hist_log.isChecked(),
            )
        else:
            gate = Gate(
                name="Population",
                kind=kind,
                x=d.combo_x.currentData(),
                y=d.combo_y.currentData(),
                params=params,
                parent=parent,
                x_log=d.chk_x_log.isChecked(),
                y_log=d.chk_y_log.isChecked(),
            )
        gate.color = ""
        added = self.session.add_gate(gate)
        self._population = added.name
        self.set_tool(None)
        self.refresh()
        self.gates_changed.emit(f"Add gate '{added.name}'")
        self.population_colors_changed.emit()

    def _on_threshold_edited(self, *_args) -> None:
        gate = self._selected_gate()
        if gate is None or gate.kind != "threshold":
            return
        params = {
            "value": self.step.spin_value.value(),
            "above": bool(self.step.combo_side.currentData()),
        }
        name = f"{short_name(gate.x)}{'+' if params['above'] else '−'}"
        self._edit(gate, params, rename_to=name if _auto_threshold_name(gate) else None)

    def _on_range_edited(self, *_args) -> None:
        gate = self._selected_gate()
        if gate is None or gate.kind != "range":
            return
        self._edit(gate, {"lo": self.step.spin_lo.value(), "hi": self.step.spin_hi.value()})

    def _edit(self, gate: Gate, params: dict, rename_to: str | None = None) -> None:
        self.session.update_gate(gate.name, params=params)
        if rename_to and rename_to != gate.name:
            self._population = self.session.rename_gate(gate.name, rename_to)
        self.refresh()
        self.gates_changed.emit(f"Edit gate '{self._population}'")
        self.population_colors_changed.emit()

    def _on_rename(self) -> None:
        gate = self._selected_gate()
        if gate is None:
            return
        text, ok = QInputDialog.getText(
            self._dialog_parent, "Rename population", "Name:", text=gate.name
        )
        if not ok or not text.strip() or text.strip() == gate.name:
            return
        self._population = self.session.rename_gate(gate.name, text)
        self.refresh()
        self.gates_changed.emit(f"Rename '{gate.name}'")

    def _on_color(self) -> None:
        gate = self._selected_gate()
        if gate is None:
            return
        color = QColorDialog.getColor(QColor(gate.color), self._dialog_parent, "Population colour")
        if not color.isValid():
            return
        self.session.update_gate(gate.name, color=color.name())
        self.refresh()
        self.gates_changed.emit(f"Recolour '{gate.name}'")
        self.population_colors_changed.emit()

    def _on_delete(self) -> None:
        gate = self._selected_gate()
        if gate is None:
            return
        before = {g.name for g in self.session.gates}
        doomed = before - {g.name for g in remove(self.session.gates, gate.name)}
        others = sorted(doomed - {gate.name})
        if others and not ask_yes_no(
            self._dialog_parent,
            "Delete population",
            f"'{gate.name}' is used by {', '.join(others)}. Delete those too?",
        ):
            return
        self.session.remove_gate(gate.name)
        self._population = gate.parent or ALL
        self.refresh()
        self.gates_changed.emit(f"Delete '{gate.name}'")
        self.population_colors_changed.emit()

    def _on_combine(self) -> None:
        names = [g.name for g in self.session.gates]
        if not names:
            self.step.lbl_tool_hint.setText("Draw at least one gate first.")
            return
        dialog = CombineDialog(names, self._dialog_parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        kind, operands = dialog.result_gate()
        label = {"and": " and ", "or": " or "}.get(kind, "")
        name = f"not {operands[0]}" if kind == "not" else label.join(operands)
        added = self.session.add_gate(Gate(name=name, kind=kind, params={"of": operands}, color=""))
        self._population = added.name
        self.refresh()
        self.gates_changed.emit(f"Add gate '{added.name}'")
        self.population_colors_changed.emit()


class CombineDialog(QDialog):
    """Pick an operation and the populations it combines."""

    def __init__(self, names: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Combine populations")
        form = QFormLayout(self)
        self.combo_kind = BioComboBox()
        self.combo_kind.addItem("Both (AND): in A and in B", "and")
        self.combo_kind.addItem("Either (OR): in A or in B", "or")
        self.combo_kind.addItem("Not: every cell outside A", "not")
        self.combo_a, self.combo_b = BioComboBox(), BioComboBox()
        for combo in (self.combo_a, self.combo_b):
            combo.addItems(names)
        if len(names) > 1:
            self.combo_b.setCurrentIndex(1)
        form.addRow("Combine:", self.combo_kind)
        form.addRow("A:", self.combo_a)
        form.addRow("B:", self.combo_b)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.combo_kind.currentIndexChanged.connect(
            lambda _i: self.combo_b.setEnabled(self.combo_kind.currentData() != "not")
        )
        self.setStyleSheet(f"QLabel {{ color: {Colors.FG_PRIMARY}; }}")

    def result_gate(self) -> tuple[str, list[str]]:
        kind = self.combo_kind.currentData()
        a, b = self.combo_a.currentText(), self.combo_b.currentText()
        return kind, [a] if kind == "not" else [a, b]


def _stat_cells(row) -> list[str]:
    def pct(value) -> str:
        return "" if math.isnan(value) else f"{value:.1f}%"

    return [str(int(row["Count"])), pct(row["PctParent"]), pct(row["PctTotal"])]


def _swatch(color: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color))
    return QIcon(pixmap)


def _auto_threshold_name(gate: Gate) -> bool:
    """True while a threshold gate still has the name it was given automatically."""
    base = short_name(gate.x)
    return gate.name.rstrip("0123456789 ") in (f"{base}+", f"{base}−")
