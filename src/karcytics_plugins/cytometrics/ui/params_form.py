"""Controls generated from a segmenter's ``parameters()``.

Size parameters are shown in µm / µm² (px / px² when uncalibrated) and
handed to the segmenter in pixels. Values the user typed are remembered by
key, so switching segmenter keeps the shared ones (channel, size filter).
"""

from __future__ import annotations

from typing import Any

from karcytics_sdk.plugin import BioComboBox, BioDoubleSpinBox, BioToggleButton
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QVBoxLayout, QWidget

from ..analysis.segment import Param
from ..analysis.units import area_unit, effective_scale, length_unit

_ADVANCED_CLOSED = "Advanced settings ▸"
_ADVANCED_OPEN = "Advanced settings ▾"


class ParamsForm(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._params: list[Param] = []
        self._widgets: dict[str, QWidget] = {}
        self._labels: dict[str, QLabel] = {}
        self._memory: dict[str, Any] = {}
        # The default of the parameter each remembered value was set on: a value only carries
        # over to a parameter with the same default (another segmenter's "Max Area" of 1e6 px²
        # for colonies is not the cells' 5000 px²). Values set explicitly carry over anywhere.
        self._memory_defaults: dict[str, Any] = {}
        self._hidden: set[str] = set()  # keys set elsewhere (e.g. the channel of a photo)
        self._channels: list[str] = []
        self._scale = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._basic = QWidget()
        self._basic_grid = _grid(self._basic)
        layout.addWidget(self._basic)

        self.btn_advanced = BioToggleButton(_ADVANCED_CLOSED)
        self.btn_advanced.toggled.connect(self._on_advanced_toggled)
        layout.addWidget(self.btn_advanced)
        self._advanced = QWidget()
        self._advanced_grid = _grid(self._advanced)
        self._advanced.setVisible(False)
        layout.addWidget(self._advanced)

    # ── Public API ───────────────────────────────────────────────────────────

    def set_params(self, params: list[Param]) -> None:
        """Rebuild the controls for `params`, keeping values the user already set."""
        self._memory.update(self.display_values())
        self._memory_defaults.update({p.key: p.default for p in self._params})
        for w in [*self._widgets.values(), *self._labels.values()]:
            w.setParent(None)
            w.deleteLater()
        self._widgets.clear()
        self._labels.clear()
        self._params = list(params)

        rows = {False: 0, True: 0}
        for p in self._params:
            grid = self._advanced_grid if p.advanced else self._basic_grid
            widget = self._make_widget(p)
            self._widgets[p.key] = widget
            r = rows[p.advanced]
            if p.kind == "bool":
                grid.addWidget(widget, r, 1)
            else:
                label = QLabel(f"{p.label}:")
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                label.setToolTip(p.help)
                self._labels[p.key] = label
                grid.addWidget(label, r, 0)
                grid.addWidget(widget, r, 1)
            rows[p.advanced] = r + 1

        self._apply_hidden()
        has_advanced = rows[True] > 0
        self.btn_advanced.setVisible(has_advanced)
        self._advanced.setVisible(has_advanced and self.btn_advanced.isChecked())
        self._apply_units()
        self._apply_dependencies()
        self.apply_theme()

    def set_channels(self, names: list[str]) -> None:
        """Offer `names` in every channel picker, keeping each picker's choice if it still exists."""
        self._channels = list(names)
        for p in self._params:
            if p.kind == "channel":
                combo = self._widgets[p.key]
                assert isinstance(combo, BioComboBox)
                current = combo.currentText() or self._memory.get(p.key, "")
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(self._channels)
                if current in self._channels:
                    combo.setCurrentText(current)
                combo.blockSignals(False)

    def set_scale(self, scale: float) -> None:
        """Relabel size controls in µm or px. The numbers the user typed are kept."""
        self._scale = scale
        self._apply_units()

    def display_values(self) -> dict[str, Any]:
        """Every control's value as shown (sizes in the displayed unit)."""
        return {p.key: self._read(p) for p in self._params}

    def set_hidden(self, keys: set[str]) -> None:
        """Hide the controls of `keys`; their values are still read and written."""
        self._hidden = set(keys)
        self._apply_hidden()

    def _apply_hidden(self) -> None:
        for key, widget in self._widgets.items():
            widget.setVisible(key not in self._hidden)
            if key in self._labels:
                self._labels[key].setVisible(key not in self._hidden)

    def set_display_values(self, values: dict[str, Any]) -> None:
        self._memory.update(values)
        for key in values:
            self._memory_defaults.pop(key, None)
        for p in self._params:
            if p.key in values:
                self._write(p, values[p.key])
        self._apply_dependencies()

    def values(self) -> dict[str, Any]:
        """The segmenter's input: sizes converted to pixels."""
        eff = effective_scale(self._scale)
        out: dict[str, Any] = {}
        for p in self._params:
            value = self._read(p)
            if p.unit == "length":
                value = value / eff
            elif p.unit == "area":
                value = value / eff**2
            if p.kind == "int":
                value = int(round(value))
            out[p.key] = value
        return out

    def widget(self, key: str) -> QWidget | None:
        return self._widgets.get(key)

    def label(self, key: str) -> QLabel | None:
        return self._labels.get(key)

    def apply_theme(self) -> None:
        """Re-colour the plain Qt widgets (the Bio* ones restyle themselves)."""
        for p in self._params:
            if p.kind == "bool":
                self._widgets[p.key].setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        for label in self._labels.values():
            label.setStyleSheet(f"color: {Colors.FG_PRIMARY};")

    # ── Internals ────────────────────────────────────────────────────────────

    def _make_widget(self, p: Param) -> QWidget:
        remembered = (
            self._memory[p.key]
            if p.key in self._memory and self._memory_defaults.get(p.key, p.default) == p.default
            else p.default
        )
        if p.kind == "bool":
            box = QCheckBox(p.label)
            box.setChecked(bool(remembered))
            box.setToolTip(p.help)
            box.toggled.connect(self._on_changed)
            return box
        if p.kind in ("channel", "choice"):
            combo = BioComboBox()
            combo.setMinimumWidth(180)
            combo.view().setMinimumWidth(300)
            combo.setToolTip(p.help)
            if p.kind == "channel":
                combo.addItems(self._channels)
                if remembered in self._channels:
                    combo.setCurrentText(remembered)
            else:
                for text, data in p.choices:
                    combo.addItem(text, data)
                index = combo.findData(remembered)
                combo.setCurrentIndex(max(index, 0))
            combo.currentIndexChanged.connect(self._on_changed)
            return combo
        spin = BioDoubleSpinBox()
        spin.setDecimals(0 if p.kind == "int" else p.decimals)
        spin.setRange(
            p.minimum if p.minimum is not None else -1e9,
            p.maximum if p.maximum is not None else 1e9,
        )
        if p.step is not None:
            spin.setSingleStep(p.step)
        if p.special_value_text:
            spin.setSpecialValueText(p.special_value_text)
        spin.setValue(float(remembered or 0))
        spin.setToolTip(p.help)
        spin.valueChanged.connect(self._on_changed)
        return spin

    def _read(self, p: Param) -> Any:
        w = self._widgets[p.key]
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, BioComboBox):
            return w.currentText() if p.kind == "channel" else w.currentData()
        assert isinstance(w, BioDoubleSpinBox)
        return int(w.value()) if p.kind == "int" else w.value()

    def _write(self, p: Param, value: Any) -> None:
        w = self._widgets[p.key]
        w.blockSignals(True)
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, BioComboBox):
            if p.kind == "channel":
                w.setCurrentText(str(value))
            else:
                w.setCurrentIndex(max(w.findData(value), 0))
        elif isinstance(w, BioDoubleSpinBox):
            w.setValue(float(value or 0))
        w.blockSignals(False)

    def _apply_units(self) -> None:
        for p in self._params:
            w = self._widgets[p.key]
            if isinstance(w, BioDoubleSpinBox) and p.unit:
                unit = length_unit(self._scale) if p.unit == "length" else area_unit(self._scale)
                w.setSuffix(f" {unit}")

    def _apply_dependencies(self) -> None:
        for p in self._params:
            if not p.enabled_by or p.enabled_by not in self._widgets:
                continue
            switch = self._widgets[p.enabled_by]
            on = isinstance(switch, QCheckBox) and switch.isChecked()
            self._widgets[p.key].setEnabled(on)
            if p.key in self._labels:
                self._labels[p.key].setEnabled(on)

    def _on_changed(self, *_args) -> None:
        self._apply_dependencies()
        self.changed.emit()

    def _on_advanced_toggled(self, checked: bool) -> None:
        self.btn_advanced.setText(_ADVANCED_OPEN if checked else _ADVANCED_CLOSED)
        self._advanced.setVisible(checked)


def _grid(parent: QWidget) -> QGridLayout:
    grid = QGridLayout(parent)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(8)
    grid.setColumnStretch(1, 1)
    return grid
