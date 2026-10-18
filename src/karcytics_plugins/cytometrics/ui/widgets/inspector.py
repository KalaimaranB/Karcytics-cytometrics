"""The selected cell, close up: thumbnail, every measurement, confidence, flag."""

from __future__ import annotations

from karcytics_sdk.plugin import BioCaptionLabel, BioTableWidget, SecondaryButton, SubtitleLabel
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

_THUMB_PX = 110


class CellInspector(QWidget):
    flag_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.noun = "Cell"  # what one object is called ("Colony"), set by the panel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(_THUMB_PX, _THUMB_PX)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(self.thumbnail)
        side = QVBoxLayout()
        self.title = SubtitleLabel("No cell selected")
        self.details = BioCaptionLabel(
            "Click a cell on the canvas, a table row or a scatter point."
        )
        self.btn_flag = SecondaryButton("🚩 Flag")
        self.btn_flag.setCheckable(True)
        self.btn_flag.setToolTip("Mark this cell to come back to. Flags are saved and exported.")
        self.btn_flag.setEnabled(False)
        self.btn_flag.toggled.connect(self._on_flag)
        side.addWidget(self.title)
        side.addWidget(self.details)
        side.addWidget(self.btn_flag)
        side.addStretch()
        top.addLayout(side, 1)
        layout.addLayout(top)
        self.values = BioTableWidget(0, 2)
        self.values.setHorizontalHeaderLabels(["Measurement", "Value"])
        self.values.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.values.verticalHeader().hide()
        self.values.setMinimumHeight(60)
        layout.addWidget(self.values, 1)
        self._quiet = False

    def show_none(self, text: str = "No cell selected") -> None:
        self.title.setText(text)
        self.details.setText("Click a cell on the canvas, a table row or a scatter point.")
        self.thumbnail.clear()
        self.values.setRowCount(0)
        self._set_flag(False, enabled=False)

    def show_cell(self, cell: dict, rows: list[tuple[str, str]], thumbnail: QPixmap | None) -> None:
        """`rows` are (name with unit, formatted value) pairs."""
        self.title.setText(f"{self.noun} {cell['id']}")
        source = cell.get("source", "auto")
        confidence = cell.get("confidence")
        details = f"Source: {source}"
        if confidence is not None:
            details += f" · AI confidence {confidence:.2f}"
        self.details.setText(details)
        if thumbnail is not None:
            self.thumbnail.setPixmap(
                thumbnail.scaled(
                    _THUMB_PX,
                    _THUMB_PX,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            )
        self.values.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            self.values.setItem(i, 0, QTableWidgetItem(name))
            self.values.setItem(i, 1, QTableWidgetItem(value))
        self._set_flag(bool(cell.get("flagged")), enabled=True)

    def _set_flag(self, flagged: bool, enabled: bool) -> None:
        self._quiet = True
        self.btn_flag.setChecked(flagged)
        self.btn_flag.setText("🚩 Flagged" if flagged else "🚩 Flag")
        self.btn_flag.setEnabled(enabled)
        self._quiet = False

    def _on_flag(self, checked: bool) -> None:
        self.btn_flag.setText("🚩 Flagged" if checked else "🚩 Flag")
        if not self._quiet:
            self.flag_toggled.emit(checked)
