"""Table items that sort by number."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QTableWidgetItem,
)


class NumericTableItem(QTableWidgetItem):
    """A custom table item that sorts numerically instead of alphabetically."""

    def __lt__(self, other):
        try:
            # Strip out "Cell " if it's the ID column, then sort by float value
            val1 = float(self.text().replace("Cell ", ""))
            val2 = float(other.text().replace("Cell ", ""))
            return val1 < val2
        except ValueError:
            return super().__lt__(other)
