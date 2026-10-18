"""Table items that sort by number."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QTableWidgetItem,
)

ID_ROLE = Qt.ItemDataRole.UserRole  # the object id behind an ID cell ("Colony 3" → 3)


class NumericTableItem(QTableWidgetItem):
    """A custom table item that sorts numerically instead of alphabetically."""

    def __lt__(self, other):
        mine, theirs = self.data(ID_ROLE), other.data(ID_ROLE)
        if mine is not None and theirs is not None:
            return mine < theirs
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)


def id_item(noun: str, object_id: int) -> NumericTableItem:
    """The ID cell of an object's row: "<noun> <id>", sorting by the id."""
    item = NumericTableItem(f"{noun} {object_id}")
    item.setData(ID_ROLE, int(object_id))
    return item


def item_id(item: QTableWidgetItem | None) -> int | None:
    """The object id of an ID cell (None for any other cell)."""
    value = item.data(ID_ROLE) if item is not None else None
    return int(value) if value is not None else None
