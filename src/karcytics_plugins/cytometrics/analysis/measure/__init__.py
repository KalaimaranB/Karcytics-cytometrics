"""Measurements: importing this package registers every measurer, in column order."""

from . import intensity, shape  # noqa: F401  (registration side effect; shape first below)
from .base import REGISTRY, Column, MeasureOptions, Measurer, ObjectView, channel_key, register
from .table import columns_for, csv_headers, measure_cells, rescale, unit_label, write_csv

# Column order: shape before intensity, whatever the import order above.
for _key in ("shape", "intensity"):
    REGISTRY[_key] = REGISTRY.pop(_key)

__all__ = [
    "REGISTRY",
    "Column",
    "MeasureOptions",
    "Measurer",
    "ObjectView",
    "channel_key",
    "columns_for",
    "csv_headers",
    "measure_cells",
    "register",
    "rescale",
    "unit_label",
    "write_csv",
]
