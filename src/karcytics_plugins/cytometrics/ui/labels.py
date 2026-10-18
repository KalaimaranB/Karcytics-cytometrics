"""How measurement columns and values are shown in tables, plots and pickers."""

from __future__ import annotations

import math

import pandas as pd

from ..analysis.measure import unit_label

_FAMILIES = ("AreaShape", "Intensity", "Location", "Texture")


def short_name(column: str) -> str:
    """A column name without its family prefix: 'AreaShape_Area' → 'Area'."""
    family, _, rest = column.partition("_")
    return rest if family in _FAMILIES and rest else column


def column_label(df: pd.DataFrame, column: str, scale: float) -> str:
    """'Area (µm²)': the short name and the unit at the current calibration."""
    unit = unit_label(df.attrs.get("units", {}).get(column, ""), scale)
    return f"{short_name(column)} ({unit})" if unit else short_name(column)


def fmt(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Measurements worth plotting (no ids, flags or positions)."""
    skip = ("ObjectNumber", "Source", "Flagged", "Location_Center_X", "Location_Center_Y")
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
