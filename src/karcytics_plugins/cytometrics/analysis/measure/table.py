"""Measure every cell into one pandas table (the ObjectTable).

Cells are rasterised from their outlines first (``cells_to_labels``), so
segmented, hand-drawn, edited and reloaded cells are all measured the same
way and the numbers are reproducible from a saved session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy import ndimage

from ..export import cells_to_labels
from ..model import ImageSet
from ..units import area_unit, ascii_unit, effective_scale, length_unit
from .base import (
    AREA,
    INTENSITY,
    LENGTH,
    NONE,
    PIXELS,
    REGISTRY,
    Column,
    MeasureOptions,
    ObjectView,
)

ID_COLUMNS = [
    Column("ObjectNumber", NONE, "The cell's ID, as shown on the canvas."),
    Column("Source", NONE, "'auto' (segmented) or 'manual' (drawn by hand)."),
    Column("Confidence", NONE, "Cellpose's mean cell probability inside the object (0–1)."),
]


def measurers(options: MeasureOptions) -> list:
    """The measurers to run, in column order."""
    out = []
    for cls in REGISTRY.values():
        enabled = getattr(cls, "enabled", None)  # optional: measurers that depend on options
        if enabled is None or enabled(options):
            out.append(cls())
    return out


def columns_for(image_set: ImageSet, options: MeasureOptions) -> list[Column]:
    cols = list(ID_COLUMNS)
    for m in measurers(options):
        cols += m.columns(image_set, options)
    return cols


def measure_cells(
    cells: list[dict],
    image_set: ImageSet,
    scale: float,
    options: MeasureOptions | None = None,
    progress: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    """One row per cell, indexed by ObjectNumber (the cell id).

    `scale` is µm/px, 0 when uncalibrated (sizes then stay in px). The unit
    of every column is in ``df.attrs["units"]`` and the scale it was
    measured at in ``df.attrs["scale"]``.
    """
    options = options or MeasureOptions()
    shape = image_set.shape
    cols = columns_for(image_set, options)
    if shape is None or not cells:
        return _empty(cols, scale)

    labels = cells_to_labels(cells, shape)  # label k = k-th cell
    margin = options.ring_gap_px + options.ring_width_px + 1
    eff = effective_scale(scale)
    active = measurers(options)
    context = {m.id: m.prepare(image_set, labels, options) for m in active if hasattr(m, "prepare")}
    rows = []
    slices = ndimage.find_objects(labels)
    for k, (cell, sl) in enumerate(zip(cells, slices, strict=False), start=1):
        row = {
            "ObjectNumber": cell["id"],
            "Source": cell.get("source", "auto"),
            "Confidence": cell.get("confidence", np.nan),
        }
        if sl is not None:  # None if a later cell covered this one completely
            crop = _grow(sl, margin, shape)
            sub = labels[crop]
            obj = ObjectView(
                number=cell["id"],
                mask=sub == k,
                others=(sub > 0) & (sub != k),
                crop=crop,
                offset=(crop[0].start, crop[1].start),
                image_set=image_set,
                scale=eff,
                options=options,
                context=context,
            )
            for m in active:
                row.update(m.measure(obj))
        rows.append(row)
        if progress:
            progress(int(100 * k / len(cells)))

    df = pd.DataFrame(rows, columns=[c.name for c in cols]).set_index("ObjectNumber", drop=False)
    return _with_attrs(df, cols, scale, options)


def rescale(df: pd.DataFrame, new_scale: float) -> pd.DataFrame:
    """The same table in the units of `new_scale` (lengths ×r, areas ×r²)."""
    ratio = effective_scale(new_scale) / effective_scale(df.attrs.get("scale", 0.0))
    out = df.copy()
    for name, unit in df.attrs.get("units", {}).items():
        if unit == LENGTH:
            out[name] = out[name] * ratio
        elif unit == AREA:
            out[name] = out[name] * ratio**2
    out.attrs = {**df.attrs, "scale": new_scale}
    return out


def unit_label(unit: str, scale: float) -> str:
    """Display unit for a column's unit kind ("" when it has none)."""
    if unit == LENGTH:
        return length_unit(scale)
    if unit == AREA:
        return area_unit(scale)
    if unit == PIXELS:
        return "px"
    if unit == INTENSITY:
        return "a.u."
    return ""


def csv_headers(df: pd.DataFrame) -> list[str]:
    """Column names with ASCII units appended: ``AreaShape_Area [um2]``."""
    scale = df.attrs.get("scale", 0.0)
    units = df.attrs.get("units", {})
    out = []
    for name in df.columns:
        unit = ascii_unit(unit_label(units.get(name, NONE), scale))
        out.append(f"{name} [{unit}]" if unit else name)
    return out


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, header=csv_headers(df), float_format="%.6g")


def _grow(sl: tuple[slice, slice], margin: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    return tuple(  # type: ignore[return-value]
        slice(max(0, s.start - margin), min(n, s.stop + margin))
        for s, n in zip(sl, shape, strict=True)
    )


def _empty(cols: list[Column], scale: float) -> pd.DataFrame:
    df = pd.DataFrame(columns=[c.name for c in cols]).set_index("ObjectNumber", drop=False)
    return _with_attrs(df, cols, scale, MeasureOptions())


def _with_attrs(
    df: pd.DataFrame, cols: list[Column], scale: float, options: MeasureOptions
) -> pd.DataFrame:
    df.attrs = {
        "scale": scale,
        "units": {c.name: c.unit for c in cols},
        "help": {c.name: c.help for c in cols},
        "options": asdict(options),
    }
    return df
