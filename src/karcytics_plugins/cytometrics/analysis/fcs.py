"""Write the object table as an FCS 3.1 file, so flow-cytometry software can gate it.

One event per cell, one parameter per numeric measurement, stored as
little-endian 32-bit floats (``$DATATYPE F``). ``$PnN`` holds the measurement
name (``AreaShape_Area``) and ``$PnS`` a readable label with its unit
(``Area (µm²)``). Written from the standard (Spidlen et al. 2010) with no
third-party code; the flow plugin opens it like any other FCS file.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_DELIM = "/"
_HEADER_LEN = 58  # "FCS3.1" + 4 spaces + six 8-character offsets
_MAX_HEADER_OFFSET = 99_999_999


def _escape(value: str) -> str:
    """Delimiters inside a keyword or value are written twice (FCS 3.1 §3.2.9)."""
    return str(value).replace(_DELIM, _DELIM * 2)


def numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """Columns that can be FCS parameters (numbers or booleans), in table order."""
    return [
        c
        for c in df.columns
        if c not in exclude
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]


def write_fcs(
    path: str | Path,
    df: pd.DataFrame,
    columns: list[str] | None = None,
    labels: dict[str, str] | None = None,
    extra: dict[str, str] | None = None,
) -> Path:
    """Write `columns` of `df` (default: every numeric one) to an FCS 3.1 file.

    `labels` maps a column to its ``$PnS`` text. Missing values are written
    as NaN, which FCS readers load as NaN. `extra` adds keywords such as
    ``$SRC`` (the image) to the TEXT segment.
    """
    path = Path(path)
    columns = columns if columns is not None else numeric_columns(df)
    labels = labels or {}
    data = (
        df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="<f4")
        if columns
        else np.zeros((len(df), 0), "<f4")
    )
    body = data.tobytes(order="C")

    keywords: dict[str, str] = {
        "$BYTEORD": "1,2,3,4",
        "$DATATYPE": "F",
        "$MODE": "L",
        "$NEXTDATA": "0",
        "$PAR": str(len(columns)),
        "$TOT": str(len(df)),
        "$CYT": "Karcytics CytoMetrics (image cytometry)",
        "$DATE": datetime.now().strftime("%d-%b-%Y").upper(),
        "$FIL": path.name,
    }
    for i, name in enumerate(columns, start=1):
        finite = data[:, i - 1][np.isfinite(data[:, i - 1])]
        top = float(finite.max()) if finite.size else 1.0
        keywords[f"$P{i}N"] = name
        keywords[f"$P{i}S"] = labels.get(name, name)
        keywords[f"$P{i}B"] = "32"
        keywords[f"$P{i}E"] = "0,0"
        keywords[f"$P{i}R"] = str(max(1, math.ceil(top) + 1))
    keywords.update(extra or {})

    # The TEXT segment states where DATA starts, which depends on TEXT's own
    # length: grow the offsets until they stop changing.
    begin_data = end_data = 0
    while True:
        offsets = {
            "$BEGINANALYSIS": "0",
            "$ENDANALYSIS": "0",
            "$BEGINSTEXT": "0",
            "$ENDSTEXT": "0",
            "$BEGINDATA": str(begin_data),
            "$ENDDATA": str(end_data),
        }
        text = (
            _DELIM
            + "".join(
                f"{_escape(k)}{_DELIM}{_escape(v)}{_DELIM}"
                for k, v in {**offsets, **keywords}.items()
            )
        ).encode("utf-8")
        text_start = _HEADER_LEN
        text_end = text_start + len(text) - 1
        new_begin = text_end + 1
        new_end = new_begin + len(body) - 1  # one before BEGINDATA when there are no events
        if (new_begin, new_end) == (begin_data, end_data):
            break
        begin_data, end_data = new_begin, new_end

    def offset(value: int) -> str:
        return f"{value if value <= _MAX_HEADER_OFFSET else 0:>8}"

    header = (
        "FCS3.1    "
        + offset(text_start)
        + offset(text_end)
        + offset(begin_data)
        + offset(end_data)
        + offset(0)
        + offset(0)
    ).encode("ascii")
    assert len(header) == _HEADER_LEN
    path.write_bytes(header + text + body)
    return path
