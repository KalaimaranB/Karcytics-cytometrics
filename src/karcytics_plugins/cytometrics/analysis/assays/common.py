"""Helpers the assays share: finding a channel's columns, percentages, peaks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..measure.base import channel_key
from ..model import color_for_name


def intensity_column(df: pd.DataFrame, channel: str | None, stat: str) -> str | None:
    """The column for `stat` ("IntegratedIntensity", "MeanIntensity") of `channel`.

    Prefers the background-corrected column when there is one. With no
    channel (or one not measured), takes the first measured channel whose
    name looks like `stat`'s usual stain; else the first channel.
    """
    keys = _channel_keys(df, stat)
    if not keys:
        return None
    key = channel_key(channel) if channel else None
    if key not in keys:
        key = keys[0]
    corrected = f"Intensity_Corrected{stat}_{key}"
    return corrected if corrected in df else f"Intensity_{stat}_{key}"


def stain_channel(names: list[str], color: str) -> str | None:
    """The first channel whose name says it's a `color` stain ("DAPI" → blue)."""
    return next((n for n in names if color_for_name(n) == color), None)


def _channel_keys(df: pd.DataFrame, stat: str) -> list[str]:
    prefix = f"Intensity_{stat}_"
    return [c[len(prefix) :] for c in df.columns if c.startswith(prefix)]


def percent(mask: np.ndarray, of: np.ndarray | None = None) -> float:
    """% of `of` (default: all rows) that is in `mask`."""
    base = np.ones(len(mask), bool) if of is None else of
    n = int(base.sum())
    return 100.0 * int((mask & base).sum()) / n if n else float("nan")


def main_peak(values: np.ndarray, min_count: int = 20) -> float:
    """The most frequent value (the highest peak of a Gaussian kernel density estimate).

    NaN with fewer than `min_count` finite values.
    """
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_count or np.ptp(values) == 0:
        return float(np.median(values)) if values.size else float("nan")
    lo, hi = np.percentile(values, [0.5, 99.5])
    grid = np.linspace(lo, hi, 512)
    density = gaussian_kde(values)(grid)
    return float(grid[int(np.argmax(density))])
