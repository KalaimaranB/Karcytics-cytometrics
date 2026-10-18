"""Helpers the assays share: finding a channel's columns, percentages, peaks."""

from __future__ import annotations

import cv2
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


_CONTRAST_PX = 600  # contrast is judged on a copy at most this many pixels across


def contrast(plane: np.ndarray) -> float:
    """How strongly small things stand out in `plane`, in multiples of its noise.

    Judged on the middle 60% (away from a dish's rim or the dark outside):
    the 99th percentile of the detail (the image minus a blur of it) over the
    pixel-to-pixel noise.
    """
    h, w = plane.shape
    factor = min(1.0, _CONTRAST_PX / max(h, w))
    image = plane.astype(np.float32)
    if factor < 1:
        size = (int(w * factor), int(h * factor))
        image = np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), np.float32)
    h, w = image.shape
    middle = image[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    if middle.size == 0:
        return 0.0
    detail = middle - cv2.GaussianBlur(middle, (0, 0), max(middle.shape) / 40)
    noise = 1.4826 * float(np.median(np.abs(np.diff(middle, axis=1)))) / np.sqrt(2)
    return float(np.percentile(np.abs(detail), 99)) / (noise + 1e-6)


def best_contrast_channel(channels: list) -> str | None:
    """The name of the channel (``model.Channel``) where objects stand out most."""
    if not channels:
        return None
    return max(channels, key=lambda c: contrast(c.data)).name
