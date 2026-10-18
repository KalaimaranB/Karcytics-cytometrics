"""One plot axis: data ↔ screen mapping on a linear or log10 scale, with tick marks."""

from __future__ import annotations

import math

import numpy as np


class Axis:
    """Maps data values to a pixel span. On a log axis, values ≤ 0 can't be shown."""

    def __init__(self, log: bool = False) -> None:
        self.log = log
        self.lo, self.hi = 0.0, 1.0  # in display space (log10 when `log`)
        self.start, self.end = 0.0, 1.0  # pixel positions of lo and hi

    def display(self, values) -> np.ndarray:
        """Values in display space: log10 on a log axis (NaN for values ≤ 0)."""
        values = np.asarray(values, dtype=float)
        if not self.log:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(values), np.nan)

    def fit(self, values) -> None:
        """Set the range to the data's, padded 5%."""
        shown = self.display(values)
        shown = shown[np.isfinite(shown)]
        if not shown.size:
            self.lo, self.hi = (0.0, 1.0)
            return
        lo, hi = float(shown.min()), float(shown.max())
        pad = (hi - lo) * 0.05 or max(abs(lo) * 0.05, 0.5 if self.log else 1.0)
        self.lo, self.hi = lo - pad, hi + pad

    def to_screen(self, value: float) -> float:
        """A data value's pixel position (``nan`` if a log axis can't show it)."""
        shown = float(self.display([value])[0])
        return self.start + (shown - self.lo) / (self.hi - self.lo) * (self.end - self.start)

    def to_screen_many(self, values) -> np.ndarray:
        shown = self.display(values)
        return self.start + (shown - self.lo) / (self.hi - self.lo) * (self.end - self.start)

    def from_screen(self, pixel: float) -> float:
        """The data value at a pixel position."""
        shown = self.lo + (pixel - self.start) / (self.end - self.start) * (self.hi - self.lo)
        return 10.0**shown if self.log else shown

    def ticks(self, max_ticks: int = 6) -> list[tuple[float, str]]:
        """``(pixel, label)`` for round values inside the range."""
        if self.log:
            low, high = math.ceil(self.lo), math.floor(self.hi)
            every = max(1, math.ceil((high - low + 1) / max_ticks))
            return [
                (self.display_to_screen(d), _format(10.0**d)) for d in range(low, high + 1, every)
            ]
        step = _nice_step((self.hi - self.lo) / max(1, max_ticks - 1))
        first = math.ceil(self.lo / step) * step
        out: list[tuple[float, str]] = []
        value = float(first)
        while value <= self.hi + step * 1e-9:
            out.append(
                (self.display_to_screen(value), _format(0.0 if abs(value) < step * 1e-9 else value))
            )
            value += step
        return out

    def display_to_screen(self, shown: float) -> float:
        """Pixel position of a value already in display space (log10 on a log axis)."""
        return self.start + (shown - self.lo) / (self.hi - self.lo) * (self.end - self.start)


def _nice_step(raw: float) -> float:
    """1, 2 or 5 × a power of ten, at least `raw`."""
    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * power >= raw:
            return m * power
    return 10 * power


def _format(value: float) -> str:
    return f"{value:.3g}"
