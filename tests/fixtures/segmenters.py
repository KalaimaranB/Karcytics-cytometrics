"""Test doubles for segmenters.

CI never downloads the ~1.3 GB Cellpose model, so anything that only needs
"a segmenter that finds objects" uses these instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from karcytics_plugins.cytometrics.analysis.segment import SegmentationResult


class FakeSegmenter:
    """Returns a fixed label mask and records what it was called with."""

    id = "fake"
    name = "Fake Segmenter"
    summary = "A test double."
    citations: tuple[str, ...] = ()

    def __init__(self, labels: np.ndarray | None = None, error: Exception | None = None) -> None:
        self._labels = labels if labels is not None else square_labels([(10, 10, 10)])
        self._error = error
        self.calls: list[tuple[Any, dict]] = []

    def parameters(self) -> list:
        return []

    def describe(self, params: dict, scale: float) -> tuple[str, dict]:
        return "Objects were found by a fake segmenter", {}

    def segment(self, image_set: Any, params: dict, progress=None) -> SegmentationResult:
        self.calls.append((image_set, params))
        if self._error is not None:
            raise self._error
        if progress:
            progress(50)
        return SegmentationResult(self._labels.copy())


def square_labels(squares: list[tuple[int, int, int]], size: int = 64) -> np.ndarray:
    """A label mask with one axis-aligned square (x, y, side) per object."""
    labels = np.zeros((size, size), np.int32)
    for i, (x, y, side) in enumerate(squares, start=1):
        labels[y : y + side, x : x + side] = i
    return labels
