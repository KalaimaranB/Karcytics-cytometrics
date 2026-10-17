"""Test doubles for segmentation pipelines.

CI never downloads the ~1.3 GB Cellpose model, so anything that only needs
"a pipeline that returns cells" uses these instead.
"""

from __future__ import annotations

from typing import Any


class FakeSegmenter:
    """Returns a fixed list of cells and records what it was called with."""

    name = "Fake Segmenter"

    def __init__(self, cells: list[dict] | None = None, error: Exception | None = None) -> None:
        self._cells = cells if cells is not None else [square_cell(0, 0, 10)]
        self._error = error
        self.calls: list[tuple[Any, dict, float]] = []

    def run(self, image_stack: Any, params: dict, scale: float = 1.0) -> list[dict]:
        self.calls.append((image_stack, params, scale))
        if self._error is not None:
            raise self._error
        return [dict(c) for c in self._cells]


def square_cell(x: int, y: int, side: int, scale: float = 1.0) -> dict:
    """A cell dict in today's pipeline output shape, for an axis-aligned square."""
    area = float(side * side * scale**2)
    perim = float(4 * side * scale)
    return {
        "points": [[x, y], [x + side, y], [x + side, y + side], [x, y + side]],
        "area": area,
        "perim": perim,
        "circ": 4 * 3.141592653589793 * area / perim**2,
    }
