"""Cells: the general workflow (size, shape and intensity of whole cells)."""

from __future__ import annotations

from .base import Assay, register

CELLS = register(
    Assay(
        id="cells",
        name="Cells",
        description="Count and measure whole cells in fluorescence or bright-field images: "
        "size, shape and the intensity of every channel.",
        icon="🔬",
        outputs="Count · size · shape · intensity",
        segmenter="cellpose",
        group="Cells",
        measure={"background": "local"},
        citations=("cpsam2025",),
    )
)
