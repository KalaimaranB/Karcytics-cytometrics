"""Nuclei: nuclear size, shape, DNA content and chromatin texture."""

from __future__ import annotations

from .base import Assay, register

NUCLEI = register(
    Assay(
        id="nuclei",
        name="Nuclei",
        description="Count and measure nuclei from a nuclear stain (DAPI, Hoechst): size, shape, "
        "DNA content (integrated intensity) and chromatin texture.",
        icon="🧬",
        segmenter="cellpose",
        group="Cells",
        measure={"background": "local", "texture": True},
        target_stain="blue",
        citations=("cpsam2025",),
    )
)
