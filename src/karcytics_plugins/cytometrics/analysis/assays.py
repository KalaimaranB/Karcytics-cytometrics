"""Assays: what the user is analysing, and the workflow that goes with it.

An assay is a preset over the pipeline: which steps the user walks through,
which segmenter and settings to start from, and which measurements to make.
The user picks one in "What are you analysing?" (one plugin, several
modes). Phase 8 adds assays with their own steps (e.g. colony counting).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import color_for_name

ALL_STEPS = ("load", "calibrate", "detect", "review", "measure", "gate", "report", "batch")
STEP_LABELS = {
    "load": "Load",
    "calibrate": "Calibrate",
    "detect": "Detect",
    "review": "Review",
    "measure": "Measure",
    "gate": "Gate",
    "report": "Report",
    "batch": "Batch",
}


@dataclass(frozen=True)
class Assay:
    id: str
    name: str
    description: str
    icon: str
    segmenter: str
    steps: tuple[str, ...] = ALL_STEPS
    settings: dict[str, Any] = field(default_factory=dict)  # Detect display values
    measure: dict[str, Any] = field(default_factory=dict)  # MeasureOptions fields
    target_stain: str | None = None  # colour of the stain to segment by default ("blue")

    def default_target(self, channel_names: list[str]) -> str | None:
        """The channel to segment by default, from the stain names (None: keep the user's)."""
        if self.target_stain is None:
            return None
        return next((n for n in channel_names if color_for_name(n) == self.target_stain), None)


REGISTRY: dict[str, Assay] = {}


def register(assay: Assay) -> Assay:
    if assay.id in REGISTRY:
        raise ValueError(f"Assay id {assay.id!r} is already registered")
    REGISTRY[assay.id] = assay
    return assay


CELLS = register(
    Assay(
        id="cells",
        name="Cells",
        description="Count and measure whole cells in fluorescence or bright-field images: "
        "size, shape and the intensity of every channel.",
        icon="🔬",
        segmenter="cellpose",
        measure={"background": "local"},
    )
)

NUCLEI = register(
    Assay(
        id="nuclei",
        name="Nuclei",
        description="Count and measure nuclei from a nuclear stain (DAPI, Hoechst): size, shape, "
        "DNA content (integrated intensity) and chromatin texture.",
        icon="🧬",
        segmenter="cellpose",
        measure={"background": "local", "texture": True},
        target_stain="blue",
    )
)

DEFAULT = CELLS.id
