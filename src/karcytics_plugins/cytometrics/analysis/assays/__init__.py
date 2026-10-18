"""The assays offered in "What are you analysing?" (see ``base`` for what an assay is).

Importing this package registers every assay, in picker order.
"""

from . import (  # noqa: F401  (registration side effect)
    bacteria,
    cells,
    colonies,
    hemocytometer,
    nuclei,
    transfection,
)
from .base import (
    ALL_STEPS,
    REGISTRY,
    STANDARD_STEPS,
    STEP_LABELS,
    ActionResult,
    Assay,
    AssayAnalysis,
    AssayContext,
    Overlay,
    ResultValue,
    register,
)

DEFAULT = "cells"

__all__ = [
    "ALL_STEPS",
    "ActionResult",
    "Overlay",
    "DEFAULT",
    "REGISTRY",
    "STANDARD_STEPS",
    "STEP_LABELS",
    "Assay",
    "AssayAnalysis",
    "AssayContext",
    "ResultValue",
    "register",
]
