"""The AI models CytoMetrics can use, with their licences and citations.

Only models whose code *and* weights carry a confirmed licence compatible
with PolyForm Noncommercial are offered (EXPANSION_PLAN.md §7). The others
are listed with the reason, so users can see what was considered and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CELLPOSE_MODEL_DIR = Path.home() / ".cellpose" / "models"


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    description: str
    file_name: str  # inside CELLPOSE_MODEL_DIR
    approx_size_mb: int
    code_license: str
    weights_license: str
    source_url: str
    citations: tuple[str, ...]
    used_by: str

    @property
    def path(self) -> Path:
        return CELLPOSE_MODEL_DIR / self.file_name

    def installed_size_mb(self) -> float | None:
        """Size on disk in MB, or None if it isn't downloaded."""
        return self.path.stat().st_size / (1024 * 1024) if self.path.is_file() else None


@dataclass(frozen=True)
class ExcludedModel:
    name: str
    reason: str


MODELS = (
    ModelInfo(
        id="cpsam",
        name="Cellpose-SAM",
        description="Generalist cell and nucleus segmentation (fluorescence, bright-field, "
        "phase contrast, any size).",
        file_name="cpsam",
        approx_size_mb=1200,
        code_license="BSD-3-Clause",
        weights_license="BSD-3-Clause",
        source_url="https://huggingface.co/mouseland/cellpose-sam",
        citations=("cpsam2025", "cellpose2021"),
        used_by="AI Smart Detect (Cellpose)",
    ),
)

# From the licence ledger (EXPANSION_PLAN.md §7).
EXCLUDED = (
    ExcludedModel("InstanSeg", "Apache-2.0 code, but the weights' licence isn't stated."),
    ExcludedModel("Trackastra", "BSD-3 code, but the weights' licence isn't stated."),
    ExcludedModel("µSAM (micro-sam)", "MIT code, but the weights' licence isn't confirmed."),
    ExcludedModel("Omnipose", "Its UW non-commercial licence isn't compatible with Karcytics'."),
    ExcludedModel("YOLO (Ultralytics)", "AGPL, which Karcytics can't ship."),
    ExcludedModel("StarDist", "Needs TensorFlow alongside PyTorch; Cellpose-SAM covers nuclei."),
)
