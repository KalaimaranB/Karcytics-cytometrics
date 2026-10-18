from typing import Any

from karcytics_sdk.plugin import AnalysisBase, PluginState, get_logger

from ..analysis.measure import MeasureOptions, measure_cells
from ..analysis.segment import run_segmentation

logger = get_logger(__name__, "cytometrics")


class CytoPipelineWorker(AnalysisBase):
    """Segments the image set and measures every cell, on a background thread."""

    def __init__(self, plugin_id: str = "cytometrics") -> None:
        super().__init__(plugin_id)
        self.pipeline: Any = None
        self.image_set: Any = None
        self.params: dict[str, Any] = {}
        self.scale = 0.0
        self.measure_options = MeasureOptions()

    def configure(self, pipeline, image_set, params, scale, measure_options=None):
        """`scale` is µm/px, or 0 when uncalibrated (everything is then in pixels)."""
        self.pipeline = pipeline
        self.image_set = image_set
        self.params = params
        self.scale = scale
        self.measure_options = measure_options or MeasureOptions()

    def run(self, state: PluginState | None = None) -> dict:
        """``{"result_cells", "measurements", "labels", "probability"}``, or ``{"error"}``.

        Progress: 0–80 segmenting, 80–100 measuring.
        """
        if not self.pipeline:
            return {"error": "No pipeline configured"}
        emit = self.signals.analysis_progress.emit
        try:
            result = run_segmentation(
                self.pipeline, self.image_set, self.params, lambda v: emit(int(v * 0.8))
            )
            for cell in result.cells:
                cell["id"] = cell["label"]
            table = measure_cells(
                result.cells,
                self.image_set,
                self.scale,
                self.measure_options,
                lambda v: emit(80 + v // 5),
            )
            return {
                "result_cells": result.cells,
                "measurements": table,
                "labels": result.labels,
                "probability": result.probability,
            }
        except Exception as e:
            logger.exception("CytoMetrics segmentation error")
            return {"error": str(e)}


def measure_func(cells, image_set, scale, options, progress_callback=None) -> dict:
    """Background task: re-measure `cells` (after an edit, recalibration or reload)."""
    return {"measurements": measure_cells(cells, image_set, scale, options, progress_callback)}


# ── Functional Task Logic (Utilities) ─────────────────────────────────


def download_model_func(progress_callback=None):
    """Downloads (and caches) the Cellpose-SAM generalist model via Cellpose's
    own model registry — cellpose.org's old cyto/cyto2/cyto3 model zoo (and its
    download URL) no longer exists as of Cellpose v4.2+.
    """
    from cellpose.models import cache_model_path

    if progress_callback:
        progress_callback(5)  # cellpose's own downloader doesn't report progress back to us

    model_path = cache_model_path("cpsam")

    if progress_callback:
        progress_callback(100)

    return {"success": True, "path": str(model_path)}


def load_libraries_func():
    """One instance of every registered segmenter, keyed by id.

    torch and cellpose are not imported here: the Cellpose segmenter loads
    its model on its first run.
    """
    from ..analysis.segment import REGISTRY

    return {"success": True, "pipelines": {sid: cls() for sid, cls in REGISTRY.items()}}
