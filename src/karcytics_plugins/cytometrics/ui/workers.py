from typing import Any

from karcytics_sdk.plugin import AnalysisBase, PluginState, get_logger

from ..analysis.segment import run_segmentation

logger = get_logger(__name__, "cytometrics")


class CytoPipelineWorker(AnalysisBase):
    """Runs one segmenter over the image set on a background thread."""

    def __init__(self, plugin_id: str = "cytometrics") -> None:
        super().__init__(plugin_id)
        self.pipeline: Any = None
        self.image_set: Any = None
        self.params: dict[str, Any] = {}
        self.scale = 1.0

    def configure(self, pipeline, image_set, params, scale):
        self.pipeline = pipeline
        self.image_set = image_set
        self.params = params
        self.scale = scale

    def run(self, state: PluginState | None = None) -> dict:
        """Returns ``{"result_cells", "labels", "probability"}``, or ``{"error"}``."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}
        try:
            result = run_segmentation(
                self.pipeline,
                self.image_set,
                self.params,
                self.scale,
                self.signals.analysis_progress.emit,
            )
            return {
                "result_cells": result.cells,
                "labels": result.labels,
                "probability": result.probability,
            }
        except Exception as e:
            logger.exception("CytoMetrics segmentation error")
            return {"error": str(e)}


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
