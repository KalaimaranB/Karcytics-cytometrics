import inspect
import logging
import sys
from typing import Any

from karcytics_sdk.plugin import AnalysisBase, PluginState
from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class InterceptorSignals(QObject):
    """Helper class to hold PyQt signals for the standard logging handler."""

    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)


class StreamCatcher(QObject):
    """Intercepts terminal output (stderr) so we can see what Cellpose is actually saying."""

    text_written = pyqtSignal(str)

    def __init__(self, original_stream):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        self.original_stream.write(text)  # Still print to your Mac terminal
        if text.strip():  # Only send if it's not an empty newline
            self.text_written.emit(text.strip())

    def flush(self):
        self.original_stream.flush()


class CellposeLogInterceptor(logging.Handler):
    """Eavesdrops on Cellpose logs to provide real-time UI updates."""

    def __init__(self):
        super().__init__()
        self.signals = InterceptorSignals()

    def emit(self, record):
        msg = self.format(record).lower()
        self.signals.status_signal.emit(record.getMessage())

        if "downloading" in msg:
            self.signals.progress_signal.emit(5)
        elif "evaluating" in msg or "network" in msg:
            self.signals.progress_signal.emit(20)
        elif "computing flows" in msg:
            self.signals.progress_signal.emit(50)
        elif "computing masks" in msg:
            self.signals.progress_signal.emit(80)


class CytoPipelineWorker(AnalysisBase):
    """Worker that runs the AI segmentation pipeline via TaskScheduler."""

    def __init__(self, plugin_id: str = "cytometrics") -> None:
        super().__init__(plugin_id)
        self.pipeline = None
        self.image_stack = None
        self.params: dict[str, Any] = {}
        self.scale = 1.0

    def configure(self, pipeline, image_stack, params, scale):
        self.pipeline = pipeline
        self.image_stack = image_stack
        self.params = params
        self.scale = scale

    def run(self, state: PluginState | None = None) -> dict:
        """Execute the segmentation on a background thread."""
        if not self.pipeline:
            return {"error": "No pipeline configured"}

        # 1. Hijack the terminal's standard error stream
        original_stderr = sys.stderr
        catcher = StreamCatcher(original_stderr)
        # We can't easily emit signals from here to the exact task,
        # but the catcher remains for stdout visibility.
        sys.stderr = catcher

        try:
            # 2. Run the AI
            result_cells = self.pipeline.run(self.image_stack, self.params, self.scale)
            return {"result_cells": result_cells}

        except Exception as e:
            logger.exception("CytoMetrics Pipeline Error")
            return {"error": str(e)}

        finally:
            # 3. Put the terminal back to normal!
            sys.stderr = original_stderr


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
    """Builds all pipeline instances. Heavy AI imports are deferred to first use."""
    from ..analysis.pipelines.cellpose_pipeline import CellposePipeline
    from ..analysis.pipelines.otsu import OtsuPipeline
    from ..analysis.pipelines.watershed import WatershedPipeline

    # NOTE: torch and cellpose are NOT imported here. CellposePipeline.__init__
    # is lightweight — it only sets self.model = None. The actual 1GB model
    # loads lazily inside _ensure_model() on the first call to .run().
    pipelines = {
        "otsu": OtsuPipeline(),
        "watershed": WatershedPipeline(),
        "cellpose": CellposePipeline(),
    }
    return {"success": True, "pipelines": pipelines}


# ── Functional Task Adapter ─────────────────────────────────────────────
# karcytics_sdk.plugin.task_scheduler only schedules AnalysisBase instances
# (via .run(state) -> dict), unlike the old FunctionalTask which could wrap
# any plain function directly. This adapts download_model_func/
# load_libraries_func — neither of which needs real PluginState — into that
# contract.


class FunctionalAnalysisTask(AnalysisBase):
    """Adapts a zero-arg or progress_callback-accepting function into an
    AnalysisBase so it can run through task_scheduler.submit().
    """

    def __init__(self, func, plugin_id: str = "cytometrics") -> None:
        super().__init__(plugin_id)
        self._func = func
        self._accepts_progress = "progress_callback" in inspect.signature(func).parameters

    def run(self, state: PluginState | None = None) -> dict:
        if self._accepts_progress:
            return self._func(progress_callback=self.signals.analysis_progress.emit)
        return self._func()
