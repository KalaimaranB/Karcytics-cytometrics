"""CytoMetrics Entry Point."""

import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from karcytics_sdk.plugin import (
    BioSplitter,
    FunctionalTask,
    HeaderLabel,
    PluginBase,
    PluginState,
    SecondaryButton,
    array_to_pixmap,
    get_double,
    get_logger,
    get_save_path,
    show_error,
    show_info,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.dialogs import SaveWorkflowDialog
from karcytics_sdk.plugin.io import PluginConfig
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.assays import DEFAULT as DEFAULT_ASSAY
from ..analysis.assays import REGISTRY as ASSAYS
from ..analysis.assays import STEP_LABELS, Assay
from ..analysis.export import write_cellpose_seg, write_imagej_rois
from ..analysis.measure import MeasureOptions, unit_label, write_csv
from ..analysis.model import ImageSet
from ..analysis.models import MODELS
from ..analysis.session import AnalysisSession, LastRun
from ..analysis.state import CytoMetricsState
from ..analysis.units import is_calibrated
from ..report import methods_markdown, write_methods_sidecars
from .dialogs.model_center import ModelCenterDialog
from .image_canvas import MultiChannelCanvas, confidence_color, value_color
from .mode_picker import ModePicker
from .results_dock import ResultsDock
from .stepper import Stepper
from .steps import CalibrateStep, DetectStep, LoadStep, MeasureStep, ReportStep, ReviewStep
from .steps.detect import PREVIEW_TEXT
from .steps.review import TOOLS
from .widgets.status_bar import SystemStatusBar
from .widgets.table_items import NumericTableItem
from .workers import (
    CytoPipelineWorker,
    load_libraries_func,
    measure_func,
    resegment_func,
)

logger = get_logger(__name__, "cytometrics")


class CytoMetricsPanel(PluginBase):
    # state_changed and status_message are now provided by PluginBase

    def __init__(self, plugin_id: str = "cytometrics", parent=None):
        super().__init__(plugin_id, parent)
        self.setMinimumWidth(380)
        self.setMinimumHeight(600)

        self.session = AnalysisSession()
        # Detect settings restored before the segmenters finished loading.
        self._pending_pipeline: str | None = None
        self._pending_values: dict[str, Any] = {}
        self._measure_generation = 0

        self.canvas = MultiChannelCanvas()

        # All pipelines are loaded in the background — start empty
        self.pipelines: dict[str, Any] = {}

        self._setup_ui()
        self._update_units()
        self.state_changed.connect(self._update_results_tab)

        # Kick off heavy AI imports as a background task
        self._loader_task = submit_task(
            FunctionalTask(load_libraries_func, "cytometrics", "Load segmentation pipelines"),
            self.state,
            on_finished=self._on_loader_finished,
            on_error=lambda message: self._on_ai_loaded(False, {}, message),
        )

    # The session owns the analysis; these keep the panel's code and the SDK's
    # `self.state` contract readable.
    @property
    def state(self) -> CytoMetricsState:
        return self.session.state

    @state.setter
    def state(self, value: CytoMetricsState) -> None:
        self.session.state = value

    @property
    def image_set(self) -> ImageSet:
        return self.session.image_set

    @property
    def measurements(self) -> pd.DataFrame:
        return self.session.measurements

    @property
    def measure_options(self) -> MeasureOptions:
        return self.session.measure_options

    def _track(self, handle):
        """Keep a running task's handle so ``cleanup`` can detach it."""
        self._tasks = [h for h in getattr(self, "_tasks", []) if not h.done]
        self._tasks.append(handle)
        return handle

    def _on_loader_finished(self, results: dict) -> None:
        self._on_ai_loaded(results.get("success", False), results.get("pipelines", {}), "Loaded")

    def cleanup(self) -> None:
        """Called when the CytoMetrics tab is closed."""
        logger.info("Cleaning up CytoMetrics panel...")

        # 1. Stop UI timers and child widgets
        if hasattr(self, "status_bar"):
            self.status_bar.stop()

        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(False)

        # 2. Cleanup key components
        if hasattr(self, "canvas") and self.canvas:
            self.canvas.cleanup()

        if hasattr(self, "channel_manager"):
            self.channel_manager.cleanup()

        # 3. Release image data
        if hasattr(self, "image_set") and self.image_set:
            self.image_set.clear()

        # 4. Stop listening to background tasks
        if hasattr(self, "_loader_task"):
            self._loader_task.detach()
        # Results of runs still in flight must not reach a closed panel.
        self._measure_generation += 1
        for handle in getattr(self, "_tasks", []):
            handle.detach()

        # 5. Base cleanup (disconnects and nulls state)
        super().cleanup()

    def shutdown(self) -> None:
        """Called for global module cleanup."""
        logger.info("Shutting down CytoMetrics module...")

        # Release the heavy AI pipelines
        if hasattr(self, "pipelines"):
            for pipe in self.pipelines.values():
                if hasattr(pipe, "model"):
                    pipe.model = None
            self.pipelines.clear()

        # Clear large state objects
        if self.state:
            self.state.cells.clear()
        if self.image_set:
            self.image_set.clear()

        # Force VRAM release if torch is loaded
        if "torch" in sys.modules:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS doesn't have an explicit clear_cache like CUDA,
                # but setting models to None and garbage collecting helps.
                import gc

                gc.collect()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_splitter = BioSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.main_splitter)
        splitter = BioSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(splitter)

        controls = QWidget()
        control_layout = QVBoxLayout(controls)
        control_layout.setContentsMargins(15, 15, 15, 10)
        control_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.addWidget(HeaderLabel("CytoMetrics"))
        header_row.addStretch()
        self.btn_assay = SecondaryButton("Choose analysis")
        self.btn_assay.setToolTip("Change what you're analysing")
        self.btn_save_session = SecondaryButton("💾")
        self.btn_save_session.setToolTip("Save this analysis to the project")
        self.btn_load_session = SecondaryButton("📂")
        self.btn_load_session.setToolTip("Open an analysis saved in this project")
        self.btn_undo = SecondaryButton("↶")
        self.btn_undo.setToolTip("Undo (Ctrl+Z)")
        self.btn_redo = SecondaryButton("↷")
        self.btn_redo.setToolTip("Redo (Ctrl+Shift+Z)")
        self.btn_undo.setEnabled(False)
        self.btn_redo.setEnabled(False)
        for button in (
            self.btn_assay,
            self.btn_undo,
            self.btn_redo,
            self.btn_save_session,
            self.btn_load_session,
        ):
            header_row.addWidget(button)
        control_layout.addLayout(header_row)

        # One page per step; the assay decides which are shown, in which order.
        self.load_step = LoadStep(self.image_set)
        self.calibrate_step = CalibrateStep()
        self.detect_step = DetectStep()
        self.review_step = ReviewStep()
        self.measure_step = MeasureStep()
        self.report_step = ReportStep()
        self.step_pages = {
            "load": self.load_step,
            "calibrate": self.calibrate_step,
            "detect": self.detect_step,
            "review": self.review_step,
            "measure": self.measure_step,
            "report": self.report_step,
        }
        self.stepper = Stepper()
        self.mode_picker = ModePicker(list(ASSAYS.values()))
        self.left_stack = QStackedWidget()
        self.left_stack.addWidget(self.mode_picker)
        self.left_stack.addWidget(self.stepper)
        control_layout.addWidget(self.left_stack, 1)
        self.status_bar = SystemStatusBar()
        control_layout.addWidget(self.status_bar)
        splitter.addWidget(controls)

        self.results = ResultsDock()
        self.results.setMinimumHeight(140)
        self.table = self.results.table
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setCollapsible(1, False)
        splitter.setSizes([3, 1])

        self.main_splitter.addWidget(self.canvas)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        # setStretchFactor only redistributes extra space; the first split needs a
        # ratio, or Qt sizes each pane by its sizeHint (far too wide a control pane).
        self.main_splitter.setSizes([1, 3])

        # Short names for the widgets the panel works with.
        self.channel_manager = self.load_step.channel_manager
        c = self.calibrate_step
        self.lbl_scale, self.btn_calibrate, self.lbl_units_note = (
            c.lbl_scale,
            c.btn_calibrate,
            c.lbl_units_note,
        )
        d = self.detect_step
        self.lbl_ai_status, self.combo_pipeline, self.btn_manage_ai = (
            d.lbl_ai_status,
            d.combo_pipeline,
            d.btn_manage_ai,
        )
        self.lbl_algo_bio, self.params_form, self.progress_bar = (
            d.lbl_algo_bio,
            d.params_form,
            d.progress_bar,
        )
        self.btn_run_pipeline = d.btn_run
        self._citation_labels = d.citation_labels
        r = self.review_step
        self.chk_show_ids, self.combo_color_by = r.chk_show_ids, r.combo_color_by
        self.chk_probability_map, self.spin_uncertain = r.chk_probability_map, r.spin_uncertain
        self.btn_next_uncertain = r.btn_next_uncertain
        m = self.measure_step
        self.combo_background, self.combo_nucleus, self.chk_texture = (
            m.combo_background,
            m.combo_nucleus,
            m.chk_texture,
        )
        self.lbl_stats = m.lbl_stats
        self.combo_hist_metric, self.histogram = (
            self.results.combo_hist_metric,
            self.results.histogram,
        )
        rp = self.report_step
        self.btn_export, self.btn_export_rois, self.btn_export_seg = (
            rp.btn_export,
            rp.btn_export_rois,
            rp.btn_export_seg,
        )
        self._review_index = -1
        self._assay: Assay | None = None
        self._wire()
        self._apply_theme_styles()
        self.undo_history.reset(self.get_state().to_dict())

    def _wire(self) -> None:
        """Connect the step pages, canvas and table to the panel."""
        self.btn_assay.clicked.connect(self._show_mode_picker)
        self.mode_picker.assay_chosen.connect(self._on_assay_chosen)
        self.stepper.step_changed.connect(self._on_step_changed)
        self.btn_save_session.clicked.connect(self._on_save_workflow)
        self.btn_load_session.clicked.connect(self._on_load_workflow)

        self.channel_manager.channels_changed.connect(self._render_composite)
        self.channel_manager.channels_changed.connect(self._update_calibrate_button_state)
        self.channel_manager.new_image_loaded.connect(self._apply_metadata_calibration)
        self.channel_manager.new_image_loaded.connect(self._apply_assay_target)
        self.canvas.load_requested.connect(self.channel_manager.prompt_add_channel)
        self.canvas.files_dropped.connect(self.channel_manager.ingest_paths)

        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.canvas.calibration_line_drawn.connect(self._on_calibration_drawn)

        self.combo_pipeline.currentIndexChanged.connect(self._on_algorithm_changed)
        self.btn_manage_ai.clicked.connect(self._open_model_manager)
        self.btn_run_pipeline.clicked.connect(self._on_run_pipeline)

        self.chk_show_ids.toggled.connect(self.canvas.set_show_ids)
        self.review_step.chk_show_outlines.toggled.connect(self.canvas.set_outlines_visible)
        self.combo_color_by.currentIndexChanged.connect(self._redraw_cells)
        self.chk_probability_map.toggled.connect(self._update_probability_map)
        self.spin_uncertain.valueChanged.connect(self._update_review_controls)
        self.btn_next_uncertain.clicked.connect(self._on_next_uncertain)
        r = self.review_step
        r.tool_group.idClicked.connect(self._on_tool_chosen)
        r.spin_brush.valueChanged.connect(lambda v: setattr(self.canvas, "brush_radius", v))
        r.btn_merge.clicked.connect(self._on_merge_selected)
        r.btn_delete.clicked.connect(self._on_delete_selected)
        self.canvas.cell_drawn.connect(self._on_cell_drawn)
        self.canvas.cell_deleted.connect(self._on_cell_deleted)
        self.canvas.selection_changed.connect(self._on_canvas_selection)
        self.canvas.split_drawn.connect(self._on_split_drawn)
        self.canvas.brush_painted.connect(self._on_brush_painted)
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.detect_step.btn_preview.clicked.connect(self._on_preview)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.signals.undo_available.connect(self.btn_undo.setEnabled)
        self.signals.redo_available.connect(self.btn_redo.setEnabled)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self.redo)

        self.combo_background.currentIndexChanged.connect(self._on_measure_options_changed)
        self.combo_nucleus.currentIndexChanged.connect(self._on_measure_options_changed)
        self.chk_texture.toggled.connect(self._on_measure_options_changed)
        self.combo_hist_metric.currentIndexChanged.connect(self._update_results_tab)

        self.btn_export.clicked.connect(self._on_export_csv)
        self.btn_export_rois.clicked.connect(self._on_export_rois)
        self.btn_export_seg.clicked.connect(self._on_export_seg)
        self.report_step.btn_copy_methods.clicked.connect(self._copy_methods)

        self.table.itemSelectionChanged.connect(self._on_table_selection)
        self.results.combo_x.currentIndexChanged.connect(self._update_scatter)
        self.results.combo_y.currentIndexChanged.connect(self._update_scatter)
        self.results.scatter.point_clicked.connect(self._on_scatter_clicked)
        self.results.inspector.flag_toggled.connect(self._on_flag_toggled)

    # ── Assays and steps ─────────────────────────────────────────────────

    def _show_mode_picker(self) -> None:
        self.left_stack.setCurrentWidget(self.mode_picker)

    def _on_assay_chosen(self, assay_id: str) -> None:
        self.set_assay(assay_id)
        self._remember_assay(assay_id)

    def set_assay(self, assay_id: str | None, apply_defaults: bool = True) -> None:
        """Switch to `assay_id`'s steps and, unless restoring a session, its default settings."""
        assay = ASSAYS.get(assay_id or "") or ASSAYS[DEFAULT_ASSAY]
        if assay is self._assay and not apply_defaults:
            return  # restoring (e.g. undo): stay on the current step
        self._assay = assay
        self.btn_assay.setText(f"{assay.icon} {assay.name} ▾")
        self.stepper.set_steps(
            [(sid, STEP_LABELS[sid], self.step_pages[sid]) for sid in assay.steps]
        )
        self.left_stack.setCurrentWidget(self.stepper)
        if not apply_defaults:
            return
        self._apply_ui_params({"pipeline": assay.segmenter, "values": dict(assay.settings)})
        self._set_measure_controls(MeasureOptions(**assay.measure))
        self._apply_assay_target()
        if self.state.cells:
            self._remeasure()

    @property
    def assay(self) -> Assay | None:
        return self._assay

    def _apply_assay_target(self, *_args) -> None:
        """Pick the assay's stain (e.g. DAPI for nuclei) as the channel to segment."""
        if self._assay is None:
            return
        target = self._assay.default_target([c.name for c in self.image_set.channels])
        if target:
            self._update_dropdowns()  # the new channel may not be offered yet
            self._pending_values = {**self._pending_values, "target_channel": target}
            self.params_form.set_display_values({"target_channel": target})

    def _project_key(self) -> str | None:
        pm = getattr(self.window(), "project_manager", None)
        directory = getattr(pm, "project_dir", None) if pm else None
        return str(directory) if directory else None

    def _remember_assay(self, assay_id: str) -> None:
        key = self._project_key()
        if key is None:
            return
        config = PluginConfig(self.plugin_id)
        remembered = dict(config.get("assay_by_project", {}) or {})
        remembered[key] = assay_id
        config.set("assay_by_project", remembered)
        config.save()

    def _remembered_assay(self) -> str | None:
        key = self._project_key()
        if key is None:
            return None
        return (PluginConfig(self.plugin_id).get("assay_by_project", {}) or {}).get(key)

    def showEvent(self, event):
        super().showEvent(event)
        # The project is only known once the panel sits in its window.
        if self._assay is None:
            remembered = self._remembered_assay()
            if remembered in ASSAYS:
                self.set_assay(remembered)

    def _on_step_changed(self, step_id: str) -> None:
        if step_id == "detect":
            self._update_dropdowns()
        elif step_id == "report":
            self._refresh_methods_preview()

    def _refresh_methods_preview(self) -> None:
        steps = self.session.methods_steps()
        if steps is None:
            self.report_step.methods_text.setPlainText(
                "Run a segmentation to see its methods text."
            )
            self.report_step.citations.set_keys(())
            return
        self.report_step.methods_text.setMarkdown(methods_markdown(steps))
        self.report_step.citations.set_keys(
            list(dict.fromkeys(k for step in steps for k in step.citations))
        )

    def _copy_methods(self) -> None:
        QApplication.clipboard().setText(self.report_step.methods_text.toPlainText())
        self.status_message.emit("Methods text copied.")

    def _apply_theme_styles(self) -> None:
        """Re-colour the plain Qt widgets; SDK Bio* widgets restyle themselves."""
        super()._apply_theme_styles()
        for page in self.step_pages.values():
            page.apply_theme()
        self.stepper.apply_theme()
        self.status_bar.apply_theme()
        self.results.apply_theme()
        # lbl_ai_status's colour depends on the load state as well as the theme.
        if self.pipelines:
            self._refresh_ai_status_label()
        else:
            self.lbl_ai_status.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 12px;")

    def _update_units(self):
        """Label every size control and readout in µm, or px when uncalibrated."""
        scale = self.state.scale
        if is_calibrated(scale):
            origin = (
                " (from image metadata)" if self.state.scale_source.startswith("metadata") else ""
            )
            self.lbl_scale.setText(f"Scale: {scale:.4f} µm/px{origin}")
        else:
            self.lbl_scale.setText("Scale: Uncalibrated (sizes in pixels)")
        self.canvas.set_scale(scale)
        self.lbl_units_note.setVisible(not is_calibrated(scale))
        self.detect_step.lbl_units_note.setVisible(not is_calibrated(scale))
        self.params_form.set_scale(scale)
        self._refresh_table()

    def _set_scale(self, scale: float, source: str) -> None:
        """Change the calibration; measured cells are converted to the new units."""
        self.session.set_scale(scale, source)
        self._update_units()
        self.push_state("Calibrate")
        self.state_changed.emit()

    def _update_calibrate_button_state(self):
        has_channels = bool(self.image_set.channels)
        self.btn_calibrate.setEnabled(has_channels)
        self.btn_calibrate.setToolTip("" if has_channels else "Add an image channel first.")

    def _open_model_manager(self):
        ModelCenterDialog(self).exec()
        self._refresh_ai_status_label()

    def _refresh_ai_status_label(self):
        size_mb = MODELS[0].installed_size_mb()
        installed = size_mb is not None
        if installed:
            self.lbl_ai_status.setText(f"✅ AI engine ready ({size_mb:.1f} MB)")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_SUCCESS}; font-size: 12px; font-weight: bold;"
            )
        else:
            self.lbl_ai_status.setText("⚠️ Cellpose-SAM isn't installed yet: open 🧠 Models")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_WARNING}; font-size: 12px; font-weight: bold;"
            )

    def _on_ai_loaded(self, success, pipelines, message):
        """Called by LibraryLoaderWorker when all libraries + pipelines are ready."""
        if success and pipelines:
            # Replace the pipeline registry with the fully constructed instances
            self.pipelines = pipelines

            self.combo_pipeline.blockSignals(True)
            self.combo_pipeline.clear()
            for key, pipe in pipelines.items():
                self.combo_pipeline.addItem(pipe.name, key)
                self.detect_step.add_citation(key, pipe.name, getattr(pipe, "citations", ()))
            self.combo_pipeline.blockSignals(False)
            self.btn_run_pipeline.setEnabled(True)
            self.detect_step.btn_preview.setEnabled(True)
            self._refresh_ai_status_label()

            # Build the settings for the current (or restored) algorithm.
            self._on_algorithm_changed()
            if self._pending_pipeline or self._pending_values:
                self._apply_ui_params(
                    {"pipeline": self._pending_pipeline, "values": self._pending_values}
                )
        else:
            self.lbl_ai_status.setText(f"⚠️ Segmentation algorithms failed to load: {message}")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_DANGER}; font-size: 12px; font-weight: bold;"
            )

    def _on_algorithm_changed(self, _index=None):
        selected = self.combo_pipeline.currentData()
        segmenter = self.pipelines.get(selected)
        self.lbl_algo_bio.setText(getattr(segmenter, "summary", ""))
        self.btn_manage_ai.setVisible(selected == "cellpose")
        if segmenter is not None:
            self.params_form.set_params(segmenter.parameters())
            self.params_form.set_channels([c.name for c in self.image_set.channels])
            self.params_form.set_scale(self.state.scale)
        self.detect_step.show_citation(selected)

    def _render_composite(self):
        composite = self.image_set.composite()
        if composite is not None:
            # Keep the user's zoom when only colours or visibility changed.
            self.canvas.set_array(composite, keep_view=self.canvas.has_image)

        self._update_dropdowns()
        self.state_changed.emit()

    def _on_run_pipeline(self):
        if not self.image_set.channels:
            show_warning(self, "No Image", "Please add a channel first.")
            return

        pipeline = self.pipelines.get(self.combo_pipeline.currentData())
        if pipeline is None:
            show_warning(self, "Not Ready", "The segmentation algorithms are still loading.")
            return
        params = self.params_form.values()
        if params.get("target_channel") not in [c.name for c in self.image_set.channels]:
            show_warning(self, "No Channel", "Choose the channel to segment.")
            return

        self.btn_run_pipeline.setEnabled(False)
        self.btn_run_pipeline.setText("⏳ Starting…")
        self.detect_step.btn_preview.setEnabled(False)

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # --- THE FIX: Wipe old cells and reset IDs on rerun ---
        self.state.cells.clear()
        self.state.cell_counter = 0

        analyzer = CytoPipelineWorker()
        analyzer.configure(pipeline, self.image_set, params, self.state.scale, self.measure_options)
        self.session.last_run = LastRun(pipeline, params)  # for the methods text

        self._track(
            submit_task(
                analyzer,
                self.state,
                on_finished=self._on_pipeline_result,
                on_error=self._on_pipeline_error,
                on_progress=self._on_run_progress,
            )
        )

    def _on_run_progress(self, value: int) -> None:
        """The worker reports 0–80 while detecting and 80–100 while measuring."""
        self.progress_bar.setValue(value)
        phase = "Measuring cells" if value >= 80 else "Detecting cells"  # noqa: PLR2004
        self.btn_run_pipeline.setText(f"⏳ {phase}… {value}%")

    def _on_pipeline_result(self, results: dict) -> None:
        # CytoPipelineWorker reports a pipeline exception as {"error": ...}
        # rather than raising, so it arrives here, not in on_error.
        if "error" in results:
            self._on_pipeline_error(results["error"])
        else:
            self._on_pipeline_finished(results)

    def _on_pipeline_finished(self, results: dict):
        self.btn_run_pipeline.setEnabled(True)
        self.detect_step.btn_preview.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)

        cells = results.get("result_cells", [])
        if not cells:
            show_info(self, "No Cells", "No cells found with current parameters.")
            return
        self.session.replace_cells(
            cells, results.get("measurements"), results.get("labels"), results.get("probability")
        )
        self.canvas.set_preview(None)
        self.set_state(self.state)
        self.push_state("Segment")
        self.state_changed.emit()

    def _on_pipeline_error(self, error_msg):
        self.btn_run_pipeline.setEnabled(True)
        self.detect_step.btn_preview.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)
        show_error(self, "Pipeline Error", f"An error occurred:\n{error_msg}")

    # ── State Management ────────────────────────────────────────────────

    def get_state(self) -> CytoMetricsState:
        """Package the workspace state for the SDK."""
        # Sync UI params before exporting
        self.state.ui_params = {
            "assay": self._assay.id if self._assay else None,
            "pipeline": self.combo_pipeline.currentData() or self._pending_pipeline,
            "values": self.params_form.display_values() or self._pending_values,
            "measure": {
                "background": self.measure_options.background,
                "texture": self.measure_options.texture,
                "nucleus_channel": self.measure_options.nucleus_channel,
            },
        }

        self.state.channels_metadata = self.session.channels_metadata()

        return self.state

    def set_state(self, state: PluginState) -> None:
        """Restore the workspace from an SDK state object."""
        if not state:
            return
        assert isinstance(state, CytoMetricsState)

        self.state = state

        # 1. Restore images if needed
        try:
            if state.channels_metadata:
                self.load_images_from_meta(state.channels_metadata)
        except Exception:
            logger.exception("Failed to restore images")

        # 2. Restore UI settings
        if state.ui_params.get("assay") in ASSAYS:
            self.set_assay(state.ui_params["assay"], apply_defaults=False)
        self._apply_ui_params(state.ui_params)

        # 3. Restore data & canvas
        self._update_units()
        if self._needs_measuring():
            self._remeasure()

        self._redraw_cells()
        self._refresh_table()
        self._update_results_tab()

    def export_state(self) -> dict:
        """Legacy export_state for backward compatibility."""
        state_obj = self.get_state()
        return state_obj.to_dict()

    def load_state(self, state_dict: dict) -> None:
        """Legacy load_state for backward compatibility."""
        if not state_dict:
            return
        state_obj = CytoMetricsState.from_dict(state_dict)
        self.set_state(state_obj)
        self.undo_history.reset(self.get_state().to_dict())  # a loaded session starts clean

    # --- ALIASES FOR HUB DASHBOARD INTEGRATION ---
    def load_workflow(self, payload: dict):
        self.load_state(payload)

    # ------------------------------------------------

    def load_images_from_meta(self, channels_meta: list[dict]) -> None:
        """Reload the saved images, then reapply the saved channel names and colours."""
        missing = self.session.reload_images(channels_meta)
        for path in missing:
            logger.warning("Saved image not found or unreadable: %s", path)
        if missing:
            self.status_message.emit(f"{len(missing)} saved image(s) could not be found.")
        self.channel_manager.refresh()
        self._render_composite()
        self._update_calibrate_button_state()

    def _refresh_table(self) -> None:
        """Show every measured column, one row per cell, with units in the headers."""
        df = self.measurements
        units = df.attrs.get("units", {})
        helps = df.attrs.get("help", {})
        columns = [c for c in df.columns if c not in ("ObjectNumber", "Source")]
        # Flags show as a 🚩 in the table rather than True/False.
        flagged = set(df.loc[df["Flagged"], "ObjectNumber"]) if "Flagged" in df else set()
        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setColumnCount(len(columns) + 1)
        self.table.setRowCount(0)
        headers = ["ID"]
        for name in columns:
            unit = unit_label(units.get(name, ""), self.state.scale)
            headers.append(f"{_short_name(name)} ({unit})" if unit else _short_name(name))
        self.table.setHorizontalHeaderLabels(headers)
        for i, name in enumerate(columns, start=1):
            header = self.table.horizontalHeaderItem(i)
            if header is not None:
                header.setToolTip(f"{name}\n{helps.get(name, '')}".strip())
        for record in df.to_dict("records"):
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, NumericTableItem(f"Cell {record['ObjectNumber']}"))
            for i, name in enumerate(columns, start=1):
                value = record[name]
                if name == "Flagged":
                    value = "🚩" if record["ObjectNumber"] in flagged else ""
                self.table.setItem(row, i, NumericTableItem(_format(value)))
        self.table.setSortingEnabled(True)
        self._update_results_tab()

    def _apply_ui_params(self, ui: dict) -> None:
        """Restore the Detect settings from saved `ui_params` (either format)."""
        if not ui:
            return
        pipeline, values = ui.get("pipeline"), dict(ui.get("values", {}))
        measure = ui.get("measure", {})
        self._update_dropdowns()  # the nucleus picker needs the channel names first
        self._set_measure_controls(
            MeasureOptions(
                background=measure.get("background", "none"),
                texture=bool(measure.get("texture", False)),
                nucleus_channel=measure.get("nucleus_channel"),
            )
        )
        self._pending_pipeline, self._pending_values = pipeline, values
        index = self.combo_pipeline.findData(pipeline)
        if index >= 0:
            self.combo_pipeline.setCurrentIndex(index)
        self.params_form.set_display_values(values)

    def _apply_metadata_calibration(self, _path: str = "") -> None:
        """Use the pixel size stored in the image file, unless a scale is already set."""
        if self.session.apply_metadata_calibration():
            self._update_units()
            self.state_changed.emit()

    def _on_calibrate_clicked(self):
        if not self.image_set.channels:
            return
        self.lbl_scale.setText("Scale: Click and drag along the image's scale bar.")
        self.canvas.set_mode("CALIBRATE")

    def _on_calibration_drawn(self, pixels: float):
        microns = get_double(
            self, "Set Scale", f"Line is {pixels:.1f} px long.\nMicrons?", 20.0, 0.001, 10000.0, 3
        )
        if microns is not None and microns > 0:
            self._set_scale(microns / pixels, "manual")
        else:
            self._update_units()

    # ── Editing ──────────────────────────────────────────────────────────

    def _on_tool_chosen(self, index: int) -> None:
        mode = TOOLS[index][0]
        if mode != "PAN" and not self.image_set.channels:
            self.review_step.tool_buttons["PAN"].setChecked(True)
            return
        self.canvas.brush_radius = self.review_step.spin_brush.value()
        self.canvas.set_mode(mode)

    def _on_canvas_selection(self, cell_ids: list) -> None:
        self._show_selection(cell_ids)
        n = len(cell_ids)
        r = self.review_step
        r.btn_merge.setEnabled(n >= 2)  # noqa: PLR2004
        r.btn_delete.setEnabled(n >= 1)
        r.lbl_selection.setText(
            f"{n} cell(s) selected." if n else "Click a cell to select it; Shift+click adds more."
        )
        self.table.blockSignals(True)
        self.table.clearSelection()
        wanted = {f"Cell {i}" for i in cell_ids}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() in wanted:
                self.table.selectRow(row)
        self.table.blockSignals(False)

    def _show_selection(self, cell_ids: list) -> None:
        """Point the inspector and the scatter plot at the selected cells."""
        self.results.scatter.set_selected(cell_ids)
        inspector = self.results.inspector
        if len(cell_ids) != 1:
            inspector.show_none(
                f"{len(cell_ids)} cells selected" if cell_ids else "No cell selected"
            )
            return
        cell = self.session.cell(cell_ids[0])
        if cell is None:
            inspector.show_none()
            return
        inspector.show_cell(cell, self._measurement_rows(cell["id"]), self._thumbnail(cell))

    def _measurement_rows(self, cell_id: int) -> list[tuple[str, str]]:
        df = self.measurements
        match = df[df["ObjectNumber"] == cell_id]
        if match.empty:
            return [("(being measured…)", "")]
        record = match.iloc[0]
        units = df.attrs.get("units", {})
        rows = []
        for name in df.columns:
            if name in ("ObjectNumber", "Source", "Confidence", "Flagged"):
                continue
            unit = unit_label(units.get(name, ""), self.state.scale)
            rows.append(
                (
                    f"{_short_name(name)} ({unit})" if unit else _short_name(name),
                    _format(record[name]),
                )
            )
        return rows

    def _thumbnail(self, cell: dict):
        """The cell and a margin around it, from the displayed composite, outlined."""
        composite = self.image_set.composite()
        if composite is None:
            return None
        pts = np.asarray(cell["points"], dtype=np.int32)
        h, w = composite.shape[:2]
        x0, y0 = pts.min(axis=0) - _THUMB_MARGIN_PX
        x1, y1 = pts.max(axis=0) + _THUMB_MARGIN_PX
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1 + 1), min(h, y1 + 1)
        crop = np.ascontiguousarray(composite[y0:y1, x0:x1])
        cv2.polylines(crop, [pts - [x0, y0]], True, (255, 220, 0), 1)
        return array_to_pixmap(crop)

    def _on_scatter_clicked(self, cell_id: int, add: bool) -> None:
        current = set(self.canvas.selection)
        selection = current ^ {cell_id} if add else {cell_id}
        self.canvas.set_selection(selection, emit=True)
        self.canvas.center_on_cell(cell_id)

    def _on_flag_toggled(self, flagged: bool) -> None:
        ids = set(self.canvas.selection)
        if not ids:
            return
        self.session.set_flag(ids, flagged)
        self._redraw_cells()
        self._refresh_table()
        self.push_state("Flag cell" if flagged else "Unflag cell")

    def _update_scatter(self, *_args) -> None:
        df = self.measurements
        x_col, y_col = self.results.combo_x.currentData(), self.results.combo_y.currentData()
        if not len(df) or x_col not in df or y_col not in df:
            self.results.scatter.set_data([], [], [], "", "")
            return
        units = df.attrs.get("units", {})

        def label(col: str) -> str:
            unit = unit_label(units.get(col, ""), self.state.scale)
            return f"{_short_name(col)} ({unit})" if unit else _short_name(col)

        self.results.scatter.set_data(
            df[x_col].to_numpy(),
            df[y_col].to_numpy(),
            df["ObjectNumber"].tolist(),
            label(x_col),
            label(y_col),
        )
        self.results.scatter.set_selected(self.canvas.selection)

    def _on_merge_selected(self) -> None:
        ids = set(self.canvas.selection)
        new_id = self.session.merge_cells(ids)
        if new_id is None:
            show_warning(self, "Can't Merge", "Only cells that touch can be merged.")
            return
        self.canvas.set_selection({new_id}, emit=True)
        self._commit_edit(f"Merge {len(ids)} cells")

    def _on_delete_selected(self) -> None:
        ids = set(self.canvas.selection)
        if not ids:
            return
        self.session.delete_cells(ids)
        self.canvas.set_selection(set(), emit=True)
        self._commit_edit(f"Delete {len(ids)} cell(s)")

    def _cell_under(self, x: float, y: float) -> int | None:
        for cell in self.state.cells:
            contour = np.asarray(cell["points"], dtype=np.float32)
            if len(contour) >= 3 and cv2.pointPolygonTest(contour, (x, y), False) >= 0:  # noqa: PLR2004
                return cell["id"]
        return None

    def _on_split_drawn(self, line) -> None:
        (x0, y0), (x1, y1) = line
        selected = self.canvas.selection
        target = (
            selected[0] if len(selected) == 1 else self._cell_under((x0 + x1) / 2, (y0 + y1) / 2)
        )
        if target is None:
            self.status_message.emit("Draw the cut across a cell (or select the cell first).")
            return
        pieces = self.session.split_cell(target, (x0, y0), (x1, y1))
        if not pieces:
            self.status_message.emit("That line doesn't cut the cell in two; draw it right across.")
            return
        self.canvas.set_selection(set(pieces), emit=True)
        self._commit_edit("Split cell")

    def _on_brush_painted(self, stroke) -> None:
        points, radius, add = stroke
        selected = self.canvas.selection
        if len(selected) != 1:
            self.status_message.emit("Select one cell to paint on (✋ Select, then click it).")
            return
        remaining = self.session.paint_cell(selected[0], points, radius, add)
        self.canvas.set_selection(set(remaining[:1]), emit=True)
        self._commit_edit("Brush" if add else "Erase")

    def _commit_edit(self, label: str) -> None:
        """An edit changed the cells: redraw, re-measure and record an undo step."""
        self._on_cells_edited()
        self.push_state(label)

    # ── Running on a region: box re-detect (kept) and preview (not kept) ──

    def _region_task(self, box, on_done) -> bool:
        """Segment inside `box` in the background; False if there's nothing to run."""
        segmenter = self.pipelines.get(self.combo_pipeline.currentData())
        if segmenter is None or not self.image_set.channels:
            return False
        task = FunctionalTask(
            partial(resegment_func, segmenter, self.image_set, self.params_form.values(), box),
            self.plugin_id,
            "Segment region",
        )
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._track(
            submit_task(
                task,
                self.state,
                on_finished=on_done,
                on_error=self._on_region_error,
                on_progress=self.progress_bar.setValue,
            )
        )
        return True

    def _on_region_error(self, message: str) -> None:
        self.progress_bar.setVisible(False)
        self._end_preview()
        show_error(self, "Segmentation Error", message)

    def _on_box_drawn(self, box) -> None:
        self._region_task(box, self._on_box_segmented)

    def _on_box_segmented(self, results: dict) -> None:
        self.progress_bar.setVisible(False)
        if "error" in results:
            self._on_region_error(results["error"])
            return
        ids = self.session.replace_region(results["box"], results["cells"])
        self.canvas.set_selection(set(ids), emit=True)
        self._commit_edit("Re-detect region")
        self.status_message.emit(f"Re-detected the region: {len(ids)} cell(s).")

    def _on_preview(self) -> None:
        if self._region_task(self.canvas.visible_region(), self._on_preview_done):
            self.detect_step.btn_preview.setEnabled(False)
            self.detect_step.btn_preview.setText("⏳ Previewing…")

    def _end_preview(self) -> None:
        self.detect_step.btn_preview.setEnabled(True)
        self.detect_step.btn_preview.setText(PREVIEW_TEXT)

    def _on_preview_done(self, results: dict) -> None:
        self.progress_bar.setVisible(False)
        self._end_preview()
        if "error" in results:
            self._on_region_error(results["error"])
            return
        self.canvas.set_preview(results["cells"])
        message = (
            f"👁 Preview: {len(results['cells'])} cell(s) found in the area on screen, "
            "drawn dashed. Nothing is kept: adjust the settings and preview again, "
            "or Run Segmentation for the whole image."
        )
        self.detect_step.lbl_ai_status.setText(message)
        self.status_message.emit(message)

    def _on_cell_drawn(self, points: list):
        self.session.add_manual_cell(points)
        self._commit_edit("Draw cell")

    def _update_dropdowns(self):
        """Offer the current channel names in the Detect settings and the nucleus picker."""
        names = [c.name for c in self.image_set.channels]
        self.params_form.set_channels(names)
        current = self.combo_nucleus.currentData()
        self.combo_nucleus.blockSignals(True)
        self.combo_nucleus.clear()
        self.combo_nucleus.addItem("Off", None)
        for name in names:
            self.combo_nucleus.addItem(name, name)
        self.combo_nucleus.setCurrentIndex(max(self.combo_nucleus.findData(current), 0))
        self.combo_nucleus.blockSignals(False)
        if current and current not in names:  # its channel was renamed or removed
            self._on_measure_options_changed()

    def _on_cell_deleted(self, cell_id):
        """Removes a cell by ID and triggers a state refresh for Undo/Redo."""
        self.session.delete_cells({cell_id})
        self._commit_edit("Delete cell")

    def _redraw_cells(self, *_args) -> None:
        """Outline the cells in one colour, by AI confidence, or by any measurement."""
        mode = self.combo_color_by.currentData()
        colors = None
        legend = ""
        if mode == _COLOR_BY_CONFIDENCE:
            colors = {
                c["id"]: confidence_color(c["confidence"])
                for c in self.state.cells
                if c.get("confidence") is not None
            }
            legend = "Red ≤ 0.5 · amber · green = 1.0"
        elif mode and mode in self.measurements.columns:
            values = self.measurements.set_index("ObjectNumber")[mode].dropna()
            if len(values):
                lo, hi = float(values.min()), float(values.max())
                span = hi - lo or 1.0
                colors = {int(i): value_color((v - lo) / span) for i, v in values.items()}
                unit = unit_label(
                    self.measurements.attrs.get("units", {}).get(mode, ""), self.state.scale
                )
                legend = f"Purple = {lo:.4g} → yellow = {hi:.4g} {unit}".rstrip()
        self.review_step.lbl_color_legend.setText(legend)
        self.canvas.draw_cells_from_state(self.state.cells, colors)
        self.canvas.set_show_ids(self.chk_show_ids.isChecked())
        self.canvas.set_outlines_visible(self.review_step.chk_show_outlines.isChecked())

    def _update_probability_map(self, *_args) -> None:
        probability = self.session.probability
        usable = probability is not None and probability.shape == self.image_set.shape
        self.chk_probability_map.setEnabled(usable)
        show = usable and self.chk_probability_map.isChecked()
        self.canvas.set_probability_map(probability if show else None)

    def _uncertain_cells(self) -> list[dict]:
        return self.session.uncertain_cells(self.spin_uncertain.value())

    def _update_review_controls(self, *_args) -> None:
        has_confidence = self.session.has_confidence()
        for widget in (
            self.review_step.lbl_uncertain,
            self.spin_uncertain,
            self.btn_next_uncertain,
        ):
            widget.setEnabled(has_confidence)
        n = len(self._uncertain_cells())
        self.btn_next_uncertain.setText(
            f"Next uncertain ({n}) ▸" if has_confidence else "Next uncertain ▸"
        )
        self.btn_next_uncertain.setEnabled(n > 0)
        self._review_index = -1

    def _on_next_uncertain(self) -> None:
        """Select, highlight and centre the next least-confident cell."""
        uncertain = self._uncertain_cells()
        if not uncertain:
            return
        self._review_index = (self._review_index + 1) % len(uncertain)
        self.select_cell(uncertain[self._review_index]["id"])

    def select_cell(self, cell_id: int) -> None:
        """Select `cell_id` in the table and centre it on the canvas."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == f"Cell {cell_id}":
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                break
        self.canvas.highlight_cell(cell_id)
        self.canvas.center_on_cell(cell_id)

    def _update_results_tab(self, *_args):
        """Refresh the metric picker, histogram and summary when cells change."""
        df = self.measurements
        numeric = [
            c
            for c in df.columns
            if c
            not in ("ObjectNumber", "Source", "Flagged", "Location_Center_X", "Location_Center_Y")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        current = self.combo_hist_metric.currentData() or "AreaShape_Area"
        self.combo_hist_metric.blockSignals(True)
        self.combo_hist_metric.clear()
        for name in numeric:
            self.combo_hist_metric.addItem(_short_name(name), name)
        self.combo_hist_metric.setCurrentIndex(max(self.combo_hist_metric.findData(current), 0))
        self.combo_hist_metric.blockSignals(False)

        self._update_color_by_options(numeric)
        self._update_scatter_options(numeric)
        metric = self.combo_hist_metric.currentData()
        lines = [f"Total Cells: {len(self.state.cells)}"]
        if metric and len(df):
            values = df[metric].dropna()
            unit = unit_label(df.attrs.get("units", {}).get(metric, ""), self.state.scale)
            if len(values):
                lines.append(f"Mean {_short_name(metric)}: {values.mean():.4g} {unit}".rstrip())
            self.histogram.update_data(values.tolist())
        else:
            self.histogram.update_data([])
        if "Confidence" in df and df["Confidence"].notna().any():
            lines.append(f"Mean AI confidence: {df['Confidence'].mean():.2f}")
        self.lbl_stats.setText("\n".join(lines))
        self._update_review_controls()
        self._update_probability_map()
        if self.combo_color_by.currentData() is not None:
            self._redraw_cells()  # colours follow the latest values

    def _update_scatter_options(self, numeric_columns: list[str]) -> None:
        defaults = (
            "AreaShape_Area",
            next((c for c in numeric_columns if c.startswith("Intensity_Mean")), None),
        )
        for combo, default in zip(
            (self.results.combo_x, self.results.combo_y), defaults, strict=True
        ):
            current = combo.currentData() or default
            combo.blockSignals(True)
            combo.clear()
            for name in numeric_columns:
                combo.addItem(_short_name(name), name)
            index = combo.findData(current)
            combo.setCurrentIndex(index if index >= 0 else min(1, combo.count() - 1))
            combo.blockSignals(False)
        self._update_scatter()

    def _update_color_by_options(self, numeric_columns: list[str]) -> None:
        current = self.combo_color_by.currentData()
        self.combo_color_by.blockSignals(True)
        self.combo_color_by.clear()
        self.combo_color_by.addItem("One colour", None)
        if self.session.has_confidence():
            self.combo_color_by.addItem("AI confidence", _COLOR_BY_CONFIDENCE)
        for name in numeric_columns:
            if name != "Confidence":
                self.combo_color_by.addItem(_short_name(name), name)
        index = self.combo_color_by.findData(current)
        self.combo_color_by.setCurrentIndex(max(index, 0))
        self.combo_color_by.blockSignals(False)
        if index < 0 and current is not None:
            self._redraw_cells()

    def _needs_measuring(self) -> bool:
        return bool(self.image_set.channels) and self.session.needs_measuring()

    def _on_cells_edited(self) -> None:
        """A cell was drawn or deleted: redraw now, re-measure in the background."""
        self._redraw_cells()
        self._refresh_table()
        self._remeasure()
        self.state_changed.emit()

    def _on_measure_options_changed(self, *_args) -> None:
        self.session.measure_options = MeasureOptions(
            background=self.combo_background.currentData(),
            texture=self.chk_texture.isChecked(),
            nucleus_channel=self.combo_nucleus.currentData(),
        )
        self._remeasure()

    def _set_measure_controls(self, options: MeasureOptions) -> None:
        """Show `options` in the Measurements controls without re-measuring."""
        widgets = (self.combo_background, self.combo_nucleus, self.chk_texture)
        for w in widgets:
            w.blockSignals(True)
        self.combo_background.setCurrentIndex(
            max(self.combo_background.findData(options.background), 0)
        )
        self.combo_nucleus.setCurrentIndex(
            max(self.combo_nucleus.findData(options.nucleus_channel), 0)
        )
        self.chk_texture.setChecked(options.texture)
        for w in widgets:
            w.blockSignals(False)
        self.session.measure_options = options

    def _remeasure(self) -> None:
        """Measure every cell again on a background thread; only the newest result is kept."""
        if not self.image_set.channels:
            return
        self._measure_generation += 1
        generation = self._measure_generation
        cells = [dict(c) for c in self.state.cells]
        task = FunctionalTask(
            partial(measure_func, cells, self.image_set, self.state.scale, self.measure_options),
            self.plugin_id,
            "Measure cells",
        )
        self._track(
            submit_task(
                task,
                self.state,
                on_finished=lambda results: self._on_measured(generation, results),
                on_error=lambda message: logger.error("Measuring failed: %s", message),
            )
        )

    def _on_measured(self, generation: int, results: dict) -> None:
        if generation != self._measure_generation:
            return  # a newer measurement is on its way
        self.session.measurements = results["measurements"]
        self._refresh_table()

    def _on_export_csv(self):
        """Saves the data table to a CSV file."""
        if not self.state.cells:
            show_info(self, "No Data", "There are no cells to export.")
            return

        path = get_save_path(
            self, "Export Results", f"{self._export_stem()}_cells.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                write_csv(self.measurements, path)
                sidecars = self._write_methods(path)
                show_info(
                    self,
                    "Success",
                    f"Data successfully exported to:\n{path}"
                    + (
                        f"\n\nMethods and citations:\n{sidecars[0].name}\n{sidecars[1].name}"
                        if sidecars
                        else ""
                    ),
                )
            except Exception as e:
                show_error(self, "Export Error", f"Could not save file:\n{str(e)}")

    def _export_stem(self) -> str:
        return self.session.export_stem()

    def _on_export_rois(self) -> None:
        if not self.state.cells:
            show_info(self, "No Data", "There are no cells to export.")
            return
        path = get_save_path(
            self, "Export ImageJ ROIs", f"{self._export_stem()}_rois.zip", "ROI Sets (*.zip)"
        )
        if not path:
            return
        try:
            written = write_imagej_rois(path, self.state.cells)
            show_info(
                self,
                "Exported",
                f"{len(self.state.cells)} ROIs written to:\n{written}\n\n"
                "In Fiji: Analyze › Tools › ROI Manager › More › Open.",
            )
        except Exception as e:
            show_error(self, "Export Error", f"Could not write ROIs:\n{e}")

    def _on_export_seg(self) -> None:
        if not self.state.cells or self.image_set.shape is None:
            show_info(self, "No Data", "There are no cells to export.")
            return
        path = get_save_path(
            self,
            "Export for Cellpose",
            f"{self._export_stem()}_seg.npy",
            "Cellpose segmentation (*_seg.npy)",
        )
        if not path:
            return
        if not path.endswith("_seg.npy"):
            path = str(Path(path).with_suffix("")) + "_seg.npy"
        try:
            written = write_cellpose_seg(
                path, self.state.cells, self.image_set.shape, self.image_set.channels[0].path
            )
            show_info(
                self,
                "Exported",
                f"Masks for {len(self.state.cells)} cells written to:\n{written}\n\n"
                "Keep it next to the image and open the image in the Cellpose GUI.",
            )
        except Exception as e:
            show_error(self, "Export Error", f"Could not write the Cellpose file:\n{e}")

    def _write_methods(self, export_path: str):
        """Write the methods paragraph and BibTeX beside an export; None if no run to describe."""
        steps = self.session.methods_steps()
        return write_methods_sidecars(export_path, steps) if steps is not None else None

    def _on_table_selection(self):
        """Grabs the selected row, extracts the ID, and highlights it on the canvas."""
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.canvas.highlight_cell(None)  # Clear highlight if they click off
            return

        # Get the row of whatever item they clicked
        row = selected_items[0].row()
        id_item = self.table.item(row, 0)  # Column 0 always holds the ID

        if id_item:
            try:
                # Strip out the word "Cell " to get the raw integer
                cell_id = int(id_item.text().replace("Cell ", ""))
                self.canvas.highlight_cell(cell_id)
            except ValueError:
                self.canvas.highlight_cell(None)

    def _on_save_workflow(self):
        pm = getattr(self.window(), "project_manager", None)
        if not pm:
            show_warning(self, "No Active Project", "Please open a project to save this workflow.")
            return

        dialog = SaveWorkflowDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        meta = dialog.get_metadata()
        try:
            pm.save_workflow(module_id=self.plugin_id, payload=self.export_state(), metadata=meta)
            show_info(self, "Success", f"Workflow '{meta['name']}' saved successfully!")
        except Exception as e:
            show_error(self, "Save Error", f"Failed to save workflow to project:\n{e}")

    def _on_load_workflow(self):
        """Pick one of this project's saved CytoMetrics workflows and restore it."""
        pm = getattr(self.window(), "project_manager", None)
        if not pm:
            show_warning(self, "No Active Project", "Please open a project to load a workflow.")
            return
        saved = [w for w in pm.workflows.list_all() if w.get("module") == self.plugin_id]
        if not saved:
            show_info(self, "No Workflows", "This project has no saved CytoMetrics workflows yet.")
            return

        newest_first = list(reversed(saved))
        labels = [
            f"{w.get('name', w['filename'])}  ({w.get('timestamp', '')[:16]})" for w in newest_first
        ]
        choice, ok = QInputDialog.getItem(self, "Load Workflow", "Workflow:", labels, 0, False)
        if not ok:
            return
        filename = newest_first[labels.index(choice)]["filename"]
        try:
            self.load_state(pm.load_workflow_payload(filename))
        except Exception as e:
            show_error(self, "Load Error", f"Could not load workflow:\n{e}")


_COLOR_BY_CONFIDENCE = "__confidence__"
_THUMB_MARGIN_PX = 6


def _short_name(column: str) -> str:
    """A column name without its family prefix: 'AreaShape_Area' → 'Area'."""
    family, _, rest = column.partition("_")
    return rest if family in ("AreaShape", "Intensity", "Location", "Texture") and rest else column


def _format(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)
