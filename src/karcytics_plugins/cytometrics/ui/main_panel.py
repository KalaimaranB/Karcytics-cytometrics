"""CytoMetrics Entry Point."""

import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
from karcytics_sdk.plugin import (
    BioComboBox,
    BioDoubleSpinBox,
    BioScrollArea,
    BioSplitter,
    BioTableWidget,
    CitationLabel,
    FunctionalTask,
    HeaderLabel,
    PluginBase,
    PluginState,
    PrimaryButton,
    SecondaryButton,
    get_double,
    get_logger,
    get_save_path,
    show_error,
    show_info,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.dialogs import SaveWorkflowDialog
from karcytics_sdk.plugin.theme_fallback import Colors
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..analysis.export import write_cellpose_seg, write_imagej_rois
from ..analysis.measure import MeasureOptions, unit_label, write_csv
from ..analysis.model import ImageSet
from ..analysis.session import AnalysisSession, LastRun
from ..analysis.state import CytoMetricsState
from ..analysis.units import is_calibrated
from ..report import write_methods_sidecars
from .channel_manager import ChannelManagerWidget
from .dialogs.model_manager import ModelManagerDialog, _check_ai_model_installed
from .image_canvas import MultiChannelCanvas, confidence_color
from .params_form import ParamsForm
from .widgets.hardware_monitor import HardwareMonitor
from .widgets.histogram import SimpleHistogram
from .widgets.scanning_indicator import ScanningIndicator
from .widgets.table_items import NumericTableItem
from .workers import (
    CytoPipelineWorker,
    load_libraries_func,
    measure_func,
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

    def _on_loader_finished(self, results: dict) -> None:
        self._on_ai_loaded(results.get("success", False), results.get("pipelines", {}), "Loaded")

    def cleanup(self) -> None:
        """Called when the CytoMetrics tab is closed."""
        logger.info("Cleaning up CytoMetrics panel...")

        # 1. Stop UI timers and child widgets
        if hasattr(self, "hw_monitor"):
            self.hw_monitor.stop()

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
        self._section_headers: list[QLabel] = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Ensure this widget fills the core app's central container
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.main_splitter = BioSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.main_splitter)

        splitter = BioSplitter(Qt.Orientation.Vertical)
        self.main_splitter.addWidget(splitter)

        controls_widget = QWidget()
        control_layout = QVBoxLayout(controls_widget)
        control_layout.setContentsMargins(15, 15, 15, 10)
        control_layout.setSpacing(10)

        header = HeaderLabel("CytoMetrics")
        control_layout.addWidget(header)

        session_layout = QHBoxLayout()

        self.btn_save_session = SecondaryButton("💾 Save Session")
        self.btn_save_session.clicked.connect(self._on_save_workflow)

        self.btn_load_session = SecondaryButton("📂 Load Session")
        self.btn_load_session.clicked.connect(self._on_load_workflow)

        session_layout.addWidget(self.btn_save_session)
        session_layout.addWidget(self.btn_load_session)
        control_layout.addLayout(session_layout)
        # -----------------------------------

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY}; border: 1px solid {Colors.BORDER}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; font-weight: bold; border-bottom-color: {Colors.BG_MEDIUM}; }}
        """)

        def create_header(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; padding-top: 15px; padding-bottom: 5px; border-bottom: 1px solid {Colors.BORDER};"
            )
            self._section_headers.append(lbl)
            return lbl

        # ==========================================
        # --- TAB 1: SETUP & CHANNELS ---
        # ==========================================
        tab_setup = QWidget()
        setup_layout = QVBoxLayout(tab_setup)
        setup_layout.setSpacing(15)

        setup_layout.addWidget(create_header("1. Import Images"))

        self.channel_manager = ChannelManagerWidget(self.image_set)
        self.channel_manager.channels_changed.connect(self._render_composite)
        self.channel_manager.channels_changed.connect(self._update_calibrate_button_state)
        self.channel_manager.new_image_loaded.connect(self._apply_metadata_calibration)
        setup_layout.addWidget(self.channel_manager)

        self.canvas.load_requested.connect(self.channel_manager.prompt_add_channel)
        self.canvas.files_dropped.connect(self.channel_manager.ingest_paths)

        setup_layout.addWidget(create_header("2. Calibrate Scale"))

        self.lbl_scale = QLabel("Scale: Uncalibrated")
        self.lbl_scale.setStyleSheet(
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; border: none;"
        )
        setup_layout.addWidget(self.lbl_scale)

        self.btn_calibrate = SecondaryButton("📏  Set Scale / Calibrate")
        self.btn_calibrate.setEnabled(False)
        self.btn_calibrate.setToolTip("Add an image channel first.")
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.canvas.calibration_line_drawn.connect(self._on_calibration_drawn)
        setup_layout.addWidget(self.btn_calibrate)

        setup_layout.addStretch()

        # ==========================================
        # --- TAB 2: DETECTION RULES ---
        # ==========================================
        tab_detect = QWidget()
        tab_layout = QVBoxLayout(tab_detect)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll Area protects the grid from being crushed
        scroll_area = BioScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(BioScrollArea.Shape.NoFrame)
        scroll_area.setStyleSheet("background: transparent;")

        detect_widget = QWidget()
        detect_layout = QVBoxLayout(detect_widget)
        detect_layout.setContentsMargins(10, 10, 10, 10)

        pipe_layout = QGridLayout()
        pipe_layout.setSpacing(8)
        pipe_layout.setColumnStretch(1, 1)

        input_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px; min-height: 24px;
            }}
            QComboBox QAbstractItemView {{ background-color: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY}; selection-background-color: {Colors.ACCENT_PRIMARY}; }}
        """

        self.lbl_ai_status = QLabel("⏳ Loading AI engine…")
        self.lbl_ai_status.setStyleSheet(f"color: {Colors.FG_SECONDARY}; font-size: 12px;")
        detect_layout.addWidget(self.lbl_ai_status)

        r = 0
        pipe_layout.addWidget(create_header("1. Algorithm"), r, 0, 1, 2)
        r += 1

        lbl_algo = QLabel("Algorithm:")
        lbl_algo.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        pipe_layout.addWidget(lbl_algo, r, 0)

        # --- NEW HORIZONTAL LAYOUT FOR DROPDOWN + BUTTON ---
        algo_row_layout = QHBoxLayout()

        self.combo_pipeline = BioComboBox()
        for key, pipe in self.pipelines.items():
            self.combo_pipeline.addItem(pipe.name, key)
        self.combo_pipeline.setStyleSheet(input_style)
        self.combo_pipeline.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_pipeline.currentIndexChanged.connect(self._on_algorithm_changed)

        self.btn_manage_ai = SecondaryButton("⚙️ Manage AI")
        self.btn_manage_ai.clicked.connect(self._open_model_manager)

        algo_row_layout.addWidget(self.combo_pipeline)
        algo_row_layout.addWidget(self.btn_manage_ai)

        pipe_layout.addLayout(algo_row_layout, r, 1)
        r += 1

        self.lbl_algo_bio = QLabel("")
        self.lbl_algo_bio.setWordWrap(True)
        self.lbl_algo_bio.setStyleSheet(
            f"color: {Colors.FG_SECONDARY}; font-size: 11px; font-style: italic; margin-bottom: 5px;"
        )
        pipe_layout.addWidget(self.lbl_algo_bio, r, 1)
        r += 1

        # "Powered by …" credit for the selected algorithm; one CitationLabel
        # per pipeline is added once the pipelines have loaded.
        self._citation_box = QWidget()
        self._citation_layout = QVBoxLayout(self._citation_box)
        self._citation_layout.setContentsMargins(0, 2, 0, 0)
        self._citation_labels: dict[str, CitationLabel] = {}
        pipe_layout.addWidget(self._citation_box, r, 1)
        r += 1

        detect_layout.addLayout(pipe_layout)
        detect_layout.addWidget(create_header("2. Settings"))
        self.params_form = ParamsForm()
        detect_layout.addWidget(self.params_form)

        self.lbl_units_note = QLabel(
            "⚠️ Not calibrated: sizes are in pixels. Set the scale in Setup for µm."
        )
        self.lbl_units_note.setWordWrap(True)
        self.lbl_units_note.setStyleSheet(
            f"color: {Colors.ACCENT_WARNING}; font-weight: bold; margin-top: 10px;"
        )
        self.lbl_units_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detect_layout.addWidget(self.lbl_units_note)

        self.progress_bar = ScanningIndicator()
        self.progress_bar.setVisible(False)
        detect_layout.addWidget(self.progress_bar)

        self.btn_run_pipeline = PrimaryButton("✨ Run Segmentation")
        self.btn_run_pipeline.clicked.connect(self._on_run_pipeline)
        detect_layout.addWidget(self.btn_run_pipeline)

        self.hw_monitor = HardwareMonitor()
        detect_layout.addWidget(self.hw_monitor)

        detect_layout.addStretch()

        scroll_area.setWidget(detect_widget)
        tab_layout.addWidget(scroll_area)

        # ==========================================
        # --- TAB 3: RESULTS & EXPORT ---
        # ==========================================
        tab_results = QWidget()
        results_tab_layout = QVBoxLayout(tab_results)
        results_tab_layout.setContentsMargins(0, 0, 0, 0)
        results_scroll = BioScrollArea()
        results_scroll.setWidgetResizable(True)
        results_scroll.setFrameShape(BioScrollArea.Shape.NoFrame)
        results_scroll.setStyleSheet("background: transparent;")
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_scroll.setWidget(results_widget)
        results_tab_layout.addWidget(results_scroll)
        results_layout.setSpacing(15)

        results_layout.addWidget(create_header("Canvas View"))
        self.chk_show_ids = QCheckBox("Show Cell ID Numbers")
        self.chk_show_ids.setChecked(True)
        self.chk_show_ids.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.chk_show_ids.toggled.connect(self.canvas.set_show_ids)
        results_layout.addWidget(self.chk_show_ids)

        self.chk_color_confidence = QCheckBox("Colour outlines by AI confidence")
        self.chk_color_confidence.setToolTip(
            "Cellpose cells only: red = uncertain, amber = borderline, green = confident."
        )
        self.chk_color_confidence.toggled.connect(self._redraw_cells)
        results_layout.addWidget(self.chk_color_confidence)

        self.chk_probability_map = QCheckBox("Show Cellpose cell-probability map")
        self.chk_probability_map.setToolTip(
            "Where the model thinks cells are, from the last Cellpose run."
        )
        self.chk_probability_map.toggled.connect(self._update_probability_map)
        results_layout.addWidget(self.chk_probability_map)

        review_row = QHBoxLayout()
        self.lbl_uncertain = QLabel("Uncertain below:")
        self.spin_uncertain = BioDoubleSpinBox()
        self.spin_uncertain.setRange(0.0, 1.0)
        self.spin_uncertain.setSingleStep(0.05)
        self.spin_uncertain.setDecimals(2)
        self.spin_uncertain.setValue(0.75)
        self.spin_uncertain.setToolTip("Cells whose AI confidence is below this need a look.")
        self.spin_uncertain.valueChanged.connect(self._update_review_controls)
        self.btn_next_uncertain = SecondaryButton("Next uncertain ▸")
        self.btn_next_uncertain.clicked.connect(self._on_next_uncertain)
        review_row.addWidget(self.lbl_uncertain)
        review_row.addWidget(self.spin_uncertain)
        review_row.addWidget(self.btn_next_uncertain)
        results_layout.addLayout(review_row)
        self._review_index = -1

        results_layout.addWidget(create_header("Measurements"))
        background_row = QHBoxLayout()
        self.lbl_background = QLabel("Intensity background:")
        self.combo_background = BioComboBox()
        for text, mode in (
            ("None (raw intensities)", "none"),
            ("Global (median outside cells)", "global"),
            ("Local ring around each cell", "local"),
        ):
            self.combo_background.addItem(text, mode)
        self.combo_background.setToolTip(
            "Subtract a background level from each cell's intensities. Global: the median of all "
            "pixels outside the cells. Local: the median of a 5 px ring, 2 px away from the cell, "
            "avoiding other cells."
        )
        self.combo_background.currentIndexChanged.connect(self._on_measure_options_changed)
        background_row.addWidget(self.lbl_background)
        background_row.addWidget(self.combo_background, 1)
        results_layout.addLayout(background_row)

        nucleus_row = QHBoxLayout()
        self.lbl_nucleus = QLabel("Nucleus / cytoplasm from:")
        self.combo_nucleus = BioComboBox()
        self.combo_nucleus.setToolTip(
            "A nuclear stain (e.g. DAPI). Each cell is split into nucleus (above the stain's Otsu "
            "threshold) and cytoplasm, and every channel's nuclear/cytoplasmic ratio is measured."
        )
        self.combo_nucleus.addItem("Off", None)
        self.combo_nucleus.currentIndexChanged.connect(self._on_measure_options_changed)
        nucleus_row.addWidget(self.lbl_nucleus)
        nucleus_row.addWidget(self.combo_nucleus, 1)
        results_layout.addLayout(nucleus_row)

        self.chk_texture = QCheckBox("Texture (Haralick features, slower)")
        self.chk_texture.setToolTip(
            "Six grey-level co-occurrence features per channel (Haralick et al., 1973): "
            "uniformity, contrast, correlation, homogeneity, entropy and variance."
        )
        self.chk_texture.toggled.connect(self._on_measure_options_changed)
        results_layout.addWidget(self.chk_texture)

        results_layout.addWidget(create_header("Distribution"))
        self.combo_hist_metric = BioComboBox()
        self.combo_hist_metric.currentIndexChanged.connect(self._update_results_tab)
        results_layout.addWidget(self.combo_hist_metric)
        self.lbl_stats = QLabel("Total Cells: 0")
        self.lbl_stats.setStyleSheet(f"color: {Colors.FG_PRIMARY}; font-size: 13px;")
        results_layout.addWidget(self.lbl_stats)

        self.histogram = SimpleHistogram()
        results_layout.addWidget(self.histogram)

        results_layout.addWidget(create_header("Export"))
        self.btn_export = SecondaryButton("💾 Export to CSV")
        self.btn_export.clicked.connect(self._on_export_csv)
        results_layout.addWidget(self.btn_export)

        self.btn_export_rois = SecondaryButton("🔷 Export ImageJ / Fiji ROIs")
        self.btn_export_rois.setToolTip("A .zip of cell outlines for ImageJ/Fiji's ROI Manager.")
        self.btn_export_rois.clicked.connect(self._on_export_rois)
        results_layout.addWidget(self.btn_export_rois)

        self.btn_export_seg = SecondaryButton("🧬 Export for Cellpose (_seg.npy)")
        self.btn_export_seg.setToolTip(
            "Masks the Cellpose GUI opens for editing, and that Cellpose can train on."
        )
        self.btn_export_seg.clicked.connect(self._on_export_seg)
        results_layout.addWidget(self.btn_export_seg)

        results_layout.addStretch()

        # ==========================================
        # --- FINALIZE LAYOUT ---
        # ==========================================
        self.tabs.addTab(tab_setup, "1. Setup")
        self.tabs.addTab(tab_detect, "⏳ Loading AI Engine...")
        self.tabs.addTab(tab_results, "3. Results")
        self.tabs.setTabEnabled(1, False)  # Locked until LibraryLoaderWorker finishes
        self.tabs.currentChanged.connect(self._on_tab_changed)
        control_layout.addWidget(self.tabs)

        # _on_algorithm_changed will be triggered once _on_ai_loaded populates the combo

        self.btn_draw = SecondaryButton("✏️  Draw Cells (Manual)")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(self._on_draw_toggled)
        self.canvas.cell_drawn.connect(self._on_cell_drawn)

        # Make sure cell_deleted is connected if you added it to canvas!
        if hasattr(self.canvas, "cell_deleted"):
            self.canvas.cell_deleted.connect(self._on_cell_deleted)

        control_layout.addWidget(self.btn_draw)

        splitter.addWidget(controls_widget)

        self.table = BioTableWidget(0, 0)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER}; border: none;"
        )
        self.table.setSortingEnabled(True)  # <--- Enable interactive sorting!
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        splitter.addWidget(self.table)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        splitter.setCollapsible(1, False)

        # Inject the canvas into the right hand side of the main splitter
        self.main_splitter.addWidget(self.canvas)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        # setStretchFactor alone only governs how *extra* space is redistributed on
        # resize — the initial split needs an explicit ratio, or Qt sizes each pane
        # by its own sizeHint() instead (which skewed the controls panel far wider
        # than intended). Values are proportional, not literal pixels — scaled to
        # whatever the splitter's actual width is at first layout.
        self.main_splitter.setSizes([1, 3])

    def _apply_theme_styles(self) -> None:
        """Re-applies theme-aware styles when the active theme changes.

        Most widgets here are plain Qt widgets with ``Colors.*`` baked into
        an f-string stylesheet at construction time, so — unlike the SDK's
        own ``Bio*`` components (buttons, ``HeaderLabel``, etc.), which
        already re-style themselves — they need to be explicitly rebuilt
        here. ``super()`` handles the panel background plus any child that
        already knows how to restyle itself.
        """
        super()._apply_theme_styles()

        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {Colors.BORDER}; border-radius: 4px; top: -1px; }}
            QTabBar::tab {{ background: {Colors.BG_DARKEST}; color: {Colors.FG_SECONDARY}; border: 1px solid {Colors.BORDER}; padding: 8px 15px; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
            QTabBar::tab:selected {{ background: {Colors.BG_MEDIUM}; color: {Colors.FG_PRIMARY}; font-weight: bold; border-bottom-color: {Colors.BG_MEDIUM}; }}
        """)

        header_style = (
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; "
            f"padding-top: 15px; padding-bottom: 5px; border-bottom: 1px solid {Colors.BORDER};"
        )
        for lbl in self._section_headers:
            lbl.setStyleSheet(header_style)

        self.lbl_scale.setStyleSheet(
            f"color: {Colors.ACCENT_PRIMARY}; font-weight: bold; border: none;"
        )

        input_style = f"""
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY};
                border: 1px solid {Colors.BORDER}; border-radius: 4px; padding: 4px; min-height: 24px;
            }}
            QComboBox QAbstractItemView {{ background-color: {Colors.BG_DARK}; color: {Colors.FG_PRIMARY}; selection-background-color: {Colors.ACCENT_PRIMARY}; }}
        """
        self.combo_pipeline.setStyleSheet(input_style)
        self.params_form.apply_theme()
        self.chk_show_ids.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        for widget in (
            self.chk_color_confidence,
            self.chk_probability_map,
            self.lbl_uncertain,
            self.lbl_background,
            self.lbl_nucleus,
            self.chk_texture,
        ):
            widget.setStyleSheet(f"color: {Colors.FG_PRIMARY};")
        self.lbl_stats.setStyleSheet(f"color: {Colors.FG_PRIMARY}; font-size: 13px;")
        self.lbl_units_note.setStyleSheet(
            f"color: {Colors.ACCENT_WARNING}; font-weight: bold; margin-top: 10px;"
        )

        self.table.setStyleSheet(
            f"background: {Colors.BG_DARKEST}; color: {Colors.FG_PRIMARY}; gridline-color: {Colors.BORDER}; border: none;"
        )

        # lbl_ai_status's color depends on AI-load state, not just the theme —
        # reuse the state-aware helper once loaded, otherwise re-apply the
        # loading-state color with the new theme's FG_SECONDARY.
        if self.tabs.isTabEnabled(1):
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
        self.params_form.set_scale(scale)
        self._refresh_table()

    def _set_scale(self, scale: float, source: str) -> None:
        """Change the calibration; measured cells are converted to the new units."""
        self.session.set_scale(scale, source)
        self._update_units()
        self.state_changed.emit()

    def _update_calibrate_button_state(self):
        has_channels = bool(self.image_set.channels)
        self.btn_calibrate.setEnabled(has_channels)
        self.btn_calibrate.setToolTip("" if has_channels else "Add an image channel first.")

    def _open_model_manager(self):
        ModelManagerDialog(self).exec()
        self._refresh_ai_status_label()

    def _refresh_ai_status_label(self):
        installed, size_mb = _check_ai_model_installed()
        if installed:
            self.lbl_ai_status.setText(f"✅ AI engine ready ({size_mb:.1f} MB)")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_SUCCESS}; font-size: 12px; font-weight: bold;"
            )
        else:
            self.lbl_ai_status.setText("⚠️ AI model not installed — click Manage AI to install")
            self.lbl_ai_status.setStyleSheet(
                f"color: {Colors.ACCENT_WARNING}; font-size: 12px; font-weight: bold;"
            )

    def _on_ai_loaded(self, success, pipelines, message):
        """Called by LibraryLoaderWorker when all libraries + pipelines are ready."""
        if success and pipelines:
            # Replace the pipeline registry with the fully constructed instances
            self.pipelines = pipelines

            # Populate the combo box with all available algorithms
            self.combo_pipeline.clear()
            for key, pipe in pipelines.items():
                self.combo_pipeline.addItem(pipe.name, key)
                label = CitationLabel(pipe.name, getattr(pipe, "citations", ()))
                label.setVisible(False)
                self._citation_layout.addWidget(label)
                self._citation_labels[key] = label

            # Unlock and rename tab 2
            self.tabs.setTabEnabled(1, True)
            self.tabs.setTabText(1, "2. Smart Detect")
            self._refresh_ai_status_label()

            # Build the settings for the current (or restored) algorithm.
            self._on_algorithm_changed()
            if self._pending_pipeline or self._pending_values:
                self._apply_ui_params(
                    {"pipeline": self._pending_pipeline, "values": self._pending_values}
                )
        else:
            # Partial load — tab stays locked with error label
            self.tabs.setTabText(1, "⚠️ Load Failed")
            self.lbl_ai_status.setText("⚠️ AI engine failed to load")
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
        for key, label in self._citation_labels.items():
            label.setVisible(key == selected)

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
        self.btn_run_pipeline.setText("⏳ Initializing...")

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # --- THE FIX: Wipe old cells and reset IDs on rerun ---
        self.state.cells.clear()
        self.state.cell_counter = 0

        analyzer = CytoPipelineWorker()
        analyzer.configure(pipeline, self.image_set, params, self.state.scale, self.measure_options)
        self.session.last_run = LastRun(pipeline, params)  # for the methods text

        submit_task(
            analyzer,
            self.state,
            on_finished=self._on_pipeline_result,
            on_error=self._on_pipeline_error,
            on_progress=self.progress_bar.setValue,
        )

    def _on_pipeline_result(self, results: dict) -> None:
        # CytoPipelineWorker reports a pipeline exception as {"error": ...}
        # rather than raising, so it arrives here, not in on_error.
        if "error" in results:
            self._on_pipeline_error(results["error"])
        else:
            self._on_pipeline_finished(results)

    def _on_pipeline_finished(self, results: dict):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)

        cells = results.get("result_cells", [])
        if not cells:
            show_info(self, "No Cells", "No cells found with current parameters.")
            return
        self.session.replace_cells(
            cells, results.get("measurements"), results.get("labels"), results.get("probability")
        )
        self.set_state(self.state)
        self.state_changed.emit()

    def _on_pipeline_error(self, error_msg):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("✨ Run Segmentation")
        self.progress_bar.setVisible(False)
        show_error(self, "Pipeline Error", f"An error occurred:\n{error_msg}")

    # ── State Management ────────────────────────────────────────────────

    def get_state(self) -> CytoMetricsState:
        """Package the workspace state for the SDK."""
        # Sync UI params before exporting
        self.state.ui_params = {
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
                self.table.setItem(row, i, NumericTableItem(_format(record[name])))
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

    def _on_draw_toggled(self, checked):
        if not self.image_set.channels:
            self.btn_draw.setChecked(False)
            return
        self.canvas.set_mode("DRAW" if checked else "PAN")
        if checked:
            # Highlight as active while drawing; restored to the normal SecondaryButton
            # look via its own theme method once the user toggles it back off.
            self.btn_draw.setStyleSheet(f"""
                QPushButton {{ background-color: {Colors.ACCENT_PRIMARY}; color: {Colors.BG_DARKEST};
                    border: none; border-radius: 6px; padding: 10px 20px; font-size: 13px;
                    font-weight: bold; text-align: left; }}
            """)
        else:
            self.btn_draw._apply_theme_styles()

    def _on_cell_drawn(self, points: list):
        self.session.add_manual_cell(points)
        self._on_cells_edited()

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

    def _on_tab_changed(self, index):
        """Force a refresh of the dropdowns when switching to the Smart Detect tab."""
        if index == 1:  # Index 1 is the second tab (Smart Detect)
            self._update_dropdowns()

    def _on_cell_deleted(self, cell_id):
        """Removes a cell by ID and triggers a state refresh for Undo/Redo."""
        self.session.delete_cells({cell_id})
        self._on_cells_edited()

    def _redraw_cells(self, *_args) -> None:
        """Outline the cells, coloured by confidence if that's switched on."""
        colors = None
        if self.chk_color_confidence.isChecked():
            colors = {
                c["id"]: confidence_color(c["confidence"])
                for c in self.state.cells
                if c.get("confidence") is not None
            }
        self.canvas.draw_cells_from_state(self.state.cells, colors)
        self.canvas.set_show_ids(self.chk_show_ids.isChecked())

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
        self.chk_color_confidence.setEnabled(has_confidence)
        for widget in (self.lbl_uncertain, self.spin_uncertain, self.btn_next_uncertain):
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
            if c not in ("ObjectNumber", "Source", "Location_Center_X", "Location_Center_Y")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        current = self.combo_hist_metric.currentData() or "AreaShape_Area"
        self.combo_hist_metric.blockSignals(True)
        self.combo_hist_metric.clear()
        for name in numeric:
            self.combo_hist_metric.addItem(_short_name(name), name)
        self.combo_hist_metric.setCurrentIndex(max(self.combo_hist_metric.findData(current), 0))
        self.combo_hist_metric.blockSignals(False)

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
        submit_task(
            task,
            self.state,
            on_finished=lambda results: self._on_measured(generation, results),
            on_error=lambda message: logger.error("Measuring failed: %s", message),
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


def _short_name(column: str) -> str:
    """A column name without its family prefix: 'AreaShape_Area' → 'Area'."""
    family, _, rest = column.partition("_")
    return rest if family in ("AreaShape", "Intensity", "Location", "Texture") and rest else column


def _format(value) -> str:
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.4g}"
    return str(value)
