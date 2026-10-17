# Plan: grow CytoMetrics from a cell outliner into an image-cytometry workbench

**Status:** Phase 0 done; Phase 1 in progress. Resume from the first unchecked checkpoint in [Phases and checkpoints](#5-phases-and-checkpoints).

**Commits:** everything goes directly on `main` in both `Karcytics-cytometrics` and `Karcytics-SDK`. No branches. One commit per checkpoint group (at least one per phase), conventional-commit messages.

**Living document.** Change it as we learn; record why in the [decision log](#8-decision-log).

---

## 1. Why

CytoMetrics today: load images → calibrate → segment (Otsu / Watershed / Cellpose-SAM) → area, perimeter, circularity → table, histogram, CSV. About 3,200 lines, 1,782 of them in `ui/main_panel.py`.

The biggest gap is that it loads multi-channel fluorescence images but **never reads a pixel intensity**, and it truncates 16-bit images to 8-bit. The goal is a tool that turns any image into a table of objects (cells, nuclei, bacteria, colonies) with many measurements per object, then lets the scientist classify, gate, aggregate across conditions, and report with the methods text and citations written for them.

```
Image(s) ─► Segment ─► Measure ─► Classify / Gate ─► Aggregate ─► Report
           (pluggable)  (many features)  (rules / ML)   (well/condition)  (+ methods & citations)
```

Each use case is an **Assay**, a preset over those five stages. One plugin, several modes.

## 2. Findings this plan fixes

### 2.1 DRY
- Area/perimeter/circularity computed 4×: three pipelines plus `_on_cell_drawn` (own shoelace).
- UI-param restore block pasted 3× (`set_state`, `_refresh_table`, `_on_load_workflow`). `set_state` also calls `_refresh_table`, so it runs twice.
- Channel lookup and border exclusion repeated per pipeline. Diameter computed in both the table and the CSV export.
- Task connect/disconnect boilerplate 3× in the panel. The same pattern exists in flow (6 files) and western-blot (3), so it goes to the SDK.
- VRAM release in both panel `shutdown()` and module `shutdown()`.
- Copies of SDK code: `FunctionalAnalysisTask` (SDK `FunctionalTask`), `SaveWorkflowDialog` (SDK `dialogs.py`).
- Version disagrees: `__init__.py` 0.2.2, `pyproject.toml` 0.3.2, README badge 0.2.2. The README and CLAUDE.md also still say "cyto3", but the code uses `cpsam`.

### 2.2 KISS
- Four names for one operation: `export_state` / `load_state` / `load_workflow` / `apply_state`.
- `set_state` used as "refresh everything" after every small mutation.
- `Channel.path` is monkey-patched and read back through three nested `getattr` fallbacks.
- `hasattr(self, "check_exclude_borders")` guards on a widget that always exists.
- The worker swaps the process-wide `sys.stderr` from a thread. Its signal has no listener, `CellposeLogInterceptor` is unused, and the progress bar never moves.
- Save goes through the project manager; load reads raw JSON from a file dialog.

### 2.3 SOLID
- **SRP:** `main_panel.py` holds 8 classes. The panel owns layout, state, geometry, CSV, JSON, TIFF metadata, task orchestration and model management.
- **OCP:** a new pipeline needs edits in four places, including `"Cellpose" in text` string matching on display names.
- **LSP/ISP:** only Otsu inherits `SegmentationPipeline`; `name` is a property in one pipeline and an attribute in the others; `get_parameters()` is never called.
- **DIP:** `self.window().project_manager` (probably `None` in the isolated process); the panel edits `channel_manager._add_row_to_ui` and `.table.cellWidget`; the analysis layer returns UI-shaped dicts.

### 2.4 Correctness and performance
- Cellpose post-processing allocates a full-image array **per cell** (`np.zeros_like(masks)` in the loop), so cost is O(cells × pixels).
- ~~Nested closures on `task_scheduler` signals run off the GUI thread~~: checked in Phase 1, not a bug (`LocalTaskScheduler` lives on, and emits from, the GUI thread). The closures were still replaced by `submit_task`.
- A pipeline exception came back as `{"error": ...}` through *finished* and was shown as "No cells found". Fixed in Phase 1.2.
- An uncalibrated manual draw silently uses a scale of 1.0.

### 2.5 Cellpose capabilities unused (installed: cellpose 4.2.1.1)
Only `diameter`, `channel_axis` and `flow_threshold` are passed today. Unused: raw 16-bit input with the `normalize` options (percentile, tile normalisation), `cellprob_threshold`, `min_size`, `max_size_fraction`, `niter`, `augment`, `batch_size`/`bsize`, the `progress` callback, list input (batch), `do_3D`/`z_axis`/`anisotropy`/`stitch_threshold`, the `flows` output (cell-probability map → confidence), `train.train_seg` (fine-tuning), `metrics.average_precision`, `io` ROI and `_seg.npy` export, and `denoise` (Cellpose3 restoration).

### 2.6 Build and test gap vs. flow
2 test files vs. 106. No functional/integration/fixtures/data tiers, no license-compliance hook, no CI split between release-gating and UI tests, no matplotlib cache, no docs site, no `.coderabbit.yaml`.

## 3. Ground rules

1. **SDK vs. plugin.** Code goes in the SDK only when ≥2 plugins use it, now or as soon as it lands. CytoMetrics-only code stays local until a second user appears. Gating stays local: flow is independent, and CytoMetrics only *writes* the standard FCS format.
2. **SDK first.** SDK changes land and get a version bump before the CytoMetrics commit that needs them; CytoMetrics then raises `karcytics-sdk>=`.
3. **Licenses: 100% sure or not at all.** A model is adopted only when both its **code and its weights** carry an explicit license compatible with PolyForm Noncommercial (permissive: BSD, MIT, Apache). If the weights' license is unstated, the model is not shipped and the feature uses its fallback. `check-licenses` gates every commit.
4. **Cite in the app wherever applicable.** Every segmenter, measurement family, assay and algorithm that rests on published work declares its citations. They show in the UI next to the feature and are written into every export (§6).
5. **Behaviour-preserving refactors are locked down by tests first** (Phase 0 golden tests).
6. **Pre-commit gates commits.** While iterating, run only the tests for the area being changed.

## 4. Target architecture

```
karcytics_sdk (shared)
  plugin/managed_task.py   FunctionalTask + progress_callback
  plugin/tasks.py          submit_task(task, state, on_finished, on_error, on_progress, on_cancelled) → TaskHandle
  contrib/imaging.py       ImageReader: bit depth kept, physical pixel size from TIFF/OME/ND2 metadata
  plugin/image_view.py     ImageView: zoom/pan/fit/drop/empty state/scale bar/calibration line, overlay layers
  plugin/citations.py      Citation, CitationRegistry, methods-text + BibTeX writer, CitationsPanel widget
  cli: verify-ledger, check-licenses

karcytics_plugins/cytometrics
  analysis/                (no Qt; strict typing; fully unit-tested)
    model.py      Channel(name, data, color, path), ImageSet, Calibration, LabelMask
    session.py    AnalysisSession — single owner of images, masks, ObjectTable, params, history
    segment/      Segmenter protocol + registry; otsu, watershed, cellpose, colony, (gated) instanseg
    measure/      Measurer protocol → pandas columns; shape, intensity, texture, ratio
    table.py      ObjectTable (pandas DataFrame wrapper; CellProfiler-style column names)
    classify/     gates (threshold, polygon, boolean), trainable classifier
    track/        overlap/Hungarian linker, (gated) trackastra
    assays/       Assay protocol + registry; one module per assay
    batch.py      folder/plate runs, condition/replicate metadata, per-condition stats
    export/       csv, fcs, imagej_rois, cellpose_seg, report (methods + BibTeX)
    citations.py  CytoMetrics' Citation entries (registered into the SDK registry)
  ui/
    panel.py      thin: wires views to the session
    views/        mode_picker, setup, detect, review, measure, report, plate
    canvas/       CytoCanvas(ImageView) + layers + edit tools
    widgets/      hardware_monitor, histogram, scatter, inspector, numeric_item, wrapping_label
    dialogs/      model_center, validation_report
    params_form.py  controls generated from Segmenter.parameters()
```

## 5. Phases and checkpoints

### Phase 0: build and test parity, safety net (cytometrics)
- [x] 0.1 Plan file (this document).
- [x] 0.2 License compliance: `scripts/check_license_compliance.py` (from flow) + pre-commit hook + CI step.
- [x] 0.3 CI parity with flow's `release.yml`: release tests (`-m "not ui"`) gate the release, UI tests are informational, matplotlib font cache, comments.
- [x] 0.4 pytest markers (`functional`, `slow`, `edge_case`) and test tiers `tests/{unit,functional,integration,ui,fixtures,data}`.
- [x] 0.5 Single-source version: `__version__` from `importlib.metadata`. Fix README (version badge, Cellpose-SAM instead of cyto3) and CLAUDE.md.
- [x] 0.6 Synthetic image fixtures (discs, touching discs, two-channel, 16-bit, RGB) and golden unit tests for `ImageStack`, Otsu and Watershed (counts, areas, circularity, border exclusion, dual-channel seeds). Found: single-channel Watershed never splits a two-cell clump (strict xfail, fix in 3.1); contour areas under-count discs by ~6% (fixed by 4.1's mask measurements).
- [x] 0.7 `FakeSegmenter` + functional test of `CytoPipelineWorker` end to end. The real Cellpose test is marked `slow` and skips when the model isn't cached.
- [x] 0.8 Integration test: spawn `ui_daemon.py` as a real subprocess via the new SDK harness `karcytics_sdk.testing.ui_daemon`. Also pre-import numpy/cv2 in `ui_daemon.py` (Windows stdin deadlock that flow already guards against).
- [x] 0.9 Docs: `mkdocs.yml`, `docs/user` and `docs/developer` skeletons, `deploy-docs.yml`, `.coderabbit.yaml`, and `docs/CITATIONS.md` listing every work the plugin builds on today.

### Phase 1: SDK shared foundations (Karcytics-SDK, then cytometrics adopts)
- [x] 1.1 `FunctionalTask(progress_callback=...)` (SDK `518e488`). Delete cytometrics' `FunctionalAnalysisTask`.
- [x] 1.2 `submit_task(...)` + `TaskHandle.cancel()/detach()` (SDK `518e488`, `8ccd52b`): one watcher `QObject` on the GUI thread, exactly one outcome callback, every connection removed however the task ends. Cytometrics adopted it. Follow-up for flow (6 files) and western-blot (3 files): listed in 1.6d.
- [ ] 1.3 `contrib/imaging.ImageReader` (tifffile, nd2; BSD-3): native dtype, channel axis, z/t axes, physical pixel size. Tests with generated files.
- [ ] 1.4 `ImageView` base (zoom/pan/fit/drop/empty state/scale bar/calibration tool/overlay layers). Migrate western-blot's `ImageCanvas` onto it.
- [ ] 1.5 `citations.py`: `Citation` (authors, title, venue, year, DOI/URL, license note), registry, `methods_text()`, `to_bibtex()`, and a `CitationsPanel` widget. Flow can register FlowKit/UMAP later.
- [ ] 1.6 CLI `karcytics-sdk verify-ledger` and `karcytics-sdk check-licenses`. Switch cytometrics' pre-commit to them; open follow-ups for the other repos.
- [ ] 1.6b `read_plugin_version(__file__, plugin_id)` (metadata, then pyproject.toml fallback): flow, western-blot (hard-coded today) and cytometrics each carry their own copy.
- [ ] 1.6c `run_ui_daemon` pre-imports numpy before starting its stdin reader (Windows deadlock, numpy/numpy#24290); delete the per-plugin workarounds in flow and cytometrics.
- [ ] 1.6d Flow and western-blot adopt the shared pieces: flow's integration test → `karcytics_sdk.testing.ui_daemon`; their task closures → `submit_task`; their version readers → `read_plugin_version`.
- [ ] 1.7 SDK version bump; cytometrics raises `karcytics-sdk>=`.

### Phase 2: domain model (cytometrics)
- [ ] 2.1 `Channel`/`ImageSet` with declared `path`, native bit depth via SDK `ImageReader`; display scaling done only in the view.
- [ ] 2.2 `Calibration` from metadata (auto) or the calibration line (manual). Uncalibrated runs are allowed and report in pixels, clearly labelled.
- [ ] 2.3 `LabelMask` as the single source of truth; contours derived for display.
- [ ] 2.4 `ObjectTable` on pandas (add `pandas` dependency).
- [ ] 2.5 `AnalysisSession` + `CytoMetricsState` v2 with a migration from v1 saved workflows.
- [ ] 2.6 Remove the duplicate `SaveWorkflowDialog`; one save/load path through the SDK project-manager accessor (fixes isolated-process save).

### Phase 3: segmenters, using Cellpose fully (cytometrics)
- [ ] 3.1 `Segmenter` protocol (`id`, `name`, `parameters()`, `citations`, `segment(ImageSet, params) -> LabelMask`) + decorator registry. Otsu and Watershed ported; shared channel lookup and border exclusion.
- [ ] 3.2 Cellpose: raw 16-bit input with the `normalize` options (percentiles, tile normalisation for uneven illumination); `cellprob_threshold`, `min_size`, `max_size_fraction`, `niter`, `augment` ("High quality"), `batch_size`/`bsize` (memory).
- [ ] 3.3 `progress` callback → real progress bar. Delete the stderr hijack and `CellposeLogInterceptor`.
- [ ] 3.4 Per-object **confidence** (mean cell probability inside the mask) from `flows`; confidence heat-map layer; "review uncertain" list.
- [ ] 3.5 Fix the O(cells × pixels) loop (`scipy.ndimage.find_objects` / `skimage.measure.regionprops`).
- [ ] 3.6 Export ImageJ ROIs and `_seg.npy` (Cellpose GUI/Fiji/QuPath interop).
- [ ] 3.7 Optional Cellpose3 `denoise` pre-step. Benchmark on synthetic noise first; adopt only if it helps Cellpose-SAM.
- [ ] 3.8 `params_form.py`: controls generated from `parameters()` (basic/advanced). Delete the display-name string matching.
- [ ] 3.9 Each segmenter shows its citations ("Powered by …") in the Detect step.

### Phase 4: measurements (cytometrics)
- [ ] 4.1 `Measurer` protocol + registry. Shape: area, perimeter, circularity, solidity, eccentricity, major/minor axis, Feret max/min, equivalent diameter.
- [ ] 4.2 Intensity per channel: mean, median, min, max, std, integrated; local-annulus or global background subtraction.
- [ ] 4.3 Texture (GLCM / Haralick, via scikit-image) and nucleus/cytoplasm ratio when a seed channel exists.
- [ ] 4.4 CellProfiler-style column names; units in headers.
- [ ] 4.5 Manual cells: polygon → mask → same measurers. Delete the shoelace copy.
- [ ] 4.6 Measurement families carry citations (Haralick 1973; CellProfiler naming).

### Phase 5: UI overhaul (cytometrics, may grow as we go)
- [ ] 5.1 Split `main_panel.py`: widgets/dialogs to their own files; panel becomes wiring only.
- [ ] 5.2 Mode picker ("What are you analysing?") → Assay selection; remembered per project.
- [ ] 5.3 Stepper (SDK `WizardPanel`/`StepIndicator`): Load › Calibrate › Detect › Review › Measure › Report; steps supplied by the assay.
- [ ] 5.4 `CytoCanvas(ImageView)`: layers (channels, masks, outlines, IDs, confidence), per-channel display range for 16-bit, colour outlines by any measurement.
- [ ] 5.5 Edit tools: delete, merge, split, brush add/erase, box-prompt re-segment (Cellpose on the drawn box), all undoable (SDK `UndoHistory`).
- [ ] 5.6 "Preview region": run the segmenter on the visible viewport only.
- [ ] 5.7 Inspector: selected object's thumbnail, every measurement, confidence, flag.
- [ ] 5.8 Results dock: Table / Histogram (any metric) / Scatter / Plate / Methods & citations.
- [ ] 5.9 Model Center (replaces Manage AI): per model, size, license, citation, download/delete, installed fine-tuned models.
- [ ] 5.10 Hardware monitor collapses into a status bar.
- [ ] 5.11 UI tests for each view (offscreen).

### Phase 6: image cytometry (cytometrics)
- [ ] 6.1 Scatter of any two measurements, **linked both ways** with the canvas.
- [ ] 6.2 Local gates: threshold, rectangle, polygon, boolean combinations; % positive; gate set saved with the workflow.
- [ ] 6.3 Population colouring on the canvas.
- [ ] 6.4 FCS 3.1 export of the object table (flow can open it, with no code shared).

### Phase 7: batch and conditions (cytometrics)
- [ ] 7.1 Folder and plate-layout import; filename → well/condition parsing; metadata editor.
- [ ] 7.2 Batch runs (Cellpose list input), cancellable, resumable.
- [ ] 7.3 Per-image / per-condition stats with replicates; plate heat-map; comparison plots.
- [ ] 7.4 Batch export (combined CSV/FCS + methods report).

### Phase 8: Assay framework + colony counter (cytometrics)
- [ ] 8.1 `Assay` protocol: segmenter + params + measurers + classifiers + steps + report spec + citations.
- [ ] 8.2 "Fluorescent cells" assay = today's workflow, migrated.
- [ ] 8.3 **Colony counter**: plate (Hough circle) detection and masking, background flattening, colony segmentation with distance-transform splitting of touching colonies, size/colour per colony, CFU/mL from dilution factor and plated volume. Algorithm reimplemented from the literature (cite OpenCFU; no code reuse).

### Phase 9: teach the model (cytometrics)
- [ ] 9.1 Training set from corrected masks (images + label masks stored in the project).
- [ ] 9.2 Fine-tune Cellpose-SAM via `train.train_seg` as a background task with progress; MPS/CUDA/CPU. Prototype MPS speed first.
- [ ] 9.3 Project-scoped custom models in the Model Center.
- [ ] 9.4 Validation report: `metrics.average_precision` vs. held-out annotated images.
- [ ] 9.5 Trainable object classifier (random forest on measurements, scikit-learn BSD-3): "click 10 live, 10 dead".

### Phase 10: more assays and models (cytometrics)
- [ ] 10.1 **Nuclei / cell cycle**: integrated DAPI intensity histogram → G1/S/G2-M. InstanSeg is offered only if the weights-license gate passes; otherwise Cellpose-SAM.
- [ ] 10.2 **Hemocytometer / viability**: grid detection, counting-square selection, trypan-blue live/dead, cells/mL and % viability.
- [ ] 10.3 **Transfection / % positive** against a negative-control image.
- [ ] 10.4 **Confluence / scratch wound**: % area covered, wound width and closure over time.
- [ ] 10.5 **Bacteria**: Cellpose-SAM (+ fine-tuning) with length/width/curvature, dividing-cell flag.
- [ ] 10.6 **Spheroid / organoid** sizing and growth over time.
- [ ] 10.7 **Z-stacks**: `do_3D` / `stitch_threshold`, volume measurements.

### Phase 11: time and interaction (cytometrics)
- [ ] 11.1 Time-lapse tracking: overlap + Hungarian linker (SciPy) by default; Trackastra if its weights-license gate passes (greedy/SCIP linking only, never Gurobi).
- [ ] 11.2 Lineages, growth curves, division events.
- [ ] 11.3 Click-to-segment with µSAM if its weights-license gate passes; otherwise the box-prompt Cellpose tool from 5.5.
- [ ] 11.4 Academy courses for CytoMetrics (SDK `academy.py`): "Count your first plate", "Gate GFP+ cells", "Teach Cellpose".

## 6. Citations in the app

- Each `Segmenter`, `Measurer`, `Assay` and algorithm declares `citations: tuple[Citation, ...]`.
- **Shown:** "Powered by …" line in the Detect step; Model Center entry; About/Citations panel; tooltip on measurement column headers where a method is published.
- **Exported:** every CSV/FCS/report export writes a sidecar `*_methods.md` (methods paragraph naming software, versions, models, key parameters) and `*_citations.bib`.

| Key | Citation | Used by |
|---|---|---|
| cellpose2021 | Stringer, Wang, Michaelos & Pachitariu. *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods 18:100–106 (2021). doi:10.1038/s41592-020-01018-x | Cellpose segmenter |
| cellpose2022 | Pachitariu & Stringer. *Cellpose 2.0: how to train your own model.* Nature Methods 19:1634–1641 (2022). doi:10.1038/s41592-022-01663-4 | Fine-tuning (Phase 9) |
| cellpose3 | Stringer & Pachitariu. *Cellpose3: one-click image restoration for improved cellular segmentation.* Nature Methods 22:592–599 (2025). doi:10.1038/s41592-025-02595-5 | Denoise pre-step |
| cpsam2025 | Pachitariu, Rariden & Stringer. *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv (2025). doi:10.1101/2025.04.28.651001 | Default model |
| pytorch2019 | Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32 (2019). arXiv:1912.01703 | Cellpose runtime |
| otsu1979 | Otsu. *A threshold selection method from gray-level histograms.* IEEE Trans. Syst. Man Cybern. 9(1):62–66 (1979). doi:10.1109/TSMC.1979.4310076 | Otsu, Watershed, colony counter |
| meyer1992 | Meyer. *Color image segmentation.* Int. Conf. on Image Processing and its Applications, 303–306, IET (1992). (OpenCV's own reference for `cv2.watershed`) | Watershed segmenter |
| opencv2000 | Bradski. *The OpenCV Library.* Dr. Dobb's Journal of Software Tools (2000). | Classical pipelines |
| skimage2014 | van der Walt et al. *scikit-image: image processing in Python.* PeerJ 2:e453 (2014). doi:10.7717/peerj.453 | Measurements |
| haralick1973 | Haralick, Shanmugam & Dinstein. *Textural features for image classification.* IEEE Trans. SMC SMC-3(6):610–621 (1973). doi:10.1109/TSMC.1973.4309314 | Texture features |
| cellprofiler4 | Stirling et al. *CellProfiler 4: improvements in speed, utility and usability.* BMC Bioinformatics 22:433 (2021). doi:10.1186/s12859-021-04344-9 | Measurement naming |
| opencfu2013 | Geissmann. *OpenCFU, a new free and open-source software to count cell colonies and other circular objects.* PLoS ONE 8(2):e54072 (2013). doi:10.1371/journal.pone.0054072 | Colony counter (method) |
| roukos2015 | Roukos, Pegoraro, Voss & Misteli. *Cell cycle staging of individual cells by fluorescence microscopy.* Nature Protocols 10:334–348 (2015). doi:10.1038/nprot.2015.016 | Cell-cycle assay |
| suarez2020 | Suarez-Arnedo et al. *An image J plugin for the high throughput image analysis of in vitro scratch wound healing assays.* PLoS ONE 15(7):e0232565 (2020). doi:10.1371/journal.pone.0232565 | Wound assay |
| instanseg2024 | Goldsborough et al. *InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation.* arXiv:2408.15954 (2024). | Gated nuclei model |
| musam2025 | Archit et al. *Segment Anything for Microscopy.* Nature Methods 22:579–591 (2025). doi:10.1038/s41592-024-02580-4 | Gated click-to-segment |
| trackastra2024 | Gallusser & Weigert. *Trackastra: Transformer-based cell tracking for live-cell microscopy.* ECCV (2024). arXiv:2405.15700 | Gated tracking |

All DOI entries were checked against Crossref on 2026-10-17; `meyer1992` against OpenCV's `doc/opencv.bib`; `opencv2000`, `pytorch2019`, `instanseg2024`, `trackastra2024` against the publisher/arXiv pages. Add new entries the same way: never from memory.

## 7. Model and dependency license ledger

| Item | Code | Weights | Status |
|---|---|---|---|
| Cellpose / Cellpose-SAM | BSD-3 (GitHub, verified) | BSD-3 (Hugging Face `mouseland/cellpose-sam`, verified) | ✅ adopted |
| SAM (base of Cellpose-SAM, µSAM) | Apache-2.0 (verified) | — | ✅ |
| InstanSeg | Apache-2.0 (verified) | **not stated** in README | ⏸ gated: adopt only if weights get an explicit permissive license |
| Trackastra | BSD-3 (verified) | **not stated** (pretrained.json downloads) | ⏸ gated; never use Gurobi |
| µSAM (micro-sam) | MIT (verified) | **not verified** | ⏸ gated |
| tifffile, nd2, bioio (core) | BSD-3 (verified) | — | ✅ (bioio readers checked one by one) |
| StarDist | BSD-3 (verified) | — | ✗ not adopted: needs TensorFlow alongside PyTorch; Cellpose-SAM / InstanSeg cover nuclei |
| Omnipose | UW NonCommercial with a redistribution clause | — | ✗ not adopted (license not 100% compatible; also pins a Cellpose fork) |
| Ultralytics YOLO | AGPL | — | ✗ incompatible |
| OpenCFU | (code not used) | — | cite the method only |
| AGAR colony dataset | license not verified | — | ✗ not used |
| Real test images | — | — | only synthetic or CC0/CC-BY sets with the license recorded in `tests/data/LICENSES.md` |

## 8. Decision log
- 2026-10-17: one plugin with modes (Assays); pandas for `ObjectTable`; gating stays local (flow independent); SDK hosts only code shared across plugins; all commits on `main` in cytometrics and the SDK; only models whose code **and** weights licenses are confirmed; citations shown in-app and written into every export.
