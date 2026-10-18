# Plan: grow CytoMetrics from a cell outliner into an image-cytometry workbench

**Status:** Phases 0, 2–7 done; Phase 1 done except 1.7 (SDK release, waiting on the user). Phase 8 done except 8.10 (z-stacks, deferred). Paused before Phase 9 for the user's review. Resume from the first unchecked checkpoint in [Phases and checkpoints](#5-phases-and-checkpoints).

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
- **DIP:** `self.window().project_manager` (works: the daemon runtime sets a `RemoteProjectManager` on the window, but the panel still reaches through its window for it); the panel edits `channel_manager._add_row_to_ui` and `.table.cellWidget`; the analysis layer returns UI-shaped dicts.

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
  plugin/image_view.py     ImageView: zoom/pan/fit/drop/empty state/scale bar/calibration line, overlay layers (done)
  plugin/citations.py      Citation, CitationRegistry, methods-text + BibTeX writer, CitationsPanel widget
  cli: verify-ledger, check-licenses

karcytics_plugins/cytometrics
  analysis/                (no Qt; strict typing; fully unit-tested)
    model.py      Channel(name, data, color, path), ImageSet, Calibration, LabelMask
    session.py    AnalysisSession — single owner of images, masks, ObjectTable, params, history
    segment/      Segmenter protocol + registry; otsu, watershed, cellpose, colony, (gated) instanseg
    measure/      Measurer protocol → pandas columns; shape, intensity, texture, ratio
    table.py      ObjectTable (pandas DataFrame wrapper; CellProfiler-style column names)
    classify/     gates (threshold, polygon, boolean)
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
- [x] 1.2 `submit_task(...)` + `TaskHandle.cancel()/detach()` (SDK `518e488`, `8ccd52b`): one watcher `QObject` on the GUI thread, exactly one outcome callback, every connection removed however the task ends. Cytometrics adopted it. Flow and western-blot adopt it on their own schedule (out of this plan's scope).
- [x] 1.3 `contrib.imaging.read_image()` (SDK `823010b`): native dtype, canonical TCZYX, pixel size/z step/interval/channel names from OME, ImageJ, cm-unit TIFF resolution, ND2 (optional `karcytics-sdk[imaging]`). Inch-unit resolution is ignored on purpose (screen DPI). Cytometrics adopts it in 2.1.
- [x] 1.4 `ImageView` (SDK `a20e041`): zoom limits, middle/Ctrl+drag pan (left-drag opt-in), fit, empty state, file drop, calibration line, scale bar, named layers. Cytometrics `9cd8f84` and western-blot `f5d9f72` (on `main`, after fast-forwarding it to `fix/theme-following`) are built on it.
- [x] 1.5 SDK `citations` + `citation_widgets` (`a852509`): `Citation`, registry, `MethodStep`, `methods_text()`, `to_bibtex()`, `CitationLabel`, `CitationsPanel`. Cytometrics registers verified entries, each pipeline declares `citations` + `describe()`, the Detect step shows a "Powered by" line for every algorithm, and CSV export writes `_methods.md` + `_citations.bib` (pulled forward from §6).
- [x] 1.6 `karcytics-sdk check-licenses` (SDK `acdbcf3`); `doctor` now flags a repo with no license hook. Cytometrics' copy deleted. **Not** done for the ledger verifier, by decision: it is a stdlib-only script so CI can run it before installing anything (the CLI imports PySide6); its four copies are identical apart from formatting.
- [x] 1.6b `read_plugin_version` (SDK `cc7b5c5`); cytometrics uses it.
- [x] 1.6c `run_ui_daemon` pre-imports numpy (SDK `cc7b5c5`); cytometrics' workaround removed. Flow keeps its own copy (it also pre-imports pandas); that's flow's call.
- [ ] 1.7 SDK version bump (proposed 3.1.0) and cytometrics `karcytics-sdk>=3.1.0`. **Waiting on the user:** the SDK's CI publishes to PyPI on push when `main` has an untagged version, and `main` also carries other unpushed work.

### Phase 2: domain model (cytometrics)
- [x] 2.1 `analysis/model.py`: `Channel` (native dtype, `path`, `as_uint8()` for display and the 8-bit OpenCV segmenters) and `ImageSet` (`add_file` via SDK `read_image`, `get(name)`, `composite()` in RGB, size check). Multi-channel files take their names from metadata (OME/ND2) and colours from stain names (DAPI → blue, GFP → green, Cy5 → magenta, …). Only t=0, z=0 for now (3-D is 8.10). `ImageStack` and its 16-bit truncation are gone; Cellpose now gets native 16-bit data.
- [x] 2.2 Calibration: pixel size from file metadata is applied automatically unless a scale is already set; the calibration line sets it manually; `state.scale_source` records which, and the methods text says so. Uncalibrated runs are allowed: every control, table header, CSV column and methods sentence reads px/px². Recalibrating converts already-measured cells.
- [x] 2.6 Save/load through the project manager only: the SDK `SaveWorkflowDialog`, a 📂 Load Session button listing this project's CytoMetrics workflows (`pm.workflows.list_all()` → `pm.load_workflow_payload`). The duplicate dialog, the raw-JSON loader, the `apply_state` alias and the three copies of the UI-parameter restore are gone. `CytoMetricsState.from_dict` ignores keys it doesn't know.
- Moved: `LabelMask` → 3.1, `ObjectTable` (pandas) → 4.1, `AnalysisSession` + state v2 migration → 5.1. They belong with the code that first needs them.

### Phase 3: segmenters, using Cellpose fully (cytometrics)
- [x] 3.1 `LabelMask` as the single source of truth (contours derived for display) and the `Segmenter` protocol (`id`, `name`, `parameters()`, `citations`, `segment(ImageSet, params) -> LabelMask`) + decorator registry. Otsu and Watershed ported; shared channel lookup and border exclusion. **Done as** `analysis/segment/` (`base.py`: `Param`, `Segmenter`, `SegmentationResult`, `@register`, `cells_from_labels`; `run_segmentation` applies border exclusion and the area filter once for every segmenter). The label mask is the result; cell outlines/measurements are derived from it (still outline-polygon measurements until 4.1). Border exclusion is now a choice for every segmenter. Watershed seeds from distance-transform peaks (`peak_local_max`, new "Min. Cell Separation" parameter) and runs scikit-image's watershed, so a two-cell clump now splits without a nuclei channel (the strict xfail is now a passing test); it cites scikit-image.
- [x] 3.2 Cellpose: raw 16-bit input with the `normalize` options (percentiles, tile normalisation for uneven illumination); `cellprob_threshold`, `min_size`, `max_size_fraction`, `niter`, `augment` ("High quality"), `batch_size`/`bsize` (memory). **Done:** all exposed except `min_size` (our area filter covers it) and `bsize` (fixed at 256 for Cellpose-SAM). `normalize` is always passed as a dict: passing a bool makes Cellpose 4.2 overwrite its module-level defaults. Every non-default setting goes into the methods text.
- [x] 3.3 Progress: Cellpose's own `progress` argument only reports in 3-D, so segmenters report stages through a `progress` callback (model loaded, network run, then per-object measuring 90–100%), shown as a percentage on the scanning indicator. The stderr hijack, `StreamCatcher` and `CellposeLogInterceptor` are deleted.
- [x] 3.4 Per-object **confidence** (mean cell probability inside the mask) from `flows`; confidence heat-map layer; "review uncertain" list. **Done:** Conf. column, mean confidence in the stats, outlines coloured red→amber→green, a cyan cell-probability overlay, and **Next uncertain ▸** (least confident first, centres the canvas). Known layout glitch for 5.8: the histogram overlaps the Export header in a short window.
- [x] 3.5 Fix the O(cells × pixels) loop (`scipy.ndimage.find_objects` / `skimage.measure.regionprops`).
- [x] 3.6 Export ImageJ ROIs and `_seg.npy` (Cellpose GUI/Fiji/QuPath interop). **Done** in `analysis/export.py`, from the current cells (deletions and hand-drawn cells included): ROIs via `roifile` (BSD-3), `_seg.npy` with `masks`, `outlines`, `ismanual`, `filename` (flows omitted; optional for the GUI). Default names follow the image (`<image>_rois.zip`, `<image>_seg.npy`).
- [~] 3.7 ~~Optional Cellpose3 `denoise` pre-step.~~ **Dropped:** Cellpose 4.2's `denoise.py` contains only the degradation helpers (`add_noise`) used to augment training; the restoration models (`DenoiseModel`, `CellposeDenoiseModel`) no longer exist, so there is nothing to adopt. Revisit only if a later Cellpose release brings them back.
- [x] 3.8 `params_form.py`: controls generated from `parameters()` (basic/advanced). Delete the display-name string matching. **Done:** `Param` declares kind, range, unit (`length`/`area`, shown in µm or px, passed in px), `advanced`, `enabled_by`, help (tooltips). Values are remembered by key across algorithm switches. `ui_params` v2 = `{pipeline: id, values}`; v1 keys are migrated on load.
- [x] 3.9 Each segmenter shows its citations ("Powered by …") in the Detect step.

### Phase 4: measurements (cytometrics)
- [x] 4.1 `ObjectTable` on pandas (add `pandas`) and the `Measurer` protocol + registry. Shape: area, perimeter, circularity, solidity, eccentricity, major/minor axis, Feret max/min, equivalent diameter. **Done** in `analysis/measure/` (`base.py` Column/Measurer/ObjectView/MeasureOptions, `shape.py`, `intensity.py`, `table.py` `measure_cells`/`rescale`/`write_csv`). Cells are rasterised from their outlines and measured per bounding box (1,585 cells on a 2048² image in 0.6 s). The worker measures after segmenting; edits, background changes and reloads re-measure in a background task (newest result wins). Recalibration rescales lengths/areas without re-measuring. The area filter now uses the same pixel area that is reported. Min Feret uses rotating calipers on the hull of pixel corners (exact for rectangles).
- [x] 4.2 Intensity per channel: mean, median, min, max, std, integrated; local-annulus or global background subtraction.
- [x] 4.3 Texture (GLCM / Haralick, via scikit-image) and nucleus/cytoplasm ratio when a seed channel exists. **Done:** `texture.py` (6 CellProfiler-named Haralick features per channel, image-wide 8-bit quantisation, only in-object pixel pairs, 4 directions averaged) and `nucleus.py` (nucleus = cell ∩ above the nuclear channel's Otsu threshold; areas, N/C area ratio, per-channel nuclear/cytoplasmic means and ratio). Both are opt-in in the Results tab (the nucleus channel is picked there, independent of the Detect seed channel). Measurers gained optional `enabled(options)` and `prepare(image_set, labels, options)` hooks.
- [x] 4.4 CellProfiler-style column names; units in headers (`AreaShape_Area [um2]` in the CSV, `Area (µm²)` in the table; tooltips carry the definitions).
- [x] 4.5 Manual cells: polygon → mask → same measurers. Delete the shoelace copy.
- [x] 4.6 Measurement families carry citations (Haralick 1973; CellProfiler naming). Crossref-verified `skimage2014`, `cellprofiler4`, `haralick1973`; the methods text adds a sentence per family used.

### Phase 5: UI overhaul (cytometrics, may grow as we go)
- [x] 5.1 Split `main_panel.py`: widgets/dialogs to their own files; panel becomes wiring only. `AnalysisSession` + `CytoMetricsState` v2 with a migration from v1 saved workflows. **Done:** `ui/widgets/` (hardware monitor, scanning indicator, histogram, table items), `ui/dialogs/model_manager.py`; `analysis/session.py` (Qt-free: calibration, cells, delete/add, uncertainty, reload, methods, export names) with the panel delegating; `state.migrate()` turns v1 saves (display-name `ui_params`, per-cell area/perim/circ) into v2. The step pages (5.3) finish the split.
- [x] 5.2 Mode picker ("What are you analysing?") → Assay selection; remembered per project. **Done:** a light `analysis/assays.py` (pulled forward from 8.1: id, name, description, steps, segmenter, Detect settings, measurement options, default stain) with **Cells** and **Nuclei**; `ui/mode_picker.py` cards; the choice is saved in the session (`ui_params.assay`) and remembered per project folder in the SDK `PluginConfig` (nothing written into the project). Restoring a session switches steps without overwriting its settings.
- [x] 5.3 Stepper (SDK `WizardPanel`/`StepIndicator`): Load › Calibrate › Detect › Review › Measure › Report; steps supplied by the assay. **Done** with a local one-line `StepBar` + page stack + Back/Next (`ui/stepper.py`): the SDK `WizardPanel` paints every child and assumes one column, and its `StepIndicator` is vertical (≈180 px). Pages in `ui/steps/` only build widgets; `CytoMetricsPanel._wire()` connects them. Report shows the live methods text (copyable) and the references (SDK `CitationsPanel`).
- [x] 5.4 `CytoCanvas(ImageView)`: layers (channels, masks, outlines, IDs, confidence), per-channel display range for 16-bit, colour outlines by any measurement. **Done** on the existing `MultiChannelCanvas` (no rename needed): per-channel visibility (checkbox in the channel table) and display range (`Channel.display_range`, auto = 0.1–99.9th percentile, display only), both saved with the session; outlines/IDs/probability layers toggle; "Colour outlines by" AI confidence or any measurement (viridis stops verified against matplotlib), with a legend.
- [x] 5.5 Edit tools: delete, merge, split, brush add/erase, box-prompt re-segment (Cellpose on the drawn box), all undoable (SDK `UndoHistory`). **Done:** geometry in `analysis/edit.py` (rasterise → change mask → re-trace; merge bridges ≤2 px gaps and refuses distant cells; split cuts with a 1 px 4-connected line; brush-add keeps only what touches the cell), session operations, canvas tools (select/Shift-select, draw, split line, brush strokes, box), Merge/Delete selected, ↶/↷ + Ctrl+Z/Ctrl+Shift+Z through `PluginBase.push_state`; runs, calibration and every edit are undo steps; a loaded session starts a clean history. Region runs keep cells cut by the box edge. Panel tasks are tracked and detached on close (a re-measure finishing after close used to hit a deleted table).
- [x] 5.6 "Preview region": run the segmenter on the visible viewport only. **Done:** 👁 Preview on visible area draws dashed outlines on a separate layer; nothing is kept, a full run clears them.
- [x] 5.7 Inspector: selected object's thumbnail, every measurement, confidence, flag. **Done:** the Cell tab of the results dock (`widgets/inspector.py`): thumbnail cut from the displayed composite with the outline, all measurements with units, source and confidence, 🚩 Flag (saved in the cell, `Flagged` column, dashed outline, undoable). Selection highlight is now white so it stands out over any colour-by.
- [x] 5.8 Results dock: Table / Histogram (any metric) / Scatter / Plate / Methods & citations. **Done** (`ui/results_dock.py`): Table · Histogram (any metric) · Scatter (any two measurements; click a point to select its cell, canvas selection highlights points: the two-way link from 6.1) · Cell. **Plate** comes with plate layouts in Phase 7; **Methods & citations** live in the Report step rather than twice.
- [x] 5.9 Model Center (replaces Manage AI): per model, size, license, citation, download/delete. **Done:** `analysis/models.py` (model registry with code/weights licences, source, citations; the excluded models and why, from §7) and `ui/dialogs/model_center.py`. Delete now removes only that model's file (the old dialog emptied the whole shared `~/.cellpose/models` cache). The "Your models" placeholder was removed when fine-tuning was dropped (see the decision log).
- [x] 5.10 Hardware monitor collapses into a status bar. **Done:** `widgets/status_bar.py`, one line under the steps (app/system CPU, RAM, GPU memory once torch is loaded; never imports torch).
- [x] 5.11 UI tests for each view (offscreen): steps/picker (`test_steps.py`), editing/undo/dock (`test_editing.py`), images/calibration/measurements (`test_panel_images.py`), params form, canvas, Model Center, plus the Qt-free session/edit/measure units. 192 tests.

### Phase 6: image cytometry (cytometrics)
- [x] 6.1 Scatter of any two measurements, **linked both ways** with the canvas. **Done:** linking came in 5.8; 6.1 added log axes (shared `ui/widgets/plot_axis.py`), population colours and gate overlays. The histogram was rewritten the same way (numpy bins, log axis, gate markers).
- [x] 6.2 Local gates: threshold, rectangle, polygon, boolean combinations; % positive; gate set saved with the workflow. **Done:** `analysis/gating.py` (Qt-free; also range and NOT; gates nest through `parent`; shape gates are tested in the log/linear space they were drawn in) and the **Gate** step (`ui/steps/gate.py`) driven by `ui/gating.py`: tools draw on the histogram/scatter, a population tree with counts, % parent and % total, exact cut-off editing, rename/colour/delete (dependants go too), Combine… dialog. Gates live in `CytoMetricsState.gates` (saved, undoable) and are described in the methods text (`report.gating_step`).
- [x] 6.3 Population colouring on the canvas. **Done:** "Population" colour-by (deepest population wins; ungated grey), switched on from the Gate step; Okabe–Ito palette.
- [x] 6.4 FCS 3.1 export of the object table (flow can open it, with no code shared). **Done:** `analysis/fcs.py` written from the standard (float32, `$PnN` = column, `$PnS` = label with unit, gates as 0/1 parameters); tests read it back with FlowIO (dev-only, BSD-3), the reader behind flow's FlowKit loader. fcsparser was not added: it pins numpy < 2. Also a population-statistics CSV and `Gate_<name>` columns in the cell CSV. Citation `fcs31` (Crossref-checked).

### Phase 7: batch and conditions (cytometrics)
- [x] 7.1 Folder and plate-layout import; filename → well/condition parsing; metadata editor. **Done:** `analysis/batch.py` (Qt-free): `find_images`, `parse_well`, `{field}` file-name templates, plate layouts as a Well list or a plate grid, per-channel files grouped by the token that tells the typical image's channel files apart. The **Batch** step (`ui/steps/batch.py`, `ui/batch.py`) lists images with editable metadata and "Set condition…".
- [x] 7.2 Batch runs, cancellable, resumable. **Done:** a `Recipe` (segmenter, params, channels, measurements, gates, scale) is taken from the current analysis; `BatchWorker` runs items one by one, writing each item's cells and table and the `batch.json` manifest after every item, so Stop + Run resumes. Changed settings ask before redoing finished items. "Open in editor" / "Save changes to batch" fixes one image. The batch reattaches with the saved session (`CytoMetricsState.batch_manifest`). **Departure:** Cellpose's list input isn't used: images go through one loaded model one at a time, which keeps per-image saving, progress and resume (list input would batch only the network, and hold every image in memory).
- [x] 7.3 Per-image / per-condition stats with replicates; plate heat-map; comparison plots. **Done:** `image_summary` (count, count and % per population, mean of every measurement), `condition_summary` (images averaged within a replicate first; n = replicates, else images; SD and SEM), `plate_matrix`/`plate_shape` (6- to 384-well). Results dock **Batch** tab: Images table, Conditions (bar ± SEM/SD with replicate dots, beside the table), Plate heat-map (viridis, hover values, click a well).
- [x] 7.4 Batch export (combined CSV/FCS + methods report). **Done:** `cells.csv`, `images.csv`, `conditions.csv` (every metric, long format), one FCS per image (metadata as `CYTOMETRICS_*` keywords) and methods/BibTeX (`report.batch_methods`: settings, gating, how conditions were summarised).

### Phase 8: assay framework and new assays (cytometrics)
An **assay** is a preset for one kind of experiment: which segmenter and settings to use, which steps to show, what to measure, what extra result to compute (CFU/mL, % viable, % positive, …) and what goes in the report and its citations. Each one is a card in the "What are you analysing?" picker. Cells and Nuclei (Phase 5) are the first two.

- [x] 8.1 `Assay` protocol. **Done:** `analysis/assays/`: an `Assay` (segmenter, Detect settings, steps, measurements, picker group, citations, the word for one object) with an optional `AssayAnalysis` (its own settings, derived columns, suggested gates, headline results, overlays, actions such as "draw the counting area", methods text). A Results step shows the headline values as cards; batch runs add them per image as `Result_<key>` columns.
- [x] 8.2 "Cells" assay = today's workflow, migrated.
- [x] 8.3 **Colony counter**. **Done:** `colonies` segmenter (Hough dish, rim excluded, blur-subtracted flattening, polarity by skew, Otsu, distance-transform watershed); CFU/mL, countable-range warning (30–300), median diameter, plate coverage. Cites OpenCFU and Breed & Dotterrer 1916.
- [x] 8.4 **Nuclei / cell cycle**. **Done:** DNA content (integrated intensity ÷ G1 peak × 2), Sub-G1/G1/S/G2-M/>4N with suggested range gates. Cellpose-SAM (InstanSeg not adopted).
- [x] 8.5 **Hemocytometer / viability**. **Done:** trypan-blue darkness (1 − mean / local background), a counting-area box, cells/mL from chamber depth and dilution, calibration from the grid line spacing (autocorrelation of the line profile).
- [x] 8.6 **Transfection / % positive**. **Done:** cut-off from a negative-control image (a percentile of its reporter intensity), a Positive gate, % positive.
- [x] 8.7 **Confluence / scratch wound**. **Done:** `texture` segmenter (local SD, Otsu, small gaps filled); Confluence %; wound area, mean and narrowest width, closure against a start area set on the first image. The batch time course (closure over time per well) moved to 9.5.
- [x] 8.8 **Bacteria**. **Done:** `rods` measurer (skeleton length, width with EDT correction, curvature, constriction, dividing flag).
- [x] 8.9 **Spheroid / organoid** sizing. **Done:** `spheroid` segmenter (smoothing, Otsu, largest objects kept); diameter, area, volume π d³/6, circularity, solidity (Zanoni 2016). Growth over time moved to 9.5.
- [ ] 8.10 **Z-stacks**: `do_3D` / `stitch_threshold`, volume measurements. **Deferred** (user, 2026-10-17: not a priority).

### Phase 9: time and interaction (cytometrics)
- [ ] 9.1 Time-lapse tracking: overlap + Hungarian linker (SciPy) by default; Trackastra if its weights-license gate passes (greedy/SCIP linking only, never Gurobi).
- [ ] 9.2 Lineages, growth curves, division events.
- [ ] 9.3 Click-to-segment with µSAM if its weights-license gate passes; otherwise the box-prompt Cellpose tool from 5.5.
- [ ] 9.5 Batch time course: a `{time}` field in file names, results over time per well/condition, closure or growth relative to the first time point (wound closure, spheroid growth).
- [ ] 9.4 Academy courses for CytoMetrics (SDK `academy.py`): "Count your first plate", "Gate GFP+ cells", "Score a scratch wound".

## 6. Citations in the app

- Each `Segmenter`, `Measurer`, `Assay` and algorithm declares `citations: tuple[Citation, ...]`.
- **Shown:** "Powered by …" line in the Detect step; Model Center entry; About/Citations panel; tooltip on measurement column headers where a method is published.
- **Exported:** every CSV/FCS/report export writes a sidecar `*_methods.md` (methods paragraph naming software, versions, models, key parameters) and `*_citations.bib`.

| Key | Citation | Used by |
|---|---|---|
| cellpose2021 | Stringer, Wang, Michaelos & Pachitariu. *Cellpose: a generalist algorithm for cellular segmentation.* Nature Methods 18:100–106 (2021). doi:10.1038/s41592-020-01018-x | Cellpose segmenter |
| cellpose2022 | Pachitariu & Stringer. *Cellpose 2.0: how to train your own model.* Nature Methods 19:1634–1641 (2022). doi:10.1038/s41592-022-01663-4 | Not used: fine-tuning dropped |
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
| roifile (ImageJ ROI export) | BSD-3-Clause (package metadata, verified) | — | ✅ |
| scipy, scikit-image | BSD-3 (verified) | — | ✅ |
| StarDist | BSD-3 (verified) | — | ✗ not adopted: needs TensorFlow alongside PyTorch; Cellpose-SAM / InstanSeg cover nuclei |
| Omnipose | UW NonCommercial with a redistribution clause | — | ✗ not adopted (license not 100% compatible; also pins a Cellpose fork) |
| Ultralytics YOLO | AGPL | — | ✗ incompatible |
| OpenCFU | (code not used) | — | cite the method only |
| AGAR colony dataset | license not verified | — | ✗ not used |
| Real test images | — | — | only synthetic or CC0/CC-BY sets with the license recorded in `tests/data/LICENSES.md` |

## 8. Decision log
- 2026-10-17: 8.10 (z-stacks) deferred at the user's request. The time-based parts of 8.7 and 8.9 (closure and growth over time in batches) moved to Phase 9 as 9.5, with the other time work.
- 2026-10-17: the old Phase 9 (fine-tuning Cellpose-SAM, training sets, custom models, validation report, trainable live/dead classifier) dropped at the user's request: not enough annotated data to train on. The old Phases 8 and 10 merged into one Phase 8 (assay framework plus every new assay, colony counter included), and the old Phase 11 became Phase 9.
- 2026-10-17: 3.7 (Cellpose3 denoise) dropped: removed from Cellpose 4.2 (only training-augmentation code remains).
- 2026-10-17: Phase 3 keeps outline-polygon measurements so the golden values only move once, in 4.1.
- 2026-10-17: the security-ledger verifier stays a per-repo stdlib script (CI runs it before installing anything); only the license check moved into the SDK CLI.
- 2026-10-17: western-blot's local `main` fast-forwarded to `fix/theme-following` (one commit, no divergence) so its canvas migration could go on `main`.
- 2026-10-17: one plugin with modes (Assays); pandas for `ObjectTable`; gating stays local (flow independent); SDK hosts only code shared across plugins; all commits on `main` in cytometrics and the SDK; only models whose code **and** weights licenses are confirmed; citations shown in-app and written into every export.
