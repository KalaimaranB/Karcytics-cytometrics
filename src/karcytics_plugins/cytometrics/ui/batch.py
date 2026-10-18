"""Batch runs in the UI: the Batch step, the background run and the Batch results tab.

The controller asks its host (the panel) for four things:

- ``build_recipe()``: the current analysis as a ``Recipe``, or raise ValueError
  with what's missing;
- ``pipelines``: the loaded segmenters, by id;
- ``open_batch_item(image_set, cells, title)``: show an image and its cells
  in the steps above;
- ``project_dir()`` and ``batch_methods(batch)``: where to put results, and
  the methods text for an export.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from karcytics_sdk.plugin import (
    ask_yes_no,
    get_directory,
    get_logger,
    get_text,
    show_error,
    show_info,
    show_warning,
    submit_task,
)
from karcytics_sdk.plugin.tasks import TaskHandle
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import QFileDialog, QTableWidgetItem

from ..analysis import batch as bt
from ..analysis.batch import DONE, FAILED, Batch, BatchItem, Recipe
from ..analysis.gating import Gate, with_gate_columns
from ..report import write_methods_sidecars
from .labels import fmt, short_name
from .steps.batch import BatchStep
from .widgets.table_items import NumericTableItem
from .workers import BatchWorker

logger = get_logger(__name__, "cytometrics")

_STATUS = {bt.PENDING: "⏳ waiting", DONE: "✅ done", FAILED: "⚠️ failed"}
_FIXED = ("Image", "Status", "Cells")


class BatchController(QObject):
    batch_changed = Signal()  # the batch was created, loaded or cleared

    def __init__(self, host, step: BatchStep, results, dialog_parent) -> None:
        super().__init__(dialog_parent)
        self.host = host
        self.step = step
        self.results = results  # BatchResults
        self._parent = dialog_parent
        self.batch: Batch | None = None
        self._layout: dict[str, dict[str, str]] = {}
        self._handle: TaskHandle | None = None
        self._started = 0.0
        self._editing: str | None = None  # id of the item open in the editor
        self._images = pd.DataFrame()
        self._wire()
        self.refresh()

    # ── Wiring ───────────────────────────────────────────────────────────────

    def _wire(self) -> None:
        s, r = self.step, self.results
        s.btn_add_folder.clicked.connect(self._on_add_folder)
        s.btn_add_files.clicked.connect(self._on_add_files)
        s.btn_layout.clicked.connect(self._on_layout)
        s.btn_clear.clicked.connect(self._on_clear)
        s.btn_pattern.clicked.connect(self._apply_metadata)
        s.edit_pattern.returnPressed.connect(self._apply_metadata)
        s.btn_set_condition.clicked.connect(self._on_set_condition)
        s.btn_open.clicked.connect(self._on_open)
        s.table.itemChanged.connect(self._on_cell_edited)
        s.table.itemDoubleClicked.connect(lambda item: self._open_row(item.row(), only_done=True))
        s.btn_run.clicked.connect(lambda: self.run(rerun=False))
        s.btn_rerun.clicked.connect(lambda: self.run(rerun=True))
        s.btn_stop.clicked.connect(self.stop)
        s.btn_save_back.clicked.connect(self._on_save_back)
        s.btn_export.clicked.connect(self._on_export)
        for combo in (r.combo_metric, r.combo_by, r.combo_error):
            combo.currentIndexChanged.connect(self._refresh_conditions)
        r.combo_plate_metric.currentIndexChanged.connect(self._refresh_plate)
        r.plate.well_clicked.connect(self._on_well_clicked)
        r.images_table.itemDoubleClicked.connect(self._on_images_row)

    # ── Batch lifecycle ──────────────────────────────────────────────────────

    def attach(self, manifest: str) -> None:
        """Reopen a saved batch (from a restored session)."""
        if not manifest or (self.batch and str(self.batch.path) == manifest):
            return
        try:
            self.batch = Batch.load(manifest)
        except (OSError, ValueError, KeyError) as e:
            logger.warning("Couldn't reopen batch %s: %s", manifest, e)
            self.batch = None
        self.refresh()

    @property
    def manifest(self) -> str:
        if self.batch is None or not self.batch.path.exists():
            return ""
        return str(self.batch.path)

    def _recipe(self) -> Recipe | None:
        try:
            return self.host.build_recipe()
        except ValueError as e:
            show_warning(self._parent, "Set Up One Image First", str(e))
            return None

    def _ensure_batch(self, recipe: Recipe) -> Batch:
        if self.batch is None:
            self.batch = Batch(folder="", recipe=recipe)
        return self.batch

    def _add_paths(self, paths: list[Path]) -> None:
        recipe = self._recipe()
        if recipe is None or not paths:
            return
        batch = self._ensure_batch(recipe)
        items, leftovers = bt.make_items(
            paths, batch.recipe, self.step.edit_pattern.text().strip(), self._layout
        )
        known = {tuple(i.files) for i in batch.items}
        new = [i for i in items if tuple(i.files) not in known]
        taken = {i.id for i in batch.items}
        for item in new:
            base, n = item.id, 2
            while item.id in taken:
                item.id = f"{base} ({n})"
                n += 1
            taken.add(item.id)
        batch.items.extend(new)
        self._save()
        self.refresh()
        message = f"Added {len(new)} image(s)."
        if leftovers:
            message += (
                f" {len(leftovers)} file(s) didn't fit the channel files of the analysis "
                f"({', '.join(batch.recipe.channel_tokens) or 'one file per image'}) and were "
                "skipped."
            )
        self.step.lbl_list.setText(message)
        self.batch_changed.emit()

    def _on_add_folder(self) -> None:
        folder = get_directory(self._parent, "Folder of images", self._start_dir())
        if folder:
            self._add_paths(bt.find_images(folder))

    def _on_add_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in bt.IMAGE_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(
            self._parent,
            "Images to analyse",
            self._start_dir(),
            f"Images ({patterns});;All files (*)",
        )
        self._add_paths([Path(p) for p in paths])

    def _on_layout(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self._parent, "Plate layout", self._start_dir(), "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            self._layout = bt.read_layout(path)
        except (OSError, ValueError, UnicodeDecodeError) as e:
            show_error(self._parent, "Plate Layout", f"Couldn't read the layout:\n{e}")
            return
        if not self._layout:
            show_warning(self._parent, "Plate Layout", "No wells found in that file.")
            return
        self._apply_metadata()
        self.step.lbl_list.setText(f"Plate layout: {len(self._layout)} wells.")

    def _apply_metadata(self) -> None:
        """Re-read every item's metadata from its name, the pattern and the layout."""
        if self.batch is None:
            return
        template = self.step.edit_pattern.text().strip()
        if template:
            names = [Path(i.files[0]).name for i in self.batch.items]
            if names and not any(bt.parse_pattern(template, n) for n in names):
                show_warning(
                    self._parent,
                    "File Name Pattern",
                    f"No file name fits '{template}'. Example name: {names[0]}",
                )
                return
        for item in self.batch.items:
            key = item.id.split(" (")[0]  # the name the item was made from
            item.metadata = {**item.metadata, **bt.metadata_for(key, template, self._layout)}
        self._save()
        self.refresh()

    def _on_clear(self) -> None:
        if self.batch is None:
            return
        if self._busy():
            return
        if not ask_yes_no(
            self._parent,
            "Clear Images",
            "Remove every image from the batch? Results already saved stay on disk.",
        ):
            return
        self.batch = None
        self._editing = None
        self.refresh()
        self.batch_changed.emit()

    def _save(self) -> None:
        if self.batch is not None and self.batch.folder:
            self.batch.save()

    def _start_dir(self) -> str:
        return str(self.host.project_dir() or "")

    def _choose_folder(self) -> bool:
        """Give the batch an output folder (in the project when there is one)."""
        assert self.batch is not None
        if self.batch.folder:
            return True
        project = self.host.project_dir()
        stamp = datetime.now().strftime("%Y-%m-%d %H%M")
        if project:
            self.batch.folder = str(Path(project) / "CytoMetrics batches" / stamp)
        else:
            parent = get_directory(self._parent, "Where should the batch results go?")
            if not parent:
                return False
            self.batch.folder = str(Path(parent) / f"CytoMetrics batch {stamp}")
        self.batch.name = stamp
        self.batch.save()
        self.batch_changed.emit()
        return True

    # ── Running ──────────────────────────────────────────────────────────────

    def _busy(self) -> bool:
        return self._handle is not None and not self._handle.done

    def run(self, rerun: bool = False) -> None:
        if self._busy():
            return
        if self.batch is None or not self.batch.items:
            show_info(self._parent, "No Images", "Add a folder or files first.")
            return
        recipe = self._recipe()
        if recipe is None:
            return
        if self.batch.recipe != recipe:
            finished = self.batch.done_count
            if finished and not rerun:
                rerun = ask_yes_no(
                    self._parent,
                    "Settings Changed",
                    f"The analysis changed since {finished} image(s) were done. Analyse every "
                    "image again with the current settings? (No: finish the rest with the "
                    "settings the batch started with.)",
                )
            if rerun or not finished:
                self.batch.recipe = recipe
        if rerun:
            self.batch.reset()
        if not self._choose_folder():
            return
        segmenter = self.host.pipelines.get(self.batch.recipe.segmenter)
        if segmenter is None:
            show_warning(
                self._parent, "Not Ready", "The segmentation algorithms are still loading."
            )
            return
        todo = len(self.batch.todo())
        if not todo:
            self.step.lbl_progress.setText("Every image is done. Use ↻ Re-run all to start over.")
            return
        self._set_running(True)
        self._started = time.perf_counter()
        self._todo_at_start = todo
        self._handle = submit_task(
            BatchWorker(self.batch, segmenter),
            None,
            on_finished=lambda _r: self._on_run_ended("Batch finished."),
            on_error=lambda message: self._on_run_ended(f"Batch stopped: {message}"),
            on_progress=self._on_progress,
            on_cancelled=lambda: self._on_run_ended("Stopped. Run again to continue."),
        )
        self.host.track_task(self._handle)

    def stop(self) -> None:
        if self._handle is not None and not self._handle.done:
            self._handle.cancel()
            self.step.lbl_progress.setText("Stopping after the current image…")

    def _set_running(self, running: bool) -> None:
        s = self.step
        for w in (
            s.btn_run,
            s.btn_rerun,
            s.btn_add_folder,
            s.btn_add_files,
            s.btn_clear,
            s.btn_open,
        ):
            w.setEnabled(not running)
        s.btn_stop.setEnabled(running)
        s.progress.setVisible(running)
        if running:
            s.progress.setValue(0)

    def _on_progress(self, value: int) -> None:
        assert self.batch is not None
        self.step.progress.setValue(value)
        done = self.batch.done_count
        total = len(self.batch.items)
        elapsed = time.perf_counter() - self._started
        text = f"{done} / {total} images · {sum(i.n_cells for i in self.batch.items):,} cells"
        if value > 2:  # noqa: PLR2004
            left = elapsed * (100 - value) / value
            text += f" · about {_duration(left)} left"
        self.step.lbl_progress.setText(text)
        self._refresh_list()

    def _on_run_ended(self, message: str) -> None:
        self._set_running(False)
        self._handle = None
        failed = [i for i in (self.batch.items if self.batch else []) if i.status == FAILED]
        if failed:
            message += f" {len(failed)} image(s) failed: hover their status for the reason."
        self.step.lbl_progress.setText(message)
        self.refresh()
        self.host.status_message.emit(message)

    # ── Editing one image ────────────────────────────────────────────────────

    def _selected_items(self) -> list[BatchItem]:
        if self.batch is None:
            return []
        rows = sorted({i.row() for i in self.step.table.selectedItems()})
        return [self.batch.items[r] for r in rows if r < len(self.batch.items)]

    def _on_open(self) -> None:
        items = self._selected_items()
        if not items:
            show_info(self._parent, "Open Image", "Select an image in the list first.")
            return
        self._open(items[0])

    def _open_row(self, row: int, only_done: bool = False) -> None:
        if self.batch is None or row >= len(self.batch.items):
            return
        item = self.batch.items[row]
        if only_done and item.status != DONE:
            return
        self._open(item)

    def _open(self, item: BatchItem) -> None:
        assert self.batch is not None
        try:
            image_set = bt.load_item(item, self.batch.recipe)
        except (OSError, ValueError) as e:
            show_error(self._parent, "Open Image", f"Couldn't open {item.id}:\n{e}")
            return
        cells = bt.read_cells(self.batch, item) if item.status == DONE else []
        self.host.open_batch_item(image_set, cells, item.id)
        self._editing = item.id if item.status == DONE else None
        self._refresh_editing()

    def _on_save_back(self) -> None:
        if self.batch is None or self._editing is None:
            return
        item = self.batch.item(self._editing)
        session = self.host.session
        if item is None:
            return
        if session.needs_measuring():
            show_info(self._parent, "Still Measuring", "Wait for the measurements to finish.")
            return
        gates = [Gate.from_dict(g) for g in self.batch.recipe.gates]
        table = with_gate_columns(gates, session.measurements)
        self.batch.cells_path(item).write_text(json.dumps(session.cells), encoding="utf-8")
        table.to_csv(self.batch.table_path(item), index=False)
        item.n_cells = len(session.cells)
        self._save()
        self.refresh()
        self.host.status_message.emit(f"Saved your changes to {item.id} in the batch.")

    def editing_ended(self) -> None:
        """The editor now shows something else (a new image or a new run)."""
        self._editing = None
        self._refresh_editing()

    def _refresh_editing(self) -> None:
        editing = self._editing is not None
        self.step.editing.setVisible(editing)
        if editing:
            self.step.lbl_editing.setText(
                f"Editing <b>{self._editing}</b> from the batch. Fix its cells in Review, then save "
                "them back here."
            )

    # ── Metadata editing ─────────────────────────────────────────────────────

    def _on_set_condition(self) -> None:
        items = self._selected_items()
        if not items:
            show_info(self._parent, "Set Condition", "Select images in the list first.")
            return
        text = get_text(
            self._parent,
            "Set Condition",
            f'Condition for {len(items)} image(s) (add ", replicate" to set both):',
            items[0].metadata.get("Condition", ""),
        )
        if text is None:
            return
        condition, _, replicate = (part.strip() for part in text.partition(","))
        for item in items:
            item.metadata["Condition"] = condition
            if replicate:
                item.metadata["Replicate"] = replicate
        self._save()
        self.refresh()

    def _on_cell_edited(self, cell: QTableWidgetItem) -> None:
        if self.batch is None:
            return
        key = self.step.table.horizontalHeaderItem(cell.column()).text()
        if key in _FIXED or cell.row() >= len(self.batch.items):
            return
        item = self.batch.items[cell.row()]
        value = cell.text().strip()
        if key == "Well" and value:
            pos = bt.well_position(value)
            value = f"{bt.ROWS[pos[0]]}{pos[1] + 1:02d}" if pos else value
        item.metadata[key] = value
        self._save()
        self.refresh()

    # ── Views ────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._refresh_recipe()
        self._refresh_list()
        self._refresh_results()
        self._refresh_editing()

    def _refresh_recipe(self) -> None:
        recipe = self.batch.recipe if self.batch else None
        if recipe is None:
            try:
                recipe = self.host.build_recipe()
            except ValueError as e:
                self.step.lbl_recipe.setText(f"⚠️ {e}")
                return
        self.step.lbl_recipe.setText(_describe(recipe, self.host))

    def _refresh_list(self) -> None:
        table = self.step.table
        table.blockSignals(True)
        items = self.batch.items if self.batch else []
        fields = self.batch.metadata_fields() if self.batch else []
        for standard in ("Well", "Condition", "Replicate"):
            if standard not in fields:
                fields.append(standard)
        headers = ["Image", *fields, "Status", "Cells"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(items))
        for row, item in enumerate(items):
            values = [item.id, *(item.metadata.get(f, "") for f in fields)]
            values += [
                _STATUS.get(item.status, item.status),
                str(item.n_cells) if item.status == DONE else "",
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if headers[col] in _FIXED:
                    cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if headers[col] == "Status" and item.error:
                    cell.setToolTip(item.error)
                if headers[col] == "Image":
                    cell.setToolTip("\n".join(item.files))
                table.setItem(row, col, cell)
        table.resizeColumnsToContents()
        table.blockSignals(False)
        done = self.batch.done_count if self.batch else 0
        if not self._busy():
            self.step.lbl_list.setText(
                f"{len(items)} image(s), {done} done." if items else "No images yet."
            )
        has_items = bool(items)
        for w in (
            self.step.btn_run,
            self.step.btn_rerun,
            self.step.btn_set_condition,
            self.step.btn_open,
        ):
            w.setEnabled(has_items and not self._busy())
        self.step.btn_export.setEnabled(done > 0)

    def _refresh_results(self) -> None:
        if self.batch is None or not self.batch.done_count:
            self._images = pd.DataFrame()
        else:
            self._images = bt.image_summary(self.batch)
        self._fill_images_table()
        metrics = bt.result_metrics(self._images) if len(self._images) else []
        for combo in (self.results.combo_metric, self.results.combo_plate_metric):
            current = combo.currentData() or "Count"
            combo.blockSignals(True)
            combo.clear()
            for m in metrics:
                combo.addItem(_metric_label(m), m)
            combo.setCurrentIndex(max(combo.findData(current), 0))
            combo.blockSignals(False)
        by = self.results.combo_by
        current = by.currentData() or "Condition"
        by.blockSignals(True)
        by.clear()
        for f in self.batch.metadata_fields() if self.batch else []:
            if f != "Replicate":
                by.addItem(f, f)
        by.setCurrentIndex(max(by.findData(current), 0))
        by.blockSignals(False)
        self._refresh_conditions()
        self._refresh_plate()

    def _fill_images_table(self) -> None:
        table = self.results.images_table
        df = self._images
        table.setSortingEnabled(False)
        table.clear()
        columns = [c for c in df.columns if c != "Status"] if len(df) else []
        table.setColumnCount(len(columns))
        table.setRowCount(len(df))
        table.setHorizontalHeaderLabels([_metric_label(c) for c in columns])
        for row, record in enumerate(df.to_dict("records")):
            for col, name in enumerate(columns):
                table.setItem(row, col, NumericTableItem(fmt(record[name])))
        table.setSortingEnabled(True)

    def _refresh_conditions(self, *_args) -> None:
        r = self.results
        metric, by, error = (
            r.combo_metric.currentData(),
            r.combo_by.currentData(),
            r.combo_error.currentData(),
        )
        if not len(self._images) or not metric or not by:
            r.comparison.set_data(pd.DataFrame(), pd.DataFrame(), "Condition", "", "")
            r.conditions_table.setRowCount(0)
            r.lbl_n.setText("")
            return
        summary = bt.condition_summary(self._images, metric, by)
        points = bt.condition_points(self._images, metric, by)
        r.comparison.set_data(summary, points, by, metric, _metric_label(metric), error)
        unit = summary["Unit"].iloc[0] if len(summary) else "images"
        r.lbl_n.setText(
            f"Bars: mean ± {error}; dots: one per {unit[:-1]}; n = number of {unit}."
            + (" Images in the same replicate were averaged first." if unit == "replicates" else "")
        )
        table = r.conditions_table
        columns = [by, "n", "Mean", "SD", "SEM"]
        table.clear()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setRowCount(len(summary))
        for row, record in enumerate(summary.to_dict("records")):
            for col, name in enumerate(columns):
                table.setItem(row, col, NumericTableItem(fmt(record[name])))

    def _refresh_plate(self, *_args) -> None:
        r = self.results
        metric = r.combo_plate_metric.currentData()
        if not len(self._images) or not metric or "Well" not in self._images:
            r.plate.set_data(bt.plate_matrix(pd.DataFrame(), ""), "")
            return
        rows, cols = bt.plate_shape(self._images["Well"])
        r.plate.set_data(bt.plate_matrix(self._images, metric, rows, cols), _metric_label(metric))

    def _on_well_clicked(self, well: str) -> None:
        if self.batch is None:
            return
        self.results.plate.set_selected(well)
        for row, item in enumerate(self.batch.items):
            if item.metadata.get("Well") == well:
                self.step.table.selectRow(row)
                break

    def _on_images_row(self, cell) -> None:
        image_id = self.results.images_table.item(cell.row(), 0)
        if self.batch is None or image_id is None:
            return
        item = self.batch.item(image_id.text())
        if item is not None and item.status == DONE:
            self._open(item)

    # ── Export ───────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if self.batch is None or not self.batch.done_count:
            return
        default = (
            str(Path(self.batch.folder) / "export") if self.batch.folder else self._start_dir()
        )
        folder = get_directory(self._parent, "Export batch results to", default)
        if not folder:
            return
        try:
            written = bt.export(self.batch, folder)
            steps = self.host.batch_methods(self.batch)
            if steps is not None:
                written += list(write_methods_sidecars(Path(folder) / "batch.csv", steps))
        except Exception as e:
            show_error(self._parent, "Export Error", f"Couldn't export the batch:\n{e}")
            return
        n_fcs = sum(p.suffix == ".fcs" for p in written)
        show_info(
            self._parent,
            "Batch Exported",
            f"Written to {folder}:\n• cells.csv (every cell)\n• images.csv (one row per image)\n"
            + (
                "• conditions.csv (per-condition statistics)\n"
                if any(p.name == "conditions.csv" for p in written)
                else ""
            )
            + f"• fcs/ ({n_fcs} FCS files)\n• batch_methods.md and batch_citations.bib",
        )


def _describe(recipe: Recipe, host) -> str:
    segmenter = host.pipelines.get(recipe.segmenter)
    name = getattr(segmenter, "name", recipe.segmenter)
    parts = [f"<b>{name}</b> on '{recipe.params.get('target_channel', '?')}'"]
    parts.append(f"{len(recipe.channel_names)} channel(s): {', '.join(recipe.channel_names)}")
    if recipe.channel_tokens:
        parts.append(f"one file per channel, told apart by {' / '.join(recipe.channel_tokens)}")
    parts.append(f"{len(recipe.gates)} gate(s)" if recipe.gates else "no gates")
    parts.append(f"{recipe.scale:.4g} µm/px" if recipe.scale else "uncalibrated (px)")
    return " · ".join(parts)


def _metric_label(column: str) -> str:
    if column == "Count":
        return "Cells per image"
    if column.startswith("Pct_"):
        return f"% {column[4:]}"
    if column.startswith("Count_"):
        return f"Cells in {column[6:]}"
    if column.startswith("Mean_"):
        return f"Mean {short_name(column[5:])}"
    return column


def _duration(seconds: float) -> str:
    if seconds < 90:  # noqa: PLR2004
        return f"{max(1, round(seconds))} s"
    if seconds < 5400:  # noqa: PLR2004
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"
