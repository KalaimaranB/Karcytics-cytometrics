"""The Batch step end to end: add a folder, read conditions, run, review, fix one image, export."""

from __future__ import annotations

import json

import flowio
import pandas as pd
import pytest
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.gating import Gate
from karcytics_plugins.cytometrics.ui import batch as batch_ui
from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui

N = len(img.ISOLATED_CENTRES)


@pytest.fixture
def plate(tmp_path):
    folder = tmp_path / "plate"
    folder.mkdir()
    for name in ("A01_ctrl.png", "A02_lps.png", "B01_lps.png"):
        Image.fromarray(img.isolated_discs()).save(folder / name)
    return folder


@pytest.fixture
def panel(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    monkeypatch.setattr(batch_ui, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    monkeypatch.setattr(batch_ui, "show_info", lambda *a, **k: None)
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    qtbot.waitUntil(lambda: bool(p.pipelines), timeout=10_000)
    p.set_assay("cells")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(p, "project_dir", lambda: str(project))
    typical = tmp_path / "typical.png"
    Image.fromarray(img.isolated_discs()).save(typical)
    p.channel_manager.ingest_path(str(typical))
    p.combo_pipeline.setCurrentIndex(p.combo_pipeline.findData("otsu"))
    p.params_form.set_display_values({"target_channel": "typical.png", "min_area_px": 50})
    p.session.set_gates([Gate("Big", "threshold", x="AreaShape_Area", params={"value": 0})])
    p.stepper.go_to("batch")
    yield p
    p.cleanup()


def _add_plate(panel, monkeypatch, plate, pattern="{well}_{condition}"):
    monkeypatch.setattr(batch_ui, "get_directory", lambda *a, **k: str(plate))
    panel.batch_step.edit_pattern.setText(pattern)
    panel.batch_step.btn_add_folder.click()


def _run(panel, qtbot):
    panel.batch_step.btn_run.click()
    qtbot.waitUntil(lambda: not panel.batch._busy(), timeout=60_000)


def test_folder_is_listed_with_conditions_from_file_names(panel, monkeypatch, plate):
    _add_plate(panel, monkeypatch, plate)

    batch = panel.batch.batch
    assert [(i.id, i.metadata) for i in batch.items] == [
        ("A01_ctrl", {"Well": "A01", "Condition": "ctrl"}),
        ("A02_lps", {"Well": "A02", "Condition": "lps"}),
        ("B01_lps", {"Well": "B01", "Condition": "lps"}),
    ]
    table = panel.batch_step.table
    assert table.rowCount() == 3
    headers = [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]
    assert headers[:4] == ["Image", "Well", "Condition", "Replicate"]
    assert (
        "Otsu" in panel.batch_step.lbl_recipe.text()
        and "1 gate" in panel.batch_step.lbl_recipe.text()
    )


def test_editing_the_list_sets_metadata(panel, monkeypatch, plate):
    _add_plate(panel, monkeypatch, plate)
    table = panel.batch_step.table
    table.item(0, 3).setText("r1")  # Replicate column
    assert panel.batch.batch.items[0].metadata["Replicate"] == "r1"

    table.selectRow(1)
    monkeypatch.setattr(batch_ui, "get_text", lambda *a, **k: "LPS 10ng, r2")
    panel.batch_step.btn_set_condition.click()
    assert panel.batch.batch.items[1].metadata == {
        "Well": "A02",
        "Condition": "LPS 10ng",
        "Replicate": "r2",
    }


def test_run_compare_and_resume(panel, monkeypatch, plate, qtbot):
    _add_plate(panel, monkeypatch, plate)
    _run(panel, qtbot)

    batch = panel.batch.batch
    assert [i.status for i in batch.items] == ["done"] * 3
    assert all(i.n_cells == N for i in batch.items)
    assert batch.path.exists() and "CytoMetrics batches" in batch.folder
    results = panel.results.batch
    assert results.images_table.rowCount() == 3
    assert results.combo_metric.findData("Pct_Big") >= 0
    assert results.conditions_table.rowCount() == 2  # ctrl, lps
    assert results.plate.shape == (2, 3)

    # Resuming only redoes what isn't finished.
    batch.reset([batch.items[1]])
    seconds = [i.seconds for i in batch.items]
    _run(panel, qtbot)
    assert batch.items[0].seconds == seconds[0]
    assert batch.items[1].status == "done"


def test_open_fix_and_save_back(panel, monkeypatch, plate, qtbot):
    _add_plate(panel, monkeypatch, plate)
    _run(panel, qtbot)
    batch = panel.batch.batch
    panel.batch_step.table.selectRow(0)
    panel.batch_step.btn_open.click()

    assert len(panel.state.cells) == N
    assert [c.name for c in panel.image_set.channels] == ["typical.png"]
    assert panel.stepper.current_id == "review"
    assert panel.batch_step.editing.isVisibleTo(panel.batch_step)

    panel.session.delete_cells({panel.state.cells[0]["id"]})
    panel._on_cells_edited()
    qtbot.waitUntil(lambda: not panel.session.needs_measuring(), timeout=10_000)
    panel.batch_step.btn_save_back.click()

    item = batch.items[0]
    assert item.n_cells == N - 1
    assert len(json.loads(batch.cells_path(item).read_text())) == N - 1
    assert "Gate_Big" in pd.read_csv(batch.table_path(item)).columns


def test_export_writes_tables_fcs_and_methods(panel, monkeypatch, plate, qtbot, tmp_path):
    _add_plate(panel, monkeypatch, plate)
    _run(panel, qtbot)
    out = tmp_path / "export"
    monkeypatch.setattr(batch_ui, "get_directory", lambda *a, **k: str(out))
    panel.batch_step.btn_export.click()

    assert len(pd.read_csv(out / "cells.csv")) == 3 * N
    assert pd.read_csv(out / "images.csv")["Count"].tolist() == [N, N, N]
    conditions = pd.read_csv(out / "conditions.csv")
    assert set(conditions["Metric"]) >= {"Count", "Pct_Big"}
    fcs = flowio.FlowData(str(out / "fcs" / "A01_ctrl.fcs"))
    assert fcs.event_count == N
    assert fcs.text["cytometrics_condition"] == "ctrl"
    methods = (out / "batch_methods.md").read_text()
    assert "applied to 3 image(s)" in methods and "manual gating" in methods


def test_batch_reattaches_with_the_saved_session(panel, monkeypatch, plate, qtbot):
    _add_plate(panel, monkeypatch, plate)
    _run(panel, qtbot)
    saved = panel.get_state().to_dict()
    assert saved["batch_manifest"].endswith("batch.json")

    panel.batch.batch = None
    panel.load_state(saved)
    assert panel.batch.batch is not None and panel.batch.batch.done_count == 3
