"""Images, calibration and workflow save/load in the panel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.state import CytoMetricsState
from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    yield p
    p.cleanup()


def _png(tmp_path, name="cells.png"):
    path = tmp_path / name
    Image.fromarray(img.isolated_discs()).save(path)
    return path


def _ome(tmp_path, um_per_px=0.5):
    path = tmp_path / "cells.ome.tif"
    tifffile.imwrite(
        path,
        img.isolated_discs().astype(np.uint16) * 256,
        metadata={"PhysicalSizeX": um_per_px, "PhysicalSizeY": um_per_px},
        ome=True,
    )
    return path


def _ready(panel, qtbot, segmenter="cellpose"):
    qtbot.waitUntil(lambda: panel.tabs.isTabEnabled(1), timeout=10_000)
    panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData(segmenter))


def test_uncalibrated_panel_labels_sizes_in_pixels_and_can_run(panel, qtbot):
    _ready(panel, qtbot)
    form = panel.params_form
    assert panel.btn_run_pipeline.isEnabled()
    assert form.widget("min_area_px").suffix() == " px²"
    assert form.widget("diameter").suffix() == " px"
    assert not panel.lbl_units_note.isHidden()
    assert panel.table.horizontalHeaderItem(1).text() == "Area (px²)"


def test_adding_an_image_shows_it_on_the_canvas(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))

    assert [c.name for c in panel.image_set.channels] == ["cells.png"]
    assert panel.channel_manager.table.rowCount() == 1
    assert panel.canvas.has_image
    assert panel.btn_calibrate.isEnabled()


def test_unreadable_image_warns_instead_of_crashing(panel, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "karcytics_plugins.cytometrics.ui.channel_manager.show_warning",
        lambda *a: warned.append(a[1]),
    )
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    panel.channel_manager.ingest_path(str(bad))

    assert warned == ["Can't Add Image"]
    assert panel.image_set.channels == []


def test_pixel_size_from_metadata_calibrates_automatically(panel, qtbot, tmp_path):
    _ready(panel, qtbot)
    panel.channel_manager.ingest_path(str(_ome(tmp_path, 0.5)))

    assert panel.state.scale == pytest.approx(0.5)
    assert panel.state.scale_source == "metadata (ome)"
    assert "from image metadata" in panel.lbl_scale.text()
    assert panel.params_form.widget("min_area_px").suffix() == " µm²"
    assert panel.lbl_units_note.isHidden()
    assert panel.image_set.channels[0].data.dtype == np.uint16


def test_metadata_does_not_override_a_manual_scale(panel, tmp_path):
    panel._set_scale(0.2, "manual")

    panel.channel_manager.ingest_path(str(_ome(tmp_path, 0.5)))

    assert panel.state.scale == pytest.approx(0.2)
    assert panel.state.scale_source == "manual"


def test_calibrating_after_a_pixel_run_converts_measured_cells(panel):
    panel.state.cells = [{"id": 1, "points": [], "area": 400.0, "perim": 80.0, "circ": 0.7}]

    panel._set_scale(0.5, "manual")

    cell = panel.state.cells[0]
    assert cell["area"] == pytest.approx(100.0)
    assert cell["perim"] == pytest.approx(40.0)
    assert cell["circ"] == pytest.approx(0.7)
    assert panel.table.item(0, 1).text() == "100.0"
    assert "µm²" in panel.lbl_stats.text()


def test_csv_headers_follow_the_units(panel, tmp_path, monkeypatch):
    export = tmp_path / "px.csv"
    monkeypatch.setattr(main_panel, "get_save_path", lambda *a, **k: str(export))
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)
    panel.state.cells = [{"id": 1, "points": [], "area": 4.0, "perim": 8.0, "circ": 0.7}]

    panel._on_export_csv()

    assert export.read_text().splitlines()[0] == (
        "Cell_ID,Area_px2,Perimeter_px,Circularity,Diameter_px"
    )


def test_state_round_trip_reloads_images_names_and_colours(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_ome(tmp_path, 0.5)))
    panel.image_set.channels[0].name = "GFP"
    panel.image_set.channels[0].color = "green"
    saved = panel.export_state()

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    other.load_state(saved)

    assert [(c.name, c.color) for c in other.image_set.channels] == [("GFP", "green")]
    assert other.channel_manager.table.item(0, 0).text() == "GFP"
    assert other.state.scale == pytest.approx(0.5)
    assert other.canvas.has_image
    other.cleanup()


def test_old_saved_state_without_newer_keys_still_loads():
    state = CytoMetricsState.from_dict({"scale": 0.3, "cells": [], "future_key": 1})

    assert state.scale == 0.3
    assert state.scale_source == ""


class _FakeProject:
    def __init__(self):
        self.saved: dict[str, dict] = {}
        self.workflows = SimpleNamespace(list_all=self._list)

    def save_workflow(self, module_id, payload, metadata):
        name = f"wf{len(self.saved)}.json"
        self.saved[name] = {
            "meta": {**metadata, "module": module_id, "filename": name},
            "payload": payload,
        }

    def _list(self):
        others = [{"module": "flow_cytometry", "filename": "flow.json", "name": "flow"}]
        return others + [w["meta"] for w in self.saved.values()]

    def load_workflow_payload(self, filename):
        return self.saved[filename]["payload"]


def test_save_and_load_go_through_the_project_manager(panel, monkeypatch):
    project = _FakeProject()
    monkeypatch.setattr(panel, "window", lambda: SimpleNamespace(project_manager=project))
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(
        main_panel.SaveWorkflowDialog,
        "exec",
        lambda self: (self.name_input.setText("Run 1"), main_panel.QDialog.DialogCode.Accepted)[1],
    )
    panel._set_scale(0.25, "manual")
    panel._on_save_workflow()
    (saved,) = project.saved.values()
    assert saved["meta"]["name"] == "Run 1"
    assert saved["meta"]["module"] == "cytometrics"

    offered = []

    def pick(_parent, _title, _label, items, *_args):
        offered.extend(items)
        return items[0], True

    monkeypatch.setattr(main_panel.QInputDialog, "getItem", pick)
    panel._set_scale(0.0, "")
    panel._on_load_workflow()

    assert len(offered) == 1 and offered[0].startswith("Run 1")
    assert panel.state.scale == pytest.approx(0.25)


@pytest.mark.parametrize("segmenter", ["otsu", "watershed"])
def test_run_button_finds_cells_and_fills_the_table(panel, qtbot, tmp_path, segmenter):
    _ready(panel, qtbot, segmenter)
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    panel.params_form.set_display_values({"min_area_px": 50})

    panel._on_run_pipeline()
    qtbot.waitUntil(lambda: panel.btn_run_pipeline.isEnabled(), timeout=10_000)

    assert panel.table.rowCount() == len(img.ISOLATED_CENTRES)
    assert panel._labels.max() == len(img.ISOLATED_CENTRES)
    assert panel.progress_bar.isHidden()


def test_settings_follow_the_chosen_algorithm(panel, qtbot):
    _ready(panel, qtbot, "otsu")
    assert panel.params_form.widget("diameter") is None
    assert panel.params_form.btn_advanced.isHidden()

    _ready(panel, qtbot, "cellpose")
    assert panel.params_form.widget("diameter") is not None
    assert panel.params_form.widget("augment") is not None
    assert not panel.params_form.btn_advanced.isHidden()


def test_shared_settings_survive_switching_algorithm(panel, qtbot, tmp_path):
    _ready(panel, qtbot, "otsu")
    panel.channel_manager.ingest_path(str(_png(tmp_path, "a.png")))
    panel.channel_manager.ingest_path(str(_png(tmp_path, "b.png")))
    panel.params_form.set_display_values({"target_channel": "b.png", "min_area_px": 42})

    _ready(panel, qtbot, "watershed")

    values = panel.params_form.display_values()
    assert values["target_channel"] == "b.png"
    assert values["min_area_px"] == 42


def test_detect_settings_round_trip_through_saved_state(panel, qtbot, tmp_path):
    _ready(panel, qtbot, "cellpose")
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    panel.params_form.set_display_values({"cellprob_threshold": -2.0, "augment": True})
    saved = panel.export_state()

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    other.load_state(saved)  # before its segmenters have loaded
    qtbot.waitUntil(lambda: other.tabs.isTabEnabled(1), timeout=10_000)

    assert other.combo_pipeline.currentData() == "cellpose"
    values = other.params_form.display_values()
    assert values["cellprob_threshold"] == -2.0
    assert values["augment"] is True
    other.cleanup()


def test_version_1_saved_settings_still_restore(panel, qtbot):
    _ready(panel, qtbot, "otsu")

    panel._apply_ui_params(
        {
            "pipeline": "AI Smart Detect (Cellpose)",
            "min_area": 25.0,
            "diameter": 12.0,
            "flow": 0.6,
            "use_dual": True,
            "exclude_borders": False,
        }
    )

    assert panel.combo_pipeline.currentData() == "cellpose"
    values = panel.params_form.display_values()
    assert (values["min_area_px"], values["diameter"], values["flow_threshold"]) == (25, 12, 0.6)
    assert values["use_dual_channel"] is True and values["exclude_borders"] is False


def _square(cell_id, x, conf=None):
    cell = {"points": [[x, 10], [x + 10, 10], [x + 10, 20], [x, 20]], "area": 100.0}
    cell.update({"perim": 40.0, "circ": 0.78, "label": cell_id})
    if conf is not None:
        cell["confidence"] = conf
    return cell


def _deliver(panel, cells, probability=None):
    panel._on_pipeline_result(
        {"result_cells": cells, "labels": np.zeros((64, 64), np.int32), "probability": probability}
    )


def test_confidence_column_stats_and_review_order(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0, 0.95), _square(2, 20, 0.40), _square(3, 40, 0.70)])

    assert panel.table.horizontalHeaderItem(5).text() == "Conf."
    assert panel.table.item(1, 5).text() == "0.40"
    assert "Mean AI confidence: 0.68" in panel.lbl_stats.text()
    assert panel.btn_next_uncertain.text() == "Next uncertain (2) ▸"

    picked = []
    for _ in range(3):
        panel._on_next_uncertain()
        picked.append(panel.table.item(panel.table.currentRow(), 0).text())
    assert picked == ["Cell 2", "Cell 3", "Cell 2"]  # least confident first, then wraps


def test_review_tools_are_off_without_confidence(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0)])

    assert not panel.chk_color_confidence.isEnabled()
    assert not panel.btn_next_uncertain.isEnabled()
    assert not panel.chk_probability_map.isEnabled()
    assert panel.table.item(0, 5).text() == ""


def test_outlines_coloured_by_confidence(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0, 0.99), _square(2, 20, 0.30)])

    panel.chk_color_confidence.setChecked(True)

    confident, unsure = panel.canvas._cell_items
    assert confident.color.green() > confident.color.red()
    assert unsure.color.red() > unsure.color.green()


def test_probability_map_overlay_toggles(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    probability = np.zeros((img.CANVAS, img.CANVAS), np.float32)
    _deliver(panel, [_square(1, 0, 0.9)], probability)

    assert panel.chk_probability_map.isEnabled()
    panel.chk_probability_map.setChecked(True)
    assert len(panel.canvas.layer("probability").childItems()) == 1
    panel.chk_probability_map.setChecked(False)
    assert panel.canvas.layer("probability").childItems() == []
