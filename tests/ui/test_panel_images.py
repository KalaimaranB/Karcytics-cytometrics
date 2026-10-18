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


def test_uncalibrated_panel_labels_sizes_in_pixels_and_can_run(panel):
    assert panel.btn_run_pipeline.isEnabled()
    assert panel.spin_min_area.suffix() == " px²"
    assert panel.spin_diameter.suffix() == " px"
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


def test_pixel_size_from_metadata_calibrates_automatically(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_ome(tmp_path, 0.5)))

    assert panel.state.scale == pytest.approx(0.5)
    assert panel.state.scale_source == "metadata (ome)"
    assert "from image metadata" in panel.lbl_scale.text()
    assert panel.spin_min_area.suffix() == " µm²"
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
    qtbot.waitUntil(lambda: panel.tabs.isTabEnabled(1), timeout=10_000)
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData(segmenter))
    panel.spin_min_area.setValue(50)

    panel._on_run_pipeline()
    qtbot.waitUntil(lambda: panel.btn_run_pipeline.isEnabled(), timeout=10_000)

    assert panel.table.rowCount() == len(img.ISOLATED_CENTRES)
    assert panel._labels.max() == len(img.ISOLATED_CENTRES)
    assert panel.progress_bar.isHidden()
