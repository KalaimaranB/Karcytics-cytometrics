"""Images, calibration and workflow save/load in the panel."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.measure import measure_cells
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
    qtbot.waitUntil(lambda: bool(panel.pipelines), timeout=10_000)
    panel.combo_pipeline.setCurrentIndex(panel.combo_pipeline.findData(segmenter))


def test_uncalibrated_panel_labels_sizes_in_pixels_and_can_run(panel, qtbot):
    _ready(panel, qtbot)
    form = panel.params_form
    assert panel.btn_run_pipeline.isEnabled()
    assert form.widget("min_area_px").suffix() == " px²"
    assert form.widget("diameter").suffix() == " px"
    assert not panel.lbl_units_note.isHidden()
    assert _headers(panel)[:4] == ["ID", "Confidence", "Flagged", "Area (px²)"]


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


def test_calibrating_after_a_pixel_run_converts_measured_cells(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0)])
    before = panel.measurements.iloc[0]

    panel._set_scale(0.5, "manual")

    after = panel.measurements.iloc[0]
    assert after["AreaShape_Area"] == pytest.approx(before["AreaShape_Area"] / 4)
    assert after["AreaShape_Perimeter"] == pytest.approx(before["AreaShape_Perimeter"] / 2)
    assert after["AreaShape_FormFactor"] == pytest.approx(before["AreaShape_FormFactor"])
    area_col = _headers(panel).index("Area (µm²)")
    assert panel.table.item(0, area_col).text() == f"{after['AreaShape_Area']:.4g}"
    assert "µm²" in panel.lbl_stats.text()


def test_csv_headers_follow_the_units(panel, tmp_path, monkeypatch):
    export = tmp_path / "px.csv"
    monkeypatch.setattr(main_panel, "get_save_path", lambda *a, **k: str(export))
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0)])

    panel._on_export_csv()

    header = export.read_text().splitlines()[0].split(",")
    assert header[:5] == [
        "ObjectNumber",
        "Source",
        "Confidence",
        "Flagged",
        "AreaShape_Area [px2]",
    ]
    assert "Intensity_MeanIntensity_cells_png [a.u.]" in header


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
    assert panel.session.labels.max() == len(img.ISOLATED_CENTRES)
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
    qtbot.waitUntil(lambda: bool(other.pipelines), timeout=10_000)

    assert other.combo_pipeline.currentData() == "cellpose"
    values = other.params_form.display_values()
    assert values["cellprob_threshold"] == -2.0
    assert values["augment"] is True
    other.cleanup()


def test_version_1_saved_settings_still_restore(panel, qtbot):
    _ready(panel, qtbot, "otsu")

    panel.load_state(
        {
            "scale": 0.0,
            "cells": [{"id": 1, "points": [[0, 0], [4, 0], [4, 4]], "area": 8.0, "circ": 0.5}],
            "ui_params": {
                "pipeline": "AI Smart Detect (Cellpose)",
                "min_area": 25.0,
                "diameter": 12.0,
                "flow": 0.6,
                "use_dual": True,
                "exclude_borders": False,
            },
        }
    )

    assert panel.combo_pipeline.currentData() == "cellpose"
    values = panel.params_form.display_values()
    assert (values["min_area_px"], values["diameter"], values["flow_threshold"]) == (25, 12, 0.6)
    assert values["use_dual_channel"] is True and values["exclude_borders"] is False
    assert panel.state.version == 2
    assert panel.state.cells == [{"id": 1, "points": [[0, 0], [4, 0], [4, 4]]}]


def _square(cell_id, x, conf=None):
    cell = {"id": cell_id, "label": cell_id}
    cell["points"] = [[x + 2, 10], [x + 12, 10], [x + 12, 20], [x + 2, 20]]
    if conf is not None:
        cell["confidence"] = conf
    return cell


def _deliver(panel, cells, probability=None):
    """Hand the panel a finished run, measured the way the worker measures it."""
    panel._on_pipeline_result(
        {
            "result_cells": cells,
            "measurements": measure_cells(cells, panel.image_set, panel.state.scale),
            "labels": np.zeros((64, 64), np.int32),
            "probability": probability,
        }
    )


def _headers(panel):
    return [panel.table.horizontalHeaderItem(i).text() for i in range(panel.table.columnCount())]


def test_confidence_column_stats_and_review_order(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0, 0.95), _square(2, 20, 0.40), _square(3, 40, 0.70)])

    assert _headers(panel)[1] == "Confidence"
    assert panel.table.item(1, 1).text() == "0.4"
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

    assert panel.combo_color_by.findData("__confidence__") == -1
    assert not panel.btn_next_uncertain.isEnabled()
    assert panel.review_step.uncertain_box.isHidden()  # only for Cellpose's confidence
    assert panel.chk_probability_map.isHidden()
    assert panel.table.item(0, 1).text() == ""


def test_outlines_coloured_by_confidence(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0, 0.99), _square(2, 20, 0.30)])

    panel.combo_color_by.setCurrentIndex(panel.combo_color_by.findData("__confidence__"))

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


def test_roi_and_cellpose_exports_use_the_image_name(panel, tmp_path, monkeypatch):
    panel.channel_manager.ingest_path(str(_ome(tmp_path)))
    _deliver(panel, [_square(1, 0), {**_square(2, 30), "source": "manual"}])
    offered = []

    def save_path(_parent, _title, default, _filter):
        offered.append(default)
        return str(tmp_path / default)

    monkeypatch.setattr(main_panel, "get_save_path", save_path)
    monkeypatch.setattr(main_panel, "show_info", lambda *a, **k: None)

    panel._on_export_rois()
    panel._on_export_seg()

    assert offered == ["cells_rois.zip", "cells_seg.npy"]
    assert (tmp_path / "cells_rois.zip").exists()
    data = np.load(tmp_path / "cells_seg.npy", allow_pickle=True).item()
    assert data["ismanual"].tolist() == [False, True]


def test_drawn_and_deleted_cells_are_remeasured(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0), _square(2, 30)])

    panel._on_cell_drawn([(40, 40), (50, 40), (50, 50), (40, 50)])
    qtbot.waitUntil(lambda: len(panel.measurements) == 3, timeout=5000)
    assert panel.measurements["Source"].tolist() == ["auto", "auto", "manual"]
    assert panel.measurements["AreaShape_Area"].iloc[2] == 121

    panel._on_cell_deleted(1)
    assert panel.measurements["ObjectNumber"].tolist() == [2, 3]
    assert panel.table.rowCount() == 2


def test_background_choice_adds_corrected_columns(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0)])

    panel.combo_background.setCurrentIndex(panel.combo_background.findData("global"))

    qtbot.waitUntil(
        lambda: "Intensity_Background_cells_png" in panel.measurements.columns, timeout=5000
    )
    assert any("CorrectedMeanIntensity" in h for h in _headers(panel))


def test_histogram_metric_can_be_any_column(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0), _square(2, 30)])

    panel.combo_hist_metric.setCurrentIndex(panel.combo_hist_metric.findData("AreaShape_Solidity"))

    assert "Mean Solidity: 1" in panel.lbl_stats.text()
    assert panel.histogram.data == [1.0, 1.0]


def test_reloaded_session_is_measured_again(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_ome(tmp_path, 0.5)))
    _deliver(panel, [_square(1, 0), _square(2, 30)])
    saved = panel.export_state()

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    other.load_state(saved)

    qtbot.waitUntil(lambda: len(other.measurements) == 2, timeout=5000)
    assert other.measurements["AreaShape_Area"].tolist() == pytest.approx(
        panel.measurements["AreaShape_Area"].tolist()
    )
    other.cleanup()


def test_nucleus_and_texture_options_add_columns_and_survive_a_reload(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path, "dapi.png")))
    panel.channel_manager.ingest_path(str(_png(tmp_path, "gfp.png")))
    _deliver(panel, [_square(1, 0), _square(2, 30)])

    panel.combo_nucleus.setCurrentIndex(panel.combo_nucleus.findData("dapi.png"))
    panel.chk_texture.setChecked(True)
    qtbot.waitUntil(
        lambda: (
            {"Intensity_NCRatio_gfp_png", "Texture_Contrast_gfp_png_3"}
            <= set(panel.measurements.columns)
        ),
        timeout=5000,
    )

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    other.load_state(panel.export_state())
    assert other.combo_nucleus.currentData() == "dapi.png"
    assert other.chk_texture.isChecked()
    qtbot.waitUntil(lambda: "Nucleus_Area" in other.measurements.columns, timeout=5000)
    other.cleanup()


def test_outlines_coloured_by_a_measurement_with_a_legend(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    small = _square(1, 0)
    big = {"id": 2, "label": 2, "points": [[30, 30], [60, 30], [60, 60], [30, 60]]}
    _deliver(panel, [small, big])

    panel.combo_color_by.setCurrentIndex(panel.combo_color_by.findData("AreaShape_Area"))

    first, second = panel.canvas._cell_items
    assert (first.color.red(), first.color.green(), first.color.blue()) == (68, 1, 84)
    assert (second.color.red(), second.color.green(), second.color.blue()) == (253, 231, 37)
    assert panel.review_step.lbl_color_legend.text().startswith("Purple = 121")


def test_outlines_layer_can_be_hidden(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_png(tmp_path)))
    _deliver(panel, [_square(1, 0)])

    panel.review_step.chk_show_outlines.setChecked(False)

    assert not panel.canvas.layer("cells").isVisible()


def test_display_contrast_changes_the_view_not_the_data(panel, tmp_path):
    panel.channel_manager.ingest_path(str(_ome(tmp_path)))
    channel = panel.image_set.channels[0]
    data_before = channel.data.copy()
    cm = panel.channel_manager
    cm.table.selectRow(0)
    assert cm.spin_low.isEnabled()

    cm.spin_high.setValue(1000)

    assert channel.display_range == (cm.spin_low.value(), 1000)
    np.testing.assert_array_equal(channel.data, data_before)
    cm.btn_auto.click()
    assert channel.display_range is None


def test_hidden_channel_and_contrast_survive_a_reload(panel, qtbot, tmp_path):
    panel.channel_manager.ingest_path(str(_ome(tmp_path)))
    channel = panel.image_set.channels[0]
    channel.display_range = (10.0, 2000.0)
    panel.channel_manager.table.item(0, 0).setCheckState(main_panel.Qt.CheckState.Unchecked)
    assert not channel.visible

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    other.load_state(panel.export_state())

    restored = other.image_set.channels[0]
    assert restored.display_range == (10.0, 2000.0)
    assert not restored.visible
    assert other.channel_manager.table.item(0, 0).checkState() == main_panel.Qt.CheckState.Unchecked
    other.cleanup()
