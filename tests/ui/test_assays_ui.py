"""Assays in the panel: the Results step, settings, actions and suggested gates."""

from __future__ import annotations

import pytest
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


def _panel(qtbot, monkeypatch):
    monkeypatch.setattr(main_panel, "show_error", lambda *a, **k: pytest.fail(a[-1]))
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    p.resize(1200, 900)
    qtbot.waitUntil(lambda: bool(p.pipelines), timeout=10_000)
    return p


def _load(p, tmp_path, name, array):
    path = tmp_path / name
    Image.fromarray(array).save(path)
    p.channel_manager.ingest_path(str(path))


def _segment(p, qtbot, segmenter, params):
    from karcytics_plugins.cytometrics.ui.workers import CytoPipelineWorker

    worker = CytoPipelineWorker()
    worker.configure(p.pipelines[segmenter], p.image_set, params, p.state.scale, p.measure_options)
    p._on_pipeline_result(worker.run())
    qtbot.waitUntil(lambda: not p.session.needs_measuring(), timeout=10_000)


def test_nuclei_results_and_suggested_gates(qtbot, monkeypatch, tmp_path):
    p = _panel(qtbot, monkeypatch)
    p.set_assay("nuclei")
    assert "results" in p.stepper.step_ids()
    _load(p, tmp_path, "DAPI.png", img.isolated_discs())
    assert p.session.assay_settings["dna_channel"] == "DAPI.png"
    _segment(p, qtbot, "otsu", {"target_channel": "DAPI.png", "min_area_px": 50})

    p.stepper.go_to("results")
    labels = [c.lbl_name.text() for c in p.results_step._cards]
    assert labels[:3] == ["G1 peak (2N)", "% Sub-G1", "% G1"]
    assert "CellCycle_DNAContent" in p.measurements

    p.results_step.btn_apply_gates.click()
    assert [g.name for g in p.session.gates] == ["Sub-G1", "G1", "S", "G2/M", ">4N"]
    assert p.can_undo
    p.cleanup()


def test_colony_counter(qtbot, monkeypatch, tmp_path):
    p = _panel(qtbot, monkeypatch)
    p.set_assay("colonies")
    offered = [p.combo_pipeline.itemData(i) for i in range(p.combo_pipeline.count())]
    assert "colonies" in offered and p.combo_pipeline.currentData() == "colonies"
    plane, _n = img.plate_photo()
    _load(p, tmp_path, "plate.png", plane)
    params = p.params_form.values()
    assert params["exclude_borders"] is False
    _segment(p, qtbot, "colonies", {**params, "target_channel": "plate.png"})

    p.results_step.form.set_display_values({"dilution_factor": 1000, "volume_ul": 50})
    p._on_assay_settings_changed()
    values = {c.lbl_name.text(): c.lbl_value.text() for c in p.results_step._cards}
    assert int(values["Colonies"]) == len(p.state.cells)
    assert values["CFU / mL"].endswith("CFU/mL")
    assert p.state.extras["dish"]
    assert p.canvas.layer("assay").childItems()  # the dish is drawn
    p.cleanup()


def test_hemocytometer_area_and_grid_calibration(qtbot, monkeypatch, tmp_path):
    import numpy as np

    p = _panel(qtbot, monkeypatch)
    p.set_assay("hemocytometer")
    grid = np.full((400, 400), 160, np.uint8)
    for v in range(10, 400, 40):
        grid[:, v] = 90
        grid[v, :] = 90
    _load(p, tmp_path, "chamber.png", grid)

    p.results_step.action_clicked.emit("calibrate_grid")
    assert p.state.scale == pytest.approx(250 / 40, rel=0.03)  # 0.25 mm per 40 px
    assert p.state.scale_source == "hemocytometer grid"

    p.results_step.action_clicked.emit("box:counting_area")
    assert p.canvas.mode == "BOX"
    p.canvas.box_drawn.emit((10, 10, 330, 330))
    assert p.session.assay_settings["counting_area"] == [10, 10, 330, 330]
    assert p.canvas.mode == "PAN"
    assert p.canvas.layer("assay").childItems()
    p.undo()
    assert "counting_area" not in p.session.assay_settings
    p.cleanup()


def test_transfection_control_button(qtbot, monkeypatch, tmp_path):
    p = _panel(qtbot, monkeypatch)
    p.set_assay("transfection")
    _load(p, tmp_path, "GFP.png", img.isolated_discs())
    assert p.session.assay_settings["reporter_channel"] == "GFP.png"
    _segment(p, qtbot, "otsu", {"target_channel": "GFP.png", "min_area_px": 50})

    p.results_step.action_clicked.emit("set_control")
    assert p.session.assay_settings["cutoff"] > 0
    assert [g.name for g in p.session.gates] == ["Positive"]
    assert "control cells" in p.results_step.lbl_action.text()
    p.cleanup()


def test_scratch_wound_start_and_closure(qtbot, monkeypatch, tmp_path):
    p = _panel(qtbot, monkeypatch)
    p.set_assay("wound")
    assert p.combo_pipeline.currentData() == "texture"
    _load(p, tmp_path, "phase.png", img.scratch_photo())
    params = p.params_form.values()
    assert params["detect"] == "gap"
    _segment(p, qtbot, "texture", {**params, "target_channel": "phase.png"})
    assert len(p.state.cells) == 1
    assert p.state.extras["wound"]["direction"] == "vertical"

    p.results_step.action_clicked.emit("set_start")
    assert p.session.assay_settings["start_area_px"] > 0
    values = {c.lbl_name.text(): c.lbl_value.text() for c in p.results_step._cards}
    assert values["Closure"] == "0 %"
    p.cleanup()


def test_spheroid_volume_card(qtbot, monkeypatch, tmp_path):
    p = _panel(qtbot, monkeypatch)
    p.set_assay("spheroids")
    assert p.combo_pipeline.currentData() == "spheroid"
    _load(p, tmp_path, "well.png", img.spheroid_photo())
    params = p.params_form.values()
    assert params["keep_largest"] == 1 and params["max_area_px"] > 1e6
    _segment(p, qtbot, "spheroid", {**params, "target_channel": "well.png"})
    assert len(p.state.cells) == 1
    values = {c.lbl_name.text(): c.lbl_value.text() for c in p.results_step._cards}
    assert values["Volume"].endswith("px³")
    p.cleanup()
