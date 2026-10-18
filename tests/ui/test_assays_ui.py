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
