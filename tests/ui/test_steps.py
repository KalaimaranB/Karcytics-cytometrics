"""Assay picker, step navigation and the Report step."""

from __future__ import annotations

import pytest
from fixtures import images as img
from PIL import Image

from karcytics_plugins.cytometrics.analysis.session import LastRun
from karcytics_plugins.cytometrics.ui import main_panel

pytestmark = pytest.mark.ui


class FakeConfig:
    store: dict = {}

    def __init__(self, _plugin_id):
        pass

    def get(self, key, default=None):
        return self.store.get(key, default)

    def set(self, key, value):
        self.store[key] = value

    def save(self):
        pass


@pytest.fixture
def panel(qtbot, monkeypatch):
    FakeConfig.store = {}
    monkeypatch.setattr(main_panel, "PluginConfig", FakeConfig)
    p = main_panel.CytoMetricsPanel()
    qtbot.addWidget(p)
    qtbot.waitUntil(lambda: bool(p.pipelines), timeout=10_000)
    yield p
    p.cleanup()


def _png(tmp_path, name):
    path = tmp_path / name
    Image.fromarray(img.isolated_discs()).save(path)
    return str(path)


def test_choosing_an_assay_shows_its_steps(panel):
    panel.mode_picker.cards["cells"].clicked.emit("cells")

    assert panel.left_stack.currentWidget() is panel.stepper
    assert panel.stepper.step_ids() == [
        "load",
        "calibrate",
        "detect",
        "review",
        "measure",
        "report",
    ]
    assert panel.btn_assay.text() == "🔬 Cells ▾"
    assert panel.combo_pipeline.currentData() == "cellpose"
    assert panel.measure_options.background == "local"


def test_back_next_and_free_navigation(panel):
    panel.set_assay("cells")
    stepper = panel.stepper

    assert not stepper.btn_back.isEnabled()
    stepper.btn_next.click()
    assert stepper.current_id == "calibrate"
    stepper.go_to("report")
    assert not stepper.btn_next.isEnabled()
    stepper.btn_back.click()
    assert stepper.current_id == "measure"


def test_nuclei_assay_segments_the_nuclear_stain_and_measures_texture(panel, tmp_path):
    panel.set_assay("nuclei")
    panel.channel_manager.ingest_path(_png(tmp_path, "GFP.png"))
    panel.channel_manager.ingest_path(_png(tmp_path, "DAPI.png"))

    assert panel.params_form.display_values()["target_channel"] == "DAPI.png"
    assert panel.measure_options.texture


def test_change_analysis_returns_to_the_picker(panel):
    panel.set_assay("cells")

    panel.btn_assay.click()

    assert panel.left_stack.currentWidget() is panel.mode_picker


def test_assay_is_remembered_per_project(panel, monkeypatch, tmp_path):
    monkeypatch.setattr(panel, "_project_key", lambda: str(tmp_path))

    panel.mode_picker.assay_chosen.emit("nuclei")

    assert FakeConfig.store["assay_by_project"] == {str(tmp_path): "nuclei"}
    assert panel._remembered_assay() == "nuclei"


def test_restoring_a_session_keeps_its_own_settings(panel, qtbot):
    panel.set_assay("nuclei")
    panel.params_form.set_display_values({"min_area_px": 77})
    saved = panel.export_state()

    other = main_panel.CytoMetricsPanel()
    qtbot.addWidget(other)
    qtbot.waitUntil(lambda: bool(other.pipelines), timeout=10_000)
    other.load_state(saved)

    assert other.assay.id == "nuclei"
    assert other.params_form.display_values()["min_area_px"] == 77
    other.cleanup()


def test_report_step_shows_methods_and_references(panel, tmp_path):
    panel.set_assay("cells")
    panel.channel_manager.ingest_path(_png(tmp_path, "cells.png"))
    panel.session.last_run = LastRun(panel.pipelines["otsu"], {"target_channel": "cells.png"})

    panel.stepper.go_to("report")

    text = panel.report_step.methods_text.toPlainText()
    assert "Otsu" in text and "Karcytics CytoMetrics" in text
    panel._copy_methods()
    assert main_panel.QApplication.clipboard().text() == text


def test_report_step_before_any_run(panel):
    panel.set_assay("cells")

    panel.stepper.go_to("report")

    assert "Run a segmentation" in panel.report_step.methods_text.toPlainText()


def test_run_button_says_which_phase_is_running(panel):
    panel._on_run_progress(40)
    assert panel.btn_run_pipeline.text() == "⏳ Detecting cells… 40%"
    panel._on_run_progress(92)
    assert panel.btn_run_pipeline.text() == "⏳ Measuring cells… 92%"


def test_preview_button_is_busy_until_the_preview_returns(panel):
    from karcytics_plugins.cytometrics.ui.steps.detect import PREVIEW_TEXT

    panel._on_preview()  # no image: nothing runs, so the button isn't left busy
    assert panel.detect_step.btn_preview.text() == PREVIEW_TEXT

    panel.detect_step.btn_preview.setText("⏳ Previewing…")
    panel.detect_step.btn_preview.setEnabled(False)
    panel._on_preview_done({"cells": [], "box": (0, 0, 1, 1)})
    assert panel.detect_step.btn_preview.isEnabled()
    assert panel.detect_step.btn_preview.text() == PREVIEW_TEXT
    assert "Nothing is kept" in panel.detect_step.lbl_ai_status.text()
