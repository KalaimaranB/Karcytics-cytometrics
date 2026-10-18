"""Model Center and the system status bar."""

from __future__ import annotations

import dataclasses

import pytest

from karcytics_plugins.cytometrics.analysis import models
from karcytics_plugins.cytometrics.ui.dialogs import model_center
from karcytics_plugins.cytometrics.ui.widgets.status_bar import SystemStatusBar

pytestmark = pytest.mark.ui


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "CELLPOSE_MODEL_DIR", tmp_path)
    model = dataclasses.replace(models.MODELS[0])
    monkeypatch.setattr(model_center, "MODELS", (model,))
    return model


def test_not_installed_model_offers_download(qtbot, fake_model):
    dialog = model_center.ModelCenterDialog()
    qtbot.addWidget(dialog)
    (card,) = dialog.cards

    assert card.lbl_status.text() == "Not installed"
    assert card.btn_download.isEnabled() and not card.btn_delete.isEnabled()


def test_installed_model_can_be_deleted_after_confirming(qtbot, fake_model, monkeypatch):
    fake_model.path.write_bytes(b"x" * 2 * 1024 * 1024)
    other = fake_model.path.parent / "someone_elses_model"
    other.write_bytes(b"keep me")
    dialog = model_center.ModelCenterDialog()
    qtbot.addWidget(dialog)
    (card,) = dialog.cards
    assert card.lbl_status.text().startswith("✅ Installed (2 MB)")

    monkeypatch.setattr(model_center, "ask_yes_no", lambda *a: True)
    card.btn_delete.click()

    assert not fake_model.path.exists()
    assert other.exists()  # only this model's file, never the whole Cellpose cache
    assert card.lbl_status.text() == "Not installed"


def test_licences_citations_and_exclusions_are_shown(qtbot, fake_model):
    dialog = model_center.ModelCenterDialog()
    qtbot.addWidget(dialog)

    text = " ".join(w.text() for w in dialog.findChildren(model_center.QLabel))
    assert "Weights: BSD-3-Clause" in text
    assert "Pachitariu" in text
    assert "InstanSeg" in text and "weights' licence isn't stated" in text


def test_status_bar_reports_cpu_and_memory(qtbot):
    bar = SystemStatusBar()
    qtbot.addWidget(bar)

    bar.refresh()

    assert bar.text().startswith("CPU ") and " GB" in bar.text()
    bar.stop()
