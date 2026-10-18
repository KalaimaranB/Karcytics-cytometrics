"""ParamsForm: controls generated from Param declarations, unit conversion, dependencies."""

from __future__ import annotations

import pytest

from karcytics_plugins.cytometrics.analysis.segment.base import Param
from karcytics_plugins.cytometrics.ui.params_form import ParamsForm

pytestmark = pytest.mark.ui

PARAMS = [
    Param("target_channel", "Target", "channel"),
    Param("use_seed", "Use seed", "bool", default=False),
    Param("seed", "Seed", "channel", enabled_by="use_seed"),
    Param("diameter", "Diameter", "float", 30.0, 0, 500, unit="length"),
    Param("min_area_px", "Min Area", "float", 100.0, 0, 1e5, unit="area"),
    Param("niter", "Iterations", "int", 0, 0, 2000, advanced=True, special_value_text="Auto"),
    Param("mode", "Mode", "choice", "fast", choices=(("Fast", "fast"), ("Careful", "careful"))),
]


@pytest.fixture
def form(qtbot):
    f = ParamsForm()
    qtbot.addWidget(f)
    f.set_params(PARAMS)
    f.set_channels(["GFP", "DAPI"])
    return f


def test_defaults_are_shown(form):
    assert form.display_values() == {
        "target_channel": "GFP",
        "use_seed": False,
        "seed": "GFP",
        "diameter": 30.0,
        "min_area_px": 100.0,
        "niter": 0,
        "mode": "fast",
    }


def test_sizes_are_converted_to_pixels_when_calibrated(form):
    form.set_scale(0.5)

    values = form.values()

    assert values["diameter"] == pytest.approx(60.0)  # 30 µm at 0.5 µm/px
    assert values["min_area_px"] == pytest.approx(400.0)  # 100 µm² / 0.25
    assert form.widget("diameter").suffix() == " µm"
    assert form.widget("min_area_px").suffix() == " µm²"


def test_uncalibrated_sizes_are_pixels(form):
    assert form.values()["diameter"] == pytest.approx(30.0)
    assert form.widget("diameter").suffix() == " px"


def test_dependent_control_follows_its_switch(form):
    assert not form.widget("seed").isEnabled()

    form.widget("use_seed").setChecked(True)

    assert form.widget("seed").isEnabled()


def test_advanced_controls_are_hidden_until_asked(form):
    assert not form.widget("niter").isVisibleTo(form)

    form.btn_advanced.setChecked(True)

    assert form.widget("niter").isVisibleTo(form)


def test_channel_choice_survives_a_channel_list_change(form):
    form.set_display_values({"target_channel": "DAPI"})

    form.set_channels(["DAPI", "GFP", "mCherry"])

    assert form.display_values()["target_channel"] == "DAPI"


def test_values_are_remembered_across_rebuilds(form):
    form.set_display_values({"min_area_px": 7.0, "niter": 300})

    form.set_params([PARAMS[0], PARAMS[4]])
    form.set_params(PARAMS)

    assert form.display_values()["min_area_px"] == 7.0
    assert form.display_values()["niter"] == 300


def test_int_values_are_ints_and_changes_are_signalled(form, qtbot):
    with qtbot.waitSignal(form.changed):
        form.widget("niter").setValue(250)

    assert form.values()["niter"] == 250
    assert isinstance(form.values()["niter"], int)
