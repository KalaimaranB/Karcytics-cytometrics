"""CytoMetricsState serialisation — saved workflows depend on this round trip."""

from __future__ import annotations

import pytest

from karcytics_plugins.cytometrics.analysis.state import CytoMetricsState

pytestmark = pytest.mark.unit


def test_state_round_trips_through_a_dict():
    state = CytoMetricsState(
        scale=0.325,
        cells=[
            {"id": 1, "points": [[0, 0], [4, 0], [4, 4]], "area": 2.6, "perim": 4.4, "circ": 0.9}
        ],
        cell_counter=1,
        channels_metadata=[{"name": "GFP", "path": "/tmp/g.tif", "color": "green"}],
        ui_params={"pipeline": "Basic Threshold (Otsu)", "min_area": 10.0},
    )

    restored = CytoMetricsState.from_dict(state.to_dict())

    assert restored == state


def test_state_rejects_undeclared_attributes():
    """PluginState forbids dynamic attributes, so every saved field must be declared."""
    state = CytoMetricsState()

    with pytest.raises(AttributeError):
        state.not_a_field = 1  # type: ignore[attr-defined]
