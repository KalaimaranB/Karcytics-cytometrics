"""Headless smoke test: the CytoMetrics panel constructs without a live Hub."""

from __future__ import annotations

import pytest


@pytest.mark.ui
def test_panel_constructs(qtbot):
    from karcytics_plugins.cytometrics.ui.main_panel import CytoMetricsPanel

    panel = CytoMetricsPanel()
    qtbot.addWidget(panel)

    assert panel.plugin_id == "cytometrics"
    assert panel.tabs.count() == 3
