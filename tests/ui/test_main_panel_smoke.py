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


@pytest.mark.ui
def test_panel_live_updates_on_theme_change(qtbot):
    """A theme switch should recolor the panel in place, with no rebuild.

    Regression test for PluginBase's theme_manager fallback being a no-op
    mock in an isolated plugin .venv (this test's own environment — see
    test_panel_constructs' docstring), which silently dropped every
    theme_changed connection and froze the whole panel at its startup theme.
    """
    from karcytics_sdk.plugin.theme_fallback import DynamicColors, theme_manager

    from karcytics_plugins.cytometrics.ui.main_panel import CytoMetricsPanel

    panel = CytoMetricsPanel()
    qtbot.addWidget(panel)

    channel_table = panel.channel_manager.table
    channel_table.blockSignals(True)
    channel_table.insertRow(0)
    from PyQt6.QtWidgets import QComboBox

    combo = QComboBox()
    combo.addItems(["gray", "magenta"])
    channel_table.setCellWidget(0, 1, combo)
    channel_table.blockSignals(False)

    assert DynamicColors.BG_DARKEST in panel.table.styleSheet()
    assert DynamicColors.BG_DARKEST in channel_table.styleSheet()
    assert panel._section_headers, "expected create_header() to have registered labels"
    dark_border = DynamicColors.BORDER

    theme_manager.set_theme("light")
    try:
        assert DynamicColors.BG_DARKEST != dark_border or DynamicColors.BORDER != dark_border
        assert DynamicColors.BG_DARKEST in panel.table.styleSheet()
        assert DynamicColors.BORDER in panel.tabs.styleSheet()
        for lbl in panel._section_headers:
            assert DynamicColors.BORDER in lbl.styleSheet()
        assert DynamicColors.BG_DARKEST in channel_table.styleSheet()
        assert DynamicColors.BG_DARK in combo.styleSheet()
    finally:
        theme_manager.set_theme("dark")
