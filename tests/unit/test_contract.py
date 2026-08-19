"""Contract tests for the CytoMetrics plugin manifest and entry point."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock

from karcytics_sdk.plugin.context import PluginContext
from karcytics_sdk.plugin.manifest import PluginManifest
from karcytics_sdk.testing import ContractTestBase


class TestCytoMetricsContract(ContractTestBase):
    PLUGIN_DIR = Path(__file__).resolve().parents[2]

    def test_headless_initialization(self, manifest: PluginManifest) -> None:
        """Overrides ContractTestBase's version.

        Its bare ``MockService()`` has no ``.info()``/``.warning()``/etc, so
        it can't stand in for the "logger" capability our ``initialize()``
        actually calls — ``MagicMock`` can.
        """
        mocked_services = {cap: MagicMock() for cap in manifest.requires}
        context = PluginContext(services=mocked_services, manifest=manifest)

        module_name, func_name = manifest.entry_point.split(":")
        module = importlib.import_module(module_name)
        init_func = getattr(module, func_name)

        plugin_instance = init_func(context)
        assert plugin_instance is not None
