"""CytoMetrics UI Daemon — hosts the module's own window in its own process.

Run by `karcytics_sdk.plugin.PluginUIDaemon` from this plugin's own `.venv`
interpreter (never imported into the Hub's process). Everything
protocol-related (frame transport, the ready handshake, request dispatch,
noticing a native window close) lives in the SDK's
`karcytics_sdk.plugin.run_ui_daemon` and is identical for every isolated
plugin; this file only does what's genuinely plugin-specific: sys.path setup
and building this plugin's `PluginContext` from the SDK's `runtime_services`
singletons.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Run directly as `python ui_daemon.py` by PluginUIDaemon rather than imported
# as part of the `karcytics_plugins` package — nothing else puts this
# plugin's own src/ on sys.path for a freestanding subprocess, so it has to
# do that for itself before it can import itself.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _build_plugin_context() -> Any:
    from karcytics_sdk.plugin.context import PluginContext
    from karcytics_sdk.plugin.manifest import PluginManifest
    from karcytics_sdk.plugin.runtime_services import event_bus, task_scheduler

    manifest = PluginManifest(
        name="cytometrics",
        entry_point="karcytics_plugins.cytometrics:initialize",
        sdk_version="2.0",
        requires=["task_scheduler", "logger", "event_bus"],
    )
    services = {
        "task_scheduler": task_scheduler,
        "logger": __import__("logging").getLogger("plugin.cytometrics"),
        "event_bus": event_bus,
    }
    return PluginContext(services=services, manifest=manifest)


def main() -> None:
    from karcytics_sdk.plugin import run_ui_daemon
    from karcytics_sdk.plugin.ui_daemon_runtime import send_event

    def _build_panel() -> Any:
        from karcytics_plugins.cytometrics import initialize

        context = _build_plugin_context()
        plugin_module = initialize(context)
        panel_class = plugin_module.get_panel_class()
        panel = panel_class()

        if hasattr(panel, "state_changed"):
            panel.state_changed.connect(lambda: send_event("state_changed", {}))
        if hasattr(panel, "status_message"):
            panel.status_message.connect(lambda msg: send_event("status_message", msg))

        return panel

    run_ui_daemon(
        _build_panel,
        window_title="CytoMetrics",
        window_size=(1400, 900),
        plugin_id="cytometrics",
    )


if __name__ == "__main__":
    main()
