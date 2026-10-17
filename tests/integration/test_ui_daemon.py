"""Real-subprocess integration test for `ui_daemon.py`.

Spawns the actual daemon under this plugin's own interpreter, the way the
Hub's `PluginUIDaemon` does, and speaks the real msgpack-over-stdio protocol
via the SDK's shared harness. Nothing is mocked: this is what proves the
plugin works standalone as `process_model = "isolated"`.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from karcytics_sdk.testing.ui_daemon import spawn_ui_daemon

REPO_ROOT = Path(__file__).resolve().parents[2]
DAEMON_SCRIPT = REPO_ROOT / "src" / "karcytics_plugins" / "cytometrics" / "ui_daemon.py"
HUB_READY_GATE_SECONDS = 45.0  # the Hub gives up on a daemon that isn't ready by then
READY_BUDGET_SECONDS = 40.0

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.timeout(90)]


@pytest.fixture
def daemon():
    with spawn_ui_daemon(DAEMON_SCRIPT, cwd=REPO_ROOT) as running:
        yield running


def test_reaches_ready_with_a_window_geometry_inside_the_hub_gate(daemon):
    start = time.monotonic()
    frame = daemon.io.wait_for_event(
        "ready", "Daemon never sent 'ready'.", timeout=HUB_READY_GATE_SECONDS
    )
    elapsed = time.monotonic() - start

    assert len(frame["payload"]["geometry"]) == 4
    assert elapsed < READY_BUDGET_SECONDS, (
        f"'ready' took {elapsed:.1f}s, close to the Hub's 45s gate"
    )


def test_focus_request_is_answered_after_ready(daemon):
    daemon.io.wait_for_event("ready", "Daemon never sent 'ready'.", timeout=HUB_READY_GATE_SECONDS)

    daemon.request(7, "focus")

    assert daemon.io.wait_for_response(7, "No reply to 'focus'.") == {
        "kind": "response",
        "request_id": 7,
        "payload": {"status": "ok"},
    }


def test_exit_request_shuts_the_process_down_cleanly(daemon):
    daemon.io.wait_for_event("ready", "Daemon never sent 'ready'.", timeout=HUB_READY_GATE_SECONDS)

    daemon.request(1, "exit")

    assert daemon.io.wait_for_response(1, "No reply to 'exit'.")["payload"] == {"status": "ok"}
    assert daemon.process.wait(timeout=10) == 0
