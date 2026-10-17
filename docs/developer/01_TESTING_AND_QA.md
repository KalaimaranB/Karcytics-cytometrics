# Testing & QA

## Test tiers

| Folder | Marker | What | Runs in |
|---|---|---|---|
| `tests/unit/` | `unit` | Headless analysis code: pipelines, image stack, state. No PySide6 widgets. | pre-commit, CI (gates release) |
| `tests/functional/` | `functional` | End to end through `CytoPipelineWorker`, with real and fake segmenters | CI (gates release) |
| `tests/integration/` | `integration`, `slow` | The real `ui_daemon.py` subprocess, via `karcytics_sdk.testing.ui_daemon` | CI (gates release) |
| `tests/ui/` | `ui` | Widgets, offscreen Qt | CI (informational, non-blocking) |

```bash
uv run pytest tests/unit -q                                   # what pre-commit runs
QT_QPA_PLATFORM=offscreen uv run pytest tests -m "not ui" -q  # what gates a release
QT_QPA_PLATFORM=offscreen uv run pytest tests -m ui -q        # UI tests
```

## Fixtures

- `tests/fixtures/images.py` generates synthetic images (isolated discs, a touching pair, a border-cut disc, nuclear seeds) with known ground truth. No downloaded data, so no data licenses to track.
- `tests/fixtures/segmenters.py` has `FakeSegmenter`, so tests never need the Cellpose model.
- Real images may only be added to `tests/data/` with a CC0 or CC-BY license recorded in `tests/data/LICENSES.md`.

## Golden values

`tests/unit/test_classical_pipelines.py` pins today's counts and areas. If a change moves one on purpose, re-derive it and update the constant with the reason. Don't loosen it into a range. Known bugs are recorded as `xfail(strict=True)` tests, so fixing one makes the suite fail until the marker is removed.

## The real Cellpose model

`test_cellpose_sam_finds_isolated_discs` is marked `slow` and skips unless `~/.cellpose/models/cpsam` already exists. CI never downloads the model.

## Pre-commit and CI

Pre-commit runs ruff, mypy, pip-audit, the license-compliance check (no GPL/AGPL/noncommercial-only dependencies under our PolyForm Noncommercial license), `tests/unit`, plugin signing, and (pre-push) security-ledger verification. CI repeats lint/type/audit/license, runs the release-gating tests on macOS and Windows, and runs UI tests without blocking the release.
