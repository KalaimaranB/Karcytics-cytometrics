# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`cytometrics` (displayed as "CytoMetrics") — a Karcytics plugin for AI-assisted multi-channel cell morphology quantification (Cellpose-SAM `cpsam` segmentation + classical CV, area/perimeter/circularity, GPU/MPS acceleration). Runs as a **separate process** (`process_model = "isolated"`), so it must never import `karcytics.*` from the Hub — only `karcytics_sdk` (sibling `Karcytics-SDK` repo, resolved via `[tool.uv.sources]` as an editable local path). Entry point: `[tool.karcytics.plugin] entry_point = "karcytics_plugins.cytometrics:initialize"`.

The AI model (~1.3GB, Cellpose) is downloaded on demand at runtime, not bundled — don't expect it present in the repo or test fixtures.

See `../Karcytics/ECOSYSTEM.md` for the full map of all 7 Karcytics repos and their relationships.

## Commands

```bash
uv run pytest tests/unit/ --tb=short -q                       # what pre-commit runs
QT_QPA_PLATFORM=offscreen uv run pytest tests/ -m "not ui" -q  # what gates a release (unit/functional/integration)
QT_QPA_PLATFORM=offscreen uv run pytest tests/ -m ui -q        # UI tests (informational in CI)
mkdocs serve                                                   # docs preview (needs mkdocs-material)
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run mypy --explicit-package-bases src/
```

Pre-commit runs ruff, mypy, pip-audit, license compliance, `tests/unit/` (not `tests/ui`), plugin re-signing, and security-ledger verification. Don't pre-run the full suite before committing.

## Structure

`src/karcytics_plugins/cytometrics/`: `analysis/` (segmentation + morphometrics, no Qt) and `ui/` (PySide6 panel, hardware/telemetry dashboard). Same split as the other Karcytics plugins — see [Karcytics-SDK](../Karcytics-SDK)'s `CLAUDE.md` for the `PluginBase`/`AnalysisBase` pattern this follows.

## Ongoing work

`docs/developer/plans/EXPANSION_PLAN.md` is the multi-phase plan (assays, image cytometry, SDK foundations, UI overhaul). Resume at its first unchecked checkpoint, and read its §2 (known DRY/KISS/SOLID problems) and §3 (ground rules: SDK only for code ≥2 plugins share, models only with confirmed code *and* weights licenses, cite published work in-app) before changing analysis or UI code. Commit directly on `main`.

Tests: `tests/fixtures/images.py` generates synthetic images with exact ground truth and `tests/fixtures/segmenters.py` has a `FakeSegmenter`. Golden values in `tests/unit/test_classical_pipelines.py` are exact on purpose; re-derive rather than loosen them. See `docs/developer/01_TESTING_AND_QA.md`.
