# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

`cytometrics` (displayed as "CytoMetrics") — a Karcytics plugin for AI-assisted multi-channel cell morphology quantification (Cellpose `cyto3` segmentation + classical CV, area/perimeter/circularity, GPU/MPS acceleration). Runs as a **separate process** (`process_model = "isolated"`), so it must never import `karcytics.*` from the Hub — only `karcytics_sdk` (sibling `Karcytics-SDK` repo, resolved via `[tool.uv.sources]` as an editable local path). Entry point: `[tool.karcytics.plugin] entry_point = "karcytics_plugins.cytometrics:initialize"`.

The AI model (~1.3GB, Cellpose) is downloaded on demand at runtime, not bundled — don't expect it present in the repo or test fixtures.

See `../Karcytics/ECOSYSTEM.md` for the full map of all 7 Karcytics repos and their relationships.

## Commands

```bash
uv run pytest tests/unit/ --tb=short -q   # what pre-commit runs
uv run pytest tests/ -q                   # full suite incl. tests/ui
uv run ruff check src/ tests/ --fix
uv run ruff format src/ tests/
uv run mypy --explicit-package-bases src/
```

Pre-commit runs ruff, mypy, pip-audit, license compliance, `tests/unit/` (not `tests/ui`), plugin re-signing, and security-ledger verification. Don't pre-run the full suite before committing.

## Structure

`src/karcytics_plugins/cytometrics/`: `analysis/` (segmentation + morphometrics, no Qt) and `ui/` (PySide6 panel, hardware/telemetry dashboard). Same split as the other Karcytics plugins — see [Karcytics-SDK](../Karcytics-SDK)'s `CLAUDE.md` for the `PluginBase`/`AnalysisBase` pattern this follows.
