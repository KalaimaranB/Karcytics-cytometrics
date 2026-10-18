# Architecture Overview

CytoMetrics runs as an **isolated** Karcytics plugin (`process_model = "isolated"`): the Hub starts `ui_daemon.py` under the plugin's own `.venv` interpreter and talks to it over msgpack frames on stdio. The plugin imports only `karcytics_sdk`, never the Hub's `karcytics` package.

## Today

```
src/karcytics_plugins/cytometrics/
  __init__.py            initialize(), get_panel_class(), __version__ (read from pyproject.toml)
  ui_daemon.py           daemon entry point (the SDK runtime pre-imports numpy)
  analysis/              no Qt
    model.py             ImageSet/Channel: files read via the SDK's read_image (native bit depth,
                         metadata pixel size and channel names), RGB composite
    units.py             µm vs px: a scale of 0 means uncalibrated and everything reads px
    state.py             CytoMetricsState (PluginState dataclass, saved in workflows; scale_source)
    pipelines/           otsu.py, watershed.py, cellpose_pipeline.py → list of cell dicts
  citations.py           verified Citation entries registered with the SDK
  report.py              methods paragraph + BibTeX sidecars for exports
  ui/
    main_panel.py        CytoMetricsPanel (PluginBase) + helper widgets and dialogs
    image_canvas.py      MultiChannelCanvas (zoom/pan, calibrate, draw, delete)
    channel_manager.py   channel table, asset import into the project
    workers.py           CytoPipelineWorker (AnalysisBase) + model download/library loading tasks
```

A pipeline returns `[{"points", "area", "perim", "circ"}, ...]`; the panel assigns IDs and stores them in `CytoMetricsState.cells`.

## Where it is going

The [Expansion Plan](plans/EXPANSION_PLAN.md) replaces the cell dicts with a label mask plus a pandas object table, makes segmenters and measurements pluggable through registries, splits `main_panel.py` into views, and moves code that other plugins also need (task helper, image reader, image view, citations registry) into the SDK. Read its §2 before changing `main_panel.py`: it lists the known duplication and design problems so new code doesn't add to them.
