# CytoMetrics

CytoMetrics is a [Karcytics](https://github.com/KalaimaranB/Karcytics) plugin that finds cells in microscopy images and measures them. It segments cells with Cellpose-SAM or classical image processing (Otsu thresholding, marker-based watershed), then reports area, perimeter, circularity and diameter for each cell, with a histogram, a sortable table and CSV export.

- **New here?** Start with [Getting Started](user/01_GETTING_STARTED.md).
- **Publishing results?** See [How to Cite](user/03_HOW_TO_CITE.md): CytoMetrics stands on published scientific work, and the authors deserve the credit.
- **Contributing?** Read the [Architecture Overview](developer/00_ARCHITECTURE_OVERVIEW.md) and the [Expansion Plan](developer/plans/EXPANSION_PLAN.md), which describes where the plugin is going: per-channel intensity, image cytometry with gating, batch runs, and assay modes such as colony counting and viability.
