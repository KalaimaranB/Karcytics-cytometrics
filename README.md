# CytoMetrics Plugin

[![Karcytics Plugin](https://img.shields.io/badge/Karcytics-Plugin-10B981?style=for-the-badge)](https://github.com/KalaimaranB/Karcytics-cytometrics)
[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FKalaimaranB%2FKarcytics-cytometrics%2Fmain%2Fpyproject.toml&query=%24.project.version&label=Version&color=blue&style=for-the-badge)](https://github.com/KalaimaranB/Karcytics-cytometrics)

**AI-assisted multi-channel cell morphology quantification.**

CytoMetrics is a high-performance Karcytics plugin designed for automated cell segmentation and morphological analysis. It leverages state-of-the-art AI (Cellpose) and traditional computer vision algorithms to provide accurate quantification of cellular features in complex multi-channel light microscopy images.

---

## ✨ Key Features

### 🧠 Advanced AI Segmentation
*   **Cellpose Integration**: Uses the Cellpose-SAM (`cpsam`) generalist model for robust segmentation of diverse cell types.
*   **Multi-Channel Awareness**: Utilize primary signal (cytoplasm) and secondary markers (nuclei) simultaneously for precise clump splitting.
*   **GPU/MPS Acceleration**: Fully optimized for macOS (Metal Performance Shaders) and NVIDIA (CUDA) for lightning-fast analysis.

### 📊 Real-Time Telemetry & Performance
*   **Hardware Monitor**: Live dashboard for System CPU, App CPU, and AI VRAM usage.
*   **Bio-Themed Animations**: Custom ECG-style loading indicators and a premium, dark-mode dashboard.
*   **Asynchronous Processing**: Heavy AI libraries and pipeline initializations run in background threads to ensure zero UI freezing.

### 📏 Quantitative Analysis
*   **Interactive Calibration**: Built-in tools for setting physical scales (µm/pixel).
*   **Morphometric Data**: Automatic calculation of **Area**, **Perimeter**, and **Circularity**.
*   **Visual Results**: Live histograms and sortable tables for immediate data exploration.

---

## 🚀 Getting Started

### 1. Installation
The plugin is installed via the Karcytics Hub's plugin manager.

### 2. Choose what you're analysing
CytoMetrics asks **What are you analysing?** when it opens: **Cells**, **Nuclei & cell cycle**, **Transfection / % positive**, **Hemocytometer & viability**, **Colony counter**, **Bacteria**, **Confluence**, **Scratch wound** or **Spheroids & organoids**. The choice sets the steps, the starting algorithm, the measurements and the headline results (CFU/mL, % viable, % positive, wound width, spheroid volume, …), and is remembered for the project. See [Assays](docs/user/02_ANALYSIS_WORKFLOW.md#9-assays). Change it any time from the button next to the title.

### 3. Work through the steps
**Load › Calibrate › Detect › Review › Measure › Gate › Report › Batch**. Use Back/Next or click any step.

- **Load**: add one image per stain or a multi-channel file (TIFF, OME-TIFF, ND2, PNG, JPG), or drop files on the canvas. Full bit depth is kept.
- **Calibrate**: files that store their pixel size are calibrated automatically; otherwise drag along a scale bar. You can also work in pixels.
- **Detect**: pick **AI Smart Detect (Cellpose-SAM)**, **Watershed** or **Otsu**; the settings below change with it (hover for help). The Cellpose-SAM model (~1.2 GB) is installed from the **🧠 Models** center, which also shows each model's licences and citations.
- **Review**: colour outlines by AI confidence, step through uncertain cells, delete or draw cells.
- **Measure**: size, shape and per-channel intensity for every cell, with optional background subtraction, nucleus/cytoplasm ratios and texture.
- **Gate**: sort cells into populations with threshold, range, rectangle, polygon and combined gates on the histogram and scatter plot, as in flow cytometry; cells on the image take their population's colour.
- **Batch**: repeat the analysis on a folder or plate; conditions from file names or a plate layout; stop and resume; per-image, per-condition (replicate-aware) and plate heat-map results; one-click export.
- **Report**: export CSV, FCS 3.1 (for flow software), population statistics (with a methods paragraph and BibTeX), ImageJ/Fiji ROIs or Cellpose `_seg.npy`; copy the methods text.

See the [user guide](docs/user/01_GETTING_STARTED.md) for details.

---

## 💻 System Support
CytoMetrics is designed for modern hardware:
- **macOS**: Fully supports M1/M2/M3 chips via Apple Silicon (MPS).
- **Windows/Linux**: Optimized for NVIDIA GPUs via CUDA.
- **CPU Fallback**: Automatic fallback for systems without compatible GPUs.

---

## 👤 Author
**Kalaimaran Balasothy**  
*Lead Developer, Karcytics Ecosystem*

---

> [!TIP]
> Use the **Hardware Monitor** to verify that your GPU/MPS is being utilized correctly during long batch runs.
