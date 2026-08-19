# CytoMetrics Plugin

[![Karcytics Plugin](https://img.shields.io/badge/Karcytics-Plugin-10B981?style=for-the-badge)](https://github.com/KalaimaranB/Karcytics-cytometrics)
[![Version](https://img.shields.io/badge/Version-0.2.2-blue?style=for-the-badge)](https://github.com/KalaimaranB/Karcytics-cytometrics)

**AI-assisted multi-channel cell morphology quantification.**

CytoMetrics is a high-performance Karcytics plugin designed for automated cell segmentation and morphological analysis. It leverages state-of-the-art AI (Cellpose) and traditional computer vision algorithms to provide accurate quantification of cellular features in complex multi-channel light microscopy images.

---

## ✨ Key Features

### 🧠 Advanced AI Segmentation
*   **Cellpose Integration**: Native support for the `cyto3` model for robust segmentation of diverse cell types.
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

### 2. Setting Up the AI Engine
The primary AI model (~1.3GB) is managed separately to optimize storage:
1.  Open the **CytoMetrics** panel.
2.  Navigate to the **Detection Rules** tab.
3.  Click **⚙️ Manage AI** and select **Download Model**.
4.  Once installed, the status will show a green checkmark indicating the AI is ready.

### 3. Loading Images
1.  Go to the **Setup & Channels** tab.
2.  Click **➕ Add Image Channel** to import TIFF, PNG, or JPG files.
3.  Assign logical colors (e.g., Green for GFP, Blue for DAPI) to each channel.

### 4. Calibration
To get measurements in micrometers (µm):
1.  Click **📏 Set Scale / Calibrate**.
2.  Click and drag a line on the image across a known distance (e.g., a scale bar).
3.  Enter the physical distance and units.

---

## 🛠 Analysis Workflow

### Detection Pipeline
Configure your segmentation rules in the **Detection Rules** tab:
- **Main Signal**: Select the channel defining the cell boundary.
- **Algorithm**: Choose between **AI Smart Detect (Cellpose)**, **Watershed**, or **Otsu**.
- **Splitting Sensitivity**: Adjust how aggressively clumps are separated.
- **Filters**: Set minimum and maximum area thresholds to exclude debris or artifacts.

### Reviewing Results
Once segmentation is complete:
- **Canvas Overlay**: View detected cells with magenta outlines and IDs on the interactive canvas.
- **Hand-Tuning**: Right-click any cell on the canvas to delete false positives.
- **Data Export**: Explore the full dataset in the **Results & Analytics** tab and export to CSV for further study.

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
