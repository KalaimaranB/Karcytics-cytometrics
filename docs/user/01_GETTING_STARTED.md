# Getting Started

## Install

Install CytoMetrics from the Karcytics Hub's Plugin Store. It runs in its own window and its own Python environment.

## Install the AI model (optional)

The Cellpose-SAM model (about 1.3 GB) is downloaded on demand, not bundled. You only need it for the **AI Smart Detect (Cellpose)** algorithm; Otsu and Watershed work without it.

1. Open the **2. Smart Detect** tab.
2. Choose **AI Smart Detect (Cellpose)** as the algorithm and click **⚙️ Manage AI**.
3. Click **⬇️ Download Model**. The status turns green when the model is ready.

## Load images

1. In **1. Setup**, click **➕ Add Image Channel**, or drop files onto the canvas. TIFF, PNG and JPG are supported.
2. An RGB file is split into blue, green and red channels. A grayscale file becomes one channel.
3. Pick a display colour for each channel (for example green for GFP, blue for DAPI).

!!! warning "16-bit images"
    16-bit images are currently reduced to 8-bit when loaded. Segmentation is unaffected for most images, but intensity resolution is lost. Keeping full bit depth is planned (Expansion Plan, Phase 2).

## Calibrate

Measurements are reported in micrometres, so CytoMetrics needs the image scale.

- If a TIFF stores its resolution in centimetre units, the scale is set automatically when it loads.
- Otherwise click **📏 Set Scale / Calibrate**, drag a line along a scale bar, and enter its length in µm.

Segmentation is disabled until the scale is set.
