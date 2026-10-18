# Getting Started

## Install

Install CytoMetrics from the Karcytics Hub's Plugin Store. It runs in its own window and its own Python environment.

## Install the AI model (optional)

The Cellpose-SAM model (about 1.3 GB) is downloaded on demand, not bundled. You only need it for the **AI Smart Detect (Cellpose)** algorithm; Otsu and Watershed work without it.

1. Open the **2. Smart Detect** tab.
2. Choose **AI Smart Detect (Cellpose)** as the algorithm and click **⚙️ Manage AI**.
3. Click **⬇️ Download Model**. The status turns green when the model is ready.

## Load images

1. In **1. Setup**, click **➕ Add Image Channel**, or drop files onto the canvas. TIFF (including OME-TIFF and ImageJ TIFF), PNG, JPG and BMP are supported; Nikon ND2 needs the optional `karcytics-sdk[imaging]` extra.
2. A grayscale file becomes one channel. A multi-channel file is split into one channel per stain, named from the file's metadata where it has names (for example "cells.ome.tif (DAPI)"). An RGB file is split into red, green and blue.
3. Each channel gets a display colour from its name where the stain is recognised (DAPI/Hoechst blue, GFP/FITC green, mCherry/RFP/Cy3 red, Cy5 magenta). Change it in the Color column.
4. Images keep their full bit depth: a 16-bit image is analysed as 16-bit, and only scaled to 8-bit for display. All channels must be the same size.

Only the first time point and z-plane of a stack are used for now.

## Calibrate

CytoMetrics reports sizes in micrometres once it knows the image scale.

- If the file stores its pixel size (OME-TIFF, ImageJ TIFF, a TIFF with centimetre resolution, ND2), the scale is set automatically when it loads. The Setup tab shows "(from image metadata)".
- Otherwise click **📏 Set Scale / Calibrate**, drag a line along a scale bar, and enter its length in µm. A scale you set by hand is never overwritten by a later file's metadata.

You can also run without calibrating. Every size field, the results table, the CSV columns and the methods text then use pixels (px, px²), and an amber note on the Detect tab says so. If you calibrate afterwards, already-measured cells are converted to µm.

## Save and reopen your work

**💾 Save Session** stores the analysis (images, channel names and colours, scale, detected cells, settings) as a workflow in the open project. **📂 Load Session** lists this project's saved CytoMetrics workflows; the Hub's dashboard can open them too.
