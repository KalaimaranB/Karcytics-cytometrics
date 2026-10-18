# Getting Started

## Install

Install CytoMetrics from the Karcytics Hub's Plugin Store. It runs in its own window and its own Python environment.

## Choose what you're analysing

When CytoMetrics opens it asks **What are you analysing?**:

- **Cells**: whole cells in fluorescence or bright-field images. Starts with Cellpose-SAM and local background subtraction.
- **Nuclei**: nuclei from a nuclear stain. Starts with Cellpose-SAM on the blue (DAPI/Hoechst) channel and adds chromatin texture.

The choice sets the steps (**Load › Calibrate › Detect › Review › Measure › Report**), the starting algorithm and the measurements, and is remembered for the project. Every setting can still be changed. To switch, click the analysis button next to the CytoMetrics title.

## Install the AI model (optional)

The Cellpose-SAM model (about 1.2 GB) is downloaded on demand, not bundled. You only need it for **AI Smart Detect (Cellpose)**; Otsu and Watershed work without it.

1. In the **Detect** step, click **🧠 Models** to open the Model Center.
2. Click **⬇️ Download**. The card turns green when the model is installed.

The Model Center lists each model's size, the licences of its code and its weights, and the papers to cite. **🗑️ Delete** removes only that model's file. It also lists models that were considered but not included, with the reason (CytoMetrics only ships models whose code *and* weights have a compatible licence).

The line under the steps shows CytoMetrics' CPU use, memory and, once a model is loaded, GPU memory.

## Load images

1. In the **Load** step, click **➕ Add Image Channel**, or drop files onto the canvas. TIFF (including OME-TIFF and ImageJ TIFF), PNG, JPG and BMP are supported; Nikon ND2 needs the optional `karcytics-sdk[imaging]` extra.
2. A grayscale file becomes one channel. A multi-channel file is split into one channel per stain, named from the file's metadata where it has names (for example "cells.ome.tif (DAPI)"). An RGB file is split into red, green and blue.
3. Each channel gets a display colour from its name where the stain is recognised (DAPI/Hoechst blue, GFP/FITC green, mCherry/RFP/Cy3 red, Cy5 magenta). Change it in the Color column.
4. Images keep their full bit depth: a 16-bit image is analysed as 16-bit, and only scaled to 8-bit for display. All channels must be the same size.

Only the first time point and z-plane of a stack are used for now.

Untick a channel to hide it. Select a channel to set its **display contrast** (the values shown as black and white, in the image's own units), or click **Auto** to stretch from the 0.1st to the 99.9th percentile. Contrast changes only what you see: detection and measurements always use the full data.

## Calibrate

CytoMetrics reports sizes in micrometres once it knows the image scale.

- If the file stores its pixel size (OME-TIFF, ImageJ TIFF, a TIFF with centimetre resolution, ND2), the scale is set automatically when it loads. The **Calibrate** step shows "(from image metadata)".
- Otherwise click **📏 Set Scale / Calibrate**, drag a line along a scale bar, and enter its length in µm. A scale you set by hand is never overwritten by a later file's metadata.

You can also run without calibrating. Every size field, the results table, the CSV columns and the methods text then use pixels (px, px²), and an amber note on the Detect tab says so. If you calibrate afterwards, already-measured cells are converted to µm.

## Save and reopen your work

**💾** (next to the title) stores the analysis (images, channel names and colours, scale, detected cells, settings) as a workflow in the open project. **📂** lists this project's saved CytoMetrics workflows; the Hub's dashboard can open them too.
