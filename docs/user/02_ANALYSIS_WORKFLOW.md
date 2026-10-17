# Analysis Workflow

## 1. Choose what to segment (2. Smart Detect)

- **Main Signal**: the channel that outlines your cells.
- **Use a 2nd signal to split clumps** + **Splitting Marker**: a nuclear stain (e.g. DAPI) whose nuclei seed one cell each. This is the most reliable way to separate touching cells with Watershed.

## 2. Pick an algorithm

| Algorithm | Best for | Method |
|---|---|---|
| **Basic Threshold (Otsu)** | Sparse, isolated cells on a clean background | Global threshold chosen by Otsu's method. Touching cells merge into one object. Cells cut by the image edge are kept. |
| **Watershed (Clump Splitting)** | Clustered, roughly round cells | Marker-based watershed (OpenCV). Works best with a splitting marker; without one, a tight two-cell clump may stay merged. Cells touching the edge are dropped. |
| **AI Smart Detect (Cellpose)** | Irregular shapes, faint extensions, crowded fields | Cellpose-SAM deep-learning segmentation. **Cell Diameter** (µm, 0 = automatic) and **Splitting Sensitivity** (flow threshold) tune it. |

## 3. Filter by size

**Min Area** and **Max Area** (µm²) drop debris and merged clumps.

## 4. Run and review

Click **✨ Run Segmentation**. Detected cells appear as outlines with ID numbers.

- **Right-click** a cell to delete a false positive.
- **✏️ Draw Cells (Manual)** lets you outline a missed cell by hand.
- Select a row in the results table to highlight that cell on the canvas.

## 5. Results and export

**3. Results** shows the cell count, average area and an area histogram. **💾 Export to CSV** writes one row per cell:

| Column | Meaning |
|---|---|
| `Cell_ID` | ID shown on the canvas |
| `Area_um2` | Area in µm² |
| `Perimeter_um` | Outline length in µm |
| `Circularity` | 4π·area / perimeter² (1 = perfect circle) |
| `Diameter_um` | Diameter of a circle with the same area |

When you publish results, cite the methods you used: see [How to Cite](03_HOW_TO_CITE.md).
