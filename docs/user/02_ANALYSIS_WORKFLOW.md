# Analysis Workflow

## 1. Pick an algorithm (2. Smart Detect)

| Algorithm | Best for | Method |
|---|---|---|
| **Basic Threshold (Otsu)** | Sparse, isolated cells on a clean background | Global threshold chosen by Otsu's method. Touching cells merge into one object. |
| **Watershed (Clump Splitting)** | Clustered, roughly round cells | Marker-based watershed (scikit-image). Splits touching cells from their shape, or from a nuclei channel when you turn one on. |
| **AI Smart Detect (Cellpose)** | Irregular shapes, faint extensions, crowded fields | Cellpose-SAM deep-learning segmentation. |

The **Settings** below the algorithm change with it. Hover over any setting for an explanation. Values you've set are kept when you switch algorithm (for example the channel and the size filter).

## 2. Settings every algorithm has

- **Target Channel**: the channel that outlines your cells.
- **Exclude cells touching image edges**: cells cut by the image edge give wrong sizes, so they're dropped by default.
- **Min Area / Max Area**: drop debris and merged clumps. Shown in µm² once the image is calibrated, px² before.

## 3. Watershed settings

- **Use a nuclei channel** + **Nuclei Channel**: a nuclear stain (e.g. DAPI) whose nuclei seed one cell each. The most reliable way to separate touching cells.
- **Min. Cell Separation**: without a nuclei channel, two cell centres closer than this are treated as one cell. Start at roughly one cell radius; lower it if clumps stay merged, raise it if single cells are cut in two.

## 4. Cellpose settings

- **Cell Diameter**: typical cell diameter. *Auto* works for most images; set it if your cells are much smaller or larger than about 30 px.
- **Detection Threshold** (Cellpose's cell-probability threshold, −6 to 6): lower it to find more and larger cells, including dim ones; raise it to keep only clear cells.
- **Shape Tolerance** (Cellpose's flow threshold): higher keeps more, less regular shapes; lower keeps only well-formed cells.
- **Use a nuclei channel**: gives Cellpose a second input channel.

Under **Advanced settings**:

- **Contrast: Low / High Percentile**: the intensity range Cellpose stretches to before segmenting (default 1–99). Lower the high percentile if a few very bright cells wash out the rest.
- **Uneven Illumination: Tile Size**: normalise contrast in tiles so cells in dark corners are found too. Try 3–5 cell diameters; *Off* by default.
- **High Quality (slower)**: test-time augmentation; runs the network on flipped copies and averages them.
- **Mask Iterations**: raise it for long, thin cells (e.g. bacteria) that come out fragmented.
- **Largest Object**: Cellpose discards masks bigger than this fraction of the image.
- **GPU Batch Size**: lower it if you run out of memory.

Each Cellpose cell also gets a **confidence**: the model's mean cell probability inside its outline.

## 5. Run and review

Click **✨ Run Segmentation**; the scanning bar shows progress. Detected cells appear as outlines with ID numbers.

- **Right-click** a cell to delete a false positive.
- **✏️ Draw Cells (Manual)** lets you outline a missed cell by hand.
- Select a row in the results table to highlight that cell on the canvas.

### Reviewing Cellpose results

Each Cellpose cell has a **Conf.** value (0–1), the model's mean cell probability inside the outline. In **3. Results**:

- **Colour outlines by AI confidence**: red = uncertain, amber = borderline, green = confident.
- **Show Cellpose cell-probability map**: a cyan overlay of where the model thinks cells are. Useful to see dim cells it nearly found (lower the Detection Threshold to include them).
- **Uncertain below** + **Next uncertain ▸**: steps through cells under the threshold, least confident first, and centres each one so you can delete or keep it.

## 6. Results and export

**3. Results** shows the cell count, average area and an area histogram. **💾 Export to CSV** writes one row per cell (units are µm, or px if the image isn't calibrated):

| Column | Meaning |
|---|---|
| `Cell_ID` | ID shown on the canvas |
| `Area_um2` | Area in µm² |
| `Perimeter_um` | Outline length in µm |
| `Circularity` | 4π·area / perimeter² (1 = perfect circle) |
| `Diameter_um` | Diameter of a circle with the same area |

Two more exports take your cells, as they are after your edits, to other tools:

- **🔷 Export ImageJ / Fiji ROIs**: a `.zip` of outlines named "Cell <ID>". In Fiji: *Analyze › Tools › ROI Manager › More › Open*. QuPath can import it too.
- **🧬 Export for Cellpose (_seg.npy)**: the masks in Cellpose's own format. Save it next to the image and open the image in the Cellpose GUI to keep editing; Cellpose can also train on it.

Every CSV export also writes a methods paragraph and BibTeX file listing the settings you used. When you publish results, cite the methods you used: see [How to Cite](03_HOW_TO_CITE.md).
