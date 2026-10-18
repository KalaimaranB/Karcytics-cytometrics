# Analysis Workflow

## 1. Pick an algorithm (Detect step)

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

Click **✨ Run Segmentation**; the button shows what it is doing (detecting, then measuring) and how far along it is. Cellpose-SAM takes tens of seconds on a large image (longer with High Quality on), and the first run also loads the model. To try settings first, zoom in on a typical area and click **👁 Preview on screen area**: only the part on screen is detected, so it is much quicker; the result is drawn dashed and nothing is kept. Detected cells appear as outlines with ID numbers.

### Editing (Review step)

Pick a tool, then work on the canvas. Every edit can be undone with **↶** (Ctrl+Z) and redone with **↷** (Ctrl+Shift+Z).

| Tool | What it does |
|---|---|
| **✋ Select** | Click a cell to select it, Shift+click to add more; drag to move around. Then **⧉ Merge selected** (touching cells only) or **🗑 Delete selected**. |
| **✏️ Draw** | Outline a missed cell by dragging around it. |
| **✂️ Split** | Draw a line right across a cell to cut it in two. |
| **🖌 Brush + / 🧽 Brush −** | Paint pixels onto, or erase them from, the selected cell (set the **Brush size**). |
| **⬚ Re-detect** | Drag a box: the current algorithm and settings run again inside it and replace the cells centred there. |

Cells you've changed are marked `edited` in the `Source` column; they lose their AI confidence, since the outline is no longer the model's.

- **Right-click** a cell to delete it straight away.
- **✏️ Draw Cells (Manual)** lets you outline a missed cell by hand.
- Select a row in the results table to highlight that cell on the canvas.

### Reviewing Cellpose results

Each Cellpose cell has a **Conf.** value (0–1), the model's mean cell probability inside the outline. In the **Review** step:

- **Colour outlines by**: *AI confidence* (red = uncertain, amber = borderline, green = confident) or any measurement (viridis: dark purple = lowest, yellow = highest; the range is shown underneath). **Outlines** and **Cell ID numbers** can be hidden.
- **Show Cellpose cell-probability map**: a cyan overlay of where the model thinks cells are. Useful to see dim cells it nearly found (lower the Detection Threshold to include them).
- **Uncertain below** + **Next uncertain ▸**: steps through cells under the threshold, least confident first, and centres each one so you can delete or keep it.

## Results panel

Under the steps, the results panel has four tabs:

- **Table**: every cell and measurement; sort by any column; select rows to select cells.
- **Histogram**: the distribution of any measurement; tick **log** for a log axis.
- **Scatter**: any two measurements against each other, each axis linear or **log**. Click a point to select its cell on the canvas (Shift+click for more); cells you select on the canvas are highlighted in the plot. Once you have gates, points take their population's colour (grey: in no gate).
- **Cell**: the selected cell close up: a thumbnail, every measurement, its source and AI confidence, and **🚩 Flag** to mark it for later (flags are saved and exported in the `Flagged` column, and flagged cells have a dashed outline).

## 6. Gate populations (Gate step)

Gating sorts cells into populations by their measurements, the way flow cytometry does, for questions such as "what % of cells are GFP-positive?" or "how many large, bright cells?".

1. In the Gate step, pick a tool:
   - **📏 Threshold**: click on the histogram where the cut-off goes. Cells above it are in the gate (switch to *below* under **Selected gate**).
   - **↔ Range**: drag across the histogram over the values to keep.
   - **▭ Rectangle**: drag a box on the scatter plot.
   - **⬠ Polygon**: click around a cluster on the scatter plot; double-click (or click the first corner) to close it. **Esc** or right-click cancels.
2. The **Populations** tree shows each population's cell count, % of its parent and % of all cells.
3. To gate **inside** a population (e.g. bright cells among the large ones), select it in the tree, then pick a tool: the plots switch to that population's cells and the new gate goes inside it.
4. **∩ Combine…** makes a population from two others: *both* (AND), *either* (OR), or *not* (every cell outside one).
5. Select a gate to see it highlighted on the plots, type an exact cut-off, **✏️ Rename**, **🎨 Colour** or **🗑 Delete** it (deleting also removes the gates that depend on it).

With **Colour cells on the image by population** on, each cell's outline takes the colour of the deepest population it belongs to; cells in no gate are grey. Gates are measured on whatever axis scale they were drawn on (log or linear), are saved with the analysis, follow undo/redo, and are described in the methods text.

## 7. Results and export

Every cell is measured from its mask, the same way whether it was detected, drawn by hand, edited or reloaded. The results table shows every measurement; hover over a column header for its definition. In the **Measure** step:

- **Intensity background**: *None* reports raw intensities. *Global* subtracts the median of all pixels outside the cells. *Local ring* subtracts the median of a 5-pixel ring 2 pixels outside each cell, skipping neighbouring cells. Either adds Background and Corrected… columns.
- **Nucleus / cytoplasm from**: pick a nuclear stain (e.g. DAPI). Each cell is split into nucleus (above the stain's Otsu threshold) and cytoplasm; you get `Nucleus_Area`, `Cytoplasm_Area`, `Nucleus_NCAreaRatio` and, per channel, `Intensity_NucleusMean_…`, `Intensity_CytoplasmMean_…` and `Intensity_NCRatio_…` (above 1 = enriched in the nucleus, e.g. for translocation assays).
- **Texture (Haralick features)**: six co-occurrence features per channel (`Texture_AngularSecondMoment/Contrast/Correlation/InverseDifferenceMoment/Entropy/Variance_<channel>_3`), at a 3-pixel distance, averaged over four directions. Slower on large images.
- **Distribution**: pick any measurement for the histogram and its mean.

**💾 Export to CSV** writes one row per cell. Column names follow [CellProfiler](https://cellprofiler-manual.s3.amazonaws.com/CellProfiler-4.2.6/modules/measurement.html), with the unit in brackets (µm when calibrated, px when not; intensities in the image's native units, "a.u."):

| Column | Meaning |
|---|---|
| `ObjectNumber` | ID shown on the canvas |
| `Source` | `auto` (detected) or `manual` (drawn) |
| `Confidence` | Cellpose's mean cell probability (empty for other algorithms) |
| `AreaShape_Area [um2]` | Pixel count × pixel area |
| `AreaShape_Perimeter [um]` | Outline length |
| `AreaShape_FormFactor` | Circularity, 4π·area / perimeter² |
| `AreaShape_Solidity` | Area / convex-hull area (1 = convex) |
| `AreaShape_Eccentricity` | 0 = circle, towards 1 = elongated |
| `AreaShape_MajorAxisLength`, `MinorAxisLength` | Axes of the ellipse with the same moments |
| `AreaShape_MaxFeretDiameter`, `MinFeretDiameter` | Longest and narrowest caliper width |
| `AreaShape_EquivalentDiameter` | Diameter of a circle with the same area |
| `Location_Center_X`, `_Y [px]` | Centroid, in image pixels |
| `Intensity_Mean/Median/Min/Max/Std/IntegratedIntensity_<channel> [a.u.]` | Pixel statistics inside the cell, per channel |
| `Intensity_Background_<channel>`, `Intensity_Corrected…_<channel>` | With a background choice: the background level and the corrected mean and integrated intensity |

With gates, the CSV has one `Gate_<population>` column per gate (True/False).

- **📈 Export FCS 3.1**: one event per cell and one parameter per measurement (plus a 0/1 `Gate_<population>` parameter per gate), as 32-bit floats. Open it in the Karcytics flow plugin, FlowJo or FCS Express to gate image data with flow tools. Parameter names (`$PnN`) are the CSV column names; the labels (`$PnS`) carry the units. Format: Spidlen et al., *Cytometry A* 77A:97–100 (2010).
- **📊 Export population statistics**: count, % of parent and % of all cells for every population.

Two more exports take your cells, as they are after your edits, to other tools:

- **🔷 Export ImageJ / Fiji ROIs**: a `.zip` of outlines named "Cell <ID>". In Fiji: *Analyze › Tools › ROI Manager › More › Open*. QuPath can import it too.
- **🧬 Export for Cellpose (_seg.npy)**: the masks in Cellpose's own format. Save it next to the image and open the image in the Cellpose GUI to keep editing; Cellpose can also train on it.

Every CSV and FCS export also writes a methods paragraph and BibTeX file listing the settings you used. When you publish results, cite the methods you used: see [How to Cite](03_HOW_TO_CITE.md).

## 8. Batch: many images, conditions and plates (Batch step)

Set the analysis up on one typical image first (Detect, Measure and, if you like, Gate). The Batch step repeats exactly that on every image: same algorithm and settings, same measurements, same gates, same calibration.

1. **Add images**: **📁 Add folder…** (every image directly in the folder) or **➕ Add files…**. If your typical image was loaded as one file per stain (e.g. `A01_DAPI.tif` + `A01_GFP.tif`), files are grouped the same way: what differs between the stain files' names ("DAPI", "GFP") tells them apart.
2. **Say what each image is**. Wells such as `A01` or `B3` in file names are found automatically. To read more from names, type how they are built in **File names**, using `{well}`, `{condition}`, `{replicate}` or any `{field}`: e.g. `{well}_{condition}_r{replicate}` reads `B03_LPS_r2.tif` as well B03, condition LPS, replicate 2. A **🧾 Plate layout** CSV gives each well its condition, either as a list (a `Well` column plus `Condition`, `Replicate`, `Dose`, …) or as a plate grid (row letters down, column numbers across). You can also type into the list, or select rows and **🏷 Set condition…**.
3. **▶ Run batch**. Each image's results are saved as soon as it's done (in the project's `CytoMetrics batches` folder), so **⏹ Stop** at any time and run again to continue. Images that can't be read are marked ⚠️ (hover for why) and the rest carry on. If you change the analysis after some images are done, you're asked whether to redo them.
4. **Check any image**: **🔍 Open in editor** (or double-click a finished image) loads it with its cells into the steps above. Fix cells in Review, then **💾 Save changes to batch**.

Results are in the **Batch** tab under the steps:

- **Images**: one row per image: its metadata, cell count, % in each population and the mean of every measurement.
- **Conditions**: any of those per condition (or any other field): a bar for the mean with **SEM** or **SD** error bars, and a dot per replicate. With a `Replicate` field, images in the same replicate are averaged first and n counts replicates (biological replicates are the unit); otherwise n counts images.
- **Plate**: a heat-map of any value by well; hover for the number, click a well to find its image.

**💾 Export batch results…** writes a folder with `cells.csv` (every cell, with its image's metadata), `images.csv`, `conditions.csv` (per-condition statistics for every value), `fcs/` (one FCS 3.1 file per image, with the metadata as keywords, for flow software) and the methods text and BibTeX.

