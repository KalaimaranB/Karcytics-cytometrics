# Sample images for trying CytoMetrics

Synthetic images, one folder per assay, for learning the workflow and checking that each assay gives sensible answers. They are drawn by [`generate_samples.py`](generate_samples.py) from a fixed seed, so they're identical on every machine. They're not committed (13 MB); make them with:

```bash
uv run python tests/data/generate_samples.py
```

This writes `tests/data/samples/` and `samples/truth.json`, which holds what each image really contains. The numbers below come from it.

They imitate real images closely enough to exercise the algorithms (blur, noise, uneven light, touching cells, debris), but they are not real microscopy. Use them to learn and to spot regressions, not to validate a method.

| Folder | Pick | Load | What's in it (truth) | Try |
|---|---|---|---|---|
| `cells/` | Cells | `cells_DAPI.tif` + `cells_GFP.tif` | 80 cells, ~15% touching; 0.65 µm/px (stored in the file) | Detect on GFP, with DAPI as the nuclei channel to split touching cells |
| `nuclei/` | Nuclei & cell cycle | `nuclei_DAPI.tif` | 240 nuclei: 2.9% Sub-G1, 57.5% G1, 18.8% S, 20.8% G2/M | Results, then ✨ to add a gate per phase |
| `transfection/` | Transfection / % positive | `control_*` first, then `sample_*` | control 0% GFP+; sample 40% (36 of 90) | Detect on DAPI and add as the negative control; 🔄 Next image, load the sample, detect, add as a sample |
| `hemocytometer/` | Hemocytometer & viability | `trypan_blue_1um_per_px.png` | 140 cells, 33 dead (76.4% viable); 0.25 mm squares are 250 px | 📐 calibrate from the grid (0.25 mm); ⬚ draw the area over one big square and add it; repeat for the other squares. All 16 squares (the whole image) give about 2.8 × 10⁶ cells/mL at dilution 2 |
| `colonies/` | Colony counter | `plate_10e-6_100uL.jpg` | 77 colonies = 7.7 × 10⁸ CFU/mL at dilution 1 000 000 and 100 µL | Set the dilution (1000000) and volume in Results, add the plate; the dense plate (any dilution) is left out as over 300 |
| | | `plate_dense.jpg` | 416 colonies: over 300, so CytoMetrics warns | |
| `bacteria/` | Bacteria | `rods_phase_contrast.tif` | 110 rods, 16 pinched in the middle (dividing); median length 3.9 µm; 0.1 µm/px | Cellpose-SAM splits about half of the pinched cells into two daughters, so expect ~119 cells and fewer flagged as dividing |
| `confluence/` | Confluence | `confluence_30pct.png`, `confluence_75pct.png` | 30% and 75% covered | Change *Texture window* and watch the coverage |
| `wound/` | Scratch wound | `A01_00h.png`, `A01_12h.png`, `A01_24h.png` | open area 287 495 → 168 658 → 50 040 px²: 41.3% then 82.6% closure; cells bridge the gap at 24 h | Detect 00h and add it as the start; load 12h and 24h, add each as a later image (named 12 h, 24 h) and read Closure |
| `spheroids/` | Spheroids & organoids | `spheroid_day1/4/7.png` | diameters 120, 210, 300 px (at 2 µm/px: 240, 420, 600 µm), plus debris | Calibrate 2 µm/px by typing it in; the debris should be ignored |
| `batch_plate/` | Transfection (or Cells) | one well's `_DAPI` + `_GFP`, then Batch › 📁 Add folder | wells A01–A03 Control (2–11% GFP+), B01–B03 Treated (38–49%) | 🧾 Plate layout: `layout.csv` gives each well its condition and replicate; compare in Conditions and Plate |

Fluorescence TIFFs store their pixel size, so CytoMetrics calibrates them when they're loaded. The PNG and JPEG photos aren't calibrated: use the grid (hemocytometer), a known size, or work in pixels.
