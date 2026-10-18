"""Batch runs: one analysis, applied the same way to many images. No Qt.

A **recipe** is everything that turns an image into measured, gated cells:
segmenter and settings, measurement options, gates, calibration and the
channels' names (images are read in the recipe's channel order and renamed
to its names, so "target channel" and friends resolve in every image).

A **batch** is a recipe, a list of **items** (one field of view each: one
multi-channel file, or one file per channel) with their metadata (well,
condition, replicate, …), and an output folder. Each finished item's cells
and measurements are written to that folder as soon as it is done, and the
manifest (``batch.json``) after every item, so a stopped run resumes where it
left off.

Statistics: per image (count, % in each population, mean of every
measurement) and per condition. When items have a ``Replicate``, images are
first averaged within each replicate and the condition's mean, SD and SEM
are over replicates (n = replicates); otherwise over images.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import gating
from .assays import REGISTRY as ASSAYS
from .assays import AssayContext
from .fcs import numeric_columns as fcs_columns
from .fcs import write_fcs
from .gating import Gate
from .measure import MeasureOptions, csv_headers, measure_cells
from .model import ImageSet
from .segment import run_segmentation

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".nd2")
MANIFEST = "batch.json"
PENDING, DONE, FAILED = "pending", "done", "failed"
ROWS = "ABCDEFGHIJKLMNOP"
META_COLUMNS = ("Image", "Well", "Condition", "Replicate")

# A well id on its own: "A1", "B03", "h12" — not part of a longer word or number.
_WELL = re.compile(r"(?<![A-Za-z0-9])([A-Pa-p])[-_ ]?0*([1-9]\d?)(?![0-9])")


# ── Recipe and items ─────────────────────────────────────────────────────────


@dataclass
class Recipe:
    segmenter: str
    params: dict[str, Any]  # segmenter input, sizes in pixels
    channel_names: list[str]
    channel_colors: list[str] = field(default_factory=list)
    channel_tokens: list[str] = field(default_factory=list)  # file-name part per channel file
    measure: dict[str, Any] = field(default_factory=dict)  # MeasureOptions fields
    gates: list[dict] = field(default_factory=list)
    scale: float = 0.0
    scale_source: str = ""
    assay: str = ""
    assay_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def files_per_item(self) -> int:
        return max(1, len(self.channel_tokens))


@dataclass
class BatchItem:
    id: str
    files: list[str]
    metadata: dict[str, str] = field(default_factory=dict)
    status: str = PENDING
    n_cells: int = 0
    error: str = ""
    seconds: float = 0.0
    results: dict[str, Any] = field(default_factory=dict)  # the assay's headline values


@dataclass
class Batch:
    folder: str
    recipe: Recipe
    items: list[BatchItem] = field(default_factory=list)
    name: str = "Batch"

    # ── Persistence ──────────────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        return Path(self.folder) / MANIFEST

    def save(self) -> Path:
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "name": self.name,
            "recipe": asdict(self.recipe),
            "items": [asdict(i) for i in self.items],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # never leave a half-written manifest
        return self.path

    @classmethod
    def load(cls, manifest: str | Path) -> Batch:
        manifest = Path(manifest)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return cls(
            folder=str(manifest.parent),
            recipe=Recipe(**data["recipe"]),
            items=[BatchItem(**i) for i in data.get("items", [])],
            name=data.get("name", "Batch"),
        )

    def cells_path(self, item: BatchItem) -> Path:
        return Path(self.folder) / "cells" / f"{_safe(item.id)}.json"

    def table_path(self, item: BatchItem) -> Path:
        return Path(self.folder) / "measurements" / f"{_safe(item.id)}.csv"

    # ── Queries ──────────────────────────────────────────────────────────────

    def item(self, item_id: str) -> BatchItem | None:
        return next((i for i in self.items if i.id == item_id), None)

    def todo(self) -> list[BatchItem]:
        return [i for i in self.items if i.status != DONE]

    @property
    def done_count(self) -> int:
        return sum(i.status == DONE for i in self.items)

    def metadata_fields(self) -> list[str]:
        """Metadata keys in use, the standard ones first."""
        keys = list(dict.fromkeys(k for i in self.items for k in i.metadata))
        standard = [k for k in ("Well", "Condition", "Replicate") if k in keys]
        return standard + [k for k in keys if k not in standard]

    def reset(self, items: Iterable[BatchItem] | None = None) -> None:
        """Mark items (default: all) to be analysed again."""
        for item in items if items is not None else self.items:
            item.status, item.n_cells, item.error, item.seconds = PENDING, 0, "", 0.0
            item.results = {}


# ── Finding and naming images ────────────────────────────────────────────────


def find_images(folder: str | Path, extensions: tuple[str, ...] = IMAGE_EXTENSIONS) -> list[Path]:
    """Image files directly in `folder` (not subfolders), sorted naturally (A2 before A10)."""
    folder = Path(folder)
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and _ext(p) in extensions
    ]
    return sorted(files, key=lambda p: _natural_key(p.name))


def parse_well(name: str) -> str | None:
    """The well in a file name, normalised to "A01" (None if there isn't exactly one)."""
    found = {f"{r.upper()}{int(c):02d}" for r, c in _WELL.findall(Path(name).name)}
    return found.pop() if len(found) == 1 else None


def well_position(well: str) -> tuple[int, int] | None:
    """Zero-based (row, column) of "A01"-style well ids."""
    m = re.fullmatch(r"([A-Pa-p])0*(\d{1,2})", well.strip())
    if not m:
        return None
    return ROWS.index(m.group(1).upper()), int(m.group(2)) - 1


def pattern_regex(template: str) -> re.Pattern[str]:
    """A regex for a file-name template such as ``{well}_{condition}_r{replicate}``.

    Each ``{field}`` matches one or more characters (as few as possible);
    everything else must match literally. Matched against the file's stem.
    """
    parts = re.split(r"(\{[A-Za-z][A-Za-z0-9_]*\})", template)
    out = []
    for part in parts:
        if part.startswith("{") and part.endswith("}"):
            out.append(f"(?P<{part[1:-1]}>.+?)")
        else:
            out.append(re.escape(part))
    return re.compile("".join(out) + "$", re.IGNORECASE)


def parse_pattern(template: str, name: str) -> dict[str, str] | None:
    """The fields of `template` in a file name (stem), or None if it doesn't fit."""
    m = pattern_regex(template).match(_stem(name))
    if not m:
        return None
    return {_field_name(k): v for k, v in m.groupdict().items()}


def read_layout(path: str | Path) -> dict[str, dict[str, str]]:
    """A plate layout: ``{"A01": {"Condition": "LPS", …}, …}`` from a CSV.

    Two shapes are accepted:

    - a list, one row per well: a ``Well`` column and any others
      (``Condition``, ``Replicate``, ``Dose``, …);
    - a plate grid: row letters down the first column, column numbers along
      the first row, the condition in each cell.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        return {}
    header = [h.strip() for h in rows[0]]
    if any(h.lower() == "well" for h in header):
        well_col = next(i for i, h in enumerate(header) if h.lower() == "well")
        out: dict[str, dict[str, str]] = {}
        for row in rows[1:]:
            pos = well_position(row[well_col]) if well_col < len(row) else None
            if pos is None:
                continue
            well = f"{ROWS[pos[0]]}{pos[1] + 1:02d}"
            out[well] = {
                _field_name(h): row[i].strip()
                for i, h in enumerate(header)
                if i != well_col and i < len(row) and h
            }
        return out
    columns = [int(h) if h.isdigit() else None for h in header[1:]]
    out = {}
    for row in rows[1:]:
        letter = row[0].strip().upper()
        if len(letter) != 1 or letter not in ROWS:
            continue
        for col, value in zip(columns, row[1:], strict=False):
            if col is not None and value.strip():
                out[f"{letter}{col:02d}"] = {"Condition": value.strip()}
    return out


def channel_tokens(file_names: list[str]) -> list[str]:
    """What tells one channel's file from another's: the part of the names that differs.

    ``["A01_DAPI.tif", "A01_GFP.tif"]`` → ``["DAPI", "GFP"]``. A single file
    (all channels in one file) gives ``[]``.
    """
    stems = [_stem(n) for n in file_names]
    if len(set(file_names)) <= 1:
        return []
    # Cut the shared parts back to a separator, so "img_c0"/"img_c1" give
    # "c0"/"c1" rather than "0"/"1" (which would match inside well numbers).
    prefix = _common_prefix(stems)
    prefix = prefix[: max((prefix.rfind(c) + 1 for c in _SEPARATORS), default=0)]
    suffix = _common_prefix([s[::-1] for s in stems])[::-1]
    cuts = [i for i in (suffix.find(c) for c in _SEPARATORS) if i >= 0]
    suffix = suffix[min(cuts) :] if cuts else ""
    tokens = [s[len(prefix) : len(s) - len(suffix) if suffix else None] for s in stems]
    if any(not t for t in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError("Can't tell the channel files apart by name: " + ", ".join(file_names))
    return tokens


def group_files(
    paths: list[Path], tokens: list[str]
) -> tuple[list[tuple[str, list[Path]]], list[Path]]:
    """Group per-channel files into fields of view.

    Returns ``([(key, files in token order), …], leftovers)``. With no
    tokens, every file is its own field of view. A file belongs to the
    token that appears in its name (the longest, if several do); files
    of incomplete groups are left over.
    """
    if not tokens:
        return [(_stem(p.name), [p]) for p in paths], []
    groups: dict[str, dict[str, Path]] = {}
    leftovers = []
    ordered = sorted(tokens, key=len, reverse=True)
    for path in paths:
        stem = _stem(path.name)
        found = next(((t, m) for t in ordered if (m := _segment(t).search(stem))), None)
        if found is None:
            leftovers.append(path)
            continue
        token, m = found
        key = stem[: m.start()] + "*" + stem[m.end() :]
        groups.setdefault(key, {})[token] = path
    complete = []
    for key, by_token in groups.items():
        if all(t in by_token for t in tokens):
            complete.append((key.replace("*", "").strip("_- ."), [by_token[t] for t in tokens]))
        else:
            leftovers.extend(by_token.values())
    complete.sort(key=lambda kv: _natural_key(kv[0]))
    return complete, leftovers


def make_items(
    paths: list[Path],
    recipe: Recipe,
    template: str = "",
    layout: dict[str, dict[str, str]] | None = None,
) -> tuple[list[BatchItem], list[Path]]:
    """Items for `paths`, with metadata from the file names and the plate layout.

    Metadata, later sources winning: the well found in the name, the
    template's fields, then the layout's row for that well.
    """
    groups, leftovers = group_files(paths, recipe.channel_tokens)
    items = []
    seen: set[str] = set()
    for key, files in groups:
        meta = metadata_for(key, template, layout)
        item_id = key
        n = 2
        while item_id in seen:
            item_id = f"{key} ({n})"
            n += 1
        seen.add(item_id)
        items.append(BatchItem(id=item_id, files=[str(f) for f in files], metadata=meta))
    return items, leftovers


def metadata_for(
    name: str, template: str = "", layout: dict[str, dict[str, str]] | None = None
) -> dict[str, str]:
    """Metadata for one image name: its well, the template's fields, then the layout's row."""
    meta: dict[str, str] = {}
    well = parse_well(name)
    if well:
        meta["Well"] = well
    if template:
        meta.update({k: v for k, v in (parse_pattern(template, name) or {}).items() if v})
        if "Well" in meta and (pos := well_position(meta["Well"])):
            meta["Well"] = f"{ROWS[pos[0]]}{pos[1] + 1:02d}"
    if layout and meta.get("Well") in layout:
        meta.update(layout[meta["Well"]])
    return meta


# ── Running one item ─────────────────────────────────────────────────────────


def load_item(item: BatchItem, recipe: Recipe) -> ImageSet:
    """The item's image set, channels renamed and coloured as in the recipe."""
    image_set = ImageSet()
    for path in item.files:
        image_set.add_file(path, Path(path).name)
    if len(image_set.channels) != len(recipe.channel_names):
        raise ValueError(
            f"{len(image_set.channels)} channel(s), but the analysis uses "
            f"{len(recipe.channel_names)} ({', '.join(recipe.channel_names)})."
        )
    for i, channel in enumerate(image_set.channels):
        channel.name = recipe.channel_names[i]
        if i < len(recipe.channel_colors):
            channel.color = recipe.channel_colors[i]
    return image_set


def analyse_item(
    item: BatchItem, recipe: Recipe, segmenter: Any, progress: Callable[[int], None] | None = None
) -> tuple[list[dict], pd.DataFrame, dict[str, Any]]:
    """Segment, measure and gate one item: its cells, measurement table and assay results."""
    report = progress or (lambda _v: None)
    image_set = load_item(item, recipe)
    result = run_segmentation(segmenter, image_set, recipe.params, lambda v: report(int(v * 0.8)))
    cells = []
    for cell in result.cells:
        cell["id"] = cell["label"]
        cells.append({k: v for k, v in cell.items() if k != "label"})
    table = measure_cells(
        cells,
        image_set,
        recipe.scale,
        MeasureOptions(**recipe.measure),
        lambda v: report(80 + v // 5),
    )
    gates = [Gate.from_dict(g) for g in recipe.gates]
    assay = ASSAYS.get(recipe.assay)
    results: dict[str, Any] = {}
    if assay is not None and assay.analysis is not None:
        settings = {**assay.default_settings(), **recipe.assay_settings}
        context = AssayContext(recipe.scale, image_set.shape, dict(result.extras))
        table = assay.analysis.derive(table, settings, context)
        for value in assay.analysis.results(table, gates, settings, context):
            results[value.key] = value.value
    return cells, gating.with_gate_columns(gates, table), results


def run_item(batch: Batch, item: BatchItem, segmenter: Any, progress=None) -> None:
    """Analyse `item` and write its results; records success or the error on the item."""
    start = time.perf_counter()
    try:
        cells, table, results = analyse_item(item, batch.recipe, segmenter, progress)
        cells_path, table_path = batch.cells_path(item), batch.table_path(item)
        cells_path.parent.mkdir(parents=True, exist_ok=True)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        cells_path.write_text(json.dumps(cells), encoding="utf-8")
        table.to_csv(table_path, index=False)
        units_path = Path(batch.folder) / "units.json"
        if not units_path.exists():
            units_path.write_text(json.dumps(table.attrs.get("units", {})), encoding="utf-8")
        item.status, item.n_cells, item.error = DONE, len(cells), ""
        item.results = results
    except Exception as e:  # one bad image must not stop the batch
        item.status, item.n_cells, item.error = FAILED, 0, str(e)
    item.seconds = round(time.perf_counter() - start, 2)


# ── Results ──────────────────────────────────────────────────────────────────


def read_cells(batch: Batch, item: BatchItem) -> list[dict]:
    return json.loads(batch.cells_path(item).read_text(encoding="utf-8"))


def read_table(batch: Batch, item: BatchItem) -> pd.DataFrame:
    df = pd.read_csv(batch.table_path(item))
    units = Path(batch.folder) / "units.json"
    df.attrs = {
        "scale": batch.recipe.scale,
        "units": json.loads(units.read_text()) if units.exists() else {},
    }
    return df


def combined_cells(batch: Batch) -> pd.DataFrame:
    """Every finished item's cells in one table, metadata columns first."""
    frames = []
    for item in batch.items:
        if item.status != DONE or not batch.table_path(item).exists():
            continue
        df = read_table(batch, item)
        meta = {"Image": item.id, **item.metadata}
        for i, (key, value) in enumerate(meta.items()):
            df.insert(i, key, value)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(META_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    first = read_table(batch, next(i for i in batch.items if i.status == DONE))
    out.attrs = first.attrs
    return out


def image_summary(batch: Batch, cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per finished item: metadata, cell count, % per population, mean measurements."""
    cells = combined_cells(batch) if cells is None else cells
    fields = batch.metadata_fields()
    gates = [Gate.from_dict(g) for g in batch.recipe.gates]
    rows = []
    for item in batch.items:
        row: dict[str, Any] = {"Image": item.id, **{f: item.metadata.get(f, "") for f in fields}}
        row["Status"] = item.status
        mine = cells[cells["Image"] == item.id] if len(cells) else cells
        row["Count"] = len(mine) if item.status == DONE else np.nan
        if item.status == DONE and len(mine):
            for gate in gates:
                inside = mine.get(f"Gate_{gate.name}")
                parent = (
                    mine.get(f"Gate_{gate.parent}")
                    if gate.parent and gate.parent != gating.ALL
                    else None
                )
                if inside is None:
                    continue
                inside = inside.astype(bool)
                n_parent = int(parent.astype(bool).sum()) if parent is not None else len(mine)
                row[f"Count_{gate.name}"] = int(inside.sum())
                row[f"Pct_{gate.name}"] = 100.0 * inside.sum() / n_parent if n_parent else np.nan
            for column in _measurement_columns(mine):
                row[f"Mean_{column}"] = float(pd.to_numeric(mine[column], errors="coerce").mean())
        for key, value in item.results.items():
            row[f"Result_{key}"] = value if item.status == DONE else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def condition_summary(images: pd.DataFrame, value: str, by: str = "Condition") -> pd.DataFrame:
    """Mean, SD, SEM and n of `value` per `by` group (replicate-aware; see module docs)."""
    done = images[images["Status"] == DONE] if "Status" in images else images
    if by not in done or value not in done or not len(done):
        return pd.DataFrame(columns=[by, "n", "Mean", "SD", "SEM", "Unit"])
    data = done[[by, value] + (["Replicate"] if "Replicate" in done else [])].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data[by] = data[by].fillna("").astype(str).replace("", "(none)")
    replicated = "Replicate" in data and data["Replicate"].astype(str).str.strip().ne("").any()
    if replicated:
        data = data.groupby([by, "Replicate"], as_index=False, sort=False)[value].mean()
    grouped = data.groupby(by, sort=False)[value]
    out = pd.DataFrame(
        {
            "n": grouped.count(),
            "Mean": grouped.mean(),
            "SD": grouped.std(ddof=1),
        }
    )
    out["SEM"] = out["SD"] / np.sqrt(out["n"])
    out["Unit"] = "replicates" if replicated else "images"
    return out.reset_index()


def condition_points(images: pd.DataFrame, value: str, by: str = "Condition") -> pd.DataFrame:
    """The points behind `condition_summary`: per replicate (or per image), for strip plots."""
    done = images[images["Status"] == DONE] if "Status" in images else images
    if by not in done or value not in done:
        return pd.DataFrame(columns=[by, value])
    cols = [by, value] + (["Replicate"] if "Replicate" in done else [])
    data = done[cols].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    data[by] = data[by].fillna("").astype(str).replace("", "(none)")
    if "Replicate" in data and data["Replicate"].astype(str).str.strip().ne("").any():
        data = data.groupby([by, "Replicate"], as_index=False, sort=False)[value].mean()
    return data


def plate_matrix(images: pd.DataFrame, value: str, rows: int = 8, cols: int = 12) -> np.ndarray:
    """`value` laid out on a plate (NaN where there's no image); several images per well are averaged."""
    grid = np.full((rows, cols), np.nan)
    if "Well" not in images or value not in images:
        return grid
    sums: dict[tuple[int, int], list[float]] = {}
    for well, v in zip(images["Well"], pd.to_numeric(images[value], errors="coerce"), strict=True):
        pos = well_position(str(well)) if isinstance(well, str) else None
        if pos and pos[0] < rows and pos[1] < cols and not math.isnan(v):
            sums.setdefault(pos, []).append(float(v))
    for (r, c), values in sums.items():
        grid[r, c] = float(np.mean(values))
    return grid


def plate_shape(wells: Iterable[str]) -> tuple[int, int]:
    """The smallest standard plate (6/12/24/48/96/384) holding every well."""
    positions = [p for w in wells if isinstance(w, str) and (p := well_position(w))]
    if not positions:
        return 8, 12
    need_r = max(p[0] for p in positions) + 1
    need_c = max(p[1] for p in positions) + 1
    for r, c in ((2, 3), (3, 4), (4, 6), (6, 8), (8, 12), (16, 24)):
        if need_r <= r and need_c <= c:
            return r, c
    return need_r, need_c


# ── Export ───────────────────────────────────────────────────────────────────


def result_metrics(images: pd.DataFrame) -> list[str]:
    """The per-image values worth comparing: count, population %, mean measurements."""
    return [
        c
        for c in images.columns
        if (c == "Count" or c.startswith(("Result_", "Pct_", "Count_", "Mean_")))
        and pd.api.types.is_numeric_dtype(images[c])
    ]


def export(batch: Batch, out_dir: str | Path) -> list[Path]:
    """Write the batch's results to `out_dir`; returns the files written.

    - ``cells.csv``: every cell of every image, metadata columns first;
    - ``images.csv``: one row per image (see ``image_summary``);
    - ``conditions.csv``: per condition, every metric (long format);
    - ``fcs/<image>.fcs``: one FCS 3.1 file per image, for flow software.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    cells = combined_cells(batch)
    cells_path = out / "cells.csv"
    cells.to_csv(cells_path, index=False, header=csv_headers(cells), float_format="%.6g")
    written.append(cells_path)
    images = image_summary(batch, cells)
    images_path = out / "images.csv"
    images.to_csv(images_path, index=False, float_format="%.6g")
    written.append(images_path)
    if "Condition" in images:
        frames = []
        for metric in result_metrics(images):
            summary = condition_summary(images, metric)
            summary.insert(0, "Metric", metric)
            frames.append(summary)
        if frames:
            conditions_path = out / "conditions.csv"
            pd.concat(frames, ignore_index=True).to_csv(
                conditions_path, index=False, float_format="%.6g"
            )
            written.append(conditions_path)
    fcs_dir = out / "fcs"
    for item in batch.items:
        if item.status != DONE or not batch.table_path(item).exists():
            continue
        table = read_table(batch, item)
        columns = fcs_columns(table, exclude=("Flagged",))
        extra = {"$SRC": ", ".join(Path(f).name for f in item.files)}
        extra.update({f"CYTOMETRICS_{k.upper()}": str(v) for k, v in item.metadata.items()})
        fcs_dir.mkdir(exist_ok=True)
        written.append(write_fcs(fcs_dir / f"{_safe(item.id)}.fcs", table, columns, extra=extra))
    return written


# ── Helpers ──────────────────────────────────────────────────────────────────


def _measurement_columns(df: pd.DataFrame) -> list[str]:
    skip = {"ObjectNumber", "Source", "Flagged", "Image", "Location_Center_X", "Location_Center_Y"}
    return [
        c
        for c in df.columns
        if c not in skip
        and not c.startswith("Gate_")
        and c not in META_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


_SEPARATORS = "_-. "


def _segment(token: str) -> re.Pattern[str]:
    """`token` as a whole part of a name: not inside a longer word or number."""
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", re.IGNORECASE)


def _ext(path: Path) -> str:
    name = path.name.lower()
    return ".tiff" if name.endswith(".ome.tiff") else path.suffix.lower()


def _stem(name: str) -> str:
    name = Path(name).name
    for ext in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def _field_name(key: str) -> str:
    """Template and layout keys as table headers: "condition" → "Condition"."""
    return key[:1].upper() + key[1:]


def _common_prefix(strings: list[str]) -> str:
    if not strings:
        return ""
    first, last = min(strings), max(strings)
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i]


def _natural_key(text: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def _safe(item_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ ()-]+", "_", item_id).strip() or "image"
