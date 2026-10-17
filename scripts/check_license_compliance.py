"""Fail CI if an installed dependency carries a license incompatible with
shipping Karcytics under the PolyForm Noncommercial License.

Why this exists: PolyForm Noncommercial forbids commercial use without a
paid license. Strong copyleft licenses (GPL, AGPL, SSPL, ...) forbid adding
*any* extra restriction on top of them — including a "noncommercial only"
one — so a GPL/AGPL dependency imported into Karcytics would make the whole
distribution illegal to ship under our own license. Weak-copyleft licenses
(LGPL, MPL) are fine: they only require the dependency itself stay
swappable, which a plain pip/uv install already satisfies.

This does NOT catch every possible license issue (declared metadata can be
wrong or missing) — it's a fast, first-pass net for CI, not a substitute
for actually reading a new dependency's LICENSE file before adding it.
"""

from __future__ import annotations

import importlib.metadata as m
import json
import re
import sys

# Substrings that, if present after LGPL/AGPL are stripped out (see below),
# mean the remaining text still names a disallowed license.
DISALLOWED_MARKERS = [
    "agpl",
    "gnu affero general public license",
    "sspl",
    "server side public license",
    "commons clause",
    "business source license",
    "busl",
    "elastic license",
    "cc-by-nc",
    "noncommercial",  # catches Creative-Commons-NC style terms on a *dependency*
]

# GPL (not LGPL) is checked separately below, after stripping "lgpl"/"agpl"
# substrings out of the text, since both contain the literal "gpl".
GPL_MARKER = "gpl"
GPL_STRIP_FIRST = ["lgpl", "agpl", "lesser general public license", "library general public license"]

# Longer than any real SPDX expression; beyond this the text is prose.
_MAX_EXPRESSION_LENGTH = 200

# Package name (case-insensitive) -> reason it's exempt despite its license
# text. Only add an entry here with a documented, verifiable reason — this
# list is a manual trust boundary, not a place to silence real findings.
NAME_ALLOWLIST = {
    "pyinstaller": "GPLv2 with an explicit bootloader exception: frozen/built "
    "output is not covered by the GPL regardless of the app's own license.",
    "pyinstaller-hooks-contrib": "Same PyInstaller bootloader exception as pyinstaller.",
}


def _dist_text(dist: m.Distribution) -> str:
    """The most authoritative license text available for this distribution.

    Priority matters: big scientific packages (numpy, pandas, matplotlib...)
    stuff their free-text ``License`` field with the full bundled-notices
    text of every vendored C library (libgfortran, FreeType, ...), which
    routinely *mentions* GPL/AGPL in passing without applying it to the
    package itself. The structured ``License-Expression`` (PEP 639) or
    ``Classifier: License ::`` trove fields are what the package actually
    declares itself under, so they take priority; the noisy free-text field
    is only consulted when neither is present (PySide6, for one, declares
    its license choice only in that field and has no classifier at all).
    """
    license_expr = dist.metadata.get("License-Expression", "") or ""
    if license_expr.strip():
        return license_expr.lower()

    classifiers = " ".join(v for k, v in dist.metadata.items() if k == "Classifier" and v.startswith("License ::"))
    if classifiers.strip():
        return classifiers.lower()

    return (dist.metadata.get("License", "") or "").lower()


def _marker_in(text: str) -> str | None:
    """The disallowed-license marker found in `text`, if any."""
    hit = next((marker for marker in DISALLOWED_MARKERS if marker in text), None)
    if hit is None:
        stripped = text
        for s in GPL_STRIP_FIRST:
            stripped = re.sub(re.escape(s), "", stripped)
        if GPL_MARKER in stripped:
            hit = "gpl"
    return hit


def _disallowed_marker(text: str) -> str | None:
    """Like `_marker_in`, but honouring a license *choice*.

    ``A OR B`` in an SPDX-style expression means the recipient picks one, so
    the dependency is usable as long as any one alternative is acceptable —
    PySide6 declares ``LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`` and we
    take it under the LGPL. Only short single-line text is treated as an
    expression; an "or" inside long free-text notices is just prose.
    """
    if "\n" in text or len(text) > _MAX_EXPRESSION_LENGTH:
        return _marker_in(text)
    hits = [_marker_in(alternative) for alternative in re.split(r"\s+or\s+", text.strip().strip("()"))]
    return None if None in hits else hits[0]


def _is_editable_first_party(dist: m.Distribution) -> bool:
    """First-party packages developed locally (this repo, the SDK, sibling
    plugins) carry our own PolyForm license intentionally and aren't a
    third-party dependency risk — skip them regardless of what their
    metadata says."""
    try:
        raw = dist.read_text("direct_url.json")
    except Exception:
        return False
    if not raw:
        return False
    try:
        data = json.loads(raw)
    except ValueError:
        return False
    return bool(data.get("dir_info", {}).get("editable"))


def find_violations() -> list[tuple[str, str, str]]:
    all_dists = list(m.distributions())

    # A package can be discoverable more than once (e.g. a stale egg-info
    # left behind by an old setuptools `pip install -e .` alongside the
    # real editable install uv manages) — if *any* discovered instance of a
    # name is a first-party editable install, treat the whole name as
    # first-party rather than flagging whichever instance lacks the marker.
    first_party_names = {
        dist.metadata.get("Name", "").lower() for dist in all_dists if _is_editable_first_party(dist)
    }

    violations = []
    seen_names: set[str] = set()
    for dist in all_dists:
        name = dist.metadata.get("Name", "?")
        version = dist.metadata.get("Version", "?")
        key = name.lower()

        if key in seen_names:
            continue
        seen_names.add(key)

        if key in NAME_ALLOWLIST or key in first_party_names:
            continue

        hit = _disallowed_marker(_dist_text(dist))

        if hit is not None:
            violations.append((name, version, hit))

    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print("License compliance check: no disallowed licenses found.")
        return 0

    print("License compliance check FAILED. The following dependencies carry", file=sys.stderr)
    print("a license incompatible with shipping Karcytics under PolyForm", file=sys.stderr)
    print("Noncommercial (see this script's module docstring for why):\n", file=sys.stderr)
    for name, version, hit in violations:
        print(f"  - {name} {version}  (matched: {hit!r})", file=sys.stderr)
    print(
        "\nIf this is a false positive (e.g. a bundled-exception library like "
        "PyInstaller), add it to NAME_ALLOWLIST in this script with a documented "
        "reason. Otherwise, find a permissively-licensed alternative.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
