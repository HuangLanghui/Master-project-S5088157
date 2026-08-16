"""Experiment dataset.

Results are reported per product, so each entry carries a stable id, its
provenance (source URL and licence), and the fields the analysis stratifies on.

The `background` field matters most. Extraction difficulty differs sharply
between a product shot on white and one shot in a styled scene, and a single
pooled mean over both hides that difference.

Validate before a run:

    python -m adflow.dataset --validate
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from PIL import Image

from .utils import log

DEFAULT_MANIFEST = Path("data/manifest.json")

# Shot types, ordered by how hard extraction is expected to be.
BACKGROUNDS = ("transparent", "white", "plain", "scene")


@dataclass
class Product:
    id: str
    image: str                       # path relative to the repository root
    name: str
    category: str = "consumer product"
    background: str = "white"
    source: str = ""                 # URL or dataset name
    licence: str = ""                # e.g. "CC0", "Unsplash", "ABO"
    notes: str = ""
    headline: str | None = None      # optional: pins copy for a qualitative figure
    tagline: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def path(self, root: Path = Path(".")) -> Path:
        return root / self.image


@dataclass
class Manifest:
    products: list[Product] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.products)

    def __iter__(self):
        return iter(self.products)

    def by_background(self, background: str) -> list[Product]:
        return [p for p in self.products if p.background == background]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.products:
            counts[p.category] = counts.get(p.category, 0) + 1
        return counts


def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> Manifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it (see data/README.md) or pass --manifest."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Manifest([Product(**entry) for entry in raw.get("products", [])])


def validate(manifest: Manifest, root: Path = Path(".")) -> list[str]:
    """Return a list of problems. Empty means the manifest is safe to run."""
    problems: list[str] = []

    seen: set[str] = set()
    for p in manifest:
        if p.id in seen:
            problems.append(f"duplicate id: {p.id!r}")
        seen.add(p.id)

        if p.background not in BACKGROUNDS:
            problems.append(
                f"{p.id}: background {p.background!r} not one of {BACKGROUNDS}")
        # A placeholder is worse than a blank: it looks filled in.
        if not p.licence or "UNKNOWN" in p.licence.upper() or "TODO" in p.licence.upper():
            problems.append(f"{p.id}: licence unresolved ({p.licence or 'blank'!r})")

        image = p.path(root)
        if not image.exists():
            problems.append(f"{p.id}: missing image {image}")
            continue
        try:
            with Image.open(image) as im:
                w, h = im.size
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{p.id}: unreadable image ({exc})")
            continue
        # Below ~512px the product occupies too few pixels for the fidelity
        # metrics to mean much after the Stage 6 resample.
        if min(w, h) < 512:
            problems.append(f"{p.id}: {w}x{h} is small; fidelity metrics will be noisy")

    return problems


def _main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Validate the experiment manifest")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    print(f"{len(manifest)} product(s)")
    for background in BACKGROUNDS:
        n = len(manifest.by_background(background))
        if n:
            print(f"  {background:<12} {n}")
    print("categories:", ", ".join(f"{k} ({v})" for k, v in manifest.categories().items()))

    problems = validate(manifest)
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nmanifest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
