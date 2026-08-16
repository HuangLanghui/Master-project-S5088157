#!/usr/bin/env python3
"""Promote reviewed candidates into the experiment dataset.

The selection below was made by eye against the criteria in data/README.md: a
single unoccluded product, at least 512px on the short edge, and for most
entries legible label detail so edge_retention has something to measure.

Exclusion reasons applied:

* pairs and groups (two mugs, a bottle beside a glass, shelf shots) break the
  one-object assumption in Stage 2's debris pruning;
* anything held in a hand or worn on a body, for the same reason;
* crops showing only part of the product, which leave no silhouette to extract;
* heavily branded supermarket goods, where trademark exposure outweighs the
  extra edge detail (see data/README.md).

    python scripts/curate.py
    python -m adflow.dataset --validate
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

CANDIDATES = Path("data/candidates/pexels/candidates.json")
PRODUCTS = Path("data/products")
MANIFEST = Path("data/manifest.json")

# (candidate stem, id, display name, category, background, notes)
SELECTION = [
    # --- white / near-white sweep: the clean control stratum -----------------
    ("sunglasses_02", "sunglasses_black", "Black Sunglasses", "eyewear", "white",
     "specular, wide, unbranded"),
    ("sunglasses_05", "sunglasses_blue", "Blue Sunglasses", "eyewear", "white",
     "specular, wide, small temple text"),
    ("headphones_00", "headphones_black", "Over-ear Headphones", "electronics", "white",
     "mixed materials, square-ish"),
    ("headphones_01", "headphones_grey", "Studio Headphones", "electronics", "white",
     "mixed materials, square-ish"),
    ("ceramic_mug_04", "mug_white", "White Mug", "homeware", "white",
     "matte, square-ish, blank - edge-retention floor control"),
    ("camera_00", "camera_dslr", "DSLR Camera", "electronics", "white",
     "mixed, wide, body lettering"),
    ("camera_02", "camera_lens", "Camera Lens", "electronics", "white",
     "dense engraved markings - strongest edge-retention case"),
    ("canvas_pack_02", "backpack_black", "Black Backpack", "bags", "white",
     "matte fabric, square-ish"),

    # --- plain: single-tone or gradient sweeps -------------------------------
    ("aluminium_can_03", "can_heyday", "Beverage Can", "beverage", "plain",
     "specular metal, tall, bold graphic type"),
    ("skincare_tube_03", "tube_lets_talk", "Cosmetic Tube", "cosmetics", "plain",
     "matte, tall, dense label type"),
    ("smartphone_04", "smartphone_blank", "Smartphone", "electronics", "plain",
     "specular glass, tall, blank screen"),
    ("ceramic_mug_03", "mug_enamel", "Enamel Mug", "homeware", "plain",
     "semi-matte, square-ish"),
    ("camera_04", "camera_mirrorless", "Mirrorless Camera", "electronics", "plain",
     "mixed, wide"),

    # --- scene: styled shots, where extraction is expected to struggle -------
    ("perfume_02", "perfume_fire", "Perfume (wood stand)", "fragrance", "scene",
     "specular glass, tall, label type"),
    ("perfume_05", "perfume_dark", "Perfume (dark cloth)", "fragrance", "scene",
     "specular glass, square-ish"),
    ("earbud_case_02", "earbuds_black", "Earbud Case", "electronics", "scene",
     "gloss plastic, wide, wood ground"),
    ("steel_flask_03", "flask_green", "Vacuum Flask (outdoor)", "drinkware", "scene",
     "specular metal, tall, foliage ground"),
    ("steel_flask_01", "flask_northland", "Vacuum Flask (snow)", "drinkware", "scene",
     "specular metal, tall, printed mark"),
    ("candle_jar_04", "candle_pink", "Candle Jar", "homeware", "scene",
     "matte label on glass, square-ish"),
]

# Already in the repository; kept because it is the hardest extraction case.
EXISTING = [{
    "id": "perfume_rose", "image": "inputs/perfume-8032808_1280.jpg", "name": "Perfume (petals)",
    "category": "fragrance", "background": "scene",
    "source": "https://pixabay.com/photos/8032808/", "licence": "",
    "notes": "specular glass; 77 debris blobs - the hard-extraction reference. "
             "CONFIRM the Pixabay licence before publication.",
}]


def main() -> int:
    index = {Path(c["file"]).stem: c
             for c in json.loads(CANDIDATES.read_text(encoding="utf-8"))["candidates"]}
    PRODUCTS.mkdir(parents=True, exist_ok=True)

    products, missing = [], []
    for stem, pid, name, category, background, notes in SELECTION:
        entry = index.get(stem)
        if entry is None:
            missing.append(stem)
            continue
        src = Path(entry["file"])
        dest = PRODUCTS / f"{pid}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        products.append({
            "id": pid,
            "image": str(dest).replace("\\", "/"),
            "name": name,
            "category": category,
            "background": background,
            "source": entry.get("source", ""),
            "licence": entry.get("licence", "Pexels License"),
            "notes": f"{notes}; photographer: {entry.get('author') or 'unknown'}",
        })

    if missing:
        print("missing candidates:", ", ".join(missing))

    payload = {
        "_comment": ("Experiment dataset. Curated from Pexels candidates by "
                     "scripts/curate.py; see data/README.md for the criteria. "
                     "Validate with: python -m adflow.dataset --validate"),
        "products": products + EXISTING,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{len(payload['products'])} product(s) -> {MANIFEST}")
    for background in ("white", "plain", "scene", "transparent"):
        n = sum(1 for p in payload["products"] if p["background"] == background)
        print(f"  {background:<12} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
