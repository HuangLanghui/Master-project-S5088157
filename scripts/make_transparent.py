#!/usr/bin/env python3
"""Build the transparent stratum from existing products.

The stratum needs inputs whose mask Stage 2 does not have to estimate, so that
any fidelity loss measured on them belongs to compositing rather than matting.
Estimating the mask separately is the obvious way to get one: distance
transform, Otsu threshold, morphological clean-up, boundary checked at 1:1.
Four attempts produced one usable mask - camera_lens_cut, which is in the
manifest and is not rebuilt here - and failed on a lens gradient, an absorbed
contact shadow, and staircase artefacts from morphology.

Constructing the case instead sidesteps the judgement entirely, and that is
what this script does for the other three. The Stage 2 cut-out of a clean
product is saved as alpha and fed back in, so Stage 2 reads the same mask it
would have produced and contributes no error at all. Exact, rather than judged.

What this does not do: say anything about matting accuracy. The mask is correct
by definition here, not by comparison with the object. Sources are limited to
products already scoring 1.0 on extraction quality so the definition is at
least a defensible one, and each derived item stays paired with its source when
sampling, since the two are the same photograph.

    python scripts/make_transparent.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from adflow.cache import file_identity  # noqa: E402
from adflow.extraction import extract_product  # noqa: E402
from adflow.utils import load_image, setup_logging  # noqa: E402

SOURCES = [
    ("sunglasses_blue", "eyewear"),
    ("headphones_black", "electronics"),
    ("backpack_black", "bags"),
]


def main() -> int:
    setup_logging(verbose=False)
    logging.getLogger("adflow").setLevel(logging.ERROR)

    manifest = ROOT / "data" / "manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in data["products"]}

    added = 0
    for pid, category in SOURCES:
        src = by_id.get(pid)
        if src is None:
            print(f"skip {pid}: not in manifest")
            continue
        new_id = f"{pid}_cut"
        if new_id in by_id:
            print(f"skip {new_id}: already present")
            continue

        image = load_image(src["image"])
        result = extract_product(image, file_identity(src["image"]))
        out = ROOT / "data" / "products" / f"{new_id}.png"
        result.cutout.save(out)

        data["products"].append({
            "id": new_id,
            "image": f"data/products/{new_id}.png",
            "name": f"{src['name']} (pre-cut)",
            "category": category,
            "background": "transparent",
            "source": src["source"],
            "licence": src["licence"],
            "notes": (
                f"synthetic pre-cut input: the Stage 2 cut-out of {pid} saved "
                "as alpha, so Stage 2 reads the mask back exactly and adds no "
                "error. Isolates compositing loss and carries no information "
                f"about matting accuracy. Derived from {pid}; sample the two "
                "as one item."),
        })
        added += 1
        print(f"wrote {out.name}  {result.cutout.size}  via {result.method}")

    if added:
        manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"{added} item(s) added; {len(data['products'])} products total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
