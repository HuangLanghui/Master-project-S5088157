#!/usr/bin/env python3
"""Extraction quality of each matting method, over the whole manifest.

The grid in `results.csv` records whichever method the provider chain actually
chose, which is `rembg` everywhere except the pre-cut `transparent` stratum. So
the GrabCut side of the comparison in the paper (5.4) has no row in that file.
This forces both providers over every product and prints the two means.

The headline comparison excludes the `transparent` stratum. A pre-cut input has
no background outside the mask, so boundary contrast is undefined and the score
collapses towards zero however good the mask is; Section 5.7 of the paper makes
the same exclusion.

    python scripts/compare_matting.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                          # noqa: E402
from PIL import Image                                       # noqa: E402

from adflow.dataset import load_manifest                    # noqa: E402
from adflow.extraction import _try_grabcut, _try_rembg      # noqa: E402
from adflow.metrics import extraction_quality               # noqa: E402

PROVIDERS = (("rembg", _try_rembg), ("grabcut", _try_grabcut))


def quality(fn, image: Image.Image) -> float:
    result = fn(image)
    if result is None:
        return float("nan")
    mask = (np.array(result.cutout)[:, :, 3] > 8).astype(np.uint8) * 255
    return extraction_quality(mask, result.bbox, image.size,
                              np.array(image.convert("RGB")))["quality"]


def main() -> None:
    manifest = load_manifest(ROOT / "data" / "manifest.json")
    rows = []
    print(f"{'product':<26}{'stratum':<14}" + "".join(f"{n:>10}" for n, _ in PROVIDERS))
    for product in manifest.products:
        with Image.open(product.path(ROOT)) as handle:
            image = handle.convert("RGBA")
        scores = {name: quality(fn, image) for name, fn in PROVIDERS}
        rows.append((product.background, scores))
        print(f"{product.id:<26}{product.background:<14}"
              + "".join(f"{scores[n]:>10.3f}" for n, _ in PROVIDERS))

    print("\nmean extraction quality")
    for name, _ in PROVIDERS:
        every = [s[name] for _, s in rows if not np.isnan(s[name])]
        screened = [s[name] for stratum, s in rows
                    if stratum != "transparent" and not np.isnan(s[name])]
        print(f"  {name:<9} all n={len(every):<3} {np.mean(every):.4f}"
              f"    excluding transparent n={len(screened):<3} {np.mean(screened):.4f}")
    print("\nThe second column is the one the paper quotes: the screen does not"
          "\napply to pre-cut inputs, so including them measures nothing useful.")


if __name__ == "__main__":
    main()
