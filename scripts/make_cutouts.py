#!/usr/bin/env python3
"""Precompute the Stage 2 cut-out of every manifest product.

This exists for the hosted harness, which cannot afford to matte. Measured on
one 867x1300 product, peak RSS for a single extraction is:

    rembg u2net     729 MB
    rembg u2netp    506 MB
    GrabCut         284 MB
    pre-cut alpha    69 MB

The free tier has 512 MB, so any onnxruntime path kills the process the first
time a visitor presses Generate. Most of that cost is the runtime rather than
the weights, which is why the small model does not rescue it either.

Writing the cut-outs here and shipping them means the hosted site reads an
alpha channel instead of estimating one, exactly as the `transparent` stratum
does. The masks are the ones rembg produces, so what the site renders is what
the pipeline renders; only the stage that produced the mask differs, and the
page says so.

    python scripts/make_cutouts.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from adflow.cache import file_identity  # noqa: E402
from adflow.dataset import load_manifest  # noqa: E402
from adflow.extraction import extract_product  # noqa: E402
from adflow.utils import load_image, setup_logging  # noqa: E402

OUT = ROOT / "data" / "cutouts"


def main() -> int:
    setup_logging(verbose=False)
    logging.getLogger("adflow").setLevel(logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)

    total = 0
    for product in load_manifest(ROOT / "data" / "manifest.json"):
        image = load_image(product.image)
        result = extract_product(image, file_identity(product.image))
        if result.method == "grabcut":
            print(f"REFUSING {product.id}: extracted by GrabCut, which fails on "
                  "styled backgrounds. Install rembg and re-run.")
            return 1
        path = OUT / f"{product.id}.png"
        result.cutout.save(path, optimize=True)
        size = path.stat().st_size
        total += size
        print(f"{product.id:<24} {result.method:<8} {str(result.cutout.size):<14} "
              f"{size / 1024:6.0f} kB")

    print(f"\n{len(list(OUT.glob('*.png')))} cut-outs, {total / 1e6:.1f} MB total "
          f"-> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
