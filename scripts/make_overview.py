#!/usr/bin/env python3
"""Render data/dataset_overview.png from the manifest.

The figure this replaces was made by hand and went stale: it showed twenty
products including one, perfume_rose, that had since been dropped from the
study, and it was missing pouch_kraft and the whole transparent stratum.
Generating it from data/manifest.json means it cannot drift again.

Pre-cut inputs are composited onto a checkerboard rather than onto white, so
that what makes the transparent stratum different - the pipeline receives an
alpha channel instead of estimating one - is visible in the figure.

    python scripts/make_overview.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from adflow.dataset import load_manifest  # noqa: E402

OUT = ROOT / "data" / "dataset_overview.png"
STRATA = ["white", "plain", "scene", "transparent"]

COLUMNS = 5
CELL = 250          # thumbnail box, square
LABEL = 26          # strip beneath each thumbnail
PAD = 12
BG = (255, 255, 255)
INK = (26, 26, 26)
CHECK = ((238, 238, 238), (214, 214, 214))


def _font(size: int = 13) -> ImageFont.ImageFont:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _checkerboard(size: tuple[int, int], square: int = 10) -> Image.Image:
    tile = Image.new("RGB", size, CHECK[0])
    d = ImageDraw.Draw(tile)
    for y in range(0, size[1], square):
        for x in range(0, size[0], square):
            if (x // square + y // square) % 2:
                d.rectangle([x, y, x + square - 1, y + square - 1], fill=CHECK[1])
    return tile


def thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        img = handle.convert("RGBA")
    img.thumbnail((CELL, CELL), Image.LANCZOS)
    base = _checkerboard(img.size) if img.getchannel("A").getextrema()[0] < 255 \
        else Image.new("RGB", img.size, BG)
    base.paste(img, (0, 0), img)
    return base


def main() -> int:
    products = sorted(load_manifest(ROOT / "data" / "manifest.json"),
                      key=lambda p: (STRATA.index(p.background), p.id))
    rows = -(-len(products) // COLUMNS)

    font = _font()
    sheet = Image.new("RGB", (COLUMNS * (CELL + PAD) + PAD,
                              rows * (CELL + LABEL + PAD) + PAD), BG)
    draw = ImageDraw.Draw(sheet)

    for i, product in enumerate(products):
        col, row = i % COLUMNS, i // COLUMNS
        x0 = PAD + col * (CELL + PAD)
        y0 = PAD + row * (CELL + LABEL + PAD)

        thumb = thumbnail(product.path(ROOT))
        sheet.paste(thumb, (x0 + (CELL - thumb.width) // 2,
                            y0 + (CELL - thumb.height) // 2))
        draw.text((x0, y0 + CELL + 6), f"{product.id} [{product.background}]",
                  fill=INK, font=font)

    sheet.save(OUT, optimize=True)
    counts = {s: sum(1 for p in products if p.background == s) for s in STRATA}
    print(f"{OUT.relative_to(ROOT)}  {sheet.width}x{sheet.height}  "
          f"{len(products)} products  " +
          "  ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
