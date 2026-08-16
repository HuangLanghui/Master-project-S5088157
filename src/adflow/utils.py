"""Shared helpers: logging, image IO, colour maths."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("adflow")


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGBA") if img.mode != "RGBA" else img


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(c) for c in rgb[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def relative_luminance(rgb) -> float:
    """WCAG relative luminance, used for contrast-aware copy colour."""
    def chan(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (chan(float(c)) for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def kmeans_palette(pixels: np.ndarray, k: int = 5, iters: int = 12, seed: int = 0):
    """Tiny dependency-free k-means for dominant-colour extraction.

    pixels: (N, 3) float array. Returns (centroids sorted by cluster size,
    normalised cluster weights).
    """
    rng = np.random.default_rng(seed)
    if len(pixels) > 4000:
        pixels = pixels[rng.choice(len(pixels), 4000, replace=False)]
    centroids = pixels[rng.choice(len(pixels), k, replace=False)].astype(float)
    for _ in range(iters):
        d = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        labels = d.argmin(axis=1)
        for i in range(k):
            members = pixels[labels == i]
            if len(members):
                centroids[i] = members.mean(axis=0)
    counts = np.bincount(labels, minlength=k).astype(float)
    order = counts.argsort()[::-1]
    return centroids[order], counts[order] / counts.sum()
