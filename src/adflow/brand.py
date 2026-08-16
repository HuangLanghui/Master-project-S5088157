"""Stage 3: Brand profile construction.

Derives a reusable brand profile from the extracted product:

* dominant colour palette (dependency-free k-means over product pixels),
* an accent / neutral split based on saturation,
* typography and tone defaults that later stages (copy rendering,
  background prompting) consume.

The profile is serialised to JSON so a campaign can be re-run
deterministically against the same brand identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image

from .understanding import ProductProfile
from .utils import kmeans_palette, log, rgb_to_hex


@dataclass
class BrandProfile:
    product_name: str
    palette: list[str]                 # hex, ordered by dominance
    palette_weights: list[float]
    accent: str
    neutral: str
    tone: str = "clean, premium, confident"
    font_stack: str = "DejaVuSans"
    prompt_fragment: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def build_brand_profile(cutout: Image.Image, mask: np.ndarray,
                        product: ProductProfile, k: int = 5) -> BrandProfile:
    arr = np.array(cutout.convert("RGB")).astype(float)
    pixels = arr[mask > 0]
    centroids, weights = kmeans_palette(pixels, k=k)

    sat = centroids.max(axis=1) - centroids.min(axis=1)
    accent = centroids[int(np.argmax(sat))]
    neutral = centroids[int(np.argmin(sat))]

    palette_hex = [rgb_to_hex(c) for c in centroids]
    fragment = (
        f"{product.category}, brand colours {', '.join(palette_hex[:3])}, "
        f"{product.audience}"
    )
    profile = BrandProfile(
        product_name=product.name,
        palette=palette_hex,
        palette_weights=[round(float(w), 4) for w in weights],
        accent=rgb_to_hex(accent),
        neutral=rgb_to_hex(neutral),
        prompt_fragment=fragment,
    )
    log.info("Stage 3: brand palette %s (accent %s)", palette_hex, profile.accent)
    return profile
