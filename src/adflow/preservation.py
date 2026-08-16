"""Stage 6: product preservation.

Holds the guarantee that the product is never re-synthesised:

* the Stage 2 cutout is only ever geometrically resampled (one LANCZOS scale
  to the planned box), with no generative model in the path;
* a per-format PreservationRecord stores the canvas region and the resampled
  reference pixels, which Stage 9 uses to verify pixel-for-pixel that
  composition did not alter the product beyond alpha blending at its boundary.

Stage 6c relights the product after the reference is recorded, which is what
makes the change measurable: the reference stays untouched, so the fidelity
drop Stage 9 reports is the cost of the relighting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .extraction import ExtractionResult
from .formats import LayoutPlan
from .utils import log


@dataclass
class PreservationRecord:
    scaled_cutout: Image.Image     # RGBA at final on-canvas size
    canvas_box: tuple[int, int, int, int]
    reference_rgb: np.ndarray      # what the product must look like in the ad
    reference_mask: np.ndarray     # uint8 {0,255}
    # What Stage 7 actually pastes. Identical to `scaled_cutout` unless Stage 6c
    # relit it. Keeping the two apart is what lets Stage 9 price the relighting:
    # `reference_rgb` remains the untouched original no matter what 6c does.
    render_cutout: Image.Image | None = None

    @property
    def to_paste(self) -> Image.Image:
        return self.render_cutout or self.scaled_cutout


def prepare_product(extraction: ExtractionResult, plan: LayoutPlan) -> PreservationRecord:
    x0, y0, x1, y1 = plan.product_box
    scaled = extraction.cutout.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    arr = np.array(scaled)
    record = PreservationRecord(
        scaled_cutout=scaled,
        canvas_box=plan.product_box,
        reference_rgb=arr[:, :, :3].copy(),
        reference_mask=(arr[:, :, 3] > 128).astype(np.uint8) * 255,
    )
    log.info("Stage 6: product locked to box %s (%dx%d px)",
             plan.product_box, scaled.width, scaled.height)
    return record
