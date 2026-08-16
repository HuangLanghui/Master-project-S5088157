"""The dissertation's central claim, as an executable assertion.

If these fail, the fidelity numbers reported in the write-up are not true of
the code that produced them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from adflow.compose import compose
from adflow.extraction import ExtractionResult
from adflow.formats import CAMPAIGN_FORMATS, plan_layout
from adflow.lighting import plan_lighting
from adflow.preservation import prepare_product


def _record_for(product_rgba, fmt):
    mask = (np.array(product_rgba)[:, :, 3] > 8).astype(np.uint8) * 255
    extraction = ExtractionResult(product_rgba, mask,
                                  (0, 0, product_rgba.width, product_rgba.height), "test")
    return prepare_product(extraction, plan_layout(fmt, product_rgba.size))


def test_opaque_product_pixels_survive_composition_exactly(product_rgba, brand_profile):
    """Where the cut-out is fully opaque, the ad must carry those exact bytes.

    This is the whole preservation contract: background, shadow and rim light
    are all painted *before* the product, so none of them may leak into it.
    """
    for fmt in CAMPAIGN_FORMATS:
        record = _record_for(product_rgba, fmt)
        plan = plan_layout(fmt, product_rgba.size)
        background = Image.new("RGB", (fmt.width, fmt.height), (200, 60, 60))

        ad = compose(background, record, plan, brand_profile,
                     "Headline", "Tagline", plan_lighting(seed=2))

        x0, y0, x1, y1 = record.canvas_box
        rendered = np.array(ad)[y0:y1, x0:x1]
        opaque = np.array(record.scaled_cutout)[:, :, 3] == 255
        assert opaque.any(), "fixture produced no fully-opaque pixels"
        assert np.array_equal(rendered[opaque], record.reference_rgb[opaque]), (
            f"{fmt.key}: composition altered preserved product pixels"
        )


def test_rim_light_stays_outside_the_mask(product_rgba, brand_profile):
    """A relighting effect that touched the product would invalidate Stage 9.

    Compose twice with rim light at its strongest and at zero; the opaque
    product region must be byte-identical between the two.
    """
    fmt = CAMPAIGN_FORMATS[0]
    record = _record_for(product_rgba, fmt)
    plan = plan_layout(fmt, product_rgba.size)
    background = Image.new("RGB", (fmt.width, fmt.height), (30, 30, 30))

    strong = plan_lighting(seed=2)                      # key-left-raking, rim 0.75
    none = strong.__class__(strong.key, strong.direction, strong.elevation,
                            strong.warmth, rim=0.0)
    assert strong.rim > 0

    lit = np.array(compose(background, record, plan, brand_profile, "H", "T", strong))
    unlit = np.array(compose(background, record, plan, brand_profile, "H", "T", none))

    x0, y0, x1, y1 = record.canvas_box
    opaque = np.array(record.scaled_cutout)[:, :, 3] == 255
    assert np.array_equal(lit[y0:y1, x0:x1][opaque], unlit[y0:y1, x0:x1][opaque]), (
        "rim light bled into preserved product pixels"
    )
    # ...and it must actually do something outside, or the test proves nothing.
    assert not np.array_equal(lit, unlit), "rim light had no visible effect at all"


def test_only_transform_is_a_single_resample(product_rgba):
    """Stage 6 must resample once, from the original cut-out, not from a chain."""
    fmt = CAMPAIGN_FORMATS[0]
    record = _record_for(product_rgba, fmt)
    x0, y0, x1, y1 = record.canvas_box

    direct = product_rgba.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    assert np.array_equal(np.array(record.scaled_cutout), np.array(direct)), (
        "scaled cut-out differs from a single LANCZOS resample of the source"
    )
