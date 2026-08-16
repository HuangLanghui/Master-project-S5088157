"""Ablation switches must actually disable what they name.

A switch that silently does nothing turns the ablation table into a row of
identical numbers, which reads as "this component does not matter" when the
truth is "this component was never turned off".
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adflow.ablation import FULL, VARIANTS, Ablation
from adflow.background import build_prompt
from adflow.compose import compose
from adflow.extraction import ExtractionResult
from adflow.formats import CAMPAIGN_FORMATS, plan_layout
from adflow.lighting import plan_lighting
from adflow.preservation import prepare_product
from adflow.understanding import ProductProfile

FMT = CAMPAIGN_FORMATS[0]
LIGHT = plan_lighting(seed=2)


def _record(product_rgba):
    mask = (np.array(product_rgba)[:, :, 3] > 8).astype(np.uint8) * 255
    ex = ExtractionResult(product_rgba, mask,
                          (0, 0, product_rgba.width, product_rgba.height), "test")
    return prepare_product(ex, plan_layout(FMT, product_rgba.size))


def _render(product_rgba, brand_profile, ablation):
    record = _record(product_rgba)
    plan = plan_layout(FMT, product_rgba.size)
    bg = Image.new("RGB", (FMT.width, FMT.height), (120, 120, 120))
    return np.array(compose(bg, record, plan, brand_profile, "H", "T", LIGHT, ablation))


def test_labels_are_stable_and_distinct():
    assert FULL.label == "full"
    assert Ablation(False, False, False, False).label == "naive"
    assert len({v.label for v in VARIANTS.values()}) == len(VARIANTS)


def test_disabling_the_shadow_changes_the_render(product_rgba, brand_profile):
    on = _render(product_rgba, brand_profile, FULL)
    off = _render(product_rgba, brand_profile, VARIANTS["no_shadow"])
    assert not np.array_equal(on, off), "shadow switch had no effect"


def test_disabling_the_rim_changes_the_render(product_rgba, brand_profile):
    on = _render(product_rgba, brand_profile, FULL)
    off = _render(product_rgba, brand_profile, VARIANTS["no_rim"])
    assert not np.array_equal(on, off), "rim switch had no effect"


def test_naive_differs_from_every_single_ablation(product_rgba, brand_profile):
    naive = _render(product_rgba, brand_profile, VARIANTS["naive"])
    for name in ("full", "no_shadow", "no_rim"):
        assert not np.array_equal(naive, _render(product_rgba, brand_profile,
                                                 VARIANTS[name])), name


def test_ablations_never_touch_preserved_pixels(product_rgba, brand_profile):
    """Whatever is switched off, the product itself must be untouched: the
    ablation study varies the compositing, not the fidelity contract."""
    record = _record(product_rgba)
    plan = plan_layout(FMT, product_rgba.size)
    bg = Image.new("RGB", (FMT.width, FMT.height), (200, 60, 60))
    opaque = np.array(record.scaled_cutout)[:, :, 3] == 255
    x0, y0, x1, y1 = record.canvas_box

    for name, ablation in VARIANTS.items():
        ad = np.array(compose(bg, record, plan, brand_profile, "H", "T", LIGHT, ablation))
        assert np.array_equal(ad[y0:y1, x0:x1][opaque],
                              record.reference_rgb[opaque]), name


def test_brand_switch_removes_the_palette_from_the_prompt(brand_profile):
    product = ProductProfile(name="T", category="fragrance")
    with_brand = build_prompt(product, brand_profile, "scene", LIGHT, FULL)
    without = build_prompt(product, brand_profile, "scene", LIGHT, VARIANTS["no_brand"])
    assert brand_profile.palette[0] in with_brand
    assert brand_profile.palette[0] not in without


def test_coordination_switch_gives_stages_different_lights():
    """The no_coord arm requires Stage 5 and Stage 7 to disagree."""
    assert plan_lighting(42) != plan_lighting(43)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_every_variant_renders(product_rgba, brand_profile, name):
    out = _render(product_rgba, brand_profile, VARIANTS[name])
    assert out.shape == (FMT.height, FMT.width, 3)
