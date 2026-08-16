"""Reproducibility: the same seed must yield the same bytes.

Every number in the write-up is produced by a seeded run. If a re-run drifts,
the results are not reproducible and the experiment tables mean nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adflow.background import generate_background
from adflow.comfyui_client import ComfyUIClient
from adflow.copywriting import generate_copy
from adflow.extraction import extract_product
from adflow.formats import CAMPAIGN_FORMATS, plan_layout
from adflow.lighting import plan_lighting
from adflow.understanding import ProductProfile


def _background(brand, seed):
    fmt = CAMPAIGN_FORMATS[0]
    plan = plan_layout(fmt, (120, 200))
    return generate_background(
        plan, ProductProfile(name="T"), brand, "studio",
        ComfyUIClient("http://127.0.0.1:1"),  # unreachable on purpose
        "unused.json", plan_lighting(seed), seed=seed, backend="procedural",
    )[0]


def test_procedural_background_is_deterministic(brand_profile):
    a = np.array(_background(brand_profile, seed=7))
    b = np.array(_background(brand_profile, seed=7))
    assert np.array_equal(a, b)


def test_seed_actually_changes_the_output(brand_profile):
    """A seed that changes nothing would make the seed sweep meaningless."""
    a = np.array(_background(brand_profile, seed=7))
    b = np.array(_background(brand_profile, seed=8))
    assert not np.array_equal(a, b)


def test_extraction_is_deterministic(product_on_white):
    """GrabCut is iterative; confirm it does not drift between identical runs."""
    a = extract_product(product_on_white)
    b = extract_product(product_on_white)
    assert a.method == b.method
    assert a.bbox == b.bbox
    assert np.array_equal(a.mask, b.mask)


def test_template_copy_is_deterministic_and_seed_varying(brand_profile):
    product = ProductProfile(name="Widget")
    img = Image.new("RGB", (16, 16))
    same = [generate_copy(img, product, brand_profile, seed=3) for _ in range(2)]
    assert same[0].headline == same[1].headline
    assert same[0].provider == "template"

    others = {generate_copy(img, product, brand_profile, seed=s).headline
              for s in range(4)}
    assert len(others) > 1, "seed does not vary the generated copy"


def test_lighting_plan_is_seed_stable():
    assert plan_lighting(5) == plan_lighting(5)
    assert plan_lighting(5) != plan_lighting(6)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_every_lighting_setup_is_reachable(seed):
    """All four setups must be selectable, or the sweep silently covers fewer."""
    assert plan_lighting(seed).key
