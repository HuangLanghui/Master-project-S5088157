"""Stage 6c: the independent variable must behave like one."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adflow.extraction import ExtractionResult
from adflow.formats import CAMPAIGN_FORMATS, plan_layout
from adflow.lighting import plan_lighting
from adflow.preservation import prepare_product
from adflow.relight import MODES, RelightSpec, relight

LIGHT = plan_lighting(seed=2)


def test_none_is_the_identity(product_rgba):
    out = relight(product_rgba, LIGHT, RelightSpec("none"))
    assert np.array_equal(np.array(out), np.array(product_rgba))


def test_zero_strength_is_the_identity(product_rgba):
    """Otherwise the 0-point of the curve is not the un-relit control."""
    out = relight(product_rgba, LIGHT, RelightSpec("graded", 0.0))
    assert np.array_equal(np.array(out), np.array(product_rgba))


def test_graded_changes_colour_but_never_alpha(product_rgba):
    out = relight(product_rgba, LIGHT, RelightSpec("graded", 0.8))
    before, after = np.array(product_rgba), np.array(out)
    assert not np.array_equal(after[:, :, :3], before[:, :, :3]), "shading did nothing"
    assert np.array_equal(after[:, :, 3], before[:, :, 3]), (
        "relighting altered the silhouette - that is a shape change, not a light change"
    )


@pytest.mark.parametrize("strength", [0.2, 0.5, 1.0])
def test_graded_is_deterministic(product_rgba, strength):
    spec = RelightSpec("graded", strength)
    a = relight(product_rgba, LIGHT, spec)
    b = relight(product_rgba, LIGHT, spec)
    assert np.array_equal(np.array(a), np.array(b))


def test_effect_grows_monotonically_with_strength(product_rgba):
    """A dial whose steps are not ordered cannot produce a tradeoff curve."""
    base = np.array(product_rgba)[:, :, :3].astype(float)
    deltas = []
    for s in (0.25, 0.5, 0.75, 1.0):
        out = np.array(relight(product_rgba, LIGHT, RelightSpec("graded", s)))
        deltas.append(np.abs(out[:, :, :3].astype(float) - base).mean())
    assert deltas == sorted(deltas), f"non-monotonic response: {deltas}"
    assert deltas[0] > 0


def test_light_direction_actually_matters(product_rgba):
    """If shading ignored the light plan, the study would vary nothing."""
    left = relight(product_rgba, plan_lighting(0), RelightSpec("graded", 1.0))
    right = relight(product_rgba, plan_lighting(1), RelightSpec("graded", 1.0))
    assert not np.array_equal(np.array(left), np.array(right))


def test_unknown_mode_is_rejected(product_rgba):
    with pytest.raises(ValueError, match="unknown relight mode"):
        relight(product_rgba, LIGHT, RelightSpec("magic", 0.5))
    assert "magic" not in MODES


def test_reference_survives_relighting(product_rgba):
    """The measurement itself: Stage 6's reference must stay un-relit, or the
    reported SSIM would compare the relit product against itself and read ~1.0
    no matter how strong the relighting was."""
    fmt = CAMPAIGN_FORMATS[0]
    mask = (np.array(product_rgba)[:, :, 3] > 8).astype(np.uint8) * 255
    extraction = ExtractionResult(product_rgba, mask,
                                  (0, 0, product_rgba.width, product_rgba.height), "test")
    record = prepare_product(extraction, plan_layout(fmt, product_rgba.size))

    before = record.reference_rgb.copy()
    record.render_cutout = relight(record.scaled_cutout, LIGHT, RelightSpec("graded", 1.0))

    assert np.array_equal(record.reference_rgb, before)
    assert not np.array_equal(np.array(record.to_paste)[:, :, :3], record.reference_rgb)
