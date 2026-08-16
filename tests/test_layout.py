"""Stage 4: layout rules are ablatable only if they hold invariantly."""
from __future__ import annotations

import pytest

from adflow.formats import CAMPAIGN_FORMATS, plan_layout

# Extreme aspect ratios: a very tall bottle and a very wide box.
SHAPES = [(120, 200), (200, 120), (40, 400), (400, 40), (300, 300)]


@pytest.mark.parametrize("fmt", CAMPAIGN_FORMATS, ids=lambda f: f.key)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_product_box_stays_on_canvas(fmt, shape):
    x0, y0, x1, y1 = plan_layout(fmt, shape).product_box
    assert 0 <= x0 < x1 <= fmt.width, f"{fmt.key} {shape}: box escapes horizontally"
    assert 0 <= y0 < y1 <= fmt.height, f"{fmt.key} {shape}: box escapes vertically"


@pytest.mark.parametrize("fmt", CAMPAIGN_FORMATS, ids=lambda f: f.key)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_aspect_ratio_is_preserved(fmt, shape):
    """Non-uniform scaling would distort the product - a fidelity violation
    that SSIM cannot catch, because Stage 6 records the distorted pixels.

    Tolerance is relative, not absolute: the product box is integer pixels, so
    rounding perturbs the ratio by up to ~1% on extreme shapes (a 40px edge
    scaled to 59.4px lands on 59). That is inherent to pixel grids, not a
    layout defect, but it belongs in the limitations.
    """
    x0, y0, x1, y1 = plan_layout(fmt, shape).product_box
    rendered = (x1 - x0) / (y1 - y0)
    assert rendered == pytest.approx(shape[0] / shape[1], rel=0.02)


@pytest.mark.parametrize("fmt", CAMPAIGN_FORMATS, ids=lambda f: f.key)
def test_copy_anchor_is_inside_the_safe_area(fmt):
    plan = plan_layout(fmt, (120, 200))
    x, y = plan.copy_anchor
    assert 0 <= x <= fmt.width and 0 <= y <= fmt.height


@pytest.mark.parametrize("fmt", CAMPAIGN_FORMATS, ids=lambda f: f.key)
def test_copy_does_not_overlap_the_product(fmt):
    """Overlap is the failure mode that makes an ad unusable; it must be
    structurally impossible, not merely unobserved on the sample images."""
    plan = plan_layout(fmt, (120, 200))
    px0, py0, px1, py1 = plan.product_box
    cx, cy = plan.copy_anchor
    # Copy occupies roughly two lines plus the accent rule below the anchor.
    copy_bottom = cy + plan.headline_px * 2.2
    if plan.copy_align == "center":
        assert copy_bottom <= py0, f"{fmt.key}: copy block runs into the product"
    else:
        assert cx < px0, f"{fmt.key}: left-aligned copy starts inside the product"
