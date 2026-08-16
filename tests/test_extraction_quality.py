"""The extraction screen - the fidelity metrics cannot see a failed cut-out.

These tests pin the behaviour on synthetic cases where the answer is known.
Real photographs are messier; see the docstrings in metrics.py for where the
screen is known to be unreliable.
"""
from __future__ import annotations

import numpy as np
import pytest

from adflow.metrics import boundary_contrast, extraction_quality

pytest.importorskip("cv2", reason="the screen degrades to NaN without OpenCV")

W = H = 400


def _scene(object_colour=(30, 30, 30), background=(230, 230, 230)):
    """A dark square object centred on a light background."""
    img = np.full((H, W, 3), background, np.uint8)
    img[120:280, 120:280] = object_colour
    return img


def _mask_of_object():
    m = np.zeros((160, 160), np.uint8)
    m[:] = 255
    return m, (120, 120, 280, 280)


def test_real_object_boundary_has_contrast():
    mask, bbox = _mask_of_object()
    assert boundary_contrast(_scene(), mask, bbox) > 50


def test_region_carved_from_uniform_background_has_none():
    """The failure mode that motivated this: a mask whose boundary exists only
    in the mask, not in the photograph."""
    img = np.full((H, W, 3), (200, 200, 200), np.uint8)
    mask = np.full((160, 160), 255, np.uint8)
    assert boundary_contrast(img, mask, (120, 120, 280, 280)) < 5


def test_low_contrast_product_is_not_distinguishable():
    """A known limitation, pinned so it is not mistaken for a regression.

    A white product on a white sweep genuinely has no boundary signal. The
    screen cannot tell it apart from a bad mask, so a low score on a
    light-on-light shot means 'inspect', not 'failed'.
    """
    img = _scene(object_colour=(238, 238, 238), background=(245, 245, 245))
    mask, bbox = _mask_of_object()
    assert boundary_contrast(img, mask, bbox) < 15


def test_quality_penalises_a_mask_that_owns_the_frame():
    mask = np.full((H, W), 255, np.uint8)
    q = extraction_quality(mask, (0, 0, W, H), (W, H))
    assert q["frame_coverage"] == pytest.approx(1.0)
    assert q["border_contact"] == pytest.approx(1.0)
    assert q["quality"] == 0.0


def test_quality_passes_a_normal_silhouette():
    mask, bbox = _mask_of_object()
    q = extraction_quality(mask, bbox, (W, H), _scene())
    assert q["quality"] > 0.9


def test_rectangular_products_are_not_penalised():
    """bbox_fill is reported but must not be scored - a phone, a pouch and a
    can all legitimately fill their bounding box."""
    mask, bbox = _mask_of_object()          # a solid square: fill == 1.0
    q = extraction_quality(mask, bbox, (W, H), _scene())
    assert q["bbox_fill"] == pytest.approx(1.0)
    assert q["quality"] > 0.9


def test_contrast_term_is_skipped_without_the_source_image():
    mask, bbox = _mask_of_object()
    q = extraction_quality(mask, bbox, (W, H))
    assert "boundary_contrast" not in q
    assert q["quality"] > 0.9


def test_grounding_detects_a_shadow_and_scores_a_paste_at_zero():
    """The metric that makes the shadow ablation measurable.

    lighting_coherence reads interior pixels only and is blind to anything
    drawn outside the silhouette, so without this the no_shadow and naive arms
    return exact nulls that look like "the shadow does not matter".
    """
    from adflow.metrics import contact_grounding

    box = (150, 100, 250, 220)
    mask = np.full((120, 100), 255, np.uint8)

    flat = np.full((H, W, 3), 200, np.uint8)
    flat[100:220, 150:250] = 40                       # the product
    assert contact_grounding(flat, box, mask) < 0.02   # nothing below it

    grounded = flat.copy()
    grounded[220:270, 130:270] = 120                   # a shadow beneath
    assert contact_grounding(grounded, box, mask) > 0.1


def test_grounding_ignores_effects_at_the_silhouette_edge():
    """A rim light is an edge treatment, not a ground contact; it must not be
    read as grounding or the two components become confounded."""
    from adflow.metrics import contact_grounding

    box = (150, 100, 250, 220)
    mask = np.full((120, 100), 255, np.uint8)
    base = np.full((H, W, 3), 200, np.uint8)
    base[100:220, 150:250] = 40

    rimmed = base.copy()
    rimmed[95:100, 145:255] = 255                      # halo above the product
    rimmed[100:220, 145:150] = 255                     # halo to its left
    assert abs(contact_grounding(rimmed, box, mask)
               - contact_grounding(base, box, mask)) < 0.02
