"""Stage 2: the mask is the preservation contract, so its edges matter."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from adflow import extraction
from adflow.extraction import _drop_debris, extract_product

cv2 = pytest.importorskip("cv2", reason="debris pruning is a no-op without OpenCV")


def _alpha_with(blobs: list[tuple[int, int, int, int]]) -> np.ndarray:
    a = np.zeros((200, 200), np.uint8)
    for x0, y0, w, h in blobs:
        a[y0:y0 + h, x0:x0 + w] = 255
    return a


def test_debris_smaller_than_the_threshold_is_removed():
    alpha = _alpha_with([(20, 20, 100, 100), (180, 180, 8, 8)])  # 10000 px vs 64 px
    cleaned = _drop_debris(alpha)
    assert cleaned[20:120, 20:120].all()
    assert not cleaned[180:188, 180:188].any()


def test_genuine_multi_part_products_are_kept():
    """A cap or a second item above the keep ratio must survive - otherwise the
    pruning silently deletes part of the product."""
    alpha = _alpha_with([(20, 20, 100, 100), (140, 20, 40, 40)])  # 10000 vs 1600 = 16%
    cleaned = _drop_debris(alpha, keep_ratio=0.15)
    assert cleaned[20:60, 140:180].all(), "16% blob dropped at a 15% threshold"


def test_threshold_boundary_is_exclusive_below_only():
    alpha = _alpha_with([(20, 20, 100, 100), (140, 20, 30, 30)])  # 900 = 9%
    assert not _drop_debris(alpha, keep_ratio=0.15)[20:50, 140:170].any()
    assert _drop_debris(alpha, keep_ratio=0.05)[20:50, 140:170].all()


def test_single_blob_is_untouched():
    alpha = _alpha_with([(20, 20, 100, 100)])
    assert np.array_equal(_drop_debris(alpha), alpha)


def test_pruning_tightens_the_bounding_box(product_on_white):
    """Debris inflates the crop, which shrinks the product in the final ad."""
    result = extract_product(product_on_white)
    x0, y0, x1, y1 = result.bbox
    assert (x1 - x0) < product_on_white.width, "bbox spans the full width"
    assert (y1 - y0) < product_on_white.height


def test_empty_mask_does_not_crash():
    assert np.array_equal(_drop_debris(np.zeros((50, 50), np.uint8)),
                          np.zeros((50, 50), np.uint8))


def test_grabcut_fallback_warns(product_on_white, monkeypatch, caplog):
    """A silent fallback is the dangerous case.

    GrabCut runs only when rembg is absent, and on a styled background it
    returns the backdrop while every fidelity metric still reports near-perfect,
    because the Stage 6 reference is the cut-out itself. The run has to say so.
    """
    pytest.importorskip("cv2")
    monkeypatch.setattr(extraction, "_try_rembg", lambda rgba: None)

    with caplog.at_level(logging.WARNING, logger="adflow"):
        result = extract_product(product_on_white)

    assert result.method == "grabcut"
    assert any("GrabCut" in r.message or "GrabCut" in r.getMessage()
               for r in caplog.records), "fallback was silent"


def test_rembg_path_does_not_warn(product_on_white, caplog):
    pytest.importorskip("rembg")
    with caplog.at_level(logging.WARNING, logger="adflow"):
        result = extract_product(product_on_white)
    assert result.method == "rembg"
    assert not [r for r in caplog.records if "GrabCut" in r.getMessage()]
