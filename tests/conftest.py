"""Shared fixtures.

Tests use a synthetic product rather than the files in inputs/ so the suite
stays fast, deterministic, and independent of whatever the operator happens to
have dropped in that folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def product_rgba() -> Image.Image:
    """An opaque product shape on a transparent field, 120x200."""
    img = Image.new("RGBA", (120, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 30, 100, 185], radius=16, fill=(38, 120, 190, 255))
    d.rectangle([48, 8, 72, 34], fill=(230, 235, 240, 255))
    return img


@pytest.fixture
def product_on_white() -> Image.Image:
    """The same product flattened onto white - what a real input looks like."""
    img = Image.new("RGB", (160, 240), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([40, 40, 120, 200], radius=16, fill=(38, 120, 190))
    return img


@pytest.fixture
def brand_profile():
    from adflow.brand import BrandProfile

    return BrandProfile(
        product_name="Test Product",
        palette=["#2678be", "#e6ebf0", "#1b5a7f", "#3f8fd0", "#14496f"],
        palette_weights=[0.4, 0.3, 0.1, 0.1, 0.1],
        accent="#2678be",
        neutral="#e6ebf0",
    )


@pytest.fixture
def mask_from(product_rgba) -> np.ndarray:
    return (np.array(product_rgba)[:, :, 3] > 128).astype(np.uint8) * 255
