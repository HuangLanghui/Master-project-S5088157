"""Stage 6b: scene lighting.

Defines one key-light direction that Stages 5 and 7 both consume, so the
background falloff, the contact shadow and the product's rim light agree on
where the light comes from. A campaign shares a single light across all three
formats.

This does not relight the product itself. Every effect here is drawn outside
the Stage 2 mask: the rim light goes onto the canvas before the preserved
cut-out is pasted, and the shadow never overlaps it. The Stage 6 guarantee
holds and Stage 9's SSIM is unaffected. Product relighting is Stage 6c.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image, ImageFilter

from .utils import log


@dataclass(frozen=True)
class LightPlan:
    key: str
    # Unit vector along which light travels (light source -> subject).
    # +x is rightward, +y is downward, matching image coordinates.
    direction: tuple[float, float]
    elevation: float   # 0 = raking/horizontal, 1 = directly overhead
    warmth: float      # -1 cool, 0 neutral, +1 warm
    rim: float         # rim-light strength, 0..1

    def to_dict(self) -> dict:
        return asdict(self)


# Four studio setups. The seed picks one, so re-rolling the seed relights the
# campaign without touching any other stage.
_SETUPS: tuple[LightPlan, ...] = (
    LightPlan("key-left-high", (0.55, 0.84), elevation=0.78, warmth=0.20, rim=0.55),
    LightPlan("key-right-high", (-0.55, 0.84), elevation=0.78, warmth=0.10, rim=0.55),
    LightPlan("key-left-raking", (0.82, 0.57), elevation=0.42, warmth=0.35, rim=0.75),
    LightPlan("top-soft", (0.10, 0.99), elevation=0.92, warmth=0.00, rim=0.35),
)


def plan_lighting(seed: int = 42) -> LightPlan:
    plan = _SETUPS[seed % len(_SETUPS)]
    log.info("Stage 6b: lighting %s (elevation %.2f, rim %.2f)",
             plan.key, plan.elevation, plan.rim)
    return plan


def describe(plan: LightPlan) -> str:
    """Phrase the light plan for a text-to-image prompt.

    Shared by Stage 5 and Stage 6c so the generated scene and any product
    relighting are asked for the same lighting, rather than each inventing its
    own wording.
    """
    dx, dy = plan.direction
    if dx > 0.25:
        side = "key light from the left"
    elif dx < -0.25:
        side = "key light from the right"
    else:
        side = "key light from directly above"

    height = "high and soft" if plan.elevation > 0.65 else "low and raking"
    if plan.warmth > 0.1:
        colour = "warm tungsten"
    elif plan.warmth < -0.1:
        colour = "cool daylight"
    else:
        colour = "neutral white"
    return f"{side}, {height}, {colour}"


def shadow_offset(plan: LightPlan, height: int) -> tuple[float, float]:
    """Where the contact shadow lands, in pixels, relative to the product base.

    A low light throws a long shadow; an overhead light pools it underneath.
    """
    length = height * 0.35 * (1.0 - plan.elevation) ** 0.7
    dx, dy = plan.direction
    return dx * length, dy * length * 0.35


def rim_mask(mask: np.ndarray, plan: LightPlan, width: int) -> Image.Image:
    """A soft halo hugging the lit edge of the product, as an 8-bit mask.

    Built by blurring the product mask and subtracting the mask itself, so the
    result lies strictly outside the product, then weighted by a gradient so
    only the edge facing the light glows.
    """
    solid = Image.fromarray(mask)
    halo = solid.filter(ImageFilter.GaussianBlur(max(2.0, width * 0.035)))

    outside = np.maximum(np.array(halo, float) - np.array(solid, float), 0.0)

    # Gradient running along the light direction; brightest where light arrives.
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = plan.direction
    norm = math.hypot(dx, dy) or 1.0
    # Project each pixel onto -direction (i.e. back toward the light source).
    proj = (-(xx / max(w - 1, 1) - 0.5) * dx - (yy / max(h - 1, 1) - 0.5) * dy) / norm
    facing = np.clip(proj * 2.0 + 0.35, 0.0, 1.0) ** 1.5

    return Image.fromarray((outside * facing * plan.rim).astype(np.uint8))


def tint(rgb: tuple[int, int, int], warmth: float, amount: float = 1.0) -> tuple[int, int, int]:
    """Shift a colour toward tungsten (warm) or daylight (cool)."""
    r, g, b = rgb
    k = warmth * amount * 28.0
    return (
        int(np.clip(r + k, 0, 255)),
        int(np.clip(g + k * 0.35, 0, 255)),
        int(np.clip(b - k * 0.8, 0, 255)),
    )
