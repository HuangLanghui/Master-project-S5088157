"""Stage 7: Composition.

Layers: background (Stage 5) -> soft contact shadow -> preserved product
(Stage 6) -> ad copy. Copy colour is chosen by WCAG contrast against the
sampled background region; a translucent scrim is added when contrast is
still insufficient, so the layout never produces unreadable text.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .ablation import FULL, Ablation
from .brand import BrandProfile
from .formats import LayoutPlan
from .lighting import LightPlan, rim_mask, shadow_offset, tint
from .preservation import PreservationRecord
from .utils import log, relative_luminance


def _font(px: int) -> ImageFont.FreeTypeFont:
    # Windows ships lowercase filenames (arialbd.ttf), Linux/mac carry DejaVu.
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf",
                 "arialbd.ttf", "segoeuib.ttf", "arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()  # type: ignore[return-value]


def _contact_shadow(canvas: Image.Image, record: PreservationRecord,
                    light: LightPlan) -> None:
    """Project the product's own silhouette onto the ground plane.

    An ellipse under the bounding box is the cheapest possible shadow and one
    of the clearest signs that a product was pasted rather than photographed:
    it has no relationship to the object's shape. Here the Stage 6 mask is
    sheared along the light direction and flattened by the light's elevation,
    which is the shadow an object of that outline would actually cast.

    Two things sell it beyond the shape. The shadow darkens where it meets the
    product and fades along its length, since a real contact shadow is sharp
    and dense at the contact point and diffuse further out; and the blur grows
    with distance for the same reason.
    """
    x0, y0, x1, y1 = record.canvas_box
    w, h = x1 - x0, y1 - y0
    ox, oy = shadow_offset(light, h)

    # Flatten toward the ground plane, then shear along the light direction.
    squash = 0.14 + 0.30 * (1.0 - light.elevation)
    dx, _ = light.direction
    shear = dx * (1.0 - light.elevation) * 1.6

    # Flip before projecting. The base of the object meets the ground at the
    # contact point and its top falls furthest away, so the silhouette has to
    # be inverted; projecting it upright puts the cap at the product's feet.
    silhouette = Image.fromarray(record.reference_mask).transpose(
        Image.FLIP_TOP_BOTTOM)
    proj_h = max(1, int(h * squash))
    proj = silhouette.resize((w, proj_h), Image.BILINEAR).transform(
        (int(w + abs(shear) * proj_h) + 1, proj_h),
        Image.AFFINE,
        (1, shear, 0 if shear >= 0 else shear * proj_h, 0, 1, 0),
        resample=Image.BILINEAR,
    )

    # Fade along the direction the shadow travels.
    arr = np.array(proj).astype(np.float64)
    ramp = np.linspace(1.0, 0.25, proj.height)[:, None]
    arr *= ramp * (0.34 + 0.30 * light.elevation)

    shadow = Image.new("L", canvas.size, 0)
    # Overlap the product's base slightly so the shadow reads as attached.
    shadow.paste(Image.fromarray(arr.astype(np.uint8)),
                 (int(x0 + ox - (shear * proj_h if shear < 0 else 0)),
                  int(y1 + oy - proj_h * 0.12)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(4, w // 55)))

    black = Image.new("RGBA", canvas.size, (0, 0, 0, 255))
    canvas.paste(black, (0, 0), shadow)


def _rim_light(canvas: Image.Image, record: PreservationRecord,
               brand: BrandProfile, light: LightPlan) -> None:
    """Paint a halo on the product's lit edge - outside the mask, so the
    preserved pixels pasted over it stay bit-identical."""
    if light.rim <= 0:
        return
    x0, y0, x1, y1 = record.canvas_box
    halo = rim_mask(record.reference_mask, light, x1 - x0)
    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow.paste(Image.new("RGB", halo.size, tint((255, 248, 236), light.warmth)),
               (x0, y0), halo)
    canvas.alpha_composite(glow)


LIGHT_INK = (250, 250, 250, 255)
DARK_INK = (22, 22, 26, 255)
# WCAG AA for large text. Below this the scrim goes in.
MIN_CONTRAST = 3.0


def _copy_box(plan: LayoutPlan, draw: ImageDraw.ImageDraw,
              headline: str, tagline: str, hl_font, tl_font) -> tuple[int, int, int, int]:
    """The area the copy block actually covers, in canvas coordinates."""
    x, y = plan.copy_anchor
    width = max(draw.textlength(headline, font=hl_font),
                draw.textlength(tagline, font=tl_font))
    height = plan.headline_px * 2.2
    left = x - width / 2 if plan.copy_align == "center" else x
    pad = plan.headline_px * 0.35
    return (int(left - pad), int(y - pad), int(left + width + pad), int(y + height + pad))


def _contrast_ratio(a: float, b: float) -> float:
    lo, hi = sorted((a, b))
    return (hi + 0.05) / (lo + 0.05)


def _copy_treatment(canvas: Image.Image, box: tuple[int, int, int, int]) -> tuple[tuple, bool]:
    """Pick the ink colour and decide whether the copy needs a scrim.

    The region is sampled from the area the text will occupy rather than a
    fixed offset from the anchor, which for centred copy sampled mostly to one
    side of the block. A generated background can be busy enough that neither
    ink reaches usable contrast; the caller then lays a scrim first.
    """
    x0, y0, x1, y1 = box
    region = np.array(canvas.convert("RGB").crop(
        (max(0, x0), max(0, y0), min(canvas.width, x1), min(canvas.height, y1))))
    if region.size == 0:
        return LIGHT_INK, False

    lum = relative_luminance(region.reshape(-1, 3).mean(axis=0))
    light = _contrast_ratio(relative_luminance(np.array(LIGHT_INK[:3])), lum)
    dark = _contrast_ratio(relative_luminance(np.array(DARK_INK[:3])), lum)
    ink = LIGHT_INK if light >= dark else DARK_INK
    return ink, max(light, dark) < MIN_CONTRAST


def _scrim(canvas: Image.Image, box: tuple[int, int, int, int], ink: tuple) -> None:
    """Lay a soft panel behind the copy so it stays legible.

    Drawn before the product is pasted, so it can never touch preserved pixels
    even if a future layout puts the two closer together. Blurred rather than
    hard-edged: a visible rectangle behind the text reads as a caption box
    rather than as part of the photograph.
    """
    x0, y0, x1, y1 = box
    pad = int((y1 - y0) * 0.4)
    panel = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(panel).rounded_rectangle(
        [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
        radius=pad, fill=150)
    panel = panel.filter(ImageFilter.GaussianBlur(pad * 0.8))

    # Scrim opposes the ink: dark panel under light text and vice versa.
    tone = (0, 0, 0) if ink is LIGHT_INK else (255, 255, 255)
    canvas.paste(Image.new("RGBA", canvas.size, tone + (255,)), (0, 0), panel)


def compose(background: Image.Image, record: PreservationRecord,
            plan: LayoutPlan, brand: BrandProfile,
            headline: str, tagline: str, light: LightPlan,
            ablation: Ablation = FULL) -> Image.Image:
    canvas = background.convert("RGBA")
    if ablation.shadow:
        _contact_shadow(canvas, record, light)
    if ablation.rim:
        _rim_light(canvas, record, brand, light)

    hl_font = _font(plan.headline_px)
    tl_font = _font(round(plan.headline_px * 0.45))
    measure = ImageDraw.Draw(canvas)
    box = _copy_box(plan, measure, headline, tagline, hl_font, tl_font)
    colour, needs_scrim = _copy_treatment(canvas, box)
    if needs_scrim:
        _scrim(canvas, box, colour)

    # Preserved pixels go on last: nothing above draws into the product mask.
    canvas.paste(record.to_paste, record.canvas_box[:2], record.to_paste)

    draw = ImageDraw.Draw(canvas)
    x, y = plan.copy_anchor
    anchor = "ma" if plan.copy_align == "center" else "la"
    # A soft shadow only where the ink is light; dark ink on a light ground
    # needs no lift, and the hard offset copy it replaced read as an artefact.
    if colour is LIGHT_INK:
        shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        sd.text((x, y + 3), headline, font=hl_font, fill=(0, 0, 0, 110), anchor=anchor)
        sd.text((x, y + round(plan.headline_px * 1.25) + 3), tagline,
                font=tl_font, fill=(0, 0, 0, 90), anchor=anchor)
        canvas.alpha_composite(shade.filter(ImageFilter.GaussianBlur(3)))
    draw.text((x, y), headline, font=hl_font, fill=colour, anchor=anchor)
    draw.text((x, y + round(plan.headline_px * 1.25)), tagline,
              font=tl_font, fill=colour, anchor=anchor)

    # Accent underline keyed to brand palette.
    bar_w = draw.textlength(tagline, font=tl_font)
    bx = x - bar_w / 2 if plan.copy_align == "center" else x
    by = y + round(plan.headline_px * 1.95)
    draw.rectangle([bx, by, bx + bar_w, by + max(4, plan.headline_px // 14)],
                   fill=brand.accent)

    log.info("Stage 7: composed %s (%s copy)", plan.fmt.key,
             "light" if colour[0] > 128 else "dark")
    return canvas.convert("RGB")
