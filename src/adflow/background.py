"""Stage 5: Background scene synthesis.

Primary path: SDXL text-to-image through a local ComfyUI instance, with the
prompt assembled from the brand profile (Stage 3) and the product's scene
keywords (Stage 1). The negative prompt explicitly discourages the model
from painting a competing product - the canvas region reserved for the real
product is filled later by Stage 6/7 with untouched pixels.

Fallback path: a deterministic procedural background built from the brand
palette (diagonal gradient + soft vignette + accent glow behind the future
product position). This keeps the whole pipeline runnable on machines
without a GPU / ComfyUI, and gives evaluation a controlled baseline.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

from . import cache, diffusers_backend
from .ablation import FULL, Ablation, NEUTRAL
from .brand import BrandProfile
from .comfyui_client import ComfyUIClient
from .formats import LayoutPlan
from .lighting import LightPlan, describe, tint
from .understanding import ProductProfile
from .utils import log, relative_luminance


# Only applied when guidance_scale > 1. SDXL-Turbo is distilled to run at
# guidance 0, where classifier-free guidance never evaluates the negative
# branch, so on that path this string has no effect at all and the positive
# prompt has to do the work on its own.
NEGATIVE_PROMPT = (
    "text, watermark, logo, product, bottle, box, packaging, people, hands, "
    "light stand, tripod, softbox, lamp, camera, cable, equipment, "
    "room, wall, floor, skirting board, door, window, ceiling, corner, "
    "furniture, horizon, perspective, "
    "clutter, low quality, deformed"
)


def build_prompt(product: ProductProfile, brand: BrandProfile, scene: str,
                 light: LightPlan | None = None,
                 ablation: Ablation = FULL) -> str:
    """Assemble the text prompt for a generated backdrop.

    Two constraints shape the wording.

    It describes a surface and its light, not a photographic set. Words like
    "studio", "backdrop" and "advertising shoot" make the model draw the
    equipment used to take such a photograph, and on the SDXL-Turbo path there
    is no working negative prompt to remove it.

    It also asks for a seamless surface with no floor line, no horizon and no
    furniture. Stage 7 pastes a front-on cut-out, so any background carrying
    its own perspective (a room corner, a receding floor) puts the product's
    implied camera at odds with the scene's, and no amount of shadow or colour
    work hides that. A gradient sweep has no perspective to contradict, which
    is why commercial pack shots use one.

    The light plan is included so Stage 5 and Stage 6b agree. Without it the
    generated scene picks its own key light while the contact shadow and rim
    light follow another, and lighting_coherence drops accordingly.
    """
    # Every word here is chosen for what it does *not* summon.
    #
    # No negations. A diffusion model does not parse "no floor line"; the
    # phrase supplies the tokens "floor" and "line" and makes the thing more
    # likely, not less. Exclusions belong in NEGATIVE_PROMPT, which only has
    # an effect above guidance 1.
    #
    # No architecture. "wall" implies the room it belongs to, and the model
    # supplies the skirting board, the door and the floor along with it. The
    # vocabulary here is abstract: gradient, haze, falloff, colour field.
    #
    # No "seamless", which reads as "tileable texture" and yields a busy
    # repeating pattern that competes with the product and the copy.
    parts = [
        "abstract colour gradient, soft atmospheric haze, defocused",
        scene,
    ]
    if ablation.brand_palette:
        parts.append(f"colour scheme {', '.join(brand.palette[:3])}")
    if light is not None:
        parts.append(describe(light))
    parts += [
        "smooth continuous tone, gentle falloff, flat colour field",
        "shallow depth of field, blurred, minimal",
    ]
    return ", ".join(parts)


def generate_background(plan: LayoutPlan, product: ProductProfile, brand: BrandProfile,
                        scene: str, client: ComfyUIClient, workflow: str,
                        light: LightPlan, seed: int = 42,
                        backend: str = "auto",
                        preset: str = diffusers_backend.DEFAULT_PRESET,
                        ablation: Ablation = FULL,
                        ) -> tuple[Image.Image, str]:
    """Returns (background RGB image at format resolution, provider name).

    Backend resolution order for "auto":
      1. diffusers (pure-Python SDXL-Turbo - the primary path)
      2. ComfyUI, if a local server is reachable (optional reference path)
      3. procedural brand background (deterministic offline baseline)
    """
    prompt = build_prompt(product, brand, scene, light, ablation)
    size = (plan.fmt.width, plan.fmt.height)

    if backend in ("auto", "diffusers") and diffusers_backend.is_available():
        log.info("Stage 5: diffusers %s - %s", preset, prompt)
        # Keyed on everything that feeds the image. The relight level is
        # absent on purpose: Stage 6c changes the product, never the backdrop,
        # so one background serves every level of a sweep.
        key = cache.digest("bg", preset, prompt, NEGATIVE_PROMPT, size, seed)
        img = cache.image(
            f"background-{preset}", key,
            lambda: diffusers_backend.generate(prompt, NEGATIVE_PROMPT,
                                               *size, seed, preset=preset))
        return img, f"diffusers-{preset}"
    if backend == "diffusers":
        log.warning("Stage 5: diffusers requested but torch/GPU unavailable")

    if backend in ("auto", "comfyui") and client.is_available():
        log.info("Stage 5: ComfyUI SDXL - %s", prompt)
        img = client.run_workflow(workflow, patches={
            "6": {"text": prompt},
            "7": {"text": NEGATIVE_PROMPT},
            "5": {"width": plan.fmt.width, "height": plan.fmt.height},
            "3": {"seed": seed},
        })
        return img.resize((plan.fmt.width, plan.fmt.height)), "comfyui-sdxl"
    if backend == "comfyui":
        log.warning("Stage 5: ComfyUI requested but server unreachable")

    log.info("Stage 5: procedural brand background (offline baseline)")
    return _procedural(plan, brand, light, seed, ablation), "procedural"


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _procedural(plan: LayoutPlan, brand: BrandProfile, light: LightPlan,
                seed: int, ablation: Ablation = FULL) -> Image.Image:
    w, h = plan.fmt.width, plan.fmt.height
    if ablation.brand_palette:
        c0 = np.array(tint(_hex_to_rgb(brand.neutral), light.warmth, 0.6), float)
        c1 = np.array(_hex_to_rgb(brand.palette[0]), float)
        accent = np.array(_hex_to_rgb(brand.accent), float)
    else:
        # Neutral grey, so the background carries no brand information.
        c0 = c1 = accent = np.array(_hex_to_rgb(NEUTRAL), float)

    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    dx, dy = light.direction
    # t = 0 on the side the light comes from, 1 on the side it travels toward.
    t = np.clip(0.5 + ((xx / w - 0.5) * dx + (yy / h - 0.5) * dy), 0.0, 1.0)

    # Orient the ramp by luminance, not by palette order. Which of the two
    # brand colours is lighter varies per product, so keying the bright end to
    # `palette[0]` put the falloff backwards for dark-neutral palettes - the
    # product then read as lit from one side and the scene from the other.
    bright, dark = ((c0, c1) if relative_luminance(c0) >= relative_luminance(c1)
                    else (c1, c0))
    base = bright[None, None, :] * (1 - t[..., None]) + dark[None, None, :] * t[..., None]

    # Accent glow, pushed off the product toward the light source.
    px = (plan.product_box[0] + plan.product_box[2]) / 2
    py = (plan.product_box[1] + plan.product_box[3]) / 2
    throw = min(w, h) * 0.22 * (1.0 - light.elevation * 0.5)
    px -= dx * throw
    py -= dy * throw
    r = math.hypot(w, h) * 0.28
    d = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    glow = np.clip(1 - d / r, 0, 1) ** 2
    base = base * (1 - 0.35 * glow[..., None]) + accent[None, None, :] * (0.35 * glow[..., None])

    # Soft vignette for depth.
    cx, cy = w / 2, h / 2
    dv = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    base *= (1 - 0.30 * np.clip(dv * 1.4, 0, 1) ** 2)[..., None]

    rng = np.random.default_rng(seed)
    base += rng.normal(0, 2.0, base.shape)  # subtle grain, avoids banding
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(1.2))
