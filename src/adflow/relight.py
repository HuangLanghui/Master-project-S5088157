"""Stage 6c: product relighting.

Four modes, in increasing order of how much they alter the product:

    none         no change; the preservation contract holds. SSIM ~0.999.
    graded       2.5-D Lambertian shading from a pseudo-normal field plus a
                 colour-temperature shift. Deterministic, CPU-only.
    resample     a control, not a treatment: the same down/upsample round trip
                 the generative path is forced through by VRAM, with no model.
                 Whatever it costs is resampling loss, not the model's doing.
    generative   diffusion img2img over the product region.

Ordering matters: Stage 6 records reference_rgb before this stage runs, so
Stage 9 compares the relit render against the untouched original and the
reported fidelity drop is the cost of relighting.

Generative mode regenerates product pixels and has been observed to alter
logotypes and package text. It is included as the low-fidelity end of the
range, not as a recommended setting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .lighting import LightPlan, describe, tint
from .utils import log

MODES = ("none", "graded", "resample", "generative")

# The diffusion pass is run at reduced resolution: SDXL img2img plus its VAE
# decode does not fit an 8 GB card at full cut-out size. Anything that
# resamples through this bound loses detail on its own, so `resample`
# exists to measure that loss separately from the model's.
GEN_MAX_EDGE = 640


@dataclass(frozen=True)
class RelightSpec:
    mode: str = "none"
    strength: float = 0.0   # 0..1; ignored when mode == "none"

    def label(self) -> str:
        return self.mode if self.mode == "none" else f"{self.mode}@{self.strength:g}"


def relight(cutout: Image.Image, light: LightPlan, spec: RelightSpec,
            seed: int = 42) -> Image.Image:
    """Return a relit RGBA cut-out. Alpha is always carried through unchanged."""
    if spec.mode == "none" or spec.strength <= 0:
        return cutout
    if spec.mode == "graded":
        out = _graded(cutout, light, spec.strength)
    elif spec.mode == "resample":
        out = _resample_roundtrip(cutout)
    elif spec.mode == "generative":
        out = _generative(cutout, light, spec.strength, seed) \
            or _graded(cutout, light, spec.strength)
    else:
        raise ValueError(f"unknown relight mode {spec.mode!r}; expected one of {MODES}")
    log.info("Stage 6c: relight %s", spec.label())
    return out


def _bounded(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    scale = min(1.0, GEN_MAX_EDGE / max(w, h))
    return max(8, int(w * scale)), max(8, int(h * scale))


def _resample_roundtrip(cutout: Image.Image) -> Image.Image:
    """Down- and upsample through the same bound the generative path uses.

    A control, not an effect. Without it, the detail lost by `generative`
    cannot be attributed: part of it is the diffusion model rewriting the
    product and part is simply the round trip through a smaller raster. This
    arm isolates the second so the first can be reported honestly.
    """
    small = _bounded(cutout.size)
    if small == cutout.size:
        return cutout
    return cutout.resize(small, Image.LANCZOS).resize(cutout.size, Image.LANCZOS)


# ---------------------------------------------------------------- graded
def _pseudo_normals(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate surface normals from the silhouette alone.

    A distance transform inside the mask gives a height field that bulges
    toward the centre, standing in for the geometry of a bottle or box; its
    gradient gives the x/y normals. This captures the rounding implied by the
    outline, not the true surface. Recovering real geometry would need a depth
    estimator.
    """
    solid = (alpha > 8).astype(np.uint8)
    try:
        import cv2
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    except ImportError:
        # Without OpenCV, fall back to a flat field: shading degrades to a
        # plain directional ramp rather than failing.
        dist = solid.astype(np.float64)

    peak = dist.max() or 1.0
    # Dome profile: height rises quickly from the edge then flattens.
    height = np.sqrt(np.clip(dist / peak, 0.0, 1.0))

    gy, gx = np.gradient(height)
    scale = 6.0  # how pronounced the implied curvature is
    nx, ny = -gx * scale, -gy * scale
    nz = np.ones_like(nx)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return nx / norm, ny / norm, nz / norm


def _graded(cutout: Image.Image, light: LightPlan, strength: float) -> Image.Image:
    arr = np.array(cutout).astype(np.float64)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]

    nx, ny, nz = _pseudo_normals(alpha)

    # Light travels along `direction`; the vector *toward* the source is -dir.
    dx, dy = light.direction
    lz = max(0.25, light.elevation)
    lx, ly = -dx, -dy
    ln = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / ln, ly / ln, lz / ln

    lambert = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    # Ambient floor; a hard zero would crush detail on the shadowed side.
    shading = 0.72 + 0.55 * lambert

    lit = rgb * shading[..., None]
    warm = np.array(tint((255, 255, 255), light.warmth, 0.5), float) / 255.0
    lit = lit * warm[None, None, :]

    blended = rgb * (1.0 - strength) + lit * strength
    out = np.dstack([np.clip(blended, 0, 255), alpha]).astype(np.uint8)
    return Image.fromarray(out)


# ------------------------------------------------------------ generative
def _generative(cutout: Image.Image, light: LightPlan, strength: float,
                seed: int) -> Image.Image | None:
    """Diffusion img2img over the product, masked back to the silhouette.

    Check that the product is still recognisable before reporting numbers from
    this mode; on the sample images it loses 30-50% of strong edges and has
    invented label text that was not in the source.

    IC-Light is the better model for this task and fits the same interface.
    This img2img path is the portable fallback, so the range has an upper end
    without a dedicated relighting checkpoint.
    """
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image
    except ImportError:
        log.warning("Stage 6c: diffusers unavailable; falling back to graded")
        return None
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        log.warning("Stage 6c: no GPU; falling back to graded")
        return None

    prompt = (f"product photograph, {describe(light)}, soft shadows, "
              "unchanged product design, photorealistic")

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    # Bound the working resolution, then restore. The `resample` mode performs
    # this round trip alone, which is what makes the two separable.
    work = cutout.convert("RGB").resize(_bounded(cutout.size), Image.LANCZOS)
    rgb = work
    generator = torch.Generator(device="cpu").manual_seed(seed)
    eff = float(np.clip(strength, 0.05, 0.6))
    # img2img runs int(steps * strength) denoising steps, so a low strength with
    # a small step count rounds to zero and the pipeline raises. Scale the step
    # count so at least two steps always survive the multiplication.
    steps = max(6, int(np.ceil(2.0 / eff)))
    result = pipe(prompt=prompt, image=rgb, strength=eff,
                  num_inference_steps=steps, guidance_scale=0.0,
                  generator=generator).images[0].resize(cutout.size, Image.LANCZOS)

    # Re-apply the original alpha. The model has no notion of the cut-out
    # boundary, so without this it redraws the silhouette and changes the
    # product's shape as well as its lighting.
    out = np.dstack([np.array(result), np.array(cutout)[:, :, 3]])
    return Image.fromarray(out.astype(np.uint8))
