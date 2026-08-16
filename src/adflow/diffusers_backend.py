"""Stage 5 backend: SDXL through HuggingFace diffusers.

No external server, so the whole generator stays one Python codebase.

Two presets, and the difference between them is not only speed:

    turbo   SDXL-Turbo, 4 steps at guidance 0. Fast, but distilled to run
            without classifier-free guidance, which means the negative prompt
            is never evaluated. Anything unwanted has to be excluded by the
            positive prompt alone.
    sdxl    SDXL base 1.0, 28 steps at guidance 6.5. Roughly an order of
            magnitude slower, and the negative prompt actually applies.

Install:  pip install torch --index-url https://download.pytorch.org/whl/cu128
          pip install -e ".[generative]"
Each preset downloads its weights on first use (~7 GB for SDXL base).
"""
from __future__ import annotations

from functools import lru_cache

from PIL import Image

from .utils import log

PRESETS: dict[str, dict] = {
    "turbo": {"model_id": "stabilityai/sdxl-turbo", "steps": 4, "cfg": 0.0},
    "sdxl": {"model_id": "stabilityai/stable-diffusion-xl-base-1.0",
             "steps": 28, "cfg": 6.5},
}
DEFAULT_PRESET = "turbo"
DEFAULT_MODEL = PRESETS[DEFAULT_PRESET]["model_id"]


def is_available() -> bool:
    try:
        import diffusers  # noqa: F401
        import torch
        return torch.cuda.is_available() or torch.backends.mps.is_available()
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=2)
def _load_pipe(model_id: str):
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "mps"
    log.info("diffusers: loading %s on %s (first run downloads the model)", model_id, device)
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=torch.float16, variant="fp16")
    if device == "cuda":
        # Keeps peak VRAM within ~8 GB laptop GPUs.
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    return pipe


def generate(prompt: str, negative: str, width: int, height: int,
             seed: int, preset: str = DEFAULT_PRESET) -> Image.Image:
    """Render a background with the named preset.

    The negative prompt is dropped below guidance 1.0. Classifier-free guidance
    interpolates between the conditioned and unconditioned predictions, and at
    guidance 0 the negative branch carries no weight, so passing one there is
    silently inert rather than an error. Turbo runs at guidance 0, which is why
    its positive prompt has to exclude unwanted content by itself.
    """
    import torch

    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    model_id = PRESETS[preset]["model_id"]
    steps = PRESETS[preset]["steps"]
    cfg = PRESETS[preset]["cfg"]

    pipe = _load_pipe(model_id)
    # SDXL works at ~1MP; render at a supported size, then resize to format.
    gw, gh = _gen_size(width, height)
    g = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(prompt=prompt,
               negative_prompt=(negative or None) if cfg > 1.0 else None,
               width=gw, height=gh, num_inference_steps=steps,
               guidance_scale=cfg, generator=g).images[0]
    return img.resize((width, height), Image.LANCZOS)


def _gen_size(w: int, h: int) -> tuple[int, int]:
    """Nearest SDXL-friendly resolution (multiple of 64, ~1MP) with same aspect."""
    aspect = w / h
    target = 1024 * 1024
    gh = int((target / aspect) ** 0.5)
    gw = int(gh * aspect)
    return max(512, gw // 64 * 64), max(512, gh // 64 * 64)
