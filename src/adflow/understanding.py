"""Stage 1: Product understanding.

Produces a structured description of the product (category, materials,
target audience, scene suggestions). Two providers:

* vlm      - Anthropic vision model (requires ANTHROPIC_API_KEY).
* manual   - falls back to metadata supplied in config/CLI. This keeps
                 the pipeline fully offline-runnable and reproducible.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field, asdict

from PIL import Image

from .utils import log


@dataclass
class ProductProfile:
    name: str
    category: str = "consumer product"
    description: str = ""
    materials: list[str] = field(default_factory=list)
    audience: str = "general consumers"
    # These reach the Stage 5 prompt verbatim. Avoid photographic vocabulary:
    # "studio" makes a text-to-image model draw light stands and softboxes
    # rather than a backdrop.
    scene_keywords: list[str] = field(
        default_factory=lambda: ["soft colour wash", "minimal", "muted tones"])
    provider: str = "manual"

    def to_dict(self) -> dict:
        return asdict(self)


_VLM_PROMPT = (
    "You are assisting an advertising pipeline. Describe the product in this "
    "image as strict JSON with keys: category, description, materials (list), "
    "audience, scene_keywords (list of 4 short scene ideas that flatter the "
    "product). Respond with JSON only."
)


def understand_product(image: Image.Image, name: str, provider: str = "auto") -> ProductProfile:
    if provider in ("auto", "vlm") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _understand_vlm(image, name)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            log.warning("VLM understanding failed (%s); using manual profile", exc)
    log.info("Stage 1: using manual/heuristic product profile")
    return ProductProfile(name=name, description=f"{name}, photographed on the provided image.")


def _understand_vlm(image: Image.Image, name: str) -> ProductProfile:
    import urllib.request

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 600,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": _VLM_PROMPT},
            ],
        }],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    text = "".join(b.get("text", "") for b in data["content"])
    parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
    log.info("Stage 1: VLM product profile obtained")
    return ProductProfile(name=name, provider="vlm", **{
        k: parsed[k] for k in ("category", "description", "materials", "audience", "scene_keywords")
        if k in parsed})
