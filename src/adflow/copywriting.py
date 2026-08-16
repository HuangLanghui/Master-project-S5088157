"""Stage 3b: ad copy.

Produces the headline and tagline that Stage 7 renders. Provider chain, first
available wins:

    manual     copy supplied in config.json. An explicit line always wins;
               generation never overrides what the operator wrote.
    vlm        Claude reads the product image and the Stage 1/3 profiles and
               proposes several variants.
    template   deterministic templates keyed off the product profile, so the
               pipeline stays offline-runnable and reproducible.

The seed picks the variant, so copy can be re-rolled without touching the rest
of the pipeline.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, asdict

from PIL import Image

from .brand import BrandProfile
from .understanding import ProductProfile
from .utils import log


@dataclass
class AdCopy:
    headline: str
    tagline: str
    provider: str

    def to_dict(self) -> dict:
        return asdict(self)


# Generic lines - safe for any product, used when the category is unknown.
_GENERIC = (
    ("Meet {name}", "Designed for every day"),
    ("Simply Better", "Every detail considered"),
    ("Made to Be Used", "Built to last"),
    ("{name}, Perfected", "No compromises"),
)

# Category-aware lines - only used when Stage 1 resolved a real category.
_CATEGORIED = (
    ("Your {category}, Elevated", "Crafted for {audience}"),
    ("Meet {name}", "{category} without compromise"),
    ("Rethink the {category}", "Made for {audience}"),
)

_DEFAULT_CATEGORY = "consumer product"

_VLM_PROMPT = (
    "You are writing advertising copy for the product in this image.\n\n"
    "Product name: {name}\n"
    "Category: {category}\n"
    "Audience: {audience}\n"
    "Brand tone: {tone}\n"
    "Brand palette: {palette}\n\n"
    "Write 3 distinct headline/tagline pairs. Headlines are at most 4 words and "
    "carry the emotional hook. Taglines are at most 7 words and state the concrete "
    "benefit. Do not invent specifications, prices, or claims you cannot see in the "
    "image. Do not use the product name in every variant."
)

_COPY_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "headline": {"type": "string"},
                    "tagline": {"type": "string"},
                },
                "required": ["headline", "tagline"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["variants"],
    "additionalProperties": False,
}


def generate_copy(image: Image.Image, product: ProductProfile, brand: BrandProfile,
                  headline: str | None = None, tagline: str | None = None,
                  seed: int = 42) -> AdCopy:
    """Resolve the ad copy. Explicit `headline`/`tagline` short-circuit generation."""
    if headline and tagline:
        log.info("Stage 3b: copy from config (manual)")
        return AdCopy(headline, tagline, "manual")

    generated = _vlm_copy(image, product, brand, seed) or _template_copy(product, seed)
    # A partially-specified config still wins for the line it did specify.
    copy = AdCopy(headline or generated.headline, tagline or generated.tagline,
                  generated.provider)
    log.info("Stage 3b: copy via %s - %r / %r", copy.provider, copy.headline, copy.tagline)
    return copy


def _template_copy(product: ProductProfile, seed: int) -> AdCopy:
    known_category = product.category and product.category != _DEFAULT_CATEGORY
    pool = _CATEGORIED + _GENERIC if known_category else _GENERIC
    headline, tagline = pool[seed % len(pool)]
    fields = {
        "name": product.name,
        "category": (product.category or "").lower(),
        "audience": (product.audience or "everyone").lower(),
    }
    return AdCopy(headline.format(**fields), tagline.format(**fields), "template")


def _vlm_copy(image: Image.Image, product: ProductProfile, brand: BrandProfile,
              seed: int) -> AdCopy | None:
    try:
        import anthropic
    except ImportError:
        return None

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    prompt = _VLM_PROMPT.format(
        name=product.name,
        category=product.category,
        audience=product.audience,
        tone=brand.tone,
        palette=", ".join(brand.palette[:3]),
    )
    try:
        response = anthropic.Anthropic().messages.create(
            model="claude-opus-5",
            max_tokens=2000,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _COPY_SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/jpeg",
                        "data": base64.b64encode(buf.getvalue()).decode()}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as exc:  # noqa: BLE001 - degrade to templates, never fail the run
        log.warning("Stage 3b: VLM copy failed (%s); using templates", exc)
        return None

    if response.stop_reason == "refusal":
        log.warning("Stage 3b: VLM declined the copy request; using templates")
        return None

    import json
    text = next((b.text for b in response.content if b.type == "text"), "")
    variants = json.loads(text).get("variants", [])
    if not variants:
        return None
    pick = variants[seed % len(variants)]
    return AdCopy(pick["headline"], pick["tagline"], "vlm")
