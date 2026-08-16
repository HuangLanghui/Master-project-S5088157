"""Stage 8: Campaign generation.

Orchestrates Stages 4-7 once per target format, producing the three-format
campaign, and hands each artefact (plus its preservation record) to
Stage 9 for evaluation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .ablation import FULL, Ablation
from .background import generate_background
from .brand import BrandProfile
from .comfyui_client import ComfyUIClient
from .compose import compose
from .extraction import ExtractionResult
from .formats import AdFormat, CAMPAIGN_FORMATS, LayoutPlan, plan_layout
from .lighting import plan_lighting
from .preservation import PreservationRecord, prepare_product
from .relight import RelightSpec, relight
from .understanding import ProductProfile
from .utils import log


@dataclass
class CampaignAsset:
    format_key: str
    image: Image.Image
    plan: LayoutPlan
    record: PreservationRecord
    background_provider: str
    path: Path | None = None
    relight: str = "none"
    ablation: str = "full"


def generate_campaign(extraction: ExtractionResult, product: ProductProfile,
                      brand: BrandProfile, headline: str, tagline: str,
                      client: ComfyUIClient, workflow: str,
                      out_dir: Path, seed: int = 42, backend: str = "auto",
                      relight_spec: RelightSpec | None = None,
                      preset: str = "turbo",
                      ablation: Ablation = FULL,
                      formats: Sequence[AdFormat] | None = None) -> list[CampaignAsset]:
    """Render one asset per format, defaulting to the whole campaign.

    `formats` exists for hosts that cannot hold three at once. Each format
    costs roughly 160 MB of peak memory, so a full campaign peaks near 390 MB
    and a single format near 230 MB; the browser harness renders one at a time
    when deployed. Seeding and lighting do not depend on the subset, so a
    format rendered alone is identical to the same format rendered as part of
    the campaign.
    """
    targets = list(formats) if formats else list(CAMPAIGN_FORMATS)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = (product.scene_keywords * 3)[:len(targets)]
    # One key light for the whole campaign, so the three formats stay coherent.
    light = plan_lighting(seed)
    # Ablating coordination gives Stage 5 a different light from Stage 7,
    # which is what an uncoordinated pipeline does by default: the backdrop
    # is generated without reference to how the product will be lit.
    bg_light = light if ablation.coordinated_light else plan_lighting(seed + 1)
    assets: list[CampaignAsset] = []

    for i, fmt in enumerate(targets):
        log.info("Stage 8: format %d/%d - %s", i + 1, len(targets), fmt.label)
        plan = plan_layout(fmt, extraction.cutout.size)
        record = prepare_product(extraction, plan)
        # After the reference is captured, so Stage 9 prices the change.
        if relight_spec and relight_spec.mode != "none":
            record.render_cutout = relight(record.scaled_cutout, light,
                                           relight_spec, seed=seed + i)
        bg, provider = generate_background(plan, product, brand, scenes[i],
                                           client, workflow, bg_light, seed=seed + i,
                                           preset=preset, ablation=ablation,
                                           backend=backend)
        ad = compose(bg, record, plan, brand, headline, tagline, light, ablation)
        path = out_dir / f"ad_{fmt.key}_{fmt.width}x{fmt.height}.png"
        ad.save(path)
        assets.append(CampaignAsset(fmt.key, ad, plan, record, provider, path,
                                    relight=(relight_spec or RelightSpec()).label(),
                                    ablation=ablation.label))
    return assets
