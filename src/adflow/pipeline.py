"""End-to-end pipeline driver used by main.py and the tests."""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from .ablation import FULL, Ablation
from .brand import build_brand_profile
from .campaign import generate_campaign
from .comfyui_client import ComfyUIClient
from .copywriting import generate_copy
from .evaluate import evaluate_campaign
from . import cache
from .extraction import extract_product
from .formats import AdFormat
from .relight import RelightSpec
from .understanding import understand_product
from .utils import load_image, log, save_json


def run_pipeline(image_path: str, name: str,
                 headline: str | None = None, tagline: str | None = None,
                 out_dir: str = "outputs", comfyui_url: str = "http://127.0.0.1:8188",
                 workflow: str = "workflows/background_sdxl_api.json",
                 provider: str = "auto", seed: int = 42, backend: str = "auto",
                 relight_spec: RelightSpec | None = None,
                 preset: str = "turbo",
                 ablation: Ablation = FULL,
                 formats: Sequence[AdFormat] | None = None) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)                                   # input
    cache_key = cache.file_identity(image_path)

    product = understand_product(image, name, provider)              # 1
    extraction = extract_product(image, cache_key)                   # 2
    brand = build_brand_profile(extraction.cutout, extraction.mask,  # 3
                                product)
    copy = generate_copy(image, product, brand,                      # 3b
                         headline, tagline, seed=seed)
    save_json({"product": product.to_dict(), "brand": brand.to_dict(),
               "copy": copy.to_dict(), "extraction_method": extraction.method},
              out / "profiles.json")
    extraction.cutout.save(out / "product_cutout.png")

    client = ComfyUIClient(comfyui_url)
    assets = generate_campaign(extraction, product, brand,           # 4-8
                               copy.headline, copy.tagline, client, workflow,
                               out, seed=seed, backend=backend,
                               relight_spec=relight_spec, preset=preset,
                               ablation=ablation, formats=formats)
    report = evaluate_campaign(assets, brand, out)                   # 9

    log.info("Pipeline finished in %.1fs - outputs in %s", time.time() - t0, out)
    return report
