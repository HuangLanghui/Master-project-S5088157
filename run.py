#!/usr/bin/env python3
"""Batch runner. Reads config.json and writes one folder per product.

Usage: python run.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adflow.pipeline import run_pipeline
from adflow.utils import setup_logging

RULE = "=" * 60


def main():
    setup_logging(verbose=True)

    config_path = Path("config.json")
    if not config_path.exists():
        print("config.json not found. See README.md.")
        return 1

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    products = config.get("products", [])
    if not products:
        print("No products listed in config.json.")
        return 1

    backend = config.get("backend", "procedural")
    seed_base = config.get("seed", 42)

    print(f"\n{RULE}")
    print(f"  {len(products)} product(s) | backend {backend} | seed {seed_base}")
    print(f"  started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{RULE}\n")

    results = {}
    for i, prod in enumerate(products):
        img_file = prod.get("image")
        name = prod.get("name", f"Product_{i + 1}")
        # Unset copy is generated in Stage 3b rather than filled with a placeholder.
        headline = prod.get("headline")
        tagline = prod.get("tagline")

        # Accepts a bare filename under inputs/ or a path relative to the repo
        # root, so config.json can point at dataset products.
        img_path = Path("inputs") / img_file
        if not img_path.exists() and Path(img_file).exists():
            img_path = Path(img_file)
        if not img_path.exists():
            print(f"[{i + 1}/{len(products)}] {name}: skipped, {img_path} not found")
            continue

        out_dir = Path("outputs") / name.replace(" ", "_").lower()
        seed = seed_base + i

        print(f"\n[{i + 1}/{len(products)}] {name}")
        print(f"   input {img_path}")
        try:
            report = run_pipeline(
                str(img_path), name, headline, tagline,
                out_dir=str(out_dir), backend=backend, seed=seed,
            )
            results[name] = {"ok": True, "metrics": report.get("summary", {})}
            print(f"   done, wrote {out_dir}/")
        except Exception as exc:  # noqa: BLE001 - one bad product must not stop the batch
            results[name] = {"ok": False, "error": str(exc)}
            print(f"   failed: {exc}")

    print(f"\n{RULE}")
    print("  Summary")
    print(RULE)
    for name, result in results.items():
        if result["ok"]:
            m = result["metrics"]
            print(f"  {name}: SSIM {m.get('mean_ssim', 'n/a')}, "
                  f"palette {m.get('mean_palette_consistency', 'n/a')}")
        else:
            print(f"  {name}: FAILED - {result['error']}")

    print(f"\nOutputs in outputs/")
    print(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
