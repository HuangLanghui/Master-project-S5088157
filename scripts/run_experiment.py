#!/usr/bin/env python3
"""Sweep the manifest and write one tidy CSV of per-format metrics.

Grid: every product, every seed, both background backends, every relight level.
One row per rendered format, which is the shape pandas and R both expect for
grouped analysis.

The run is resumable. Completed rows are keyed by
(product, seed, backend, relight, format) and skipped on a restart, and the CSV
is flushed after every pipeline call, so an interrupted sweep loses at most one
product-seed combination.

Rendered ads are written to a scratch directory that each run overwrites; at
1800 images a full grid would otherwise leave several gigabytes behind. A small
subset is kept under experiments/samples for figures.

    python scripts/run_experiment.py
    python scripts/run_experiment.py --seeds 42 --backends procedural  # quick check
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from adflow import cache  # noqa: E402
from adflow.ablation import VARIANTS  # noqa: E402
from adflow.dataset import load_manifest  # noqa: E402
from adflow.extraction import extract_product  # noqa: E402
from adflow.metrics import extraction_quality  # noqa: E402
from adflow.pipeline import run_pipeline  # noqa: E402
from adflow.relight import RelightSpec  # noqa: E402
from adflow.utils import load_image, setup_logging  # noqa: E402

RELIGHT_LEVELS = [
    RelightSpec("none", 0.0),
    RelightSpec("graded", 0.25),
    RelightSpec("graded", 0.5),
    RelightSpec("graded", 0.75),
    RelightSpec("graded", 1.0),
    # Control and treatment for the generative arm. `resample` isolates the
    # detail lost to the VRAM-imposed round trip so the model's own damage can
    # be reported separately.
    RelightSpec("resample", 1.0),
    RelightSpec("generative", 0.2),
]

FIELDS = [
    "product", "category", "stratum", "seed", "backend", "relight_mode",
    "relight_strength", "ablation", "format", "provider", "extraction_method",
    "extraction_quality", "boundary_contrast", "ssim", "psnr_db", "lpips",
    "edge_retention", "hist_correlation", "lighting_coherence",
    "contact_grounding",
    "palette_consistency",
]

# Keep renders only for these, so the figures have material without the sweep
# filling the disk.
SAMPLE_STRENGTHS = {0.0, 0.5}


class KeepAwake:
    """Stop the machine sleeping mid-sweep, without changing its settings.

    A grid run is hours of GPU work with no keyboard activity, so Windows
    suspends the machine and the sweep dies partway through. SetThreadExecutionState
    asserts the requirement only for the lifetime of this process and Windows
    drops it when the process exits, so an interrupted run leaves no residue.
    powercfg would work too but edits the user's global power plan, which is
    not this script's business.

    ES_DISPLAY_REQUIRED is not set, so the screen may sleep even though the
    machine may not.
    """

    def __enter__(self):
        self.ok = False
        if sys.platform != "win32":
            return self
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            self.ok = bool(ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        except Exception:  # noqa: BLE001 - never block the run over this
            self.ok = False
        print("sleep inhibited" if self.ok else "warning: could not inhibit sleep")
        return self

    def __exit__(self, *exc):
        if self.ok:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def _existing(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {(r["product"], r["seed"], r["backend"],
                 r["relight_mode"], r["relight_strength"],
                 r.get("ablation", "full"), r["format"])
                for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--out", default="experiments/results.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--backends", nargs="+", default=["procedural", "diffusers"])
    ap.add_argument("--preset", default="turbo")
    ap.add_argument("--variants", nargs="+", default=["full"],
                    help="ablation arms; 'all' runs every variant")
    ap.add_argument("--relight", type=float, nargs="+", default=None,
                    help="restrict to these relight strengths")
    ap.add_argument("--modes", nargs="+", default=["none", "graded"],
                    help="relight modes to include")
    args = ap.parse_args()

    setup_logging(verbose=False)
    import logging
    for name in ("adflow", "httpx", "diffusers", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)

    variants = (list(VARIANTS) if args.variants == ["all"] else args.variants)
    levels = [s for s in RELIGHT_LEVELS if s.mode in set(args.modes)]
    if args.relight is not None:
        levels = [s for s in levels if s.strength in set(args.relight)]

    manifest = load_manifest(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path("experiments/_scratch")
    samples = Path("experiments/samples")

    done = _existing(out)
    if done:
        print(f"resuming: {len(done)} row(s) already present")

    total = (len(manifest) * len(args.seeds) * len(args.backends)
             * len(levels) * len(variants))
    written, skipped, failed = 0, 0, 0
    started = time.time()

    with KeepAwake(), out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if not done:
            writer.writeheader()

        for product in manifest:
            # Extraction quality is a property of the product, not of any one
            # run, so it is measured once and repeated onto every row.
            image = load_image(product.image)
            ex = extract_product(image, cache.file_identity(product.image))
            eq = extraction_quality(ex.mask, ex.bbox, image.size,
                                    np.array(image.convert("RGB")))

            for seed in args.seeds:
                for backend in args.backends:
                    for variant in variants:
                      ablation = VARIANTS[variant]
                      for spec in levels:
                        key_prefix = (product.id, str(seed), backend,
                                      spec.mode, str(spec.strength), ablation.label)
                        if all((*key_prefix, f) in done
                               for f in ("square", "story", "banner")):
                            skipped += 1
                            continue

                        keep = (seed == args.seeds[0]
                                and spec.strength in SAMPLE_STRENGTHS)
                        out_dir = (samples / f"{product.id}_{backend}_"
                                             f"{spec.label()}_{ablation.label}"
                                   if keep else scratch)
                        try:
                            report = run_pipeline(
                                product.image, product.name,
                                product.headline, product.tagline,
                                out_dir=str(out_dir), seed=seed,
                                backend=backend, relight_spec=spec,
                                preset=args.preset, ablation=ablation,
                            )
                        except Exception as exc:  # noqa: BLE001
                            failed += 1
                            print(f"  FAILED {product.id} seed={seed} "
                                  f"{backend} {spec.label()} {ablation.label}: {exc}")
                            continue

                        for fmt_key, m in report["formats"].items():
                            pf = m["product_fidelity"]
                            writer.writerow({
                                "product": product.id,
                                "category": product.category,
                                "stratum": product.background,
                                "seed": seed,
                                "backend": backend,
                                "relight_mode": spec.mode,
                                "relight_strength": spec.strength,
                                "ablation": ablation.label,
                                "format": fmt_key,
                                "provider": m["background_provider"],
                                "extraction_method": ex.method,
                                "extraction_quality": eq["quality"],
                                "boundary_contrast": eq.get("boundary_contrast", ""),
                                "ssim": pf["ssim"],
                                "psnr_db": pf["masked_psnr_db"],
                                "lpips": pf.get("lpips", ""),
                                "edge_retention": pf["edge_retention"],
                                "hist_correlation": pf["hist_correlation"],
                                "lighting_coherence":
                                    m["scene_integration"]["lighting_coherence"],
                                "contact_grounding":
                                    m["scene_integration"]["contact_grounding"],
                                "palette_consistency": m["palette_consistency"],
                            })
                            written += 1
                        fh.flush()

            elapsed = time.time() - started
            pace = elapsed / max(1, written / 3)
            remaining = (total - written / 3 - skipped) * pace
            print(f"{product.id:<20} rows={written:<5} "
                  f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.0f}m")

    print(f"\n{written} row(s) -> {out}")
    if skipped:
        print(f"{skipped} combination(s) already present")
    if failed:
        print(f"{failed} combination(s) failed")
    print("cache:", cache.stats())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
