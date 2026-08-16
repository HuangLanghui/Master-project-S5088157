#!/usr/bin/env python3
"""Can a detector tell a composite from a real photograph?

A substitute for the perceptual study, not an equal of it. It answers whether
the compositing leaves a machine-detectable trace, not which image a person
prefers, and the two questions come apart: an effect can be detectable and
still invisible, or visible and hard to detect.

Method. For each ablation arm, crop the product region plus a margin from the
render, and crop the matching region from the source photograph. A classifier
is asked to separate the two. The lower its accuracy, the harder that arm is to
tell from a real photograph, so accuracy reads as an inverse realism score with
0.5 as the ceiling.

Two design points decide whether the number means anything.

Splits are grouped by product. The same product appears on both sides of the
comparison, so a random split lets the model memorise products instead of
learning the compositing artefact, and accuracy inflates towards 1.0.

The crop is the product and its immediate surround, not the whole canvas. Given
the full frame the model separates a generated backdrop from a photographic one
and never looks at the junction, which is the thing under test.

Known confound: the source crop and the render differ in provenance as well as
in compositing. Both have been through the same resize, but the render also
carries the layout scale. Read a high accuracy as "distinguishable", not
necessarily as "the compositing is visible".

    python scripts/discriminability.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from adflow.dataset import load_manifest  # noqa: E402

ARMS = ["full", "no_shadow", "no_rim", "no_brand", "no_coord", "naive"]
CROP = 224


def _features(batch: list[Image.Image]) -> np.ndarray:
    """ResNet-18 penultimate activations, or raw pixels if torch is absent.

    A pretrained backbone is used rather than raw pixels because the artefacts
    of interest are textural (edge haloes, missing contact shading) and survive
    into mid-level features, while raw pixels are dominated by product colour.
    """
    try:
        import torch
        from torchvision import models, transforms
    except ImportError:
        return np.stack([np.asarray(im.resize((32, 32)).convert("L"),
                                    dtype=np.float64).ravel() / 255 for im in batch])

    global _NET, _TF
    try:
        net, tf = _NET, _TF
    except NameError:
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval()
        tf = transforms.Compose([
            transforms.Resize((CROP, CROP)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _NET, _TF = net, tf

    with torch.no_grad():
        x = torch.stack([tf(im.convert("RGB")) for im in batch])
        return net(x).numpy()


def _product_crop(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    mx, my = int((x1 - x0) * 0.25), int((y1 - y0) * 0.25)
    return img.crop((max(0, x0 - mx), max(0, y0 - my),
                     min(img.width, x1 + mx), min(img.height, y1 + my)))


def collect(samples: Path, manifest, arm: str, backend: str = "procedural"):
    """Paired crops: the render and the source photo it was built from."""
    from adflow.cache import file_identity
    from adflow.extraction import extract_product
    from adflow.formats import CAMPAIGN_FORMATS, plan_layout
    from adflow.utils import load_image

    images, labels, groups = [], [], []
    for product in manifest:
        folder = samples / f"{product.id}_{backend}_none_{arm}"
        render = folder / "ad_square_1080x1080.png"
        if not render.exists():
            continue
        source = load_image(product.image)
        ex = extract_product(source, file_identity(product.image))
        plan = plan_layout(CAMPAIGN_FORMATS[0], ex.cutout.size)

        with Image.open(render) as r:
            images.append(_product_crop(r.copy(), plan.product_box))
        labels.append(1)
        groups.append(product.id)

        # The matching region of the original photograph, at its own scale.
        images.append(_product_crop(source, ex.bbox))
        labels.append(0)
        groups.append(product.id)
    return images, np.array(labels), np.array(groups)


def score_arms(samples: Path, manifest, backend: str = "procedural") -> list[dict]:
    """One row per arm: accuracy with a binomial CI, or a note saying why not.

    Returned rather than printed so `analyse.py` can fold the result into
    `analysis.txt` alongside the metric sections. Nothing else reproduces these
    numbers, so a run that only prints them leaves the claim unevidenced.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from scipy import stats

    rows = []
    for arm in ARMS:
        imgs, y, groups = collect(samples, manifest, arm, backend)
        if len(set(groups)) < 4:
            rows.append({"arm": arm, "n": len(y), "note": "too few products"})
            continue
        pred = cross_val_predict(
            LogisticRegression(max_iter=2000, C=0.1),
            _features(imgs), y, groups=groups,
            cv=GroupKFold(n_splits=min(5, len(set(groups)))))
        hits = int((pred == y).sum())
        res = stats.binomtest(hits, len(y), 0.5)
        lo, hi = res.proportion_ci(0.95)
        rows.append({"arm": arm, "n": len(y), "accuracy": hits / len(y),
                     "ci": (float(lo), float(hi)), "p": res.pvalue})
    return rows


def report(rows: list[dict]) -> None:
    print("Composite vs photograph, grouped by product (chance = 0.50)")
    print(f"{'arm':<12}{'n':>5}{'accuracy':>10}{'95% CI':>18}{'p':>10}")
    for r in rows:
        if "accuracy" not in r:
            print(f"{r['arm']:<12}{r['n']:>5}  {r['note']}")
            continue
        lo, hi = r["ci"]
        print(f"{r['arm']:<12}{r['n']:>5}{r['accuracy']:>10.3f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>18}{r['p']:>10.1e}")

    print("\nLower accuracy = harder to tell from a photograph. Compare arms to"
          "\neach other, not against an absolute standard: the crops differ in"
          "\nprovenance as well as in compositing.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", default="experiments/samples")
    ap.add_argument("--backend", default="procedural")
    args = ap.parse_args()

    manifest = load_manifest(ROOT / "data" / "manifest.json")
    report(score_arms(Path(args.samples), manifest, args.backend))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
