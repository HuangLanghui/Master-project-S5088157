# Experiment dataset

Reported results come from the products listed in `manifest.json`. `inputs/` is
a scratch folder for ad-hoc runs of `run.py` and is not read by the experiment
harness.

Validate before every run:

```bash
python -m adflow.dataset --validate
```

## Adding products

Put image files under `data/products/` and add one entry each:

```json
{
  "id": "sony_a6600",
  "image": "data/products/sony_a6600.jpg",
  "name": "Sony Alpha 6600",
  "category": "camera",
  "background": "white",
  "source": "https://unsplash.com/photos/XXXX",
  "licence": "Unsplash Licence",
  "notes": ""
}
```

| Field | Purpose |
|---|---|
| `id` | Stable key for result tables and output folders. Renaming one breaks comparability with earlier runs. |
| `background` | `transparent`, `white`, `plain` or `scene`. The main stratification variable: extraction difficulty dominates fidelity results, so pooling a white-background shot with a styled scene hides the effect. |
| `category` | Secondary stratification. Also feeds Stage 1 when no VLM is configured. |
| `source`, `licence` | Recorded per image at collection time. The validator rejects blank or placeholder licences. |

## Collection criteria

Twenty products, spread across `background` rather than concentrated in easy
white-background shots. The informative failures are in the `scene` stratum.

- **Resolution**: at least 512px on the short edge. Below that the product
  occupies too few pixels after the Stage 6 resample for the fidelity metrics
  to separate methods. The validator warns.
- **One unoccluded product per image.** Groups, and products held in a hand,
  break the single-object assumption in `_drop_debris`.
- **Vary the aspect ratio**: tall bottles, wide boxes, roughly square items.
  Layout rules branch on shape.
- **Vary the lighting direction in the source photo.** The relighting study is
  most informative where the product's original lighting disagrees with the
  generated scene.
- **At least 12 products carrying legible label text or a graphic mark.**
  `edge_retention` is computed from the product's own high-frequency content;
  a set of smooth unbranded bottles gives it nothing to measure, and the
  finding that generative relighting erases roughly half the strong edges
  cannot be supported on such a set.
- **Include specular materials** (glass, polished metal, gloss plastic)
  alongside matte ones, roughly half and half. The `graded` mode uses
  Lambertian shading, which assumes a diffuse surface; specular products are
  where that assumption breaks, and that boundary is a result to report. An
  all-matte set flatters the method and an all-glass set penalises it.
- **Licence**: CC0, Unsplash, Pexels, or a citable research set such as Amazon
  Berkeley Objects.

## Strata

The four axes are crossed, not summed: one coffee-bag photo can be `white`,
matte, branded and wide at once.

| Axis | Target split | Rationale |
|---|---|---|
| `background` | white 8, plain 5, scene 5, transparent 2 | Extraction difficulty dominates fidelity; report per stratum. |
| Material | matte 8, specular 8, mixed 4 | Probes where the Lambertian assumption holds. |
| Branding | 12+ with legible marks | Gives `edge_retention` something to measure. |
| Shape | tall 8, wide 5, square-ish 7 | Layout rules branch on aspect ratio. |

The `transparent` stratum is small but worth keeping: pre-cut PNGs give an
extraction-error-free reference, which lets fidelity loss be attributed to
compositing rather than matting. Hand-mask these; a matting model's output is
not error-free by definition.

## Categories that satisfy the quotas

Matte and strongly branded: coffee bag, cereal carton, book cover, canvas
backpack, skincare tube, candle jar with a paper label.

Specular: perfume bottle, aluminium can, wine bottle, stainless flask,
sunglasses, smartphone, watch.

Mixed or complex geometry: sneaker, camera, over-ear headphones, earbud case,
laptop, ceramic mug, cosmetic pot.

Avoid official brand product photography. Trademarked imagery is a problem in a
published dissertation generally, and a sharper one here: the study's main
finding is that a diffusion model fabricates brand marks on the products it
relights.
