# AdFlow: product-preserving advertisement generation

Generates a three-format ad campaign from a single product photograph. The
background is synthesised; the product is not. Stage 6 records the product
pixels before composition so Stage 9 can verify, pixel for pixel, that nothing
downstream altered them.

## About this repository

This is a published snapshot rather than a development history. It begins at a
single commit, made once the dataset provenance, licensing and dependency sets
had been settled, so the log records the state of the code and not the route to
it.

The dissertation the code was written for is in `docs/`. The `Section N`
references throughout this file point into it, and it is the place to look for
why the pipeline is built the way it is, what the measures do, and what the
results do and do not support.

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[matting,eval,experiments]"
```

`matting` pulls in rembg. Install it: GrabCut, the offline fallback, fails on
styled backgrounds in a way none of the fidelity metrics detect.

For the diffusion background and generative relighting, install a CUDA build of
torch first, then the extra:

```bash
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[generative]"
```

## Run

Edit `config.json`, then:

```bash
python run.py
```

```json
{
  "products": [
    {"image": "your_product.jpg", "name": "Product Name"},
    {"image": "data/products/camera_lens.jpeg", "name": "Camera Lens",
     "headline": "Optional", "tagline": "Optional"}
  ],
  "backend": "procedural",
  "seed": 42
}
```

`image` accepts a bare filename under `inputs/` or a path relative to the
repository root. Omit `headline` and `tagline` and Stage 3b generates them.
`seed` selects the lighting setup and the copy variant, so re-rolling it
changes both without touching anything else.

Each product gets a folder under `outputs/`:

```
ad_square_1080x1080.png     1:1, social feed
ad_story_1080x1920.png      9:16, stories
ad_banner_1920x1080.png     16:9, web
product_cutout.png          the Stage 2 mask, for inspecting extraction
profiles.json               product, brand, copy and extraction method
evaluation.json/.md         metrics
```

## Pipeline

| Stage | Module | Produces |
|---|---|---|
| 1 | `understanding.py` | product profile (category, audience, scene keywords) |
| 2 | `extraction.py` | RGBA cut-out and mask, the preservation contract |
| 3 | `brand.py` | palette, accent and neutral colours |
| 3b | `copywriting.py` | headline and tagline |
| 4 | `formats.py` | per-format layout plan |
| 5 | `background.py` | background at format resolution |
| 6 | `preservation.py` | reference pixels recorded before composition |
| 6b | `lighting.py` | one key-light direction shared across formats |
| 6c | `relight.py` | optional product relighting, the study variable |
| 7 | `compose.py` | finished ad |
| 8 | `campaign.py` | all three formats |
| 9 | `evaluate.py` | metric report |

Section 3 of the dissertation in `docs/` covers the contracts between stages;
Section 4 covers the measures and how they were validated.

## Experiments

The experiment dataset is `data/manifest.json`, not `inputs/`. Twenty-four
products stratified by background type, each with a recorded source and licence.

```bash
python -m adflow.dataset --validate
```

`data/README.md` documents the collection criteria. Four scripts maintain the
dataset, and all of them are derived from `data/manifest.json`, so the manifest
is the only thing to edit by hand:

| Script | Produces |
|---|---|
| `scripts/fetch_products.py` | Candidates from stock-photo APIs, recording provenance as it downloads |
| `scripts/curate.py` | Promotes a reviewed selection into the manifest |
| `scripts/make_transparent.py` | The constructed members of the `transparent` stratum |
| `scripts/make_overview.py` | `data/dataset_overview.png` |
| `scripts/make_cutouts.py` | `data/cutouts/`, the pre-extracted masks the hosted harness reads |

The overview figure is generated rather than assembled by hand because the hand
made one went stale without anyone noticing: it showed a product that had been
dropped from the study and was missing a stratum that had been added.

`experiments/README.md` accounts for what the sweeps leave behind, including why
two superseded results grids are kept alongside the current one.

## Browser harness

A local test site that drives the real pipeline from a browser. Nothing on it
is mocked: every image it shows came out of `run_pipeline` with the settings
printed beside it, and every number came out of Stage 9.

```bash
pip install -e ".[web]"          # fastapi, uvicorn, python-multipart
python webapp/app.py             # http://127.0.0.1:8000
```

It binds to loopback only and there is no deployed instance; run it yourself to
look at it. Install the `experiments` extra too if you want `/results` to work,
since that page reads the grid with pandas.

| Route | Purpose |
|---|---|
| `/` | Pick a dataset product or upload your own, choose backend, seed, relight strength and ablation arm, render all three formats, and read the metrics back beside the images |
| `/results` | The experiment grid read live from `experiments/results.csv`: the relight sweep, per-stratum aggregates and the paired contrasts |
| `/study` | The perceptual study, five two-alternative comparisons (below) |
| `/health` | Liveness check, for when you have it running behind something |
| `/runs`, `/samples` | Static mounts serving the renders the other pages link to |

What it writes, all of it gitignored:

| Path | Contents |
|---|---|
| `webapp/_runs/<id>/` | The three renders and the metric report for one `/` submission |
| `webapp/_uploads/` | Product images uploaded through `/` |
| `experiments/human_study.csv` | One row per `/study` trial (see below) |

Two design points worth knowing before you read the code. `/` runs the pipeline
synchronously rather than queueing, because how long a render actually takes is
part of what the harness is for: the procedural backend answers in a couple of
seconds, diffusers takes about 30 seconds per format cold and under two warm.
And study sessions live in memory, so a restart costs whichever rater was
mid-sitting and nothing else, since every trial is written to disk as it is
answered.

### Hosting it

`render.yaml` is a Render blueprint: dashboard, New, Blueprint, pick this
repository. `ADFLOW_DEPLOY=1` puts the app in hosted mode; locally none of it
applies and the harness behaves as it always did.

Everything below follows from one number. The free tier has 512 MB, and the
pipeline does not fit in it by default, so the hosted instance is a set of
concessions rather than the same thing on a different machine. Each was
measured, and each cost something worth stating.

**torch.** Stage 9 loads LPIPS whenever torch is importable, which takes one
render from ~350 MB to ~1.3 GB. `requirements-render.txt` leaves it out. The
cost is the LPIPS column: `perceptual_distance` returns None when the import
fails, so the hosted table shows a blank cell rather than a fabricated score.

**Matting.** Peak RSS for a single Stage 2 extraction:

| Path | Peak RSS |
|---|---|
| rembg, u2net | 729 MB |
| rembg, u2netp | 506 MB |
| GrabCut | 284 MB |
| pre-cut alpha | **69 MB** |

The cost is onnxruntime rather than the weights, so the small model does not
rescue it, and downscaling the input does nothing because u2net resizes to
320x320 internally anyway. `scripts/make_cutouts.py` writes `data/cutouts/`,
24 files and 7 MB, and the hosted site reads those as an alpha channel exactly
as the `transparent` stratum does. rembg is not installed there at all, since
installing it would only let a visitor kill the process. The masks are still
the ones rembg produced, so the render is unchanged; only the stage that
produced the mask differs. Uploads have no precomputed mask and fall to
GrabCut, which is weaker on busy backgrounds.

**Formats.** Peak scales with how many a run holds at once: 390 MB for three,
350 MB for two, 230 MB for one. `generate_campaign` and `run_pipeline` take an
optional `formats` subset, and the hosted form gains a selector so all three
stay reachable one render at a time. Seed and key light do not depend on the
subset, so a format rendered alone is identical to the same format rendered as
part of a campaign. `python run.py` locally still produces all three at once.

**The GPU backend.** Not available, so the hosted form drops it and `/generate`
forces `procedural` server-side rather than trusting the field. Section 5.6
reports the procedural backend scoring better on every fidelity measure anyway.

Uploads are capped at 8 MB. Measured on the deployed instance, a render takes
7-19 seconds depending on canvas, against 1-3 seconds locally, because the CPU
is shared. And the instance sleeps after 15 minutes idle, with the next visitor
waiting roughly a minute for the wake: point an uptime pinger at `/health` every
5-10 minutes before anyone you care about clicks the link.

Moving to an instance with more memory removes the need for all of this except
the GPU backend. The concessions are keyed off `ADFLOW_DEPLOY`, so lifting them
means changing that block rather than unpicking anything.

## Perceptual study

**This study was not run.** It is specified and implemented, and the write-up
treats the resulting absence of human evaluation as its main limitation. What
follows describes what the code does, not something that was carried out.

The automated metrics were twice caught measuring the wrong thing, so agreement
with human judgement is the only independent check on the corrected ones. The
study is five two-alternative comparisons, about 12 minutes per rater:

| Arm | Contrast | What it settles |
|---|---|---|
| `rim` | full vs no_rim | No metric can see the rim light at all |
| `relight` | none vs graded@0.5 | Whether the main variable is perceptible |
| `pipeline` | full vs naive | Whether the pipeline is worth having |
| `shadow` | full vs no_shadow | Whether `contact_grounding` predicts people |
| `identity` | none vs generative | Attention check, and the identity claim |

The first four ask which image looks more like a real photograph; the last asks
which shows the product more accurately. Those are different axes and are not
pooled. `identity` doubles as the screen: generative relighting removes 44
points of edge retention, so a rater near chance there was not looking, and
their other answers carry no information either.

Responses go to `experiments/human_study.csv` - a random session id, the trial,
the choice and the reaction time, with no personal data. `scripts/analyse.py`
applies the screen, reports preference rates with binomial CIs, and then
correlates the per-stimulus metric difference against the per-stimulus
preference. It also runs `scripts/discriminability.py` - the machine substitute
for the study, which did not work - so that result is recorded in
`experiments/analysis.txt` too instead of only on a terminal. That correlation is
why the data would be worth collecting: it is what would give the metrics
external validity, and equally what would falsify them.

If you do run it, note what `render.yaml` makes easy: `/study` is a live
human-data collection endpoint, and deploying the harness puts it on the public
internet. Running it with recruited participants requires ethics approval first,
through your institution's research ethics process. That the responses carry no
personal data does not exempt it.

## Tests

```bash
pytest
```

The suite pins the preservation contract as an executable assertion, along with
determinism, debris pruning thresholds, layout bounds and the relighting dial.
If those fail, reported fidelity numbers do not describe the code that produced
them.

## Licence

MIT, covering the software. See `LICENSE`.

The dissertation in `docs/` is covered too, being my own work.

The product photographs under `data/products/`, and the cut-outs derived from
them, are the exception: they are not mine to license. They are redistributed
under the Pexels License, and `data/manifest.json` records the source and
licence for each one separately.
