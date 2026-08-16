#!/usr/bin/env python3
"""Local test harness: drive the pipeline from a browser.

This runs the real pipeline, not a mock. Every image on the results page came
out of run_pipeline with the settings shown beside it, so what the page reports
is what the code does.

Two views:

    /          pick a dataset product or upload one, choose backend, seed,
               relight strength and ablation arm, then render and read the
               metrics back
    /results   the experiment grid: per-stratum aggregates and the paired
               contrasts, read live from experiments/results.csv

Start it with:

    python webapp/app.py            # http://127.0.0.1:8000

Generation is synchronous. A queue would hide how long a real render takes,
which is part of what the harness reports. The
procedural backend answers in a couple of seconds; diffusers takes about 30
seconds per format on a cold cache and under two on a warm one.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time
import uuid
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from fastapi import FastAPI, Form, UploadFile, File  # noqa: E402
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
import uvicorn  # noqa: E402

from adflow.ablation import VARIANTS  # noqa: E402
from adflow.dataset import load_manifest  # noqa: E402
from adflow.formats import CAMPAIGN_FORMATS  # noqa: E402
from adflow.pipeline import run_pipeline  # noqa: E402
from adflow.relight import RelightSpec  # noqa: E402
from adflow.utils import setup_logging  # noqa: E402

import study  # noqa: E402

RUNS = ROOT / "webapp" / "_runs"
RUNS.mkdir(parents=True, exist_ok=True)
UPLOADS = ROOT / "webapp" / "_uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

# Hosted mode. Locally this is all off and the harness behaves as before.
#
# The constraint that shapes it is memory. One render peaks at about 350 MB
# without torch, and at 1.3 GB with it, because Stage 9 loads LPIPS. A 512 MB
# host therefore has to run without torch, which costs the LPIPS column and
# nothing else: `perceptual_distance` already returns None when the import
# fails, so the metric table shows a blank rather than a fabricated score.
DEPLOYED = os.environ.get("ADFLOW_DEPLOY") == "1"
MAX_UPLOAD_BYTES = int(os.environ.get("ADFLOW_MAX_UPLOAD", 8 * 1024 * 1024))

# Stage 2 is the other thing that does not fit. Peak RSS for one extraction is
# 729 MB through rembg's u2net, 506 MB through the small u2netp, 284 MB through
# GrabCut and 69 MB reading a pre-cut alpha: the cost is onnxruntime itself, not
# the weights, so the small model does not rescue it. A hosted instance
# therefore serves the cut-outs made by scripts/make_cutouts.py, which are the
# masks rembg produced locally, and never loads a matting model for a dataset
# product. Uploads have no precomputed mask and fall to GrabCut.
CUTOUTS = ROOT / "data" / "cutouts"


def precut(product_path: str) -> Path | None:
    """The shipped cut-out for a manifest product, if there is one."""
    if not DEPLOYED or not product_path:
        return None
    candidate = CUTOUTS / f"{Path(product_path).stem}.png"
    return candidate if candidate.exists() else None

app = FastAPI(title="AdFlow test harness")
app.mount("/runs", StaticFiles(directory=str(RUNS)), name="runs")
SAMPLES = ROOT / "experiments" / "samples"
if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")

# Sessions live in memory: the study is a single-sitting task and losing
# an in-flight session to a restart costs one rater, not the dataset,
# which is already on disk after every trial.
SESSIONS: dict[str, list] = {}

FORMATS = [("square", "1080x1080", "1:1 social feed"),
           ("story", "1080x1920", "9:16 story"),
           ("banner", "1920x1080", "16:9 banner")]

STYLE = """
:root { color-scheme: light dark; --bg:#fbfbfc; --fg:#16181d; --muted:#5b6270;
        --line:#dfe3e8; --card:#fff; --accent:#1f6feb; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --fg:#e8eaed; --muted:#9aa3b2; --line:#262b33; --card:#161a21; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system,
       "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 32px 0 12px; }
.sub { color: var(--muted); margin: 0 0 28px; }
nav a { color: var(--accent); text-decoration: none; margin-right: 16px; }
form { background: var(--card); border: 1px solid var(--line); border-radius: 10px;
       padding: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px; }
label { display: block; font-size: 13px; color: var(--muted); margin-bottom: 5px; }
select, input[type=file], input[type=number] { width: 100%; padding: 7px 9px;
        border: 1px solid var(--line); border-radius: 6px; background: var(--bg);
        color: var(--fg); font-size: 14px; }
button { margin-top: 18px; padding: 9px 20px; border: 0; border-radius: 6px;
         background: var(--accent); color: #fff; font-size: 14px; cursor: pointer; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.ads { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
       gap: 18px; margin-top: 8px; }
.ads figure { margin: 0; }
.ads img { width: 100%; border: 1px solid var(--line); border-radius: 8px;
           background: var(--card); }
figcaption { font-size: 12px; color: var(--muted); margin-top: 6px; }
.note { color: var(--muted); font-size: 13px; }
pre { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
      padding: 14px; overflow-x: auto; font-size: 12.5px; }
"""


def page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{STYLE}</style></head><body><main>
<nav><a href="/">Generate</a><a href="/results">Experiment results</a><a href="/study">Study</a></nav>
{body}</main></body></html>""")


@app.get("/", response_class=HTMLResponse)
def index():
    try:
        products = list(load_manifest(ROOT / "data" / "manifest.json"))
    except Exception:
        products = []
    options = "\n".join(
        f'<option value="{p.image}">{p.name} ({p.background})</option>' for p in products)
    variants = "\n".join(f'<option value="{k}">{k}</option>' for k in VARIANTS)

    # The cache note is about the diffusers backend, which hosted mode does not
    # offer, so hosted gets its own timing instead of advice about a control
    # that is not on the page.
    timing_note = (" A render here takes roughly 7-19 seconds depending on the"
                   " canvas, the CPU being shared."
                   if DEPLOYED else
                   " The first diffusers render of a given prompt takes about"
                   " 30s per format; after that it is served from the on-disk"
                   " cache.")

    # Hosted, one format per request: three at once peak near 390 MB against a
    # 512 MB instance, one near 230 MB. The visitor picks, so all three are
    # still reachable, just not in a single render.
    format_field = ("<div><label>Format</label><select name=\"fmt\">"
                    + "".join(f'<option value="{k}">{label} &middot; {size}</option>'
                              for k, size, label in FORMATS)
                    + "</select></div>") if DEPLOYED else ""

    backends = ('<option value="procedural">procedural (fast, offline)</option>'
                if DEPLOYED else
                '<option value="procedural">procedural (fast, offline)</option>'
                '<option value="diffusers">diffusers (SDXL-Turbo, GPU)</option>')
    hosted_note = ("""
<p class="note"><strong>Hosted instance</strong>, on 512 MB and no GPU, which
constrains it in three ways. Backgrounds are procedural only, the diffusers
backend needing a GPU; Section 5.6 reports the procedural backend scoring
better on every fidelity measure regardless. LPIPS is blank, because computing
it would load torch and take one render from 350 MB to 1.3 GB. And Stage 2 is
not run for the products listed below: matting peaks at 729 MB, so their
cut-outs were computed offline and are read back here as an alpha channel, the
same trick the <code>transparent</code> stratum uses. Uploads have no
precomputed mask and fall back to GrabCut, which is weaker on busy backgrounds.
Everything after Stage 2 is the real pipeline, run per request.</p>"""
                   if DEPLOYED else "")

    return page("AdFlow test harness", f"""
<h1>AdFlow test harness</h1>
<p class="sub">Runs the real pipeline. {len(products)} dataset products available.</p>
{hosted_note}
<form action="/generate" method="post" enctype="multipart/form-data">
  <div class="grid">
    <div><label>Dataset product</label><select name="product">{options}</select></div>
    <div><label>or upload an image</label><input type="file" name="upload" accept="image/*"></div>
    <div><label>Background backend</label><select name="backend">{backends}</select></div>
    <div><label>Relight strength</label><select name="relight">
      <option value="0.0">0.00 (none)</option><option value="0.25">0.25</option>
      <option value="0.5" selected>0.50</option><option value="0.75">0.75</option>
      <option value="1.0">1.00</option>
    </select></div>
    <div><label>Ablation arm</label><select name="ablation">{variants}</select></div>
    <div><label>Seed</label><input type="number" name="seed" value="42"></div>
    {format_field}
  </div>
  <button type="submit">Generate</button>
</form>
<p class="note">Seed selects both the lighting setup and the copy variant.{timing_note}</p>""")


@app.post("/generate", response_class=HTMLResponse)
async def generate(product: str = Form(""), backend: str = Form("procedural"),
                   relight: float = Form(0.5), seed: int = Form(42),
                   ablation: str = Form("full"), fmt: str = Form(""),
                   upload: UploadFile | None = File(None)):
    setup_logging(verbose=False)

    formats = None
    if DEPLOYED:
        backend = "procedural"          # the only one this host can serve
        chosen = [f for f in CAMPAIGN_FORMATS if f.key == fmt] or [CAMPAIGN_FORMATS[0]]
        formats = chosen

    source = precut(product) or (ROOT / product if product else None)
    if upload is not None and upload.filename:
        blob = await upload.read()
        if len(blob) > MAX_UPLOAD_BYTES:
            return page("Too large", "<h1>Upload rejected</h1>"
                        f"<p>{len(blob) // 1024} kB exceeds the "
                        f"{MAX_UPLOAD_BYTES // 1024} kB limit for this instance. "
                        "Pick a dataset product, or run the harness locally "
                        "where there is no limit.</p>"
                        '<p><a href="/">Back</a></p>')
        source = UPLOADS / f"{uuid.uuid4().hex[:8]}_{Path(upload.filename).name}"
        source.write_bytes(blob)
    if source is None or not source.exists():
        return RedirectResponse("/", status_code=303)

    run_id = uuid.uuid4().hex[:10]
    out_dir = RUNS / run_id
    spec = RelightSpec("graded", relight) if relight > 0 else RelightSpec("none")

    started = time.time()
    try:
        report = run_pipeline(str(source), source.stem.replace("_", " ").title(),
                              out_dir=str(out_dir), seed=seed, backend=backend,
                              relight_spec=spec, ablation=VARIANTS[ablation],
                              formats=formats)
    except Exception as exc:  # noqa: BLE001
        return page("Failed", f"<h1>Generation failed</h1><pre>{exc}</pre>"
                              '<p><a href="/">Back</a></p>')
    elapsed = time.time() - started

    # Only the formats this run actually produced. A hosted run renders one.
    rendered = set(report["formats"])
    ads = "\n".join(
        f'<figure><img src="/runs/{run_id}/ad_{k}_{size}.png" alt="{label}">'
        f"<figcaption>{label} &middot; {size}</figcaption></figure>"
        for k, size, label in FORMATS if k in rendered)
    if DEPLOYED and len(rendered) < len(FORMATS):
        ads += ('<p class="note">One format per render here: three at once peak '
                'near 390 MB against this host\'s 512 MB, one near 230 MB. Pick '
                'another format and generate again to see it - the seed and the '
                'key light do not depend on the subset, so each format is '
                'identical to the one the full campaign would produce. '
                '<code>python run.py</code> locally renders all three at once.</p>')

    rows = ""
    for key, m in report["formats"].items():
        pf = m["product_fidelity"]
        rows += (f"<tr><td>{key}</td><td>{m['background_provider']}</td>"
                 f"<td class='num'>{pf['ssim']:.4f}</td>"
                 f"<td class='num'>{pf['masked_psnr_db']:.1f}</td>"
                 f"<td class='num'>{pf.get('lpips', float('nan')):.4f}</td>"
                 f"<td class='num'>{pf['edge_retention']:.4f}</td>"
                 f"<td class='num'>{m['scene_integration']['lighting_coherence']:.4f}</td>"
                 f"<td class='num'>{m['palette_consistency']:.4f}</td></tr>")

    return page("Result", f"""
<h1>{source.stem}</h1>
<p class="sub">{backend} &middot; relight {spec.label()} &middot; ablation {ablation}
 &middot; seed {seed} &middot; rendered in {elapsed:.1f}s</p>
<div class="ads">{ads}</div>
<h2>Metrics</h2>
<table><thead><tr><th>Format</th><th>Provider</th><th>SSIM</th><th>PSNR dB</th>
<th>LPIPS</th><th>Edge ret.</th><th>Light coh.</th><th>Palette</th></tr></thead>
<tbody>{rows}</tbody></table>
<p class="note">SSIM, PSNR and edge retention measure the product against the
Stage 6 reference recorded before compositing, so they price whatever the
relighting changed. Lighting coherence and palette consistency measure how well
the result sits in its background.</p>
<h2>Extraction</h2>
<div class="ads"><figure><img src="/runs/{run_id}/product_cutout.png" alt="cut-out">
<figcaption>Stage 2 cut-out. Inspect this first when the metrics look wrong:
a failed extraction leaves every fidelity number near-perfect.</figcaption>
</figure></div>
<p><a href="/">Generate another</a></p>""")


@app.get("/results", response_class=HTMLResponse)
def results():
    csv_path = ROOT / "experiments" / "results.csv"
    if not csv_path.exists():
        return page("Results", "<h1>Experiment results</h1>"
                               "<p class='note'>No results.csv yet. Run "
                               "<code>scripts/run_experiment.py</code>.</p>")
    import pandas as pd

    df = pd.read_csv(csv_path)
    for c in ("ssim", "lpips", "edge_retention", "lighting_coherence",
              "palette_consistency"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ablation" not in df:
        df["ablation"] = "full"

    full = df[(df.ablation == "full")]
    by_strength = (full.groupby("relight_strength")[
        ["ssim", "lpips", "edge_retention", "lighting_coherence"]].mean())
    sweep = "".join(
        f"<tr><td>{i:.2f}</td><td class='num'>{r.ssim:.4f}</td>"
        f"<td class='num'>{r.lpips:.4f}</td>"
        f"<td class='num'>{r.edge_retention:.4f}</td>"
        f"<td class='num'>{r.lighting_coherence:.4f}</td></tr>"
        for i, r in by_strength.iterrows())

    by_backend = (full[full.relight_strength == 0.0].groupby("backend")[
        ["ssim", "lpips", "lighting_coherence", "palette_consistency"]].mean())
    backends = "".join(
        f"<tr><td>{i}</td><td class='num'>{r.ssim:.4f}</td>"
        f"<td class='num'>{r.lpips:.4f}</td>"
        f"<td class='num'>{r.lighting_coherence:.4f}</td>"
        f"<td class='num'>{r.palette_consistency:.4f}</td></tr>"
        for i, r in by_backend.iterrows())

    arms = df[df.relight_strength == 0.0].groupby("ablation")[
        ["lighting_coherence", "palette_consistency", "ssim"]].mean()
    ablation = "".join(
        f"<tr><td>{i}</td><td class='num'>{r.lighting_coherence:.4f}</td>"
        f"<td class='num'>{r.palette_consistency:.4f}</td>"
        f"<td class='num'>{r.ssim:.4f}</td></tr>"
        for i, r in arms.iterrows())

    return page("Experiment results", f"""
<h1>Experiment results</h1>
<p class="sub">{len(df)} rows &middot; {df['product'].nunique()} products &middot;
seeds {sorted(df.seed.unique())} &middot; backends {sorted(df.backend.unique())}</p>

<h2>Relighting sweep (full pipeline)</h2>
<table><thead><tr><th>Strength</th><th>SSIM</th><th>LPIPS</th>
<th>Edge retention</th><th>Lighting coherence</th></tr></thead>
<tbody>{sweep}</tbody></table>

<h2>Backend (no relight)</h2>
<table><thead><tr><th>Backend</th><th>SSIM</th><th>LPIPS</th>
<th>Lighting coherence</th><th>Palette</th></tr></thead>
<tbody>{backends}</tbody></table>

<h2>Ablation arms</h2>
<table><thead><tr><th>Arm</th><th>Lighting coherence</th><th>Palette</th>
<th>SSIM</th></tr></thead><tbody>{ablation}</tbody></table>
<p class="note">Means only. The paired contrasts, effect sizes and p-values are
in experiments/analysis.txt, which is what the write-up should quote: between-product
variance is far larger than any treatment effect, so unpaired means understate
what a within-items design detects.</p>""")


@app.get("/study", response_class=HTMLResponse)
def study_intro():
    trials = study.build_session()
    if not trials:
        why = ("The comparisons are drawn from pre-rendered stimuli, about "
               "1.6 GB of them, which are not carried in the repository and so "
               "are not on this host either. The study is implemented and its "
               "design is fixed; it was not run within the project, and the "
               "write-up treats the absence of human evaluation as its main "
               "limitation."
               if DEPLOYED else
               "No stimuli found. Run <code>scripts/run_experiment.py "
               "--variants all</code> to render them into "
               "<code>experiments/samples/</code>.")
        return page("Study", f"""
<h1>Image comparison study</h1>
<p class="note">{why}</p>
<h2>What it would ask</h2>
<table><thead><tr><th>Arm</th><th>Contrast</th><th>What it settles</th></tr></thead>
<tbody>
<tr><td>rim</td><td>full vs no_rim</td><td>No metric can see the rim light at all</td></tr>
<tr><td>relight</td><td>none vs graded@0.5</td><td>Whether the main variable is perceptible</td></tr>
<tr><td>pipeline</td><td>full vs naive</td><td>Whether the pipeline is worth having</td></tr>
<tr><td>shadow</td><td>full vs no_shadow</td><td>Whether contact grounding predicts people</td></tr>
<tr><td>identity</td><td>none vs generative</td><td>Attention check, and the identity claim</td></tr>
</tbody></table>
<p class="note">The first four ask which image looks more like a real
photograph; the last asks which shows the product more accurately. Those are
different axes and are not pooled.</p>""")
    return page("Perceptual study", f"""
<h1>Image comparison study</h1>
<p class="sub">About 12 minutes, {len(trials)} comparisons.</p>
<form action="/study/start" method="post">
  <h2>What you will do</h2>
  <p class="note">You will see two advertisement images side by side and pick
  one. There are no right or wrong answers for most of them; go with your first
  impression rather than studying each pair.</p>
  <h2>Your data</h2>
  <p class="note">Only your choices and how long each took are recorded, under a
  random session number. No personal information is collected, and the data
  cannot be linked back to you. You may close the page at any point and your
  partial responses will simply be excluded.</p>
  <button type="submit">I understand, begin</button>
</form>""")


@app.post("/study/start")
def study_start():
    session = study.new_session_id()
    SESSIONS[session] = study.build_session()
    return RedirectResponse(f"/study/trial/{session}/0", status_code=303)


@app.get("/study/trial/{session}/{index}", response_class=HTMLResponse)
def study_trial(session: str, index: int):
    trials = SESSIONS.get(session)
    if trials is None:
        return RedirectResponse("/study", status_code=303)
    if index >= len(trials):
        return page("Thank you", """
<h1>Finished</h1>
<p class="sub">Thank you. Your responses have been recorded.</p>
<p class="note">You can close this page.</p>""")

    t = trials[index]
    left = study.image_url(t["product"], t["backend"], t["left"], t["format"])
    right = study.image_url(t["product"], t["backend"], t["right"], t["format"])
    return page("Comparison", f"""
<h1>{t["question"]}</h1>
<p class="sub">{index + 1} of {len(trials)}</p>
<form action="/study/answer" method="post" id="f">
  <input type="hidden" name="session" value="{session}">
  <input type="hidden" name="index" value="{index}">
  <input type="hidden" name="rt_ms" id="rt" value="0">
  <div class="ads">
    <figure><button type="submit" name="chose" value="left" style="all:unset;cursor:pointer;width:100%">
      <img src="{left}" alt="option A"></button><figcaption>Click to choose</figcaption></figure>
    <figure><button type="submit" name="chose" value="right" style="all:unset;cursor:pointer;width:100%">
      <img src="{right}" alt="option B"></button><figcaption>Click to choose</figcaption></figure>
  </div>
</form>
<script>
const t0 = performance.now();
document.getElementById("f").addEventListener("submit", () => {{
  document.getElementById("rt").value = Math.round(performance.now() - t0);
}});
</script>""")


@app.post("/study/answer")
def study_answer(session: str = Form(...), index: int = Form(...),
                 chose: str = Form(...), rt_ms: int = Form(0)):
    trials = SESSIONS.get(session)
    if trials is None or index >= len(trials):
        return RedirectResponse("/study", status_code=303)
    study.record(session, index, trials[index], chose, rt_ms)
    return RedirectResponse(f"/study/trial/{session}/{index + 1}", status_code=303)


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    # Loopback by default: the harness reads and writes the repository, so it
    # should not appear on a network unless someone says so. Hosts that hand
    # you a port in the environment (Render, Fly, Cloud Run) set both.
    host = os.environ.get("HOST", "0.0.0.0" if DEPLOYED else "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    if not DEPLOYED:
        print(f"http://{host}:{port}")
    # One worker, deliberately. Generation is synchronous and a single render
    # peaks near the memory a small instance has, so a second concurrent worker
    # is what would take the process down.
    uvicorn.run(app, host=host, port=port, log_level="warning", workers=1)
