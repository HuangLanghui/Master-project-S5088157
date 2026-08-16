"""Two-alternative forced choice study, served from the same app.

Not run in the project this was written for, which is why the write-up treats
the absence of human evaluation as its main limitation. It is implemented and
its design is fixed, so it can be run as it stands.

Before doing that: this module collects data from people, and deploying the
harness with render.yaml puts the endpoint on the public internet. Recruiting
participants needs ethics approval from your institution first. Collecting
nothing identifiable, which is the case here, does not exempt it - what
triggers review is involving human participants, not the sensitivity of what
they leave behind.

Why 2AFC rather than a rating scale: the effects under test are small (the
relighting moves the metric by dz 0.24), and a five-point scale adds
between-rater variance that would swamp them. Asked to pick one of two, raters
agree with themselves and with each other far more.

Five comparisons, each targeting something the automated metrics leave open:

    rim         full vs no_rim        no metric can see the rim light at all
    relight     none vs graded@0.5    is the study's main variable perceptible
    pipeline    full vs naive         is the pipeline worth having
    shadow      full vs no_shadow     does contact_grounding predict people
    identity    none vs generative    attention check, and the identity claim

The first four ask which image looks more like a real photograph. The last asks
which shows the product more accurately. Those are different axes and must not
be pooled: an image can be more convincing as a photograph while showing a
product that is no longer the one in the catalogue.

`identity` doubles as the screen for inattentive raters. Generative relighting
removes 44 points of edge retention, so the correct answer is unambiguous;
anyone near chance on those trials should have their whole session discarded.

Responses land in experiments/human_study.csv, one row per trial, with no
personal data: a random session id, the trial, the choice, and the reaction
time. Reaction times are recorded to identify clicking-through, not to analyse
speed.

The analysis this is for is not the preference rate on its own. It is whether
the per-stimulus metric difference predicts the per-stimulus preference. That
is what would give the metrics external validity, and it is also what would
falsify them.
"""
from __future__ import annotations

import csv
import random
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "experiments" / "samples"
RESPONSES = ROOT / "experiments" / "human_study.csv"

FIELDS = ["session", "trial_index", "comparison", "product", "backend",
          "format", "left", "right", "chose", "correct_side", "rt_ms", "timestamp"]

REALISM = "Which looks more like a real photograph?"
ACCURACY = "Which shows the product more accurately?"

# (key, left-arm suffix, right-arm suffix, question, arm expected to win)
COMPARISONS = [
    ("rim", "none_full", "none_no_rim", REALISM, None),
    ("relight", "none_full", "graded@0.5_full", REALISM, None),
    ("pipeline", "none_full", "none_naive", REALISM, None),
    ("shadow", "none_full", "none_no_shadow", REALISM, None),
    # The only comparison with a defensible right answer, used as the screen.
    ("identity", "none_full", "generative@0.2_full", ACCURACY, "none_full"),
]

FORMATS = {"square": "1080x1080", "story": "1080x1920", "banner": "1920x1080"}
TRIALS_PER_COMPARISON = 6


def _available(suffix: str) -> list[tuple[str, str]]:
    """(product, backend) pairs that have a render for this arm.

    Empty when the stimulus directory is absent. It is gitignored, being some
    1.6 GB of renders, so a checkout that has never run the sweep does not have
    it and neither does a deployment. Returning nothing lets the caller say so;
    iterating a missing directory raised FileNotFoundError and took the page
    down with a 500 instead.
    """
    if not SAMPLES.is_dir():
        return []
    out = []
    for d in SAMPLES.iterdir():
        if not d.is_dir():
            continue
        for backend in ("procedural", "diffusers"):
            marker = f"_{backend}_{suffix}"
            if d.name.endswith(marker):
                out.append((d.name[: -len(marker)], backend))
    return out


def build_session() -> list[dict]:
    """One randomised block of trials.

    Stimuli are drawn per rater rather than fixed, so product-specific quirks
    average out across the sample instead of concentrating in whichever few
    products a fixed set happened to use.
    """
    trials: list[dict] = []
    for key, left_arm, right_arm, question, expected in COMPARISONS:
        shared = sorted(set(_available(left_arm)) & set(_available(right_arm)))
        if not shared:
            continue
        random.shuffle(shared)
        for product, backend in shared[:TRIALS_PER_COMPARISON]:
            fmt = random.choice(list(FORMATS))
            # Randomise which side each arm appears on, or raters who develop a
            # side preference would bias every comparison the same way.
            flip = random.random() < 0.5
            a, b = (right_arm, left_arm) if flip else (left_arm, right_arm)
            trials.append({
                "comparison": key, "product": product, "backend": backend,
                "format": fmt, "question": question,
                "left": a, "right": b,
                "correct_side": (None if expected is None else
                                 ("left" if a == expected else "right")),
            })
    random.shuffle(trials)
    return trials


def image_url(product: str, backend: str, arm: str, fmt: str) -> str:
    return f"/samples/{product}_{backend}_{arm}/ad_{fmt}_{FORMATS[fmt]}.png"


def record(session: str, index: int, trial: dict, chose: str, rt_ms: int) -> None:
    RESPONSES.parent.mkdir(parents=True, exist_ok=True)
    new = not RESPONSES.exists()
    with RESPONSES.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "session": session, "trial_index": index,
            "comparison": trial["comparison"], "product": trial["product"],
            "backend": trial["backend"], "format": trial["format"],
            "left": trial["left"], "right": trial["right"],
            "chose": chose, "correct_side": trial["correct_side"] or "",
            "rt_ms": rt_ms, "timestamp": int(time.time()),
        })


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
