#!/usr/bin/env python3
"""Turn results.csv into the tables and figures the write-up needs.

Two things are deliberate here.

Comparisons are paired. Lighting coherence varies far more between products
than any treatment moves it (sd 0.37 against effects of order 0.02), so an
unpaired comparison drowns the effect in between-product variance. Every
contrast below is computed within (product, seed, backend, format) and then
averaged, which is also what a reader should expect from a within-items design.

Effect sizes are reported next to p-values. With n=432 pairs almost anything
reaches significance; Cohen's dz and the share of pairs that improved say
whether the effect is worth acting on.

    python scripts/analyse.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

PAIR_KEY = ["product", "seed", "backend", "format"]
NUMERIC = ["ssim", "psnr_db", "lpips", "edge_retention", "hist_correlation",
           "lighting_coherence", "contact_grounding", "palette_consistency",
           "extraction_quality"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ablation" not in df:
        df["ablation"] = "full"
    return df


def paired(df: pd.DataFrame, column: str, key: list[str],
           split: str, a, b) -> dict:
    """Contrast two arms of `split` within each `key` group."""
    left = df[df[split] == a].set_index(key)[column]
    right = df[df[split] == b].set_index(key)[column]
    joined = pd.concat([left, right], axis=1, join="inner", keys=["a", "b"]).dropna()
    if joined.empty:
        return {}
    d = joined["b"] - joined["a"]
    t, p = stats.ttest_rel(joined["b"], joined["a"])
    return {
        "n": len(d),
        "mean_a": joined["a"].mean(),
        "mean_b": joined["b"].mean(),
        "delta": d.mean(),
        "sd": d.std(),
        "dz": d.mean() / d.std() if d.std() else float("nan"),
        "p": p,
        "improved": (d > 0).mean(),
    }


def _fmt(r: dict) -> str:
    if not r:
        return "no paired data"
    return (f"n={r['n']:<4} {r['mean_a']:.4f} -> {r['mean_b']:.4f}  "
            f"delta {r['delta']:+.4f}  dz {r['dz']:+.2f}  "
            f"p={r['p']:.1e}  improved {100 * r['improved']:.0f}%")


def relight_section(df: pd.DataFrame, out: Path) -> None:
    full = df[(df.ablation == "full")
              & (df.relight_mode.isin(["none", "graded"]))]
    if full.empty:
        return
    print("\n== Relighting: fidelity cost against scene-integration gain ==")
    rows = []
    for s in sorted(full.relight_strength.unique()):
        r = full[full.relight_strength == s]
        rows.append({
            "strength": s,
            "ssim": r.ssim.mean(), "ssim_sd": r.ssim.std(),
            "lpips": r.lpips.mean(),
            "edge": r.edge_retention.mean(),
            "coh": r.lighting_coherence.mean(), "coh_sd": r.lighting_coherence.std(),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nPaired, relight 0 -> 1:")
    print("  overall   ", _fmt(paired(full, "lighting_coherence",
                                      PAIR_KEY, "relight_strength", 0.0, 1.0)))
    print("  ssim cost ", _fmt(paired(full, "ssim",
                                      PAIR_KEY, "relight_strength", 0.0, 1.0)))
    for b in sorted(full.backend.unique()):
        sub = full[full.backend == b]
        print(f"  {b:<10}", _fmt(paired(sub, "lighting_coherence",
                                        PAIR_KEY, "relight_strength", 0.0, 1.0)))
    for s in sorted(full.stratum.unique()):
        sub = full[(full.stratum == s) & (full.backend == "diffusers")]
        print(f"  {s:<10}", _fmt(paired(sub, "lighting_coherence",
                                        PAIR_KEY, "relight_strength", 0.0, 1.0)))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(table.strength, table.coh, yerr=table.coh_sd / np.sqrt(len(full) / 5),
                marker="o", color="#c0392b", label="lighting coherence")
    ax.set_xlabel("relight strength")
    ax.set_ylabel("lighting coherence", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax.twinx()
    ax2.errorbar(table.strength, table.ssim, yerr=table.ssim_sd / np.sqrt(len(full) / 5),
                 marker="s", color="#2c3e50", label="SSIM")
    ax2.set_ylabel("SSIM (product fidelity)", color="#2c3e50")
    ax2.tick_params(axis="y", labelcolor="#2c3e50")
    ax.set_title("Relighting trades product fidelity for scene integration")
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png", dpi=160)
    plt.close(fig)


def ablation_section(df: pd.DataFrame, out: Path) -> None:
    base = df[(df.relight_strength == 0.0) & (df.relight_mode == "none")]
    arms = [a for a in base.ablation.unique() if a != "full"]
    if not arms:
        print("\n== Ablation: not run yet ==")
        return

    print("\n== Ablation: each arm against the full pipeline ==")
    key = ["product", "seed", "backend", "format"]
    summary = []
    for arm in ["no_shadow", "no_rim", "no_brand", "no_coord", "naive"]:
        if arm not in arms:
            continue
        sub = base[base.ablation.isin(["full", arm])]
        row = {"arm": arm}
        for metric in ("lighting_coherence", "contact_grounding",
                       "palette_consistency", "ssim"):
            r = paired(sub, metric, key, "ablation", "full", arm)
            row[metric] = r.get("delta", float("nan"))
            row[metric + "_p"] = r.get("p", float("nan"))
        summary.append(row)
        print(f"  {arm:<10} coherence "
              f"{_fmt(paired(sub, 'lighting_coherence', key, 'ablation', 'full', arm))}")
        print(f"  {'':<10} grounding "
              f"{_fmt(paired(sub, 'contact_grounding', key, 'ablation', 'full', arm))}")
        print(f"  {'':<10} palette   "
              f"{_fmt(paired(sub, 'palette_consistency', key, 'ablation', 'full', arm))}")

    if not summary:
        return
    tab = pd.DataFrame(summary).set_index("arm")

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(tab))
    ax.bar(x - 0.27, tab.lighting_coherence, 0.26, label="lighting coherence", color="#c0392b")
    ax.bar(x, tab.contact_grounding, 0.26, label="contact grounding", color="#27ae60")
    ax.bar(x + 0.27, tab.palette_consistency, 0.26, label="palette consistency", color="#2980b9")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, tab.index, rotation=20)
    ax.set_ylabel("change vs full pipeline")
    ax.set_title("Ablation: what each component contributes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ablation.png", dpi=160)
    plt.close(fig)


def generative_section(df: pd.DataFrame) -> None:
    """The generative arm, against its own resampling control.

    `resample` performs the down/upsample round trip the diffusion path is
    forced through by VRAM, with no model. Without it the detail loss cannot be
    attributed between the two.
    """
    d = df[(df.ablation == "full") & (df.backend == "procedural")]
    key = ["product", "seed", "format"]
    arms = [("none", 0.0), ("resample", 1.0), ("generative", 0.2)]
    if not all(len(d[(d.relight_mode == m) & (d.relight_strength == s)]) for m, s in arms):
        return

    print("\n== Generative relighting against a resampling control ==")
    for mode, s in arms:
        r = d[(d.relight_mode == mode) & (d.relight_strength == s)]
        print(f"  {mode + '@' + str(s):<16} SSIM {r.ssim.mean():.4f}  "
              f"LPIPS {r.lpips.mean():.4f}  edge {r.edge_retention.mean():.4f}  "
              f"PSNR {r.psnr_db.mean():.1f}")

    def contrast(a, b, col):
        A = d[(d.relight_mode == a[0]) & (d.relight_strength == a[1])].set_index(key)[col]
        B = d[(d.relight_mode == b[0]) & (d.relight_strength == b[1])].set_index(key)[col]
        j = pd.concat([A, B], axis=1, join="inner", keys=["a", "b"]).dropna()
        diff = j["b"] - j["a"]
        t_, p_ = stats.ttest_rel(j["b"], j["a"])
        sd = diff.std()
        return (f"n={len(diff):<4} delta {diff.mean():+.4f}  "
                f"dz {diff.mean() / sd if sd else float('nan'):+.2f}  "
                f"p={p_:.1e}  worse in {100 * (diff < 0).mean():.0f}%")

    print("  edge retention, paired:")
    print("    resample vs none      ", contrast(("none", 0.0), ("resample", 1.0), "edge_retention"))
    print("    generative vs none    ", contrast(("none", 0.0), ("generative", 0.2), "edge_retention"))
    print("    generative vs resample", contrast(("resample", 1.0), ("generative", 0.2), "edge_retention"))


# Which results.csv rows correspond to each arm of a study comparison.
# (ablation, relight_mode, relight_strength)
STUDY_ARMS = {
    "none_full": ("full", "none", 0.0),
    "none_no_rim": ("no_rim", "none", 0.0),
    "none_no_shadow": ("no_shadow", "none", 0.0),
    "none_naive": ("naive", "none", 0.0),
    "graded@0.5_full": ("full", "graded", 0.5),
    "generative@0.2_full": ("full", "generative", 0.2),
}

# Which metric each comparison is expected to move, for the validation step.
STUDY_METRIC = {
    "rim": "lighting_coherence",
    "relight": "lighting_coherence",
    "pipeline": "contact_grounding",
    "shadow": "contact_grounding",
    "identity": "edge_retention",
}


def study_section(df: pd.DataFrame, path: Path) -> None:
    """Rater preferences, and whether the metrics predict them.

    The preference rates are the smaller half of this. Human judgements are
    collected for external validity: two of the automated metrics were caught
    measuring the wrong thing, so agreement with people is the only independent
    check that the corrected ones measure anything real.

    Raters are screened on the identity comparison before anything is reported.
    Generative relighting removes 44 points of edge retention, so a rater near
    chance there was not looking at the images, and their other answers carry no
    information either.
    """
    if not path.exists():
        print("\n== Human study: no responses yet ==")
        print(f"  collect with the /study route, writes to {path}")
        return

    resp = pd.read_csv(path)
    if resp.empty:
        return
    print(f"\n== Human study: {len(resp)} trials, "
          f"{resp.session.nunique()} rater(s) ==")

    checks = resp[resp.comparison == "identity"]
    if len(checks):
        hit = checks.chose == checks.correct_side
        rate = hit.groupby(checks.session).mean()
        keep = set(rate[rate >= 0.7].index)
        print(f"  attention check: {100 * hit.mean():.0f}% correct overall; "
              f"{len(keep)}/{len(rate)} rater(s) pass at 70%")
        resp = resp[resp.session.isin(keep)]
    else:
        print("  attention check: no identity trials recorded")

    if resp.empty:
        print("  no rater passed the screen")
        return

    # `left`/`right` record which arm was shown where, so the preference is
    # recovered per trial rather than assuming a fixed layout.
    resp["chosen_arm"] = np.where(resp.chose == "left", resp.left, resp.right)

    print("\n  preference for the reference arm (none_full):")
    for comp in sorted(resp.comparison.unique()):
        sub = resp[resp.comparison == comp]
        wins = int((sub.chosen_arm == "none_full").sum())
        n = len(sub)
        if not n:
            continue
        res = stats.binomtest(wins, n, 0.5)
        lo, hi = res.proportion_ci(0.95)
        print(f"    {comp:<10} {wins}/{n} = {wins / n:.2f}  "
              f"95% CI [{lo:.2f}, {hi:.2f}]  p={res.pvalue:.3f}")

    print("\n  do the metrics predict the preferences?")
    for comp in sorted(resp.comparison.unique()):
        metric = STUDY_METRIC.get(comp)
        sub = resp[resp.comparison == comp]
        if metric is None or sub.empty:
            continue

        rows = []
        for (product, backend, fmt), grp in sub.groupby(
                ["product", "backend", "format"]):
            arms = {grp.left.iloc[0], grp.right.iloc[0]}
            other = next((a for a in arms if a != "none_full"), None)
            if other is None or other not in STUDY_ARMS:
                continue

            def value(arm):
                ab, mode, strength = STUDY_ARMS[arm]
                m = df[(df["product"] == product) & (df.backend == backend)
                       & (df.format == fmt) & (df.ablation == ab)
                       & (df.relight_mode == mode)
                       & (df.relight_strength == strength)][metric]
                return m.mean() if len(m) else np.nan

            delta = value("none_full") - value(other)
            pref = (grp.chosen_arm == "none_full").mean()
            if not np.isnan(delta):
                rows.append((delta, pref))

        if len(rows) < 4:
            print(f"    {comp:<10} too few cells to correlate ({len(rows)})")
            continue
        deltas, prefs = zip(*rows)
        if np.std(deltas) < 1e-9:
            print(f"    {comp:<10} {metric} is identical across cells; "
                  "the metric cannot see this component at all")
            continue
        if np.std(prefs) < 1e-9:
            print(f"    {comp:<10} every cell preferred the same arm; "
                  "no variance in preference to correlate")
            continue
        r, p = stats.pearsonr(deltas, prefs)
        print(f"    {comp:<10} {metric:<19} r={r:+.2f} p={p:.3f} "
              f"over {len(rows)} product/format cells")


def discriminability_section(samples: Path, backend: str) -> None:
    """Fold `discriminability.py` into this report.

    It is the only substitute for the perceptual study that was actually run,
    and it used to print to a terminal and nowhere else, which left the numbers
    quoted in the write-up with no artefact behind them. It is skipped rather
    than fatal when the renders or the optional dependencies are missing, since
    the metric sections above do not need either.
    """
    print("\n== Machine discrimination: composite vs photograph ==")
    if not samples.exists():
        print(f"  {samples} not found; render them with "
              "scripts/run_experiment.py --variants all")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from discriminability import report, score_arms
        from adflow.dataset import load_manifest
    except ImportError as exc:
        print(f"  skipped: {exc}")
        return

    try:
        rows = score_arms(samples, load_manifest(), backend)
    except ImportError as exc:                  # sklearn is an optional extra
        print(f"  skipped: {exc}")
        return
    report(rows)

    scored = [r for r in rows if "accuracy" in r]
    if len(scored) > 1:
        lo = min(scored, key=lambda r: r["accuracy"])
        hi = max(scored, key=lambda r: r["accuracy"])
        print(f"\n  range {lo['accuracy']:.3f} ({lo['arm']}) to "
              f"{hi['accuracy']:.3f} ({hi['arm']}), n={scored[0]['n']} per arm")


def extraction_section(df: pd.DataFrame) -> None:
    print("\n== Extraction quality by stratum ==")
    per = df.groupby(["stratum", "product"]).extraction_quality.first().reset_index()
    print(per.groupby("stratum").extraction_quality
          .agg(["count", "mean", "min"]).to_string(float_format=lambda v: f"{v:.3f}"))
    low = per[per.extraction_quality < 0.5]
    if len(low):
        print("  flagged for review:", ", ".join(low["product"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="experiments/results.csv")
    ap.add_argument("--out", default="experiments/figures")
    ap.add_argument("--study", default="experiments/human_study.csv")
    ap.add_argument("--samples", default="experiments/samples")
    ap.add_argument("--backend", default="procedural",
                    help="backend whose renders the discrimination check reads")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(Path(args.results))

    print(f"{len(df)} rows | {df['product'].nunique()} products | "
          f"seeds {sorted(df.seed.unique())} | backends {sorted(df.backend.unique())} | "
          f"ablation arms {sorted(df.ablation.unique())}")

    print("\n== Backend comparison (full pipeline, no relight) ==")
    base = df[(df.ablation == "full") & (df.relight_strength == 0.0)]
    print(base.groupby("backend")[["ssim", "lpips", "edge_retention",
                                   "lighting_coherence", "palette_consistency"]]
          .agg(["mean", "std"]).to_string(float_format=lambda v: f"{v:.4f}"))

    relight_section(df, out)
    ablation_section(df, out)
    generative_section(df)
    study_section(df, Path(args.study))
    discriminability_section(Path(args.samples), args.backend)
    extraction_section(df)
    print(f"\nfigures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
