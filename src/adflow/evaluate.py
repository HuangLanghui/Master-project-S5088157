"""Stage 9: Brand-consistency evaluation.

Quantifies the pipeline's two promises:

1. **Product fidelity** - inside each ad's product box, compare the
   rendered pixels against the Stage-6 reference (the only permitted
   transform was LANCZOS scaling + boundary alpha blending):
   * masked SSIM (scikit-image if installed, otherwise a windowed
     numpy implementation),
   * masked MSE / PSNR,
   * per-channel colour-histogram correlation.
2. **Palette consistency** - dominant colours of the full ad vs the brand
   palette, scored as mean nearest-neighbour distance in RGB space
   (normalised so 1.0 = identical palette, 0.0 = maximally distant).

Emits machine-readable JSON plus a human-readable Markdown report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .brand import BrandProfile
from .campaign import CampaignAsset
from .metrics import (contact_grounding, edge_retention, lighting_coherence,
                      perceptual_distance)
from .utils import kmeans_palette, log, save_json


# ---------------------------------------------------------------- fidelity
def _masked_mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    return float(np.mean((a[m].astype(float) - b[m].astype(float)) ** 2))


def _psnr(mse: float) -> float:
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
        return float(structural_similarity(a, b, channel_axis=2, data_range=255))
    except ImportError:
        return _ssim_numpy(a, b)


def _ssim_numpy(a: np.ndarray, b: np.ndarray, win: int = 8) -> float:
    """Windowed SSIM on the luma channel (fallback, no scipy/skimage)."""
    def luma(x):
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    x, y = luma(a.astype(float)), luma(b.astype(float))
    h, w = x.shape
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for i in range(0, h - win + 1, win):
        for j in range(0, w - win + 1, win):
            px, py = x[i:i + win, j:j + win], y[i:i + win, j:j + win]
            mx, my = px.mean(), py.mean()
            vx, vy = px.var(), py.var()
            cov = ((px - mx) * (py - my)).mean()
            vals.append(((2 * mx * my + c1) * (2 * cov + c2)) /
                        ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2)))
    return float(np.mean(vals)) if vals else 1.0


def _hist_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    corrs = []
    m = mask > 0
    for ch in range(3):
        ha, _ = np.histogram(a[..., ch][m], bins=32, range=(0, 255), density=True)
        hb, _ = np.histogram(b[..., ch][m], bins=32, range=(0, 255), density=True)
        if ha.std() == 0 or hb.std() == 0:
            corrs.append(1.0)
        else:
            corrs.append(float(np.corrcoef(ha, hb)[0, 1]))
    return float(np.mean(corrs))


# ---------------------------------------------------------------- palette
def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def palette_consistency(ad: Image.Image, brand: BrandProfile, k: int = 5) -> float:
    pixels = np.array(ad.convert("RGB")).reshape(-1, 3).astype(float)
    ad_pal, _ = kmeans_palette(pixels, k=k)
    brand_pal = np.stack([_hex_to_rgb(h) for h in brand.palette])
    d = np.linalg.norm(ad_pal[:, None, :] - brand_pal[None, :, :], axis=2).min(axis=1)
    max_d = np.linalg.norm([255, 255, 255])
    return float(np.clip(1 - d.mean() / max_d, 0, 1))


# ---------------------------------------------------------------- driver
def evaluate_campaign(assets: list[CampaignAsset], brand: BrandProfile,
                      out_dir: Path) -> dict:
    report: dict = {"formats": {}, "brand_palette": brand.palette}
    for asset in assets:
        x0, y0, x1, y1 = asset.record.canvas_box
        rendered = np.array(asset.image.convert("RGB"))[y0:y1, x0:x1]
        ref, mask = asset.record.reference_rgb, asset.record.reference_mask

        mse = _masked_mse(rendered, ref, mask)
        # SSIM operates on rectangular arrays; outside the product mask the
        # reference RGB is undefined (alpha=0), so fill it with the rendered
        # pixels - SSIM then measures the product region only.
        ref_for_ssim = ref.copy()
        ref_for_ssim[mask == 0] = rendered[mask == 0]
        metrics = {
            "background_provider": asset.background_provider,
            "relight": asset.relight,
            "ablation": asset.ablation,
            "product_fidelity": {
                "masked_mse": round(mse, 3),
                "masked_psnr_db": round(_psnr(mse), 2),
                "ssim": round(_ssim(rendered, ref_for_ssim), 4),
                "hist_correlation": round(_hist_corr(rendered, ref, mask), 4),
                "edge_retention": round(edge_retention(rendered, ref, mask), 4),
            },
            "scene_integration": {
                # What relighting is meant to buy, priced on the same run.
                "lighting_coherence": round(
                    lighting_coherence(np.array(asset.image.convert("RGB")),
                                       asset.record.canvas_box, mask), 4),
                # lighting_coherence reads interior pixels only, so it cannot
                # see the shadow. This does.
                "contact_grounding": round(
                    contact_grounding(np.array(asset.image.convert("RGB")),
                                      asset.record.canvas_box, mask), 4),
            },
            "palette_consistency": round(palette_consistency(asset.image, brand), 4),
        }
        # Same masking as SSIM: outside the product the "reference" holds the
        # source image's pixels while the render holds background, so an
        # unmasked comparison scores the background, not the product.
        lpips_score = perceptual_distance(rendered, ref_for_ssim)
        if lpips_score is not None:
            metrics["product_fidelity"]["lpips"] = round(lpips_score, 4)
        report["formats"][asset.format_key] = metrics
        log.info("Stage 9: %s - SSIM %.4f, palette %.3f", asset.format_key,
                 metrics["product_fidelity"]["ssim"], metrics["palette_consistency"])

    fmts = list(report["formats"].values())
    report["summary"] = {
        "mean_ssim": round(float(np.mean([f["product_fidelity"]["ssim"] for f in fmts])), 4),
        "mean_edge_retention": round(float(np.mean(
            [f["product_fidelity"]["edge_retention"] for f in fmts])), 4),
        "mean_lighting_coherence": round(float(np.mean(
            [f["scene_integration"]["lighting_coherence"] for f in fmts])), 4),
        "mean_contact_grounding": round(float(np.mean(
            [f["scene_integration"]["contact_grounding"] for f in fmts])), 4),
        "mean_palette_consistency": round(float(np.mean([f["palette_consistency"] for f in fmts])), 4),
    }
    lpips_scores = [f["product_fidelity"].get("lpips") for f in fmts]
    if all(s is not None for s in lpips_scores):
        report["summary"]["mean_lpips"] = round(float(np.mean(lpips_scores)), 4)
    save_json(report, out_dir / "evaluation.json")
    _write_markdown(report, out_dir / "evaluation.md")
    return report


def _write_markdown(report: dict, path: Path) -> None:
    lines = ["# Brand-consistency evaluation", "",
             f"Brand palette: {', '.join(report['brand_palette'])}", "",
             "| Format | Provider | Relight | SSIM | PSNR (dB) | LPIPS | Edge ret. | Light coh. | Palette |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key, m in report["formats"].items():
        pf = m["product_fidelity"]
        lpips = pf.get("lpips")
        lines.append(f"| {key} | {m['background_provider']} | {m.get('relight', 'none')} | "
                     f"{pf['ssim']} | {pf['masked_psnr_db']} | "
                     f"{'-' if lpips is None else lpips} | {pf['edge_retention']} | "
                     f"{m['scene_integration']['lighting_coherence']} | "
                     f"{m['palette_consistency']} |")
    s = report["summary"]
    lines += ["",
              "Fidelity (higher is better except LPIPS): SSIM, PSNR, LPIPS, edge retention.",
              "Scene integration (higher is better): lighting coherence, palette consistency.",
              "",
              f"**Mean SSIM:** {s['mean_ssim']} · "
              f"**Mean edge retention:** {s['mean_edge_retention']} · "
              f"**Mean lighting coherence:** {s['mean_lighting_coherence']} · "
              f"**Mean palette consistency:** {s['mean_palette_consistency']}"]
    path.write_text("\n".join(lines), encoding="utf-8")
