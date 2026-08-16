"""Supplementary metrics for Stage 9.

The original metrics (SSIM, PSNR, histogram correlation, palette consistency)
all measure fidelity or brand match, both of which the pipeline is built to
maximise. None of them can express what relighting buys, so on their own they
can only ever show that relighting makes things worse.

    lighting_coherence   agreement between the product's shading direction and
                         the background's. Unsupervised, no reference needed.
    perceptual_distance  LPIPS against the untouched reference; tracks human
                         judgement better than PSNR. Needs torch and lpips.
    edge_retention       share of the reference's strong edges surviving.
                         Logotypes and package text are high-frequency detail
                         that neither SSIM nor LPIPS isolates.
    extraction_quality   screen for a failed Stage 2 cut-out, which none of the
                         fidelity metrics can detect.
"""
from __future__ import annotations

import numpy as np


def boundary_contrast(image_rgb: np.ndarray, mask: np.ndarray,
                      bbox: tuple[int, int, int, int], band: int = 3) -> float:
    """Mean colour difference across the silhouette boundary, in RGB units.

    A real silhouette separates two different things, so pixels just inside it
    differ from those just outside. A region carved out of a smooth backdrop
    has the same colour on both sides: the boundary exists in the mask but not
    in the photograph. frame_coverage and border_contact miss that case, since
    a mid-frame slab of backdrop neither reaches the edges nor fills the frame.

    Sampled from the source photograph rather than the cut-out, because some
    matting backends zero the RGB outside the mask, which would make the outer
    band uniformly black.

    Both bands skip a transition zone. Soft mattes ramp alpha over several
    pixels, and on a large photograph that ramp exceeds a fixed 3px band;
    sampling inside it reads background colour on both sides and reports
    near-zero contrast for a good cut-out. band therefore scales with the
    shorter image edge.
    """
    try:
        import cv2
    except ImportError:
        return float("nan")

    x0, y0, x1, y1 = bbox
    if (y1 - y0, x1 - x0) != mask.shape[:2]:
        return float("nan")

    # Work in source-image coordinates. The mask arrives cropped to its own
    # bounding box, so within that crop there is no room outside the silhouette
    # to sample - dilation cannot grow past the array edge, and for a
    # rectangular product the outer band would be empty entirely. Placing the
    # mask back on a full-size canvas gives the outer band real background to
    # measure against, all the way around the object.
    region = image_rgb.astype(np.float64)
    solid = np.zeros(image_rgb.shape[:2], np.uint8)
    solid[y0:y1, x0:x1] = (mask > 0).astype(np.uint8)

    step = max(band, min(y1 - y0, x1 - x0) // 120)
    k_skip = np.ones((2 * step + 1,) * 2, np.uint8)
    k_wide = np.ones((2 * step * 3 + 1,) * 2, np.uint8)

    # Skip the transition, then take a band of the same width beyond it.
    inner_edge = cv2.erode(solid, k_skip, iterations=1)
    inner_core = cv2.erode(solid, k_wide, iterations=1)
    outer_edge = cv2.dilate(solid, k_skip, iterations=1)
    outer_far = cv2.dilate(solid, k_wide, iterations=1)

    inner_band = (inner_edge > 0) & (inner_core == 0)
    outer_band = (outer_far > 0) & (outer_edge == 0)
    if inner_band.sum() < 64 or outer_band.sum() < 64:
        return float("nan")

    return float(np.abs(region[inner_band].mean(axis=0)
                        - region[outer_band].mean(axis=0)).mean())


def extraction_quality(mask: np.ndarray, bbox: tuple[int, int, int, int],
                       image_size: tuple[int, int],
                       image_rgb: np.ndarray | None = None) -> dict:
    """Unsupervised signals that a cut-out is not a product silhouette.

    The fidelity metrics cannot detect a failed extraction. SSIM compares the
    render against the Stage 6 reference, and that reference is the cut-out, so
    when Stage 2 returns the backdrop instead of the product every fidelity
    number stays near-perfect. A pooled mean over a dataset silently includes
    those cases.

    No ground-truth mask is required:

        frame_coverage   share of the photograph the mask claims. A product
                         rarely owns more than half the frame.
        border_contact   share of the bounding box's sides lying on the image
                         edge. Advertising subjects are not cropped by the
                         frame; a grabbed backdrop reaches several edges.
        bbox_fill        share of its own bounding box the mask fills.
                         Reported but not scored: a failed grab is a near-solid
                         rectangle, but so are a phone, a pouch and a can.
                         Scoring it flagged four correctly extracted products
                         and still let a real failure through.

    Returns the statistics plus quality in [0, 1]. Below about 0.5, inspect the
    cut-out before trusting anything computed from it.

    Two known limits: a light product on a light sweep has no boundary signal
    to measure, so it scores low while being correctly extracted; and a
    mid-frame grab that straddles real edges can still score high. Use this to
    select cases for review, not as an automatic gate.
    """
    x0, y0, x1, y1 = bbox
    width, height = image_size
    solid = mask > 0
    area = float(solid.sum())

    frame_coverage = area / max(1.0, float(width * height))
    bbox_fill = area / max(1.0, float((x1 - x0) * (y1 - y0)))

    margin = max(2, min(width, height) // 100)
    touching = sum((x0 <= margin, y0 <= margin,
                    x1 >= width - margin, y1 >= height - margin))
    border_contact = touching / 4.0

    # Each term is 1.0 in the healthy range and decays outside it. `bbox_fill`
    # is absent by design; see the docstring.
    coverage_ok = 1.0 if frame_coverage <= 0.45 else max(
        0.0, 1.0 - (frame_coverage - 0.45) / 0.35)
    border_ok = max(0.0, 1.0 - border_contact / 0.75)

    stats = {
        "frame_coverage": round(frame_coverage, 4),
        "bbox_fill": round(bbox_fill, 4),
        "border_contact": round(border_contact, 4),
    }

    contrast_ok = 1.0
    if image_rgb is not None:
        contrast = boundary_contrast(image_rgb, mask, bbox)
        if not np.isnan(contrast):
            stats["boundary_contrast"] = round(contrast, 2)
            # Below ~8 RGB units the two sides are the same material; ramp to
            # full credit by ~20, which every correctly matted product clears.
            contrast_ok = float(np.clip((contrast - 8.0) / 12.0, 0.0, 1.0))

    stats["quality"] = round(coverage_ok * border_ok * contrast_ok, 4)
    return stats


def _luma(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float64)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _shading_direction(luma: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """Direction brightness increases in, as a least-squares plane fit.

    An earlier version averaged the numerical gradient over the region. That
    was wrong in a way worth recording: the integral of a gradient over an area
    is fixed by its boundary, so the average was dominated by the step between
    the product and whatever lay just outside it, not by the shading inside.
    Painting a rim light on the silhouette edge moved the measured "product
    lighting direction" by more than the relighting under study did, and the
    ablation duly reported that removing the rim light improved coherence.

    Fitting a plane to the luminance of the interior pixels only measures what
    the name claims: the smooth trend across the region. Boundary pixels never
    enter the fit, so effects painted outside the mask cannot influence it.
    """
    ys, xs = np.nonzero(region)
    if len(xs) < 64:
        return 0.0, 0.0
    values = luma[ys, xs]
    design = np.column_stack([xs.astype(np.float64), ys.astype(np.float64),
                              np.ones(len(xs))])
    try:
        coeffs, *_ = np.linalg.lstsq(design, values, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    return float(coeffs[0]), float(coeffs[1])


def lighting_coherence(ad_rgb: np.ndarray, box: tuple[int, int, int, int],
                       mask: np.ndarray) -> float:
    """Agreement between the product's shading direction and the background's.

    1.0 = the two gradients point the same way (the product looks lit by the
    scene); 0.0 = they oppose (the classic "pasted-on" look). Returns 0.5 when
    either region is too flat for a direction to be meaningful - a flat
    background genuinely carries no lighting cue, and scoring that as agreement
    would reward the procedural baseline for saying nothing.
    """
    x0, y0, x1, y1 = box
    product = _luma(ad_rgb[y0:y1, x0:x1])

    # Interior pixels only. Boundary pixels carry the alpha blend against the
    # backdrop and any halo painted just outside the silhouette, neither of
    # which is the product's own shading.
    inside = (mask > 0).astype(np.uint8)
    try:
        import cv2
        margin = max(2, min(mask.shape) // 40)
        inside = cv2.erode(inside, np.ones((2 * margin + 1,) * 2, np.uint8))
    except ImportError:
        pass

    # Background sampled from a ring around the product, not the whole canvas:
    # copy and accent furniture elsewhere would pollute the estimate.
    pad_x, pad_y = (x1 - x0) // 2, (y1 - y0) // 2
    bx0, by0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    bx1, by1 = min(ad_rgb.shape[1], x1 + pad_x), min(ad_rgb.shape[0], y1 + pad_y)
    ring = _luma(ad_rgb[by0:by1, bx0:bx1])
    ring_mask = np.ones(ring.shape, bool)
    ring_mask[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] = False

    pgx, pgy = _shading_direction(product, inside)
    bgx, bgy = _shading_direction(ring, ring_mask)

    pn, bn = np.hypot(pgx, pgy), np.hypot(bgx, bgy)
    if pn < 1e-6 or bn < 1e-6:
        return 0.5
    cosine = (pgx * bgx + pgy * bgy) / (pn * bn)
    return float((cosine + 1.0) / 2.0)


def contact_grounding(ad_rgb: np.ndarray, box: tuple[int, int, int, int],
                      mask: np.ndarray) -> float:
    """How much darker the product's immediate surroundings are than the wider
    background: the trace a real object leaves on the surface it rests on.

    This exists because `lighting_coherence` cannot see the compositing at all.
    Fixing that metric restricted it to interior pixels, which is correct for
    measuring the product's own shading but leaves the shadow and the rim light
    - both painted outside the silhouette - unmeasurable. An ablation of those
    components against interior-only metrics can only ever return null.

    This measures a property of photographs rather than checking that a shadow
    was drawn, which would be circular: an object occludes ambient light and
    casts a contact shadow, so the surface touching it is darker than the same
    surface further away. A cut-out dropped onto a backdrop shows no such
    gradient and scores zero.

    Returns the relative darkening in [0, 1]. Around 0 means the ground below
    the product looks like the rest of the backdrop, which is the signature of
    a paste.

    The sampling region is the band directly beneath the product, sized to the
    shadow's actual reach. An earlier version compared a thin annulus hugging
    the silhouette against a wider one and reported the shadow as *reducing*
    grounding: the projected shadow is far longer than that annulus, so it
    darkened the comparison region rather than the sample, while the rim light
    brightened the sample. Both bands have to be chosen from where the effect
    physically lands.

    The reference is the median luminance of the remaining backdrop rather than
    its mean, so the copy block and the accent rule do not drag it around.
    """
    x0, y0, x1, y1 = box
    height, width = ad_rgb.shape[:2]
    product_h = y1 - y0

    below_top = y1
    below_bottom = min(height, y1 + int(product_h * 0.45))
    if below_bottom - below_top < 8:
        return float("nan")
    # Widen a little: a raking light shears the shadow sideways.
    left = max(0, x0 - int((x1 - x0) * 0.35))
    right = min(width, x1 + int((x1 - x0) * 0.35))

    luma = _luma(ad_rgb)
    below = luma[below_top:below_bottom, left:right]

    rest = np.ones((height, width), bool)
    rest[max(0, y0 - product_h // 4):below_bottom, left:right] = False
    if rest.sum() < 256 or below.size < 64:
        return float("nan")

    reference = float(np.median(luma[rest]))
    if reference <= 1e-6:
        return 0.0
    return float(np.clip((reference - below.mean()) / reference, 0.0, 1.0))


def edge_retention(rendered: np.ndarray, reference: np.ndarray,
                   mask: np.ndarray, percentile: float = 90.0) -> float:
    """Fraction of the reference's strongest edges that survive in the render.

    Thresholding at a percentile of the *reference's* own gradient makes this
    scale-free: it asks "is the detail that was there still there", not "is
    there detail".
    """
    inside = mask > 0
    if not inside.any():
        return 1.0

    def strength(img):
        gy, gx = np.gradient(_luma(img))
        return np.hypot(gx, gy)

    ref_edges, out_edges = strength(reference), strength(rendered)
    threshold = np.percentile(ref_edges[inside], percentile)
    if threshold <= 0:
        return 1.0

    strong = inside & (ref_edges >= threshold)
    if not strong.any():
        return 1.0
    # An edge counts as retained if it keeps at least half its magnitude.
    retained = out_edges[strong] >= (ref_edges[strong] * 0.5)
    return float(retained.mean())


def perceptual_distance(rendered: np.ndarray, reference: np.ndarray) -> float | None:
    """LPIPS (AlexNet backbone). Lower is more similar. None if unavailable.

    Returned as None rather than 0.0 when the dependency is missing, so a run
    without torch produces a blank cell in the results table instead of a
    fabricated perfect score.
    """
    try:
        import torch
        import lpips
    except ImportError:
        return None

    global _LPIPS
    try:
        net = _LPIPS
    except NameError:
        net = _LPIPS = lpips.LPIPS(net="alex", verbose=False)

    def to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        return float(net(to_tensor(rendered), to_tensor(reference)).item())
