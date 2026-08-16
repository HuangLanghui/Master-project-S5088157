"""Stage 2: product extraction.

Separates the product from its original background and returns an RGBA cut-out
plus a binary mask. Provider chain, first available wins:

    rembg     U^2-Net matting. Required for reported results; GrabCut fails on
              styled scenes in a way no fidelity metric detects.
    grabcut   OpenCV GrabCut initialised from a centred rectangle. Offline
              fallback, adequate on plain backgrounds only.
    alpha     trusts an existing alpha channel, e.g. a pre-cut PNG.

The mask produced here is the contract for Stage 6: these are the pixels the
pipeline promises never to regenerate. Check it with
metrics.extraction_quality before trusting downstream numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import cache
from .utils import log


@dataclass
class ExtractionResult:
    cutout: Image.Image        # RGBA, tight-cropped product
    mask: np.ndarray           # uint8 {0,255}, same size as cutout
    bbox: tuple[int, int, int, int]
    method: str


def extract_product(image: Image.Image,
                    cache_key: tuple | None = None) -> ExtractionResult:
    """Run the provider chain, optionally reusing a cached cut-out.

    Extraction does not depend on the seed, the background backend or the
    relight level, so in a grid sweep the same product is matted many times
    over. Pass cache_key (usually cache.file_identity(path)) to do it once.
    """
    rgba = image.convert("RGBA")

    # A source that already carries transparency has its mask supplied, not
    # estimated, so re-running a matting model would discard the given alpha
    # and reintroduce the error the caller took care to avoid. This is what
    # makes the `transparent` stratum an extraction-error-free reference; with
    # the chain in its usual order those inputs were silently re-matted.
    order = ((_try_alpha, _try_rembg, _try_grabcut)
             if _has_transparency(rgba) else (_try_rembg, _try_grabcut, _try_alpha))

    def produce() -> tuple[Image.Image, dict]:
        for fn in order:
            result = fn(rgba)
            if result is not None:
                return result.cutout, {"bbox": list(result.bbox), "method": result.method}
        raise RuntimeError("No extraction method succeeded")

    if cache_key is None:
        cutout, meta = produce()
    else:
        cutout, meta = cache.image_with_meta(
            "extraction", cache.digest("extract", *cache_key), produce)

    # The mask is the cut-out's own alpha, so it survives the PNG round-trip
    # rather than needing to be stored separately.
    mask = (np.array(cutout)[:, :, 3] > 8).astype(np.uint8) * 255
    result = ExtractionResult(cutout, mask, tuple(meta["bbox"]), meta["method"])
    log.info("Stage 2: extraction via %s (bbox=%s)", result.method, result.bbox)
    if result.method == "grabcut":
        # Reaching the fallback means rembg is not installed, since it is tried
        # first on any opaque input. Worth a warning rather than a log line: on
        # a styled background GrabCut returns the backdrop and the props, and
        # every fidelity metric downstream still reports near-perfect, because
        # the Stage 6 reference is this cut-out. The run looks fine and is not.
        log.warning("Stage 2: fell back to GrabCut. It fails on styled "
                    "backgrounds in a way the fidelity metrics cannot detect - "
                    "install rembg (pip install -e '.[matting]') before "
                    "trusting anything downstream, or check the cut-out by eye.")
    return result


def _has_transparency(rgba: Image.Image) -> bool:
    """True when the alpha channel carries real information, not a flat 255."""
    alpha = np.array(rgba)[:, :, 3]
    return bool(alpha.min() < 255 and (alpha > 8).any())


def _drop_debris(alpha: np.ndarray, keep_ratio: float = 0.15) -> np.ndarray:
    """Zero out blobs far smaller than the main subject.

    Both matting backends keep scene clutter (petals, reflections, glass
    shards) that is disconnected from the product. Anything under keep_ratio of
    the largest blob is treated as debris. The threshold is loose enough that
    genuine multi-part products, such as a bottle with a separated cap, survive.
    """
    try:
        import cv2
    except ImportError:
        return alpha
    binary = (alpha > 8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 2:  # background + at most one blob: nothing to prune
        return alpha
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {i + 1 for i, area in enumerate(areas) if area >= areas.max() * keep_ratio}
    if len(keep) == count - 1:
        return alpha
    cleaned = alpha.copy()
    cleaned[~np.isin(labels, list(keep))] = 0
    log.info("Stage 2: pruned %d debris blob(s) from mask", count - 1 - len(keep))
    return cleaned


def _finalise(rgba: np.ndarray, method: str) -> ExtractionResult | None:
    rgba = rgba.copy()
    rgba[:, :, 3] = _drop_debris(rgba[:, :, 3])
    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = rgba[y0:y1, x0:x1]
    mask = (crop[:, :, 3] > 8).astype(np.uint8) * 255
    return ExtractionResult(Image.fromarray(crop), mask, (int(x0), int(y0), int(x1), int(y1)), method)


def _try_rembg(rgba: Image.Image) -> ExtractionResult | None:
    try:
        from rembg import remove
    except ImportError:
        return None
    out = remove(rgba)
    return _finalise(np.array(out), "rembg")


def _try_grabcut(rgba: Image.Image) -> ExtractionResult | None:
    try:
        import cv2
    except ImportError:
        return None
    bgr = np.array(rgba.convert("RGB"))[:, :, ::-1].copy()
    h, w = bgr.shape[:2]
    rect = (int(w * 0.05), int(h * 0.05), int(w * 0.9), int(h * 0.9))
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    out = np.dstack([np.array(rgba.convert("RGB")), fg])
    return _finalise(out, "grabcut")


def _try_alpha(rgba: Image.Image) -> ExtractionResult | None:
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    # Only meaningful when the image genuinely has transparency.
    if alpha.min() == 255:
        # Opaque image and no other method available: keep everything.
        return _finalise(arr, "opaque-passthrough")
    return _finalise(arr, "alpha")
