"""Stage 4: Ad format planning.

Declares the campaign's target formats and computes a layout plan for each:
product placement, scale, safe margins and copy anchor. Layout is
rule-based and transparent (each rule is inspectable), which
matters for the dissertation's reproducibility argument.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .utils import log


@dataclass(frozen=True)
class AdFormat:
    key: str
    label: str
    width: int
    height: int
    channel: str


CAMPAIGN_FORMATS: tuple[AdFormat, ...] = (
    AdFormat("square", "Social feed 1:1", 1080, 1080, "Instagram/WeChat feed"),
    AdFormat("story", "Vertical story 9:16", 1080, 1920, "Stories/RedNote"),
    AdFormat("banner", "Landscape banner 16:9", 1920, 1080, "Web display"),
)


@dataclass
class LayoutPlan:
    fmt: AdFormat
    product_box: tuple[int, int, int, int]   # x0, y0, x1, y1 on canvas
    copy_anchor: tuple[int, int]
    copy_align: str                          # "left" | "center"
    headline_px: int
    margin: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fmt"] = asdict(self.fmt)
        return d


def plan_layout(fmt: AdFormat, product_wh: tuple[int, int]) -> LayoutPlan:
    pw, ph = product_wh
    margin = round(min(fmt.width, fmt.height) * 0.07)
    portrait_canvas = fmt.height >= fmt.width

    # Product occupies ~55% of the shorter canvas side, preserving aspect.
    target = min(fmt.width, fmt.height) * 0.55
    scale = min(target / pw, target / ph)
    w, h = round(pw * scale), round(ph * scale)

    if portrait_canvas:
        # Centre horizontally, sit in lower-middle band; copy above.
        x0 = (fmt.width - w) // 2
        y0 = round(fmt.height * 0.42)
        copy_anchor = (fmt.width // 2, margin + round(fmt.height * 0.06))
        align = "center"
    else:
        # Product right, copy left - classic Z-scan layout.
        x0 = fmt.width - w - margin
        y0 = (fmt.height - h) // 2
        copy_anchor = (margin, round(fmt.height * 0.30))
        align = "left"

    y0 = min(y0, fmt.height - h - margin)
    plan = LayoutPlan(
        fmt=fmt,
        product_box=(x0, y0, x0 + w, y0 + h),
        copy_anchor=copy_anchor,
        copy_align=align,
        headline_px=round(min(fmt.width, fmt.height) * 0.085),
        margin=margin,
    )
    log.info("Stage 4: %s layout - product %s, copy %s/%s",
             fmt.key, plan.product_box, plan.copy_anchor, align)
    return plan
