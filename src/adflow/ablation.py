"""Component switches, used for both the baseline and the ablation study.

Every number reported so far compares the pipeline against other settings of
itself, which cannot show that the pipeline is worth having. Turning components
off supplies the missing reference:

    full        everything on, the configuration under study
    no_shadow   Stage 7 casts no contact shadow
    no_rim      Stage 7 paints no rim light
    no_brand    Stage 5 ignores the Stage 3 palette; the procedural background
                becomes neutral grey and the generated prompt drops the colour
                scheme
    no_coord    Stage 5 and Stage 7 use different key lights, breaking the
                coordination Stage 6b exists to provide
    naive       all of the above at once: the product pasted onto an
                uncoordinated neutral background. This is the baseline, and it
                is what the pipeline has to beat.

`naive` is what direct compositing gives you without any of the stages above,
so the gap between it and `full` measures the pipeline's contribution.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

NEUTRAL = "#8a8a8a"


@dataclass(frozen=True)
class Ablation:
    shadow: bool = True
    rim: bool = True
    brand_palette: bool = True
    coordinated_light: bool = True

    @property
    def label(self) -> str:
        if all((self.shadow, self.rim, self.brand_palette, self.coordinated_light)):
            return "full"
        if not any((self.shadow, self.rim, self.brand_palette, self.coordinated_light)):
            return "naive"
        off = [n for n, v in (("shadow", self.shadow), ("rim", self.rim),
                              ("brand", self.brand_palette),
                              ("coord", self.coordinated_light)) if not v]
        return "no_" + "_".join(off)


FULL = Ablation()
VARIANTS: dict[str, Ablation] = {
    "full": FULL,
    "no_shadow": replace(FULL, shadow=False),
    "no_rim": replace(FULL, rim=False),
    "no_brand": replace(FULL, brand_palette=False),
    "no_coord": replace(FULL, coordinated_light=False),
    "naive": Ablation(False, False, False, False),
}
