"""AI-AdFlow: a product-preserving advertisement generation pipeline.

Nine-stage pipeline:
  1. Product understanding      (adflow.understanding)
  2. Product extraction         (adflow.extraction)
  3. Brand profile construction (adflow.brand)
  4. Ad format planning         (adflow.formats)
  5. Background scene synthesis (adflow.background)
  6. Product preservation       (adflow.preservation)
  7. Composition                (adflow.compose)
  8. Campaign generation        (adflow.campaign)
  9. Brand-consistency eval     (adflow.evaluate)
"""

__version__ = "0.1.0"
