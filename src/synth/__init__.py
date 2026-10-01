"""Synthetic-example generation pathway for AutoMISC (Direction 2).

A large open teacher model writes NEW labeled utterances conditioned on the
MISC 2.5 codebook — broad across codes, with rare codes oversampled — to enrich
the small student's training set. Each synthetic sample is a tiny conversation
(a short preceding exchange + a target utterance) so the existing context
builder sees real neighbours; only the target utterance carries a gold label.

Two stages:
  * `generate.py` — produce synthetic samples per target code.
  * `verify.py`   — round-trip re-code each target utterance and keep it only if
                    the label is recovered (quality gate), then emit an
                    augmented training CSV in the manual schema.

Targets are bare labels only: the FT-Rat result showed that imitated rationales
hurt, so no synthetic rationale is attached.
"""
