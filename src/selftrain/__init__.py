"""Self-training / pseudo-labeling pathway for AutoMISC (Direction 1).

Label a large real *unlabeled* pool of MI transcripts with MISC 2.5 codes, gate
the confident predictions, add them to the human training set, retrain the small
Qwen2.5-7B student, and (optionally) iterate. The labeler is either the student
itself (1a, true self-training) or a stronger open model (1b, distillation).

Everything downstream of labeling reuses the existing fine-tune + eval harness:
augmented data is written in the manual-CSV schema so it flows through
`automisc_ft.data.build_tier_examples` and is scored by `baseline.eval` on the
fixed 821-utterance expert set.
"""
