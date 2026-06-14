"""AutoMISC + LoRA fine-tuning fair head-to-head experiment.

Runs the original AutoMISC hierarchical (T1 -> T2) annotator as a zero-shot
baseline and compares it against the same pipeline with two LoRA adapters
(one for T1, one for T2), trained and evaluated with conversation-level
cross-validation on human-consensus labels.
"""
