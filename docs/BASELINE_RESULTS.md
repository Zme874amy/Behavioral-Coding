# Model Scale x Adaptation x Rationale Alignment — Results

Grid: 2 model tiers x 4 adaptation arms (`ZS`, `FS`, `FT-Bare`, `FT-Rat`) x 2 inference styles (`Inf-Bare`, `Inf-CoT`).

Evaluation set: `data/manual/MIV6.3A_manual.csv` (821 human-consensus utterances). Few-shot exemplars and fine-tuning data both come from `data/manual/HLQC_balanced_manual.csv`, held out from evaluation. Exemplars, distilled rationales, and fine-tuning prompts are all frozen per context length so training and evaluation contexts always match.

`FT-Bare` trains on bare-label targets; `FT-Rat` trains on gpt-4o distilled rationale + label targets. Those rationales are post-hoc, generated conditioned on the gold label, so they may not be faithful to any reasoning that would independently produce the label.

`Macro-F1 (gold)` averages F1 only over codes that occur in the gold labels — the number comparable to the paper. `Macro-F1 (all)` also counts codes the model predicted but that never occur in gold (each contributing 0).

## Paper reference (GPT-4.1, hierarchical, 3 context volleys, clean set)

| Tier | Scope | Metric | Paper |
|---|---|---|---:|
| T1 | counsellor | accuracy | 0.82 |
| T1 | client | accuracy | 0.88 |
| T2 | counsellor | accuracy | 0.68 |
| T2 | counsellor | macro-F1 | 0.42 |
| T2 | client | accuracy | 0.76 |
| T2 | client | macro-F1 | 0.41 |

## T1 results — context = 3 volleys

### GPT-4o (frontier)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.804 | 0.770 | 0.791 | 0.791 |
| ZS-Bare | counsellor | 580 | 0.779 | 0.715 | 0.759 | 0.759 |
| ZS-Bare | client | 241 | 0.863 | 0.780 | 0.854 | 0.854 |
| ZS-CoT | all | 821 | 0.815 | 0.783 | 0.767 | 0.767 |
| ZS-CoT | counsellor | 580 | 0.797 | 0.738 | 0.729 | 0.729 |
| ZS-CoT | client | 241 | 0.859 | 0.767 | 0.845 | 0.845 |
| FS-Bare | all | 821 | 0.809 | 0.776 | 0.799 | 0.799 |
| FS-Bare | counsellor | 580 | 0.788 | 0.726 | 0.771 | 0.771 |
| FS-Bare | client | 241 | 0.859 | 0.772 | 0.855 | 0.855 |
| FS-CoT | all | 821 | 0.822 | 0.792 | 0.776 | 0.776 |
| FS-CoT | counsellor | 580 | 0.807 | 0.751 | 0.738 | 0.738 |
| FS-CoT | client | 241 | 0.859 | 0.769 | 0.852 | 0.852 |

## T1 results — context = 5 volleys

### GPT-4o (frontier)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.790 | 0.755 | 0.780 | 0.780 |
| ZS-Bare | counsellor | 580 | 0.769 | 0.702 | 0.751 | 0.751 |
| ZS-Bare | client | 241 | 0.842 | 0.749 | 0.839 | 0.839 |
| ZS-CoT | all | 821 | 0.810 | 0.778 | 0.767 | 0.767 |
| ZS-CoT | counsellor | 580 | 0.786 | 0.724 | 0.721 | 0.721 |
| ZS-CoT | client | 241 | 0.867 | 0.782 | 0.859 | 0.859 |
| FS-Bare | all | 821 | 0.799 | 0.765 | 0.716 | 0.716 |
| FS-Bare | counsellor | 580 | 0.778 | 0.712 | 0.651 | 0.651 |
| FS-Bare | client | 241 | 0.851 | 0.761 | 0.845 | 0.845 |
| FS-CoT | all | 821 | 0.822 | 0.792 | 0.793 | 0.793 |
| FS-CoT | counsellor | 580 | 0.798 | 0.739 | 0.748 | 0.748 |
| FS-CoT | client | 241 | 0.880 | 0.806 | 0.881 | 0.881 |

### Qwen2.5-7B-Instruct (SLM)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.599 | 0.535 | 0.512 | 0.460 |
| ZS-Bare | counsellor | 580 | 0.538 | 0.422 | 0.427 | 0.366 |
| ZS-Bare | client | 241 | 0.747 | 0.550 | 0.681 | 0.681 |
| ZS-CoT | all | 821 | 0.664 | 0.613 | 0.602 | 0.602 |
| ZS-CoT | counsellor | 580 | 0.616 | 0.521 | 0.517 | 0.517 |
| ZS-CoT | client | 241 | 0.780 | 0.650 | 0.771 | 0.771 |
| FS-Bare | all | 821 | 0.636 | 0.576 | 0.562 | 0.506 |
| FS-Bare | counsellor | 580 | 0.566 | 0.447 | 0.448 | 0.384 |
| FS-Bare | client | 241 | 0.805 | 0.674 | 0.790 | 0.790 |
| FS-CoT | all | 821 | 0.653 | 0.599 | 0.594 | 0.534 |
| FS-CoT | counsellor | 580 | 0.609 | 0.510 | 0.515 | 0.441 |
| FS-CoT | client | 241 | 0.759 | 0.611 | 0.752 | 0.752 |
| FT-Bare_Inf-Bare | all | 821 | 0.789 | 0.753 | 0.704 | 0.704 |
| FT-Bare_Inf-Bare | counsellor | 580 | 0.771 | 0.706 | 0.648 | 0.648 |
| FT-Bare_Inf-Bare | client | 241 | 0.834 | 0.719 | 0.816 | 0.816 |
| FT-Bare_Inf-CoT | all | 821 | 0.777 | 0.739 | 0.695 | 0.695 |
| FT-Bare_Inf-CoT | counsellor | 580 | 0.750 | 0.679 | 0.629 | 0.629 |
| FT-Bare_Inf-CoT | client | 241 | 0.842 | 0.731 | 0.826 | 0.826 |
| FT-Rat_Inf-Bare | all | 821 | 0.629 | 0.570 | 0.556 | 0.556 |
| FT-Rat_Inf-Bare | counsellor | 580 | 0.566 | 0.457 | 0.455 | 0.455 |
| FT-Rat_Inf-Bare | client | 241 | 0.780 | 0.620 | 0.759 | 0.759 |
| FT-Rat_Inf-CoT | all | 821 | 0.725 | 0.679 | 0.624 | 0.624 |
| FT-Rat_Inf-CoT | counsellor | 580 | 0.703 | 0.624 | 0.575 | 0.575 |
| FT-Rat_Inf-CoT | client | 241 | 0.776 | 0.600 | 0.722 | 0.722 |

## T2 results — context = 3 volleys

### GPT-4o (frontier)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.665 | 0.639 | 0.534 | 0.465 |
| ZS-Bare | counsellor | 580 | 0.645 | 0.606 | 0.529 | 0.494 |
| ZS-Bare | client | 241 | 0.714 | 0.608 | 0.540 | 0.439 |
| ZS-CoT | all | 821 | 0.687 | 0.661 | 0.529 | 0.446 |
| ZS-CoT | counsellor | 580 | 0.681 | 0.644 | 0.533 | 0.467 |
| ZS-CoT | client | 241 | 0.701 | 0.578 | 0.524 | 0.426 |
| FS-Bare | all | 821 | 0.676 | 0.651 | 0.544 | 0.473 |
| FS-Bare | counsellor | 580 | 0.657 | 0.620 | 0.541 | 0.505 |
| FS-Bare | client | 241 | 0.722 | 0.617 | 0.546 | 0.444 |
| FS-CoT | all | 821 | 0.699 | 0.675 | 0.536 | 0.452 |
| FS-CoT | counsellor | 580 | 0.702 | 0.668 | 0.576 | 0.504 |
| FS-CoT | client | 241 | 0.693 | 0.569 | 0.492 | 0.399 |

## T2 results — context = 5 volleys

### GPT-4o (frontier)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.639 | 0.612 | 0.499 | 0.435 |
| ZS-Bare | counsellor | 580 | 0.621 | 0.580 | 0.494 | 0.461 |
| ZS-Bare | client | 241 | 0.685 | 0.574 | 0.505 | 0.410 |
| ZS-CoT | all | 821 | 0.664 | 0.637 | 0.501 | 0.423 |
| ZS-CoT | counsellor | 580 | 0.650 | 0.611 | 0.514 | 0.449 |
| ZS-CoT | client | 241 | 0.697 | 0.577 | 0.487 | 0.396 |
| FS-Bare | all | 821 | 0.669 | 0.643 | 0.539 | 0.485 |
| FS-Bare | counsellor | 580 | 0.648 | 0.610 | 0.532 | 0.532 |
| FS-Bare | client | 241 | 0.718 | 0.616 | 0.546 | 0.443 |
| FS-CoT | all | 821 | 0.680 | 0.655 | 0.543 | 0.458 |
| FS-CoT | counsellor | 580 | 0.666 | 0.629 | 0.543 | 0.476 |
| FS-CoT | client | 241 | 0.714 | 0.607 | 0.542 | 0.440 |

### Qwen2.5-7B-Instruct (SLM)

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 (gold) | Macro-F1 (all) |
|---|---|---:|---:|---:|---:|---:|
| ZS-Bare | all | 821 | 0.381 | 0.341 | 0.256 | 0.209 |
| ZS-Bare | counsellor | 580 | 0.297 | 0.251 | 0.245 | 0.190 |
| ZS-Bare | client | 241 | 0.585 | 0.356 | 0.268 | 0.217 |
| ZS-CoT | all | 821 | 0.442 | 0.401 | 0.325 | 0.274 |
| ZS-CoT | counsellor | 580 | 0.384 | 0.321 | 0.301 | 0.248 |
| ZS-CoT | client | 241 | 0.581 | 0.442 | 0.352 | 0.305 |
| FS-Bare | all | 821 | 0.402 | 0.364 | 0.299 | 0.252 |
| FS-Bare | counsellor | 580 | 0.298 | 0.248 | 0.243 | 0.212 |
| FS-Bare | client | 241 | 0.651 | 0.501 | 0.359 | 0.292 |
| FS-CoT | all | 821 | 0.470 | 0.434 | 0.394 | 0.313 |
| FS-CoT | counsellor | 580 | 0.417 | 0.363 | 0.325 | 0.240 |
| FS-CoT | client | 241 | 0.598 | 0.466 | 0.469 | 0.381 |
| FT-Bare_Inf-Bare | all | 821 | 0.660 | 0.629 | 0.424 | 0.369 |
| FT-Bare_Inf-Bare | counsellor | 580 | 0.634 | 0.590 | 0.445 | 0.366 |
| FT-Bare_Inf-Bare | client | 241 | 0.722 | 0.581 | 0.402 | 0.373 |
| FT-Bare_Inf-CoT | all | 821 | 0.646 | 0.614 | 0.397 | 0.357 |
| FT-Bare_Inf-CoT | counsellor | 580 | 0.612 | 0.567 | 0.426 | 0.373 |
| FT-Bare_Inf-CoT | client | 241 | 0.726 | 0.581 | 0.365 | 0.339 |
| FT-Rat_Inf-Bare | all | 821 | 0.510 | 0.472 | 0.377 | 0.318 |
| FT-Rat_Inf-Bare | counsellor | 580 | 0.450 | 0.397 | 0.320 | 0.263 |
| FT-Rat_Inf-Bare | client | 241 | 0.656 | 0.477 | 0.437 | 0.379 |
| FT-Rat_Inf-CoT | all | 821 | 0.620 | 0.587 | 0.410 | 0.346 |
| FT-Rat_Inf-CoT | counsellor | 580 | 0.597 | 0.554 | 0.456 | 0.375 |
| FT-Rat_Inf-CoT | client | 241 | 0.676 | 0.483 | 0.361 | 0.313 |

## Instruction compliance

Whether each cell obeyed its inference instruction: `Inf-CoT` should produce a rationale, `Inf-Bare` should not. `Followed` is the share of utterances that did the requested thing.

Rows marked `constrained` were decoded through a JSON schema that forced the output shape, so the model had no opportunity to disobey and the figure is structural rather than behavioural.

### Context = 3 volleys

| Tier | Condition | Level | n | Rationale expected | Rationale emitted | Followed | Unparseable | Decoding |
|---|---|---|---:|---|---:|---:|---:|---|
| GPT-4o (frontier) | ZS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |

### Context = 5 volleys

| Tier | Condition | Level | n | Rationale expected | Rationale emitted | Followed | Unparseable | Decoding |
|---|---|---|---:|---|---:|---:|---:|---|
| GPT-4o (frontier) | ZS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | ZS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| GPT-4o (frontier) | FS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | constrained |
| Qwen2.5-7B-Instruct (SLM) | ZS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 1 (0.1%) | free |
| Qwen2.5-7B-Instruct (SLM) | ZS-Bare | T2 | 821 | no | 3 (0.4%) | 99.6% | 46 (5.6%) | free |
| Qwen2.5-7B-Instruct (SLM) | ZS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | ZS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 2 (0.2%) | free |
| Qwen2.5-7B-Instruct (SLM) | FS-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 26 (3.2%) | free |
| Qwen2.5-7B-Instruct (SLM) | FS-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 4 (0.5%) | free |
| Qwen2.5-7B-Instruct (SLM) | FS-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 27 (3.3%) | free |
| Qwen2.5-7B-Instruct (SLM) | FS-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 14 (1.7%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Bare_Inf-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Bare_Inf-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Bare_Inf-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Bare_Inf-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Rat_Inf-Bare | T1 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Rat_Inf-Bare | T2 | 821 | no | 0 (0.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Rat_Inf-CoT | T1 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | free |
| Qwen2.5-7B-Instruct (SLM) | FT-Rat_Inf-CoT | T2 | 821 | yes | 821 (100.0%) | 100.0% | 0 (0.0%) | free |

Per-code precision/recall/F1 reports: `outputs/baseline_eval/<tier>_<arm>_inf_<style>_ctx<N>_<t1|t2>_report.csv`.
