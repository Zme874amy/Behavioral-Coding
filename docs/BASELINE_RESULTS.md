# AutoMISC Baseline Reproduction — Results

Model: Azure OpenAI `gpt-4o`, temperature 0, hierarchical T1→T2 prompts, interval context of 5 turns.
Evaluation set: `data/manual/MIV6.3A_manual.csv` (human consensus labels).
Few-shot exemplars and fine-tuning data: `data/manual/HLQC_balanced_manual.csv` (held out, no leakage).

## T1 results

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Zero-shot + rationales (original AutoMISC) | all | 821 | 0.810 | 0.778 | 0.767 |
| Zero-shot + rationales (original AutoMISC) | counsellor | 580 | 0.786 | 0.724 | 0.721 |
| Zero-shot + rationales (original AutoMISC) | client | 241 | 0.867 | 0.782 | 0.859 |
| Zero-shot, no rationales | all | 821 | 0.790 | 0.755 | 0.780 |
| Zero-shot, no rationales | counsellor | 580 | 0.769 | 0.702 | 0.751 |
| Zero-shot, no rationales | client | 241 | 0.842 | 0.749 | 0.839 |
| Few-shot + rationales | all | 821 | 0.822 | 0.792 | 0.793 |
| Few-shot + rationales | counsellor | 580 | 0.798 | 0.739 | 0.748 |
| Few-shot + rationales | client | 241 | 0.880 | 0.806 | 0.881 |
| Few-shot, no rationales | all | 821 | 0.799 | 0.765 | 0.716 |
| Few-shot, no rationales | counsellor | 580 | 0.778 | 0.712 | 0.651 |
| Few-shot, no rationales | client | 241 | 0.851 | 0.761 | 0.845 |

## T2 results

| Condition | Scope | n | Accuracy | Cohen's kappa | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Zero-shot + rationales (original AutoMISC) | all | 821 | 0.664 | 0.637 | 0.423 |
| Zero-shot + rationales (original AutoMISC) | counsellor | 580 | 0.650 | 0.611 | 0.449 |
| Zero-shot + rationales (original AutoMISC) | client | 241 | 0.697 | 0.577 | 0.396 |
| Zero-shot, no rationales | all | 821 | 0.639 | 0.612 | 0.435 |
| Zero-shot, no rationales | counsellor | 580 | 0.621 | 0.580 | 0.461 |
| Zero-shot, no rationales | client | 241 | 0.685 | 0.574 | 0.410 |
| Few-shot + rationales | all | 821 | 0.680 | 0.655 | 0.458 |
| Few-shot + rationales | counsellor | 580 | 0.666 | 0.629 | 0.476 |
| Few-shot + rationales | client | 241 | 0.714 | 0.607 | 0.440 |
| Few-shot, no rationales | all | 821 | 0.669 | 0.643 | 0.485 |
| Few-shot, no rationales | counsellor | 580 | 0.648 | 0.610 | 0.532 |
| Few-shot, no rationales | client | 241 | 0.718 | 0.616 | 0.443 |

Per-code precision/recall/F1 reports: `outputs/baseline_eval/<condition>_<tier>_report.csv`.
