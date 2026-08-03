# Legacy SFT run summaries (Qwen2.5-0.5B)

Summary artifacts from the earlier `outputs/sft_runs/qwen05b_full/` experiment,
kept here because `outputs/` is now gitignored and the checkpoint directory it
lived in was untracked to stop ~40MB of tokenizer and optimizer blobs growing
the repository.

| File | Source |
|---|---|
| `qwen05b_eval_metrics.json` | `outputs/sft_runs/qwen05b_full/eval/sft_evaluation/metrics.json` |
| `qwen05b_lodo_summary.json` | `outputs/sft_runs/qwen05b_full/sft_artifacts/lodo_summary.json` |
| `qwen05b_sft_metadata.json` | `outputs/sft_runs/qwen05b_full/sft_artifacts/sft_metadata.json` |

The full checkpoint, including the LoRA adapter, remains recoverable from git
history at commit `da3a59f`. The narrative results are in `docs/RESULTS.md`.

This run predates the Model Scale x Adaptation x Rationale Alignment grid and
is not part of it; see `docs/BASELINE_RESULTS.md` for the current numbers.
