# P5 student pilot: choosing the third model family

**Started** 2026-10-06 · **Status** rule fixed, runs pending · **Plan** [RERUN_PLAN.md](../RERUN_PLAN.md) P5

## The question

Qwen3.5-9B (primary) and Gemma-4-12B were chosen as current, ungated, Apache-2.0
students. Which model should be the **third family**? The choice is made by a
pilot on data that is not the test set, so the thesis can justify it, rather
than on paper.

## Candidates

| Candidate | Released | Licence | Params | bf16 on disk | Notes |
|---|---|---|---|---|---|
| `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 2025-12 | Apache-2.0 | 8.9B | ~18 GB | multimodal checkpoint, used text-only; BF16 repo (default is FP8) |
| `allenai/Olmo-3-7B-Instruct` | 2025-11 | Apache-2.0 | 7.3B | ~15 GB | fully open data and training recipe |
| `microsoft/phi-4` | 2024-12 | MIT | 14.7B | ~29 GB | newest Phi instruct model (later Phi releases are reasoning/vision); does not fit the disk headroom next to the others |
| `Qwen/Qwen2.5-7B-Instruct` | 2024-09 | Apache-2.0 | 7.6B | cached | reference only: the previous student, not a candidate |

## Rule (fixed before any run; not changed after seeing results)

Data: HLQC gold only. Zero-shot runs on the GRPO validation fold (val_folds 7,
fold 0, seed 42: 369 rows). The LoRA check trains on the first 300 HLQC rows
outside that fold. **MIV6.3A and CASAA are never read.**

1. **Compliance gate:** ≥ 99% of rows give a parseable T1 and T2 label
   zero-shot, with prompt set v2.
2. **Training gate:**
   - the 1-epoch LoRA run (main-setting recipe, 1 adapter, mixed regime, bf16) finishes on one 40 GB A100;
   - the mean training loss over the last quarter of logged steps is below the first quarter;
   - LoRA sits only on language-model layers ("LoRA outside LM" = 0, or the target modules are fixed and rerun).
3. **Choice:** among candidates passing both gates, the highest **robust
   learnable T2 macro-F1** (mean of counsellor and client) zero-shot.
   Robust = FA+FI merged, and for counsellors SR/CR merged, removing HLQC's
   known convention noise (DATASETS.md section 4).
4. **Ties:** if the top two differ by < 0.02, take the cheaper one (disk, then
   seconds per utterance).
5. **None passes:** fall back to two families (Qwen3.5, Gemma-4) and say so.

Zero-shot F1 on 369 rows is a coarse signal. It is used only to rank
candidates for the third slot, never as a result.

## How to run

```bash
# MLeRP, from the clean clone /mnt/userdata4/jia-wen/rerun (grpo-env)
bash scripts/stage_model.sh allenai/Olmo-3-7B-Instruct
MODEL=/mnt/userdata4/jia-wen/models/Olmo-3-7B-Instruct sbatch scripts/mlerp_student_pilot.slurm
MODEL=Qwen/Qwen2.5-7B-Instruct STEP=zs sbatch scripts/mlerp_student_pilot.slurm
# locally, after copying outputs/student_pilot/ back
PYTHONPATH=src python -m baseline.student_pilot report
```

## Results

*Pending.*
