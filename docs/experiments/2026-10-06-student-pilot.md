# P5 student pilot: choosing the third model family

**Started** 2026-10-06 · **Status** done: **Phi-4 is the third family** · **Plan** [RERUN_PLAN.md](../RERUN_PLAN.md) P5

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

**Deviation 1 (2026-10-06, after the Qwen2.5-7B reference and OLMo-3-7B
zero-shot runs, before any candidate was ranked).** Under the strict parser, the
compliance gate failed even for the reference model (94.9%; OLMo 89.4%). Every
strict failure but one was a valid code written as its full name ("Affirm",
"Facilitate", "Direct", "label: Support"). The parser now also resolves a code's
full name when it is unique among the allowed codes (`schemes.misc.code_from_name`;
the ambiguous client "Desire" stays unresolved unless the T1 group fixes it).
The change is applied identically to every candidate by re-parsing the saved raw
generations. Both figures are reported: "parseable % (strict)" and "parseable %"
(gate). The rule itself is unchanged. This also fixes every future zero-shot arm.

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

## Results (jobs 170650–170652, 170656; clean clone at dc53d7d / 91a5cf8, grpo-env)

HLQC validation fold (369 rows; sessions high_127, low_031), zero-shot, prompt v2,
plus a 1-epoch LoRA run on 300 training rows (150 steps). F1 = learnable T2
macro-F1. Robust = FA+FI merged (and SR/CR merged for counsellors).

| Student | parseable % strict | parseable % (gate) | counsellor F1 | client F1 | mean F1 | **robust mean F1** | T1 acc | s/utt | LoRA | loss first→last quarter | peak GB | LoRA off LM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| **microsoft/phi-4** (14.7B) | 99.5 | **99.5** ✓ | 0.243 | 0.405 | 0.324 | **0.364** | 0.596 | 3.00 | ok ✓ | 2.99→0.10 | 31.0 | 0 ✓ |
| allenai/Olmo-3-7B-Instruct | 89.4 | **100.0** ✓ | 0.137 | 0.159 | 0.148 | 0.183 | 0.458 | 1.01 | ok ✓ | 2.09→0.18 | 17.3 | 0 ✓ |
| mistralai/Ministral-3-8B-Instruct-2512-BF16 | 96.7 | **96.7** ✗ | 0.111 | 0.165 | 0.138 | 0.164 | 0.455 | 1.46 | ok | 1.32→0.09 | 21.2 | 168 ✗ |
| *Qwen/Qwen2.5-7B-Instruct (reference)* | 94.9 | 99.7 | 0.160 | 0.189 | 0.174 | 0.186 | 0.534 | 1.02 | – | – | – | – |

**Decision (rule step 3): Phi-4.** It passes both gates and has the highest robust
F1 (0.364 vs OLMo-3 0.183), far outside the 0.02 tie band. OLMo-3 passes both
gates but scores at the level of the old Qwen2.5-7B reference. Ministral-3 fails
the compliance gate: these are genuine format failures, not parser gaps. It
sometimes ignores "label only" and writes a chain of thought up to the token cap;
it answers "D" with no sign; and some answers are empty. Its default LoRA target
modules also attach 168 layers to the vision tower. That would be fixable, but
it is moot after the compliance failure.

Caveats:

- Phi-4 is the largest candidate (14.7B vs 7–9B). Part of its lead is size,
  which also puts it in the same class as Gemma-4-12B.
- It costs about 3× the zero-shot time per utterance (3.0 s vs 1.0 s).
- It needs 28 GB of disk.
- It trains in bf16 LoRA at 31 GB peak, which fits a 40 GB A100 with no QLoRA.
- It is the oldest model in the set (Dec 2024): the newest Phi instruct model,
  since later Phi releases are reasoning/vision models.
- 369 rows from 2 sessions rank the candidates; they are not a result.

Disk: Phi-4 stays staged (`/mnt/userdata4/jia-wen/models/phi-4`). OLMo-3, Ministral-3
and `gpt-oss-20b` were deleted with the user's approval (losers + gpt-oss-20b),
so the user directory went from 80 to ~94 GB. That leaves no room to stage
Qwen3.5-9B (~19 GB) and Gemma-4-12B (~24 GB) together. Stage 1 must therefore
stage one student at a time.

Raw outputs: `outputs/student_pilot/` (zero-shot CSVs with raw generations,
`*_lora.json` with loss curves, job logs, `report.md`).
