# Re-run preparation (P0–P4): record, data, prompts, infrastructure, scorer

**Started** 2026-10-05 · **Status** P0–P4 done locally (CPU only; no GPU jobs);
P5–P7 not started · **Plan** [RERUN_PLAN.md](../RERUN_PLAN.md)

## The question

Can every Stage-1 number be produced from one documented, leak-checked,
reproducible pipeline? Before this, nothing read `data/splits/`, prompts were
unversioned, arms overwrote each other's files, and several design constants
had been chosen by looking at the test set.

## What changed

| Phase | Change | Check |
|---|---|---|
| P0 record | Correct stale docs; add the never-logged campaigns to EXPERIMENT_LOG; mark the NO-GO arms "dropped"; track run sidecars; GRPO ladder no longer uses panther/tabby | `expansion_table` regenerated with no number changed |
| P1 data | `src/schemes/` is the one home of every code list and mapping (9 copies removed); `eda.clean_apply` builds the cleaned-HLQC ablation copy (77 FI→FA, 6 T1 fixes) | v1 loaders and the datasets notebook unchanged (all checks pass). The CASAA MITI proxy moved slightly because Reframe is now dropped, not forced to NC: 0.246→0.247 and 0.335→0.339 |
| P2 prompts | Prompt set v2 (v1 bugs fixed, v1 frozen); MITI 4.2.1 and AnnoMI prompt families; `REGISTRY.yaml` with sha256; seeded per-fold exemplar draws | v1 renders byte-identically (28 renders); seed 42 reproduces the frozen exemplars |
| P3 infra | `baseline.folds` (manifest → train/test with namespaced `uid`), `baseline.rerun` (one path per cell), `local_arm --manifest` mode, `fold_stats` (rare/learnable/never-seen from training rows), `mlerp_run.slurm` + `submit_stage.py` + `conf/stages/stage1.yaml` | LOSO and seed-tied 5-fold each cover 821/821 MIV utterances once; CPU smoke test (Qwen2.5-0.5B, LOSO fold 0) passed end to end |
| P4 eval | `baseline.rerun_eval`: learnable/gold/tail macro-F1, never-seen, robust variants, session-level ICC, paired seed-matched comparison with a permutation test; CASAA scored in MITI space | reproduces `comparison.csv` exactly on the legacy `ft1mix_bare` files |

## Findings on the way

- **The test-informed rare-code list was not what the training data says.** The
  rule "< 20 rows or < 1% of the speaker's training labels" gives, for HLQC
  counsellors, SU, RF, RCP, GI, CO and DI. The hand-picked list (SU, EC, AF, GI)
  included AF and EC, which are common in HLQC but common in the test set too.
- **With 10 test sessions, the session permutation test is stricter than the
  bootstrap.** For synth v2 proto, the client gain has a bootstrap CI that
  excludes 0 ([0.012, 0.098]), and 2 of 3 seeds agree. Yet the sign-flip
  permutation gives p = 0.14. The re-run reports both. Pooled CV does not add
  test sessions (it is still the same 10), so this limit stays.
- **Run provenance was weaker than it looked.** Every MLeRP sidecar records git
  sha `3a04564` (a stale tree copied over by scp), and retrain sidecars record
  `seed: null`. Re-run jobs now refuse a dirty tree and record the sha, the tag
  and the dirty flag.
- **The MITI ceiling for a MISC model is low.** Even gold MISC codes mapped to
  MITI score 0.60 macro-F1 on CASAA, because MITI Q/NC rows have no MISC T2. Any
  MITI-space transfer number should be read against that ceiling.

## Not done yet (carried into the stages that need them)

- The Welivita eval rebuilt from stage-I-agreed labels (weak transfer test; after Stage 1).
- AnnoMI prompt wording against the paper: marked `pending-source-check` (the paper was unreachable).
- Rationale-prompt de-duplication and the `synth/judge.py` v3 columns (needed only by Stages 2 and 4).
- Learning-side constants (retriever, self-training, synthesis) switching to `fold_stats`: they matter only when Stage 4 rebuilds those arms.
- QLoRA settings for Gemma-3-12B are wired in the P5 pilot.

## How to run

```bash
PYTHONPATH=src python -m components.prompts.registry --check
python scripts/submit_stage.py conf/stages/stage1.yaml            # dry run (100 Qwen cells)
PYTHONPATH=src python -m baseline.rerun_eval --stage conf/stages/stage1.yaml
```
