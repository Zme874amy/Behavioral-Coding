# Re-run pipeline plan (refined from the user's outline)

## Status (2026-10-05)
| Phase | State |
|---|---|
| P0 record | done, except **push + merge to main (awaiting your OK)** |
| P1 data | done |
| P2 prompts | done, except the de-duplicated rationale prompt and the judge/verify fixes (Stages 2/4) and the AnnoMI wording check |
| P3 infrastructure | done for zs/fs/SFT-bare arms; GRPO/rationale/agentic/self-training entry points join with their stages |
| P4 evaluation | done, except the Welivita eval rebuild |
| P5 models | students swapped 2026-10-06; **third family = Phi-4**, chosen by the pilot ([doc](experiments/2026-10-06-student-pilot.md)); jobs run from the clean clone `/mnt/userdata4/jia-wen/rerun` in grpo-env; Qwen3.5-9B and Gemma-4-12B still to stage and pilot |
| P6–P7 | not started |

Details: [experiments/2026-10-05-rerun-prep.md](experiments/2026-10-05-rerun-prep.md).

## Context
Every past result used one inherited protocol (train HLQC gold → test MIV6.3A gold). The 2026-10-04/05 audits then found several problems:

| Area | Problems found |
|---|---|
| Labels | HLQC label problems (FI catch-all, under-coded change talk, 6 T1/T2 contradictions); the SR/CR line differs between coder groups |
| Data | AnnoMI tests overlap HLQC train sessions (15/21/44/53); 22 of 41 few-shot exemplars come from test-duplicated sessions |
| Design and seeds | test-set-informed design constants; most arms single-seed |
| Prompts | 29 unversioned prompts with two bugs (client T2/flat prompts say "counsellor"; spec YAML teaches ADWP/CON/DIR/RCWP) |
| Pipeline | nothing reads `data/splits/` yet; pooling HLQC+MIV collides on `corp_utt_idx`; some arms overwrite each other's results; CASAA is never scored and nothing scores in MITI space; GRPO checkpoint selection leaks (warm-start trained on the val fold); SFT has no checkpoint selection at all |
| Docs and repo | stale docs (THESIS_SOURCE, GRPO_TWOCALL numbers, teacher-screen headline built on the discredited HLQC gate); 18 unpushed commits on a GRPO-named branch; tracked-by-design files left untracked |

The user wants a solid, reproducible, traceable re-run.

**User decisions (2026-10-05):**
- report **both** MISC protocols equally (pooled CV and HLQC→MIV cold-start);
- **staged, core-first** scope;
- HLQC fixes as a **derived "cleaned" copy, used as an ablation only**;
- students: **Qwen2.5-7B + Gemma (3-12B / 4 class) + Llama-3.1-8B**; **swapped 2026-10-06** (user: too outdated) to Qwen3.5-9B (primary) + Gemma-4-12B + Phi-4 (third family chosen by the P5 pilot over OLMo-3-7B and Ministral-3-8B), with Qwen2.5-7B kept as a cold-start bridge (P5).

## Refined pipeline (the user's outline, reordered so each stage only depends on earlier ones)

```
P0 Record & repo hygiene ─▶ P1 Data prep ─▶ P2 Prompts ─▶ P3 Run infrastructure ─▶ P4 Evaluation harness
        ─▶ P5 Model selection ─▶ P6 Stage-1 core runs ─▶ gate ─▶ P7 Stage-2..4 (CoT / RL / augmentation)
Cross-cutting at every stage: validation (seeds), rare classes, prompt consistency, traceability, docs+commit
```

### P0. Fix the record and the repo (no GPU)
- **Git:**
  - push the 18 local commits;
  - merge `feat/two-call-grpo` into `main` (it now carries all the data/EDA work), then branch per stage (`rerun/p1-data`, …);
  - track the files `.gitignore` already whitelists: `outputs/baseline_eval/{compliance,coverage}.csv`, `outputs/grpo/{calibration_*,faithfulness_sc_grpo_seed0}.json`, and `data/manual/{AnnoMI_eval,Welivita_eval}.csv` (AnnoMI is public; Welivita CC BY-NC-SA, non-commercial OK).
- **Docs** (list from the audit):

| Doc | Fix |
|---|---|
| THESIS_SOURCE.md | refresh: stale pending jobs and seeds, recovery file "missing", Gemma/Qwen3.5 "not run" |
| GRPO_TWOCALL doc | numbers don't match BASELINE_RESULTS |
| EXPERIMENT_LOG | add missing rows: AutoMISC HLQC CV 168271, LM Studio runs, single-call SFT, 3-seed baseline, 2-adapter synthesis; fix the retrieval/classic counts and row order |
| Teacher screen | re-word the headline: HLQC gate invalid; Gemma-4-31B 0.583 on MIV |
| AUTOMISC_FT.md | the run was on HLQC, not MIV |
| EXPANSION_RESULTS | mark the "pending" self-training v2 round-2 / 32B rows as "dropped (NO-GO)" |
| BASELINE_RESULTS | Welivita is a MITI variant |
| SYNTHETIC_DATA | the "independent verifier" is the same 32B model |
| EXPERIMENTS.md / BASELINE.md | "not tuned on test" → disclose the test-informed constants |
| Split review | checkpoint statement: only GRPO selects on HLQC val; SFT uses the final epoch |
| AnnoMI transfer numbers | flag as including training sessions until re-run |

- **Script bug:** `scripts/submit_grpo_ladder.sh` still routes to the `panther` QOS (lines 41–48, 93), the one that strands GPU jobs.

### P1. Data prep
1. **Cleaning: the derived "cleaned" ablation copy**, `data/clean/misc.hlqc.gold.cleaned.csv`, built by a new `src/eda/clean_apply.py` from the rules in `src/eda/quality.py`. The original stays the default.
   - Standalone acknowledgements coded FI → FA (rule `quality.ACK`, manual p.22).
   - The 6 T1/T2 contradictions → T1 := group(T2).
   - No relabelling of change talk or SR/CR (those need humans). Document them as known noise.
   - A changelog CSV lists every changed row.
2. **Splits.** The manifests already exist (`src/eda/splits.py` → `data/splits/*.json`):
   - **MISC:** `misc_main` (cold-start) and `misc_pooled_cv` (LOSO headline, seed-tied 5-fold exploratory). Both protocols are reported for every core arm, per the user's decision.
   - **AnnoMI and Welivita:** grouped 5-fold CV.
   - **MITI:** `miti_scheme` (train Welivita(agreed)+MISC mapped → test CASAA 18 sessions).
   - Exclusions apply to any model that saw HLQC by any route.
   - **Fix:** `misc_main` claims HLQC-val checkpoint selection, which SFT doesn't do. Decide the SFT rule in P3 and update the manifest text.
3. **Scheme mapping in one place.** New `src/schemes/` package (`misc.py`, `miti.py`, `annomi.py`, `mappings.py`), the single source of truth.
   - Move in `automisc_ft.data.{COUNSELLOR,CLIENT}_GROUPS`, `eda.registry.HANDBOOK`/`WELIVITA_TO_MITI`, `eda.quality.MISC_TO_MITI`, `prep_casaa.MITI_TO_MISC`, `selftrain.ingest.MITI_TO_MISC_T2`, and the three `LABEL_ALIASES` copies plus `fewshot.T2_GROUPS`.
   - Everything else imports from it.
   - Unmapped codes return `None` explicitly; the silent "NC" default in `eda.split_eval` goes.

### P2. Prompts (consistent, versioned, scheme-specific)
- **Fix the bugs**, creating prompt set **v2**. Prompt v1 is frozen as-is, for the record.
  - client `t2.j2` and `flat.j2` say "counsellor's";
  - `specs/counsellor_t2.yaml` and `counsellor/flat.j2` use ADWP/CON/DIR/RCWP → ADW/CO/DI/RCW;
  - reconcile the CQ/OQ and ST wording with the MISC 2.5 manual: drop the extra-manual "always CQ if ambiguous" rule, or cite it as a deliberate choice;
  - typos;
  - `synth/judge.py` reads v2 scenario columns, so v3 windows are judged "(not specified)";
  - `synth/verify` context uses utterances, not volleys.
- **One prompt per scheme and setting**, since prompts must differ when the schema differs:

| Scheme | Inference prompt family | Used for |
|---|---|---|
| MISC 2.5 | two-call T1→T2 (bare, CoT) × 2 speakers = 8 templates; joint `t12` ×2 for single-call/GRPO | MIV, HLQC |
| MITI 4.2.1 (new) | single-tier counsellor (bare, CoT) = 2; codebook from the MITI manual | CASAA, Welivita-as-MITI |
| AnnoMI (new) | counsellor main behaviour (+subtype) and client talk type (bare) = 2 | AnnoMI own-scheme |

  Plus the shared `user_prompt.j2`, the rationale-distillation prompt (1, de-duplicated), the parser (1), the synthesis v4 prompts (generator 3 + segment 1 + judge 1), and the agent prompts (3, if Stage 4 keeps retrieval).
  - **Final total: ~25 active templates**, each listed in `src/components/prompts/REGISTRY.yaml` with id, version, scheme, setting and sha256.
  - Cross-scheme transfer tests can still run the MISC prompt, but they are labelled "MISC-prompt transfer". The same-schema runs use the scheme's own prompt.
- **"Final prompt" per setting**, stated in the registry:
  - zero-shot = `{scheme}/t*_bare`;
  - few-shot = the same + exemplar block;
  - SFT bare = `t*_bare` (training target = label);
  - CoT = `t*` (target = rationale + label);
  - single-call / GRPO = `t12`.
- **Traceability:** every result's `.meta.json` records the prompt ids and sha256, the exemplar-draw seed, the manifest name, design, fold and seed, the config hash and the git sha. Code goes to MLeRP from a **clean committed tree**: `git pull` of a tag, replacing scp onto a dirty tree. Each stage starts from a tag such as `rerun-p6-<date>`.
- **Exemplars:**
  - `fewshot.build_exemplars(seed=…, source=<manifest train side>)`, with ≥3 draws (seeds 42/1/2);
  - drawn per fold from the training side only (HLQC by default; in-domain MIV exemplars as a labelled arm);
  - exemplar sessions are dropped when scoring on the exemplars' own corpus;
  - rationales for exemplars come from the de-duplicated distillation prompt.

### P3. Run infrastructure (make the pipeline manifest-driven)
- **Fold loader:** new `src/baseline/folds.py`, `load_fold(manifest, design, fold, seed) -> (train_df, test_df, meta)`. It generalises `oof_t1._split` and the `cv_anchor` train-fold/predict-fold/report pattern.
  - All train/predict entry points accept `--manifest/--design/--fold` (`local_arm`, `sc_arm`/`grpo`, `grpo_tc`, `selftrain` retrain, `classic`).
- **ID collision:** namespace `corp_utt_idx` as `"<dataset>:<idx>"` when frames are pooled. This affects the rationale store, the OOF store, `cv_anchor` done/report, `local_arm.cmd_predict` resume and `eval.collect_paired`.
- **One naming scheme:** `results/<protocol>/<design>/<arm>/<student>/seed<S>/fold<k>.csv`, plus a pooled file and `.meta.json`.
  - This fixes the overwrites in `sc_arm`, `grpo_tc`, agentic stems and the retrain seed.
  - Separate adapter roots by protocol/design/fold/seed.
  - `save_strategy=no`; delete fold adapters after prediction, keeping only the merged predictions (disk quota).
- **Seeds:** unify on 42/1/2 for all arms. GRPO currently uses 0/1/2, so map or re-run; record which.
- **Checkpoint and hyperparameter rule:**
  - SFT: fixed epochs, no selection (state it).
  - GRPO: select on the HLQC val fold (0 of 7) only after the warm-start is retrained **without** that fold. This fixes the selection leak.
  - Under pooled CV, never select on MIV.
- **Test-informed constants become per-fold, from training data only:** `agentic.retriever.RARE_T2`, `agent_arm --tail-only` (remove it: it filters eval rows by gold label), `selftrain.select.RARE_DEFAULT`, `synth` `RARE_CODES` / `ontology.MIXES["eval"]`, `classic.TAIL`, `oof_t1.REFERENCE_T1_ACC`, `eval.SEED_NOISE_FLOOR`. The rule is "rare = < 1% or < 20 in the fold's training labels". Keep `v3_mivmix` only as a disclosed legacy arm.
- **SLURM:** one generic `scripts/mlerp_run.slurm` (ARM, STUDENT, MANIFEST, DESIGN, FOLD, SEED, STAGE) plus a submitter that expands a stage config into a job array respecting the 4-job cap. Add the missing `cv_anchor`-style pooled-report step.

### P4. Evaluation harness (extend `src/baseline/eval.py`)
- **Metrics:**

| Metric | Definition |
|---|---|
| Primary | **T2 macro-F1 per speaker over codes learnable in that fold's training data** (per-fold `TRAINABLE_CODES`, replacing the global HLQC list) |
| Secondary | accuracy, Cohen's κ, T1 accuracy, per-class F1 with CIs, **tail macro-F1** (codes < 20 in training), and a **never-seen** list (RCP, TS−: reported as N/A) |
| Added: session-level fidelity agreement | MI summary scores per session from predicted vs gold codes, compared by ICC/MAE. This is the use case (feedback to counsellors), and it is less sensitive to single-label noise |

  Session-level summary scores: R:Q, %CR, %OQ, %MIC (MICO/(MICO+MIIN)) and %CT for MISC; MITI R:Q, %CR and technical/relational proxies for MITI.
- **Robust variants** for known label noise, reported alongside, never replacing: FA+FI merged, SR+CR at T1 (reflection), and AnnoMI questions at T1.
- **Statistics:**
  - pooled out-of-fold scoring (821 MIV utterances);
  - session-cluster bootstrap 95% CI;
  - paired difference across matched seeds, folds and exemplar draws;
  - **a permutation test** (missing today; memory says headline F1 needs it);
  - verdict "real" only if the CI excludes 0 **and** the sign holds for ≥ 2/3 seeds.
- **Schema × domain grid** (each cell names its prompt and mapping):

| | Same schema | Different schema |
|---|---|---|
| Same domain | MISC on MIV (pooled CV) | — |
| Domain transfer | MISC: HLQC→MIV (cold-start); MITI: Welivita (written)→CASAA (spoken); AnnoMI own-scheme CV | MISC model → AnnoMI (T1-level, exclusions) / CASAA (MITI-mapped) / Welivita (MITI-mapped) |

- **Code to add:**
  - `casaa` and `miti` entries in `CROSS_SCHEME` (they're missing, so `_dscasaa` is never reported);
  - drop `hlqc_overlap` rows;
  - MITI-space scoring using `schemes.mappings`;
  - rebuild `Welivita_eval.csv` from the stage-I-agreed labels so it matches `welivita_own`.

### P5. Model selection
- **Students**, LoRA on MLeRP:

Swapped 2026-10-06 (the 2024-era list was outdated). All three are ungated (Apache-2.0, Phi-4 MIT) and load in MLeRP's grpo-env. Qwen3.5 and Gemma-4 are multimodal checkpoints used text-only; Phi-4 is text-only. The third family was picked by the P5 pilot, not on paper.

| Role | Student | Released | Params / bf16 on disk | Training mode (to confirm in the pilot) | Notes |
|---|---|---|---|---|---|
| **primary** (headline, full LOSO) | `Qwen/Qwen3.5-9B` | 2026-03 | 9.65B / ~19 GB | bf16 LoRA | thinks by default: `chat_template_kwargs` passes `enable_thinking=False`; `AutoModelForCausalLM` maps to `Qwen3_5ForCausalLM` |
| second family | `google/gemma-4-12B-it` | 2026 | 11.96B / ~24 GB | bf16 LoRA if it fits 40 GB, else QLoRA | `gemma4_unified` architecture: supported by MLeRP's grpo-env (transformers 5.14.1), not by the DSKS env (4.55), so all re-run jobs use grpo-env; thinking only with `<\|think\|>` in the system prompt, which ours never has |
| third family (**chosen by the P5 pilot**, robust zero-shot F1 0.364 vs OLMo-3 0.183; Ministral-3 failed compliance) | `microsoft/phi-4` | 2024-12 | 14.7B / ~28 GB | bf16 LoRA (31 GB peak in the pilot) | MIT; text-only `Phi3ForCausalLM`; staged at `/mnt/userdata4/jia-wen/models/phi-4` |
| bridge only | `Qwen/Qwen2.5-7B-Instruct` | 2024 | 7.6B / ~15 GB | bf16 LoRA (as before) | cold start, ft1mix_bare, 3 seeds prompt v2 + 1 seed prompt v1: ties every earlier number to the new pipeline |

Why no Llama: Meta's current generation (Llama 4) is MoE only (Scout 109B total), with no small dense model; Llama-3.1-8B (2024) is what was outdated. LoRA target modules must be restricted to the language model (no vision tower); the pilot checks the trainable-parameter count.

  - **Disk plan:** usage is 61 GB of a ~110 GB quota. The three new students add ~71 GB (bf16: 19 + 24 + 28), so stage one at a time with `scripts/stage_model.sh`, keep only the active student, and drop adapters after pooled prediction. The user approves any deletion.
  - **Pilot each new student:** 1 cold-start run, seed 42, to check the chat template, the answer parser, the compliance rate and the wall time. This also calibrates the per-run cost used below.
- **Teachers** (MLeRP Ollama shared store, no quota cost): Gemma-4-31B, Qwen3.6-27B, gpt-oss-120b; plus Qwen2.5-32B-AWQ (vLLM, already staged). Qwen3-VL is excluded (thinking build).
  - **New teacher gate, not on any test set:** few-shot (3 exemplar draws) on the HLQC val fold scored with the **robust** metrics (FA+FI merged, SR/CR at T1). That removes HLQC's convention penalty.
  - MIV and CASAA are report-only.
  - Teachers serve as in-context baselines (Stage 1), synthesis generators (Stage 4) and pseudo-labellers (Stage 4).
- **Hyperparameters:** a small grid per student (lr {5e-5, 1e-4, 2e-4} × epochs {2, 3}; LoRA r {16, 32} only for the primary student). Run it once, under the cold-start protocol, on the HLQC val fold, then freeze it for every protocol. It is never touched with MIV.

### P6. Stage 1: core runs (both protocols, 3 seeds, all students)
| Arm | Setting | Protocols | Seeds / draws |
|---|---|---|---|
| zero-shot | in-context, MISC prompt v2 | cold-start + pooled (same predictions, scored both ways) | 1 (greedy) |
| few-shot | in-context | both (exemplars per fold) | 3 exemplar draws |
| teachers zs/fs | in-context (Ollama) | both | 1 / 3 draws |
| SFT `ft1mix_bare` | 1-adapter mixed fine-tune | cold-start + **pooled LOSO** | 42/1/2 |
| SFT on cleaned HLQC | ablation (primary student only) | both | 42/1/2 |
| bridge: Qwen2.5-7B `ft1mix_bare` | old student, new pipeline | cold start | 42/1/2 (v2) + 42 (v1) |
| best synthesis (v2 proto, v3 hlqcmix) | data augmentation, existing files | both | 42/1/2 |
| classic (TF-IDF/RoBERTa) | cheap reference | both | 1 |

**Cost estimate** (calibrated by the P5 pilots):
- per student, SFT = 3 seeds × (1 cold-start + 10 LOSO) = 33 trainings;
- Stage 1 (`conf/stages/stage1.yaml`, all students enabled): 178 cells: 106 train an adapter (Qwen3.5 66, Gemma-4 18, Phi-4 18, bridge 4) and 72 are zs/fs (no training); plus the synthesis arms on the primary student.

Use **seed-tied 5-fold for the Gemma-4/Phi-4 exploratory pass** (3 × 5 = 15 per student). Run LOSO for whichever student and arms enter the headline table. This roughly halves the cost.

**Gate to Stage 2:** the Stage-1 table is complete, and checks pass (coverage 821/821 per pooled file, seeds complete, prompt hashes match the registry).

### P7. Later stages (each runs only if its gate passes; same protocols, seeds and rules)
- **Stage 2, reasoning (CoT):**
  - `ft1mix_rat` with gpt-4o rationales generated **per training fold** (MIV rows need new rationales; generate only for training-side rows of each fold);
  - Inf-CoT inference;
  - faithfulness probe on the best CoT arm.
  - Gate: CoT ≥ bare within the CI on the cold-start protocol (history says CoT hurt).
- **Stage 3, RL:**
  - single-call GRPO and two-call GRPO (mix×dec only; decoupled ≈ joint before);
  - "different reasoning styles": GRPO over the CoT arm with diversity across rationale formats;
  - dynamic teacher forcing only as a 1-arm check (null twice before).
  - Fix the GRPO val-fold leak first (P3). Gate: Stage 2 shows any CoT benefit, or RL is run purely as a negative-result replication on 1 protocol.
- **Stage 4, augmentation:**
  - **Synthesis v4** (P2 prompt fixes; topic mix and rare-code list from fold training data): compare **generator teachers** (Qwen2.5-32B vs the best gated Ollama teacher). Ollama is greedy-only, so it can't be used for likelihood scoring.
  - **Self-training**:
    - *offline*: pseudo-label the pool once with the gated teacher or student, then retrain;
    - *online*: K rounds with re-labelling by the current student, i.e. the planned v2 round 2.

    Run self-training only if a teacher clears the gate. The last result was negative (−0.047), from majority amplification.
  - **Rare-class remedies**, evaluated on the tail F1: class-weighted SFT loss, synthesis targeting fold-rare codes, and in-domain exemplars. Never-seen codes (RCP, TS−) are reported as synthetic-only.

## Cross-cutting rules (apply at every stage)
- **Validation:**
  - ≥ 3 matched seeds (42/1/2) for any claim; paired by seed, fold and exemplar draw;
  - session bootstrap and a permutation test;
  - one seed = exploratory, labelled as such.
- **Leakage:** manifests plus `variance_rules` / `exemplar_rules` / `leakage_rules`; per-fold statistics; no MIV selection; AnnoMI/CASAA exclusions for any HLQC exposure.
- **Prompts:** registry plus sha256 in every `.meta.json`; a scheme-specific prompt for same-schema evaluation; transfer runs labelled.
- **Documentation:**
  - each stage gets an EXPERIMENT_LOG row, a `docs/experiments/` doc and a commit plus tag;
  - generated tables come only from `baseline.eval` / `expansion_table`;
  - `THESIS_SOURCE` is regenerated at the end.

## Critical files
- **New:** `src/schemes/{misc,miti,annomi,mappings}.py`, `src/baseline/folds.py`, `src/eda/clean_apply.py`, `src/components/prompts/REGISTRY.yaml` (+ v2 templates under `templates/v2/`, `templates/miti/`, `templates/annomi/`), `scripts/mlerp_run.slurm`, `scripts/submit_stage.py`, `conf/stages/stage1.yaml`.
- **Edit:**
  - `src/baseline/{local_arm,sc_arm,grpo,grpo_tc,eval,fewshot,rationalize,oof_t1,cv_anchor}.py`
  - `src/agentic/{retriever,agent_arm}.py`
  - `src/selftrain/{select,label_pool,loop}.py`
  - `src/synth/{ontology,generate,judge,verify}.py`
  - `src/components/prompts/specs/counsellor_t2.yaml` (v2 copy)
  - `scripts/submit_grpo_ladder.sh`
  - the docs listed in P0.
- **Reuse:** `automisc_ft.data.assign_folds/load_manual`, `cv_anchor` report pattern, `eval._clustered_bootstrap_ci/_paired_tables/aggregate_seeds`, `eda.splits` / `eda.quality` / `eda.split_eval`, `scripts/stage_model.sh`, `selftrain.bootstrap`.

## Verification
- **P1–P4 (local, CPU):**
  - unit checks: the fold loader reproduces every manifest partition, with no train/test overlap and namespaced IDs unique;
  - `eda.splits` checks pass;
  - a smoke run of `local_arm` with Qwen2.5-0.5B on LOSO fold 0 end-to-end → pooled report → eval prints primary + robust + session metrics;
  - prompt registry hashes match the files;
  - the cleaned copy changelog lists exactly the rule-hit rows (77 FI→FA acknowledgements, 6 T1 fixes);
  - eval reproduces the published cold-start `ft1mix_bare` numbers from the existing CSVs. This is a regression check that P4 changed no old metric.
- **P5:** each student's pilot run completes with compliance ≥ 99% and its measured wall time is recorded; the teacher gate table is produced without reading MIV.
- **P6:**
  - every pooled file covers 821/821;
  - seeds complete;
  - `.meta.json` has prompt sha, manifest, fold and git tag;
  - the cold-start Qwen `ft1mix_bare` re-run lands within the old seed spread (0.673 ± ~0.01 T2 acc) given the prompt v2 change, or the difference is explained by the prompt fix alone (run 1 seed with prompt v1 as a bridge).
- **Each stage:** EXPERIMENT_LOG row + campaign doc + tagged commit; the notebook checks still pass.
