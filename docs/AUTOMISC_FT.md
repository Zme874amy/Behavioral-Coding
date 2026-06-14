# AutoMISC + LoRA fine-tuning: fair head-to-head

Runs the **original AutoMISC hierarchical (T1 -> T2) annotator** as a zero-shot
baseline and compares it against the **same pipeline with two LoRA adapters**
(one for T1, one for T2). Same model, same original prompts, same prior-volley
context, same decoding, same per-speaker metrics - the only difference between
conditions is the adapter weights.

## Design

- **Data:** `data/manual/MIV6.3A_manual.csv` (821 human-consensus utterances,
  10 conversations). The parser is out of scope (E1); we score against the
  human-segmented consensus utterances, exactly as the paper does.
- **CV:** conversation-level 5-fold (no conversation leaks). Zero-shot runs once
  over all rows; fine-tuned predictions are pooled out-of-fold (every utterance
  predicted by adapters trained on the other folds).
- **Adapters:** per fold, a T1 adapter (predicts the Tier-1 grouping) and a T2
  adapter (predicts the fine-grained code; trained with the gold-T1 spec
  injected, and at inference uses the predicted-T1 spec).
- **Prompts:** the production templates in
  `src/components/prompts/templates/{counsellor,client}/{t1,t2}.j2` plus
  `user_prompt.j2` and the `specs/*_t2.yaml` code lists, rendered through the
  model's chat template.
- **Metrics:** per-speaker Tier-1 accuracy and Tier-2 macro-F1 / accuracy, plus
  a client Change/Sustain Talk breakdown.

## Run (CUDA / HPC)

```bash
bash scripts/check_data.sh          # must pass first
source scripts/env.sh
python src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
sbatch scripts/mlerp_automisc_ft.slurm
```

Artifacts land in `<run_dir>/automisc_ft_eval/`:
`predictions.csv`, `metrics.json`, `comparison.txt`, and per-fold adapters under
`<run_dir>/adapters/foldK/{t1,t2}/`.

## Smoke test (tiny model, CPU)

```bash
PYTHONPATH=src python src/automisc_ft/main.py \
  model.base_model=Qwen/Qwen2.5-0.5B-Instruct \
  limit_convs=2 cv.n_folds=2 annotator.num_context_turns=0 \
  inference.force_cpu=true inference.max_new_tokens=8 inference.max_input_len=1024 \
  training.num_train_epochs=1 training.max_length=1024 model.bf16=false \
  hydra.run.dir=outputs/automisc_ft/smoke
```

## Results (fill after the HPC run)

Paper reference: GPT-4.1, hierarchical, 3 context volleys (MIV6.3A clean set).

| Speaker | Tier | Metric | Zero-shot | Fine-tuned | Delta | Paper (GPT-4.1) |
|---|---|---|---:|---:|---:|---:|
| Counsellor | T1 | accuracy | | | | 0.82 |
| Client | T1 | accuracy | | | | 0.88 |
| Counsellor | T2 | macro-F1 | | | | 0.42 |
| Counsellor | T2 | accuracy | | | | 0.68 |
| Client | T2 | macro-F1 | | | | 0.41 |
| Client | T2 | accuracy | | | | 0.76 |

Client long-tail:

| Subset | Zero-shot F1 | Fine-tuned F1 | Delta | Support |
|---|---:|---:|---:|---:|
| Change Talk | | | | |
| Sustain Talk | | | | |
