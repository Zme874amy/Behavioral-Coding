# Hugging Face setup (gated models + HPC)

Required before fine-tuning **Gemma** or other gated checkpoints.

## 1. Install dependencies

```bash
pip install -U "transformers>=4.51" peft accelerate huggingface_hub
```

## 2. Authenticate

On the machine that runs training (login node or GPU node):

```bash
huggingface-cli login
```

Or set `HF_TOKEN` in the environment (do not commit tokens to git).

## 3. Accept model licenses

While logged into the same HF account in a browser, open each model page and accept the license:

- [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) (maps to proposal `gemma-4-e4b`)
- [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) (fallback if Gemma 3n + LoRA fails)

`Qwen/Qwen2.5-7B-Instruct` is not gated.

## 4. Preflight (fail fast)

```bash
cd /path/to/localproject
PYTHONPATH=src python src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
PYTHONPATH=src python src/sft/preflight_hf.py --model google/gemma-3n-E4B-it
```

## 5. Run SFT with a model preset (CUDA HPC)

```bash
PYTHONPATH=src python src/sft/main.py \
  +sft_models=qwen7b \
  sft_device=cuda \
  hydra.run.dir=outputs/sft_runs/qwen7b_rich
```

Mac smoke test only:

```bash
PYTHONPATH=src python src/sft/main.py \
  +sft_models=qwen05b \
  sft_device=mps \
  training.train_subset_size=50 \
  evaluation.sample_size=20
```

## Proposal name → HF repo

| Proposal / LM Studio | HF repo for LoRA |
|---|---|
| `google/gemma-4-e4b` | `google/gemma-3n-E4B-it` |
| `qwen/qwen3.5-9b` | `Qwen/Qwen2.5-7B-Instruct` |
