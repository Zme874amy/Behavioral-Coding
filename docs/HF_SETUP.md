# Setup (MLeRP + Hugging Face)

## Environment

```bash
cd /mnt/userdata4/$USER/localproject
bash scripts/env.sh setup     # once  (DSKS + pip --user extras on MLeRP; .venv locally)
source scripts/env.sh         # every session
source scripts/env.sh
python -c "import hydra, torch, transformers, peft; print('ok')"
```

`source scripts/env.sh` puts the DSKS conda env on `PATH` and sets `PYTHONPATH=src`.
DSKS is read-only, so extras (`requirements-extras.txt`) install to `~/.local` via `pip --user`.

Override the auto-detected env: `export MLERP_DSKS=/apps/mambaforge/envs/dsks_2024.06`

## Hugging Face

```bash
python -m huggingface_hub.cli login
```

Gemma is gated — accept the license on huggingface.co while logged into the same account:
[gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it),
[gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it). Qwen is open.

## Run

```bash
source scripts/env.sh
python src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
python src/sft/main.py +sft_models=qwen7b sft_device=cuda \
  hydra.run.dir=outputs/sft_runs/qwen7b_rich
```

Batch (edit `--qos` if your account differs):

```bash
export PRESET=qwen7b
sbatch scripts/mlerp_sft.slurm
```

## Presets → HF repo

| Preset | HF repo |
|---|---|
| `qwen7b` | `Qwen/Qwen2.5-7B-Instruct` |
| `gemma3n_e4b` | `google/gemma-3n-E4B-it` |
| `gemma2_9b` | `google/gemma-2-9b-it` |
| `qwen05b` | `Qwen/Qwen2.5-0.5B-Instruct` (Mac smoke test) |
