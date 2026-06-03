# Hugging Face + HPC

## MLeRP — two commands

```bash
cd /mnt/userdata4/jia-wen/localproject
bash scripts/env.sh setup
source scripts/env.sh
python3 src/sft/preflight_hf.py --model Qwen/Qwen2.5-7B-Instruct
export PRESET=qwen7b && sbatch scripts/mlerp_sft.slurm
```

Uses `/apps/mambaforge/envs/dsks_2025.08` (PyTorch, transformers) + `pip install --user` for hydra/peft.

If `dsks_2025.08` missing, edit `DSKS=` in `scripts/env.sh` to `dsks_2024.06`.

## HF login (once)

```bash
python3 -m huggingface_hub.cli login
```

Accept Gemma licenses on huggingface.co for gated models.

## Mac

Same `setup` / `source` — creates local `.venv` from `requirements.txt`.
