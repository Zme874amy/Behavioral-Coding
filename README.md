# AutoMISC

Clinical MI coding toolkit with two pipelines:

- Annotation pipeline (AutoMISC-style): `python src/main.py`
- Human-label SFT pipeline: `python src/sft/main.py +sft_models=qwen7b sft_device=cuda`

```bash
bash scripts/env.sh setup && source scripts/env.sh   # once, then every session
```

Docs: [guide](docs/README.md) · [setup](docs/HF_SETUP.md) · [results](docs/RESULTS.md)
