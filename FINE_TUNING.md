# Fine-Tuning Architecture

## Overview

AutoMISC supports two fine-tuning workflows:

1. **Local Fine-Tuning** (provider: `local`)
   - Uses Hugging Face `transformers` + optional `peft` (LoRA)
   - Runs on your machine or compute cluster
   - Complete control over training process

2. **OpenAI Fine-Tuning** (provider: `openai`)
   - Submits jobs to OpenAI's fine-tuning API
   - Requires `OPENAI_API_KEY`
   - Managed training infrastructure

---

## Local Fine-Tuning Modes

### Full Fine-Tuning (use_peft: false)
- **All model parameters are updated** during training
- Pros: Potentially highest quality results
- Cons: Higher memory usage, slower, higher compute cost
- **When to use**: Large dataset, sufficient GPU memory, want best possible performance
- Example with GPT2:
  ```yaml
  fine_tuning:
    enabled: true
    provider: local
    base_model: gpt2
    use_peft: false
  ```

### LoRA Fine-Tuning (use_peft: true)
- **Only ~1-5% of parameters are updated** via Low-Rank Adaptation
- Pros: Fast, memory-efficient, cheaper, still effective
- Cons: Slightly reduced model capacity
- **When to use**: Limited GPU memory, want rapid iteration, working with large models
- Requires: `pip install peft`
- Example with Llama 2:
  ```yaml
  fine_tuning:
    enabled: true
    provider: local
    base_model: meta-llama/Llama-2-7b
    use_peft: true
    peft_r: 8
    peft_alpha: 16
    target_modules: [q_proj, v_proj, k_proj, v_proj]
  ```

---

## Module Structure

```
src/components/fine_tuning/
├── __init__.py                  # Main entry point
├── core.py                      # Orchestration logic
├── dataset_builder.py           # Data loading and preprocessing
├── local_trainer.py             # Hugging Face Transformers trainer
└── openai_trainer.py            # OpenAI API integration
```

### Key Components

- **`dataset_builder.py`**: Handles data loading from annotated CSV, building prompt-completion pairs
- **`local_trainer.py`**: Implements local training loop (full or LoRA)
- **`openai_trainer.py`**: Handles OpenAI API communication
- **`core.py`**: Routes to correct provider, orchestrates workflow

---

## Configuration

See `conf/config.yaml` for all available options. Key settings:

### Local Training
```yaml
fine_tuning:
  enabled: boolean                    # Enable/disable fine-tuning
  provider: local                     # Use local training
  base_model: string                  # Hugging Face model ID
  use_peft: boolean                   # Enable LoRA
  
  # Training hyperparameters
  num_train_epochs: float
  learning_rate: float
  per_device_train_batch_size: int
  # ... (see config.yaml for all options)
```

### OpenAI
```yaml
fine_tuning:
  enabled: boolean
  provider: openai
  base_model: string                  # OpenAI model (e.g., gpt-3.5-turbo)
  max_epochs: int
  batch_size: int
  # ... requires OPENAI_API_KEY env var
```

---

## Dataset Format

Training data comes from annotated utterances in `data/annotated/`.

**Prompt-Completion Pair Format:**

```
Prompt:
  Speaker: counsellor
  
  Utterance: "That's a great observation"
  
  Label:

Completion:
   MI_QUESTION
```

The label source is configurable:
- `label_field: auto` → uses `t2_label_auto` (tiered) or `label_auto` (flat)
- `label_field: t2_label_auto` → uses tier-2 labels
- `label_field: t1_label_auto` → uses tier-1 labels

---

## Workflow

1. **Data Preparation**
   - Load annotated CSV
   - Build prompt-completion pairs
   - Split into train/validation sets

2. **Local Training Path**
   - Load tokenizer and model from Hugging Face Hub
   - Optionally apply LoRA configuration
   - Run training loop with HF Transformers Trainer
   - Save model to `data/fine_tuning/local_finetuned_model/`

3. **OpenAI Path**
   - Convert examples to OpenAI JSONL format
   - Upload files to OpenAI
   - Create and monitor fine-tune job
   - Save job metadata

---

## Usage Examples

### Quick Start: LoRA Fine-Tuning with GPT2
```bash
python src/main.py fine_tuning.enabled=true fine_tuning.use_peft=true
```

### Full Fine-Tuning with Larger Model
```bash
python src/main.py \
  fine_tuning.enabled=true \
  fine_tuning.use_peft=false \
  fine_tuning.base_model=meta-llama/Llama-2-7b \
  fine_tuning.learning_rate=1e-4 \
  fine_tuning.num_train_epochs=5
```

### OpenAI Fine-Tuning
```bash
export OPENAI_API_KEY="sk-..."
python src/main.py \
  fine_tuning.enabled=true \
  fine_tuning.provider=openai \
  fine_tuning.base_model=gpt-3.5-turbo
```

---

## Notes

- **LoRA Rank (`peft_r`)**: Higher = more capacity but more parameters. Start with 8, try 16 for bigger models.
- **Target Modules**: Which attention heads to apply LoRA to. Common choices: `[q_proj, v_proj]` or `[q_proj, v_proj, k_proj, v_proj]`
- **Learning Rate**: Often lower for fine-tuning (2e-4 to 5e-5) vs pre-training
- **Batch Size**: Depends on GPU memory. Smaller = slower but less memory
