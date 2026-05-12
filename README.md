# AutoMISC
Automatic MISC 2.5 Annotation of Motivational Interviewing Transcripts

## Installation

### Requirements

- Python >= 3.11
- Conda (recommended)

### Set up Conda environment (recommended)

```bash
conda create -n automisc python=3.11
conda activate automisc
```

### Install required packages:
```bash
pip install -r requirements.txt
```

## Data Preparation

Ensure your `data/` directory has the following structure:
```text
data/
├── parsed/
│   ├── AnnoMI_parsed.csv
│   ├── HLQC_parsed.csv
│   ├── MIV6.3A_parsed.csv
│   └── MIV6.3B_parsed.csv
├── AnnoMI.csv
├── HLQC.csv
├── MIV6.3A.csv
└── MIV6.3B.csv
```

## LM Studio

If using LM Studio models, ensure that the application is running and the required models are downloaded locally.

## Usage

### Single Model Run (Legacy)
Run the main script with your desired configuration:
```bash
python src/main.py
```

### Multi-Model Testing (New)
Test multiple model combinations automatically with progress saving:

1. **Configure multiple models** in `conf/config.yaml`:
   ```yaml
   parser:
     models:
       - provider: lmstudio
         model: google/gemma-4-e4b
         temperature: 0.0
       - provider: lmstudio
         model: qwen/qwen3.5-9b
         temperature: 0.0
   
   annotator:
     models:
       - provider: lmstudio
         model: google/gemma-4-e4b
         context_mode: interval
         num_context_turns: 5
         class_structure: tiered
         temperature: 0.0
       - provider: lmstudio
         model: qwen/qwen3.5-9b
         context_mode: interval
         num_context_turns: 5
         class_structure: tiered
         temperature: 0.0
   
   fine_tuning:
     enabled: true
     configs:
       - provider: local
         base_model: gpt2
         use_peft: true
       - provider: local
         base_model: gpt2
         use_peft: false
   ```

2. **Run multi-model experiments**:
   ```bash
   python src/main.py
   ```
   This will automatically test all combinations of parser × annotator × fine-tuning configs.

3. **Resume interrupted runs**:
   ```bash
   python src/main.py experiment.resume_from_checkpoint=true
   ```

### Evaluation
Run unified evaluation across all experiments:
```bash
python src/unified_eval.py
```

This project uses Hydra for configuration management. Annotated corpora are saved `data/annotated/` in `.csv` format, and all experiment artifacts are saved to the default hydra output directory (`outputs/<date>/<time>/`).

## Multi-Model Testing

AutoMISC now supports testing multiple model combinations automatically with progress saving and unified evaluation.

### Configuration

Update `conf/config.yaml` to specify multiple models:

```yaml
parser:
  models:
    - provider: lmstudio
      model: google/gemma-4-e4b
      temperature: 0.0
    - provider: lmstudio
      model: qwen/qwen3.5-9b
      temperature: 0.0

annotator:
  models:
    - provider: lmstudio
      model: google/gemma-4-e4b
      context_mode: interval
      num_context_turns: 5
      class_structure: tiered
      temperature: 0.0
    - provider: lmstudio
      model: qwen/qwen3.5-9b
      context_mode: interval
      num_context_turns: 5
      class_structure: tiered
      temperature: 0.0

fine_tuning:
  enabled: true
  configs:
    - provider: local
      base_model: gpt2
      use_peft: true
    - provider: local
      base_model: gpt2
      use_peft: false

experiment:
  checkpoint_dir: outputs/checkpoints
  resume_from_checkpoint: false
  run_evaluation: true
  evaluation_methods: [IRR, relate_outcomes, fine_tuning_comparison]
```

### Running Multi-Model Experiments

```bash
# Run all combinations
python src/main.py

# Resume from checkpoint if interrupted
python src/main.py experiment.resume_from_checkpoint=true

# Run unified evaluation
python src/unified_eval.py
```

### Features

- **Automatic Combination Generation**: Tests all parser × annotator × fine-tuning combinations
- **Progress Checkpointing**: Saves progress automatically, resume interrupted runs
- **Unified Evaluation**: Compare results across all experiments
- **Backward Compatibility**: Legacy single-model configs still work

### Defining run configuration

The default configuration is located at `conf/config.yaml` and can be modified directly. Individual settings may be overriden via the command line.

Available config options are:

```yaml
input_dataset:
  name: [MIV6.3A, MIV6.3B, AnnoMI, HLQC]
  subset: [lowconf, highconf, HI, LO]

n_conversations: <int>

parser:
  model: [openai/gpt-4o, google/gemma-3-12b, qwen/qwen3-30b-a3b]
  temperature: <float>

annotator:
  model: [openai/gpt-4o, google/gemma-3-12b, qwen/qwen3-30b-a3b]
  context_mode: [all, cumulative, interval]
  num_context_turns: <int> (only when context_mode is interval)
  class_structure: [tiered, flat]
  temperature: <float>

fine_tuning:
  enabled: [true, false]
  provider: [openai]
  base_model: <str>
  suffix: <str>
  max_epochs: <int>
  batch_size: <int>
  validation_split: <float>
  label_field: [auto, label_auto, t2_label_auto, t1_label_auto]
  output_dir: <str>
  seed: <int>
```

## Fine-tuning

AutoMISC supports fine-tuning language models on your annotated MI data using either local training (Hugging Face) or OpenAI's API.

### Quick Start

1. **Enable fine-tuning** in `conf/config.yaml`:
   ```yaml
   fine_tuning:
     enabled: true
     provider: local  # or 'openai'
     use_peft: true   # Use LoRA for faster training
   ```

2. **Run the full pipeline** (parsing → annotation → fine-tuning):
   ```bash
   python src/main.py
   ```

3. **Evaluate the results**:
   ```bash
   python src/evaluate_fine_tuning.py
   ```

### Fine-tuning Options

#### Local Fine-tuning (Recommended)
- **Pros**: Private, customizable, no API costs
- **Cons**: Requires local compute (GPU recommended)
- **Requirements**: `pip install torch transformers datasets accelerate peft`

```yaml
fine_tuning:
  enabled: true
  provider: local
  base_model: gpt2                    # Any HF model
  use_peft: true                     # LoRA mode (faster, less memory)
  num_train_epochs: 3
  learning_rate: 2e-4
  per_device_train_batch_size: 4
```

#### OpenAI Fine-tuning
- **Pros**: Managed infrastructure, easy to use
- **Cons**: API costs, less control
- **Requirements**: `OPENAI_API_KEY` environment variable

```yaml
fine_tuning:
  enabled: true
  provider: openai
  base_model: gpt-3.5-turbo-0613
  max_epochs: 4
  batch_size: 8
```

### Evaluation and Comparison

After fine-tuning, evaluate and compare to the baseline:

```bash
# Run evaluation
python src/evaluate_fine_tuning.py

# Results are saved to outputs/<date>/<time>/fine_tuning_evaluation/
# - evaluation_results.json: Full results
# - evaluation_summary.json: Key metrics
```

The evaluation will:
1. **Load your fine-tuned model** (if available)
2. **Run inference** on a sample of test utterances
3. **Compare to baseline** (original model performance)
4. **Report metrics**: accuracy, F1-score, classification report

**Expected Output:**
```
==================================================
FINE-TUNING EVALUATION RESULTS
==================================================
Baseline Model (gpt2):
  Accuracy: 0.723
  F1 Macro: 0.689
  F1 Weighted: 0.701

Fine-tuned Model:
  Accuracy: 0.845 (+0.122 improvement)
  F1 Macro: 0.821 (+0.132 improvement)
  F1 Weighted: 0.833 (+0.132 improvement)
```

### Advanced Usage

#### Custom Base Models
```bash
# Use a different Hugging Face model
python src/main.py fine_tuning.base_model=microsoft/DialoGPT-medium

# Use OpenAI GPT-4 fine-tuning
OPENAI_API_KEY=sk-... python src/main.py fine_tuning.provider=openai fine_tuning.base_model=gpt-4-0613
```

#### LoRA Configuration
```bash
# Adjust LoRA parameters for better performance
python src/main.py \
  fine_tuning.use_peft=true \
  fine_tuning.peft_r=16 \
  fine_tuning.peft_alpha=32 \
  fine_tuning.target_modules="[q_proj,v_proj,k_proj,o_proj]"
```

#### Evaluation on Different Data
Modify `src/evaluate_fine_tuning.py` to load your own test dataset with ground truth labels.

See [FINE_TUNING.md](FINE_TUNING.md) for detailed technical documentation.
