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

Run the main script with your desired configuration:
```bash
python src/main.py
```
This project uses Hydra for configuration management. Annotated corpora are saved `data/annotated/` in `.csv` format, and all experiment artifacts are saved to the default hydra output directory (`outputs/<date>/<time>/`).

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
```