"""Dataset preparation for fine-tuning."""
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from hydra.utils import log


from components.artifact_paths import get_annotated_csv_path  # noqa: F401 — re-export


def resolve_label_field(cfg) -> str:
    """Determine which label field to use for fine-tuning."""
    if cfg.fine_tuning.label_field != 'auto':
        return cfg.fine_tuning.label_field

    ann = cfg.annotator
    if hasattr(ann, "models") and not hasattr(ann, "class_structure"):
        ann = ann.models[0]
    return 't2_label_auto' if ann.class_structure == 'tiered' else 'label_auto'


def build_examples(cfg, df: pd.DataFrame) -> List[Dict[str, str]]:
    """Build prompt-completion examples from annotated utterances."""
    label_field = resolve_label_field(cfg)
    if label_field not in df.columns:
        raise ValueError(
            f"Fine-tuning label field '{label_field}' not found in annotated data columns: {list(df.columns)}"
        )

    examples: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        label = row.get(label_field)
        if pd.isna(label) or str(label).strip() == '':
            continue

        prompt_parts = [f"Speaker: {row['speaker']}", f"Utterance: {row['utt_text']}"]
        prompt = "\n\n".join(prompt_parts) + "\n\nLabel:"
        completion = f" {label}\n"

        examples.append({
            'prompt': prompt,
            'completion': completion,
        })

    if not examples:
        raise ValueError("No valid training examples could be built from annotated data.")

    log.info(f"Built {len(examples)} fine-tuning examples using field '{label_field}'.")
    return examples


def split_examples(
    cfg, examples: List[Dict[str, str]]
) -> tuple[List[Dict[str, str]], Optional[List[Dict[str, str]]]]:
    """Split examples into train and validation sets."""
    split = float(cfg.fine_tuning.validation_split)
    if split <= 0.0:
        return examples, None

    seed = int(cfg.fine_tuning.seed)
    random.Random(seed).shuffle(examples)
    n_valid = int(len(examples) * split)
    
    if n_valid < 1:
        log.warning(
            "Validation split requested, but dataset is too small. Using all examples for training."
        )
        return examples, None

    return examples[n_valid:], examples[:n_valid]


def write_jsonl(examples: List[Dict[str, str]], output_path: Path) -> None:
    """Write examples to JSONL format (used for OpenAI fine-tuning)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + '\n')

    log.info(f"Saved JSONL file to {output_path}")


def load_annotated_data(cfg) -> pd.DataFrame:
    """Load annotated CSV file."""
    annotated_path = get_annotated_csv_path(cfg)
    if not annotated_path.exists():
        raise FileNotFoundError(f"Annotated CSV file not found: {annotated_path}")
    return pd.read_csv(annotated_path)
