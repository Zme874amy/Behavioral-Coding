"""Human-labeled data loaders and Leave-One-Dataset-Out (LODO) splitter.

Schema produced by every loader:
    dataset, conv_id, speaker, utt_text, t1_label, t2_label

Speaker is normalized to {"counsellor", "client"}.
Labels missing in a corpus are stored as None (NaN) and dropped at vocab time.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# MISC 2.5 label vocabularies (source of truth = components/prompts/response_formats.py)
# -----------------------------------------------------------------------------
COUNSELLOR_T1 = ["CRL", "SRL", "IMC", "IMI", "Q", "O"]
CLIENT_T1 = ["C", "S", "N"]

COUNSELLOR_T2 = [
    "CR", "AF", "SU", "RF", "EC",
    "SR",
    "ADP", "RCP", "GI",
    "ADW", "CO", "DI", "RCW", "WA",
    "OQ", "CQ",
    "FA", "FI", "ST",
]
CLIENT_T2 = [
    "O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+",
    "O-", "D-", "AB-", "R-", "N-", "C-", "AC-", "TS-",
    "N",
]

# Long-tail clinical groupings (the thesis focus)
CHANGE_TALK_T2 = ["O+", "D+", "AB+", "R+", "N+", "C+", "AC+", "TS+"]
SUSTAIN_TALK_T2 = ["O-", "D-", "AB-", "R-", "N-", "C-", "AC-", "TS-"]


def vocab(class_structure: str) -> List[str]:
    if class_structure == "t1":
        return COUNSELLOR_T1 + CLIENT_T1  # note: "O" appears only on counsellor side; "N" is client-only
    if class_structure == "t2":
        # Client T2 contains a bare "N" that collides with counsellor speakers,
        # but at training time we always know the speaker, so collisions are
        # disambiguated by `speaker`. The label set we predict per-example is
        # the union; per-speaker validity is enforced by `valid_for_speaker`.
        return list(dict.fromkeys(COUNSELLOR_T2 + CLIENT_T2))
    raise ValueError(f"Unknown class_structure: {class_structure}")


def valid_for_speaker(label: str, speaker: str, class_structure: str) -> bool:
    if class_structure == "t1":
        if speaker == "counsellor":
            return label in COUNSELLOR_T1
        return label in CLIENT_T1
    # t2
    if speaker == "counsellor":
        return label in COUNSELLOR_T2
    return label in CLIENT_T2


# -----------------------------------------------------------------------------
# Speaker normalization
# -----------------------------------------------------------------------------
def _normalize_speaker(s: str) -> str:
    s = str(s).strip().lower()
    if s in {"counsellor", "counselor", "therapist", "clinician"}:
        return "counsellor"
    if s in {"client", "patient", "speaker"}:
        return "client"
    return s


# -----------------------------------------------------------------------------
# Per-corpus loaders. Every loader returns the normalized schema.
# -----------------------------------------------------------------------------
def load_miv63a_manual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "dataset": "miv63a",
            "conv_id": df["conv_id"].astype(str),
            "speaker": df["speaker"].map(_normalize_speaker),
            "utt_text": df["utt_text"].astype(str),
            "t1_label": df.get("t1_label_GT"),
            "t2_label": df.get("t2_label_GT"),
        }
    )
    return out


def load_hlqc_manual(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "dataset": "hlqc",
            "conv_id": df["conv_id"].astype(str),
            "speaker": df["speaker"].map(_normalize_speaker),
            "utt_text": df["utt_text"].astype(str),
            "t1_label": df.get("t1_label_GT"),
            "t2_label": df.get("t2_label_GT"),
        }
    )
    return out


# AnnoMI label -> normalized T1 label. AnnoMI does not encode Change Talk
# +/- distinctions, so it cannot supply T2 client labels. For the counsellor
# side, AnnoMI's CQ/CR/SR/GI/OQ are T2 codes; we keep them only when
# `class_structure == "t2"` is allowed (they map 1:1). For T1, we collapse
# AnnoMI's counsellor codes to the T1 vocabulary.
_ANNOMI_TO_T1_COUNSELLOR = {
    "CR": "SRL",   # Complex Reflection -> Simple/Complex Reflection -> SRL (T1 reflection bucket)
    "SR": "SRL",   # Simple Reflection
    "CQ": "Q",     # Closed Question -> T1 Question
    "OQ": "Q",     # Open Question -> T1 Question
    "GI": "IMC",   # Giving Information -> T1 Imparting MI Consistent
    "O":  "O",     # Other
    "ADVI": "IMC", # Advise -> Imparting MI Consistent (MI consistent advice)
    "NEGO": "O",   # Negotiate -> Other (no direct T1)
    "OPTI": "O",   # Optimism -> Other (no direct T1)
}
_ANNOMI_TO_T2_COUNSELLOR = {
    # AnnoMI counsellor codes that already match T2 vocabulary
    "CR": "CR", "SR": "SR", "CQ": "CQ", "OQ": "OQ", "GI": "GI",
    # The rest do not have a clean T2 mapping; drop those rows.
}


def load_annomi(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    def _first_label(x):
        if not isinstance(x, str):
            return None
        x = x.strip()
        if x.startswith("["):
            try:
                lst = ast.literal_eval(x)
                return lst[0] if lst else None
            except Exception:
                return None
        return x

    df = df.copy()
    df["lbl"] = df["AnnoMI Label"].map(_first_label)
    df["speaker"] = df["speaker"].map(_normalize_speaker)
    df["utt_text"] = df["vol_text"].astype(str)
    df["conv_id"] = df["conv_id"].astype(str)

    # T1 mapping: counsellor via lookup, client = label as-is (already T1: C/S/N).
    def _t1(row):
        sp = row["speaker"]
        lbl = row["lbl"]
        if lbl is None:
            return None
        if sp == "counsellor":
            return _ANNOMI_TO_T1_COUNSELLOR.get(lbl)
        return lbl if lbl in CLIENT_T1 else None

    # T2 mapping: AnnoMI lacks +/- Change Talk; we cannot recover T2 client labels.
    # Counsellor side: keep only the codes that already match T2 vocab.
    def _t2(row):
        sp = row["speaker"]
        lbl = row["lbl"]
        if lbl is None:
            return None
        if sp == "counsellor":
            return _ANNOMI_TO_T2_COUNSELLOR.get(lbl)
        return None  # AnnoMI client labels have no T2 equivalent

    out = pd.DataFrame(
        {
            "dataset": "annomi",
            "conv_id": df["conv_id"],
            "speaker": df["speaker"],
            "utt_text": df["utt_text"],
            "t1_label": df.apply(_t1, axis=1),
            "t2_label": df.apply(_t2, axis=1),
        }
    )
    return out


CORPUS_LOADERS = {
    "miv63a": (load_miv63a_manual, Path("data/manual/MIV6.3A_manual.csv")),
    "hlqc":   (load_hlqc_manual,   Path("data/manual/HLQC_balanced_manual.csv")),
    "annomi": (load_annomi,        Path("data/AnnoMI.csv")),
}


def load_dataset(name: str) -> pd.DataFrame:
    if name not in CORPUS_LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(CORPUS_LOADERS)}")
    loader, path = CORPUS_LOADERS[name]
    if not path.exists():
        raise FileNotFoundError(f"Manual data file not found: {path}")
    return loader(path)


# -----------------------------------------------------------------------------
# Prompt construction (identical for training and zero-shot evaluation so that
# the comparison is fair: same instructions, only the model weights differ).
# -----------------------------------------------------------------------------
def system_block(class_structure: str) -> str:
    if class_structure == "t1":
        return (
            "You are a clinical Motivational Interviewing (MI) behavior classifier.\n"
            "Assign exactly one MISC 2.5 Tier-1 code based on the speaker and utterance.\n"
            f"Counsellor codes: {', '.join(COUNSELLOR_T1)}.\n"
            f"Client codes: {', '.join(CLIENT_T1)}.\n"
            "Reply with the code only, no explanation."
        )
    return (
        "You are a clinical Motivational Interviewing (MI) behavior classifier.\n"
        "Assign exactly one MISC 2.5 Tier-2 code based on the speaker and utterance.\n"
        f"Counsellor codes: {', '.join(COUNSELLOR_T2)}.\n"
        f"Client codes: {', '.join(CLIENT_T2)}.\n"
        "Client '+' codes indicate Change Talk; '-' codes indicate Sustain Talk.\n"
        "Reply with the code only, no explanation."
    )


def _build_bare_prompt(speaker: str, utt_text: str, class_structure: str) -> str:
    sys = system_block(class_structure)
    return (
        f"{sys}\n\n"
        f"Speaker: {speaker}\n"
        f"Utterance: {utt_text}\n\n"
        f"Label:"
    )


def _build_rich_prompt(speaker: str, utt_text: str, class_structure: str) -> str:
    """Render the production `flat.j2` system prompt from src/components/prompts
    and append the target utterance plus a code-only output instruction.

    Used so training, zero-shot eval, and fine-tuned eval all see the same
    rich-prompt format. The single difference between zero-shot and
    fine-tuned then is the LoRA adapter.
    """
    from components.prompts.loader import render_prompt  # local to avoid heavy import on module load

    speaker_key = "counsellor" if speaker == "counsellor" else "client"
    system_prompt = render_prompt(speaker=speaker_key, structure="flat")
    # The production flat.j2 ends with an "Output Format" block that asks for
    # JSON {"explanation": ..., "label": ...}. We're calling raw `generate()`
    # so swap that for a short, machine-parseable instruction.
    cut_idx = system_prompt.find("## **Output Format**")
    if cut_idx != -1:
        system_prompt = system_prompt[:cut_idx].rstrip() + "\n\n"
    instructions = (
        "## **Output Format**\n"
        "Reply with exactly one abbreviated label (e.g. `SR`, `OQ`, `R+`, `N`) "
        "and nothing else. Do not include an explanation.\n\n"
        "---\n\n"
    )
    return (
        f"{system_prompt}{instructions}"
        f"## **Target Utterance**\n"
        f"Speaker: {speaker_key}\n"
        f"Utterance: {utt_text}\n\n"
        f"Label:"
    )


def build_prompt(
    speaker: str, utt_text: str, class_structure: str, prompt_style: str = "bare"
) -> str:
    """Construct the SFT/zero-shot prompt.

    Args:
        speaker: 'client' or 'counsellor'
        utt_text: the target utterance
        class_structure: 't1' or 't2'
        prompt_style: 'bare' (terse inline) or 'rich' (production flat.j2 with
            per-code descriptions). Rich is only supported for the flat schema
            so far, which matches the t2 codebook we ship.
    """
    if prompt_style == "rich":
        return _build_rich_prompt(speaker, utt_text, class_structure)
    if prompt_style == "bare":
        return _build_bare_prompt(speaker, utt_text, class_structure)
    raise ValueError(f"Unknown prompt_style: {prompt_style!r} (use 'bare' or 'rich')")


def build_completion(label: str) -> str:
    # Leading space + newline = stable formatting for prompt-completion SFT.
    return f" {label}\n"


# -----------------------------------------------------------------------------
# Build supervised examples from human labels.
# -----------------------------------------------------------------------------
@dataclass
class Example:
    dataset: str
    conv_id: str
    speaker: str
    utt_text: str
    label: str
    prompt: str
    completion: str


def to_examples(
    df: pd.DataFrame, class_structure: str, prompt_style: str = "bare"
) -> List[Dict[str, str]]:
    """Filter to rows whose human label belongs to the vocab for the given
    speaker, then build prompt-completion examples."""
    label_col = f"{class_structure}_label"
    keep_rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        speaker = row["speaker"]
        if speaker not in {"counsellor", "client"}:
            continue
        label = row.get(label_col)
        if not isinstance(label, str):
            continue
        label = label.strip()
        if not label:
            continue
        if not valid_for_speaker(label, speaker, class_structure):
            continue
        text = str(row["utt_text"]).strip()
        if not text:
            continue
        keep_rows.append(
            {
                "dataset": str(row["dataset"]),
                "conv_id": str(row["conv_id"]),
                "speaker": speaker,
                "utt_text": text,
                "label": label,
                "prompt": build_prompt(speaker, text, class_structure, prompt_style),
                "completion": build_completion(label),
            }
        )
    return keep_rows


# -----------------------------------------------------------------------------
# LODO split.
# -----------------------------------------------------------------------------
def build_lodo(
    train_datasets: Iterable[str],
    test_dataset: str,
    class_structure: str,
    prompt_style: str = "bare",
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, int]]:
    train_datasets = [d.lower() for d in train_datasets]
    test_dataset = test_dataset.lower()
    if test_dataset in train_datasets:
        raise ValueError("test_dataset must not appear in train_datasets (Leave-One-Dataset-Out)")

    train_frames = [load_dataset(d) for d in train_datasets]
    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test_df = load_dataset(test_dataset)

    train_examples = to_examples(train_df, class_structure, prompt_style)
    test_examples = to_examples(test_df, class_structure, prompt_style)

    summary = {
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
    }
    for d in train_datasets:
        summary[f"train_rows[{d}]"] = sum(1 for ex in train_examples if ex["dataset"] == d)
    summary[f"test_rows[{test_dataset}]"] = sum(1 for ex in test_examples if ex["dataset"] == test_dataset)

    return train_examples, test_examples, summary
