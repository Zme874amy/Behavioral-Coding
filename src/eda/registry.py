"""One canonical ID, one loader and one metadata record per available dataset.

Why this exists: the same corpus lives in several files under unrelated names
(`HLQC.csv`, `parsed/HLQC_parsed.csv`, `manual/HLQC_balanced_manual.csv` are a
volley-level pool, an utterance-level pool and a 10-session gold subset), and
`synth/aug/*_proto_eval.csv` is synthetic TRAINING data, not an eval set. Our
generated sets are registered here too (ids `synth.*`) but kept out of `load_all()`
by default and documented in docs/SYNTHETIC_DATA.md, because they change with
every synthesis campaign. Every
analysis in `eda.*` and the notebook goes through the IDs below, so a dataset is
always named the same way and its lineage is explicit.

Every loader returns the same long format, one row per coded unit:
    dataset, conv_id, speaker ('counsellor'|'client'), vol_idx, utt_idx, text,
    native   label in the dataset's own scheme (None if uncoded),
    t1, t2   MISC 2.5 label where an exact mapping exists (else None),
plus dataset-specific columns (annotator, session metadata, ...).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
D = REPO / "data"
EXT = D / "external"

DOWNLOADS = {
    EXT / "annomi" / "AnnoMI-full.csv": "https://raw.githubusercontent.com/uccollab/AnnoMI/HEAD/AnnoMI-full.csv",
    EXT / "annomi" / "AnnoMI-simple.csv": "https://raw.githubusercontent.com/uccollab/AnnoMI/HEAD/AnnoMI-simple.csv",
    EXT / "welivita" / "MI_Dataset.csv":
        "https://raw.githubusercontent.com/anuradha1992/Motivational-Interviewing-Dataset/HEAD/MI%20Dataset.csv",
}


def ensure_downloads() -> None:
    """Fetch the public releases that the repo only holds in derived form."""
    for dst, url in DOWNLOADS.items():
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dst)


# --------------------------------------------------------------------- metadata
@dataclass
class Meta:
    id: str
    title: str
    files: Dict[str, str]                 # repo path -> what that file is
    lineage: str                          # where the data came from, what derives from what
    domain: str
    modality: str                         # spoken-transcribed / chatbot text / written forum / synthetic
    transcription: str
    session_type: str
    language: str
    scheme: str
    unit: str
    annotators: str
    reliability: str                      # paper-reported; measured values live in the notebook
    licence: str
    source: str
    role: str                             # how we use it
    known_issues: List[str] = field(default_factory=list)


META: Dict[str, Meta] = {m.id: m for m in [
    Meta(
        id="misc.miv63a.gold", title="MIBot v6.3A, expert MISC consensus (TEST SET)",
        files={"data/manual/MIV6.3A_manual.csv": "10 sessions, utterance-segmented, *_GT = human consensus, *_auto = AutoMISC GPT-4.1"},
        lineage="10 of the 173 pool.miv63a sessions; labels by 3 interns + 1 grad student aligned to MI clinicians (AutoMISC thesis).",
        domain="smoking cessation", modality="chatbot text (LLM counsellor, human client)", transcription="none (typed)",
        session_type="real participants (Prolific) with MIBot v6.3A", language="English",
        scheme="MISC 2.5 + AutoMISC extension (Activation AC+/-; AutoMISC T1 grouping), both speakers", unit="utterance (thought unit), single label",
        annotators="4 trained coders, consensus", reliability="Fleiss kappa >= 0.6 after alignment (thesis App. B)",
        licence="AutoMISC repo (research)", source="https://github.com/cimhasgithub/AutoMISC",
        role="fixed test set for all MISC models",
        known_issues=["only 10 sessions, no dev split", "TS+ / AC- occur here but not in HLQC train"]),
    Meta(
        id="misc.hlqc.gold", title="HLQC balanced subset, MISC 2.5 (TRAIN SET)",
        files={"data/manual/HLQC_balanced_manual.csv": "10 sessions (5 high, 5 low), utterance-segmented, *_GT human"},
        lineage="10 of the 257 pool.hlqc sessions; labels by the AutoMISC team (released with NLPAI4Health 2025).",
        domain="mixed health behaviour (alcohol, diet, exercise, smoking, ...)", modality="spoken, transcribed",
        transcription="ASR (Pérez-Rosas et al. 2019), no casing/punctuation", session_type="MI demonstration / role-play videos",
        language="English", scheme="MISC 2.5 + AutoMISC extension (Activation AC+/-; AutoMISC T1 grouping), both speakers", unit="utterance, single label",
        annotators="AutoMISC team", reliability="not reported for this subset",
        licence="AutoMISC repo (research)", source="https://github.com/cimhasgithub/AutoMISC",
        role="training set for all MISC models; few-shot / synthesis exemplars",
        known_issues=["FI vs FA coded inconsistently", "noisy ASR text", "1,925 rows vs 1,924 in the paper",
                      "high_121 = CASAA 'Emmy's First Encounter'"]),
    Meta(
        id="pool.hlqc", title="HLQC full corpus (UNLABELLED pool)",
        files={"data/HLQC.csv": "257 sessions, volley level (one row per speaker turn)",
               "data/parsed/HLQC_parsed.csv": "same 257 sessions, split into utterances by the AutoMISC parser"},
        lineage="Pérez-Rosas et al. 2019 'What makes a good counselor?'; session id prefix high_/low_ = session-level quality label.",
        domain="mixed health behaviour", modality="spoken, transcribed", transcription="ASR",
        session_type="MI demonstration videos (YouTube/Vimeo)", language="English",
        scheme="none at utterance level (session high/low quality only)", unit="volley / utterance",
        annotators="-", reliability="-", licence="research", source="Pérez-Rosas et al. 2019",
        role="unlabelled pool (retrieval, self-training); contains misc.hlqc.gold",
        known_issues=["contains the 10 gold sessions", "MI-TAGS authors report duplicate / mismatched transcripts"]),
    Meta(
        id="pool.miv63a", title="MIBot v6.3A full run (UNLABELLED pool)",
        files={"data/MIV6.3A.csv": "173 sessions, volley level",
               "data/parsed/MIV6.3A_parsed.csv": "utterance level",
               "data/2024-11-14-MIV6.3A-...merged.csv": "per-participant readiness/importance/confidence pre/post/week-later + demographics (join on 'Participant id' = conv_id)"},
        lineage="MIBot v6.3A study (Mahmood et al. 2025); contains the 10 misc.miv63a.gold sessions.",
        domain="smoking cessation", modality="chatbot text", transcription="none", session_type="real participants with MIBot",
        language="English", scheme="none", unit="volley / utterance", annotators="-", reliability="-",
        licence="AutoMISC repo (research)", source="https://github.com/cimhasgithub/AutoMISC",
        role="unlabelled pool; MUST exclude the 10 test sessions",
        known_issues=["contains the test set", "templated chatbot lines repeat across sessions"]),
    Meta(
        id="pool.miv63b", title="MIBot v6.3B run (UNLABELLED pool)",
        files={"data/MIV6.3B.csv": "165 sessions, volley level", "data/parsed/MIV6.3B_parsed.csv": "utterance level",
               "data/2024-11-19-MIV6.1B_...merged.csv": "outcomes file (named 6.1B; verify it is this run)"},
        lineage="A different MIBot run from 6.3A (different participants).",
        domain="smoking cessation", modality="chatbot text", transcription="none", session_type="real participants with MIBot",
        language="English", scheme="none", unit="volley / utterance", annotators="-", reliability="-",
        licence="AutoMISC repo (research)", source="https://github.com/cimhasgithub/AutoMISC",
        role="unlabelled pool (self-training v2 'Step F')",
        known_issues=["shares templated chatbot utterances with the MIV6.3A test set (22/652 long test utts verbatim)"]),
    Meta(
        id="annomi.gold", title="AnnoMI (expert-annotated MI demonstrations)",
        files={"data/external/annomi/AnnoMI-full.csv": "official full release, one row per (utterance, annotator)",
               "data/external/annomi/AnnoMI-simple.csv": "official simple release, one row per utterance",
               "data/AnnoMI.csv": "AutoMISC's MISC-mapped volley labels (different row order; join on conv_id+utterance_id)",
               "data/manual/AnnoMI_eval.csv": "our cross-scheme eval file built from data/AnnoMI.csv",
               "data/parsed/AnnoMI_parsed.csv": "utterance-split text, unlabelled"},
        lineage="Wu et al. 2022/2023; 133 YouTube/Vimeo demonstrations, professionally transcribed.",
        domain="mixed (alcohol, smoking, diet, ...); 'topic' column", modality="spoken, transcribed",
        transcription="manual (professional)", session_type="MI demonstrations (110 high, 23 low quality)", language="English",
        scheme="AnnoMI: therapist main behaviour + subtypes; client change/neutral/sustain", unit="speaker turn",
        annotators="10 MI experts; 126 transcripts single-annotated, 7 by all 10", reliability="paper: none per item; measured by us (notebook)",
        licence="public (GitHub, no explicit licence)", source="https://github.com/uccollab/AnnoMI",
        role="own-scheme train/dev/test; cross-scheme transfer test for MISC models",
        known_issues=["strong annotator effect (complex-reflection share 0.19-0.82 on the same items)",
                      "client-talk kappa 0.47", "may overlap HLQC (same YouTube sources) - see overlap map"]),
    Meta(
        id="welivita.gold", title="Welivita & Pu MI dataset (peer-support forums)",
        files={"data/external/welivita/MI_Dataset.csv": "official release: ann1, ann2, judge stages, final agreed label",
               "data/external/welivita_mi_parsed.csv": "our ingest (selftrain.ingest), weak_miti = final agreed label, weak_t2 = MISC map",
               "data/manual/Welivita_eval.csv": "our eval file built from the ingest"},
        lineage="Welivita & Pu, COLING 2022; CounselChat + Reddit threads. Labels are THEIRS (verified 100% match), not ours.",
        domain="mental-health peer support (written)", modality="written forum Q&A", transcription="none",
        session_type="online forum threads (2-6 turns)", language="English",
        scheme="MITI-derived (MITI 2.0 / 4.2.1), 15 listener codes; maps to MITI 4.2.1 (57% exact, 34% approximate, 8% none); seekers uncoded", unit="sentence-level listener segment",
        annotators="2 MTurk crowd workers + 2 expert judge stages", reliability="measured by us: ann1 vs ann2 kappa 0.34",
        licence="CC BY-NC-SA 3.0", source="https://github.com/anuradha1992/Motivational-Interviewing-Dataset",
        role="own-scheme train/dev/test (agreed subset); weak cross-scheme test; self-training pool",
        known_issues=["weak gold", "written domain far from spoken MI", "~2% of rows differ from the release only in whitespace"]),
    Meta(
        id="miti.casaa.gold", title="CASAA MITI 4 coded training transcripts",
        files={"data/external/casaa/CASAA_eval.csv": "load_manual schema, MISC-mapped gold where exact",
               "data/external/casaa/casaa_turns.csv": "one row per turn, raw code cell + coder notes",
               "data/external/casaa/casaa_globals.csv": "MITI global ratings (9 sessions)"},
        lineage="UNM CASAA coder-training PDFs, parsed by baseline.prep_casaa.",
        domain="mixed (alcohol, smoking, diabetes, IPV, parenting, ...)", modality="spoken, transcribed",
        transcription="manual", session_type="MI training demonstrations / role-plays", language="English",
        scheme="MITI 4 (counsellor only)", unit="speaker turn (multi-code turns split by sentence when counts match)",
        annotators="MITI developers' lab (reference coding)", reliability="reference standard (no kappa)",
        licence="CASAA training material, research use with citation; not redistributed", source="https://casaa.unm.edu/tools/miti.html",
        role="test only (MITI; MISC on 6 exact T2 codes + T1)",
        known_issues=["Emmy = HLQC high_121 (train!)", "Rounder = HLQC high_072", "turn 25 missing in 3 source PDFs"]),
] + [
    Meta(
        id=f"synth.{k}", title=f"Synthetic MISC ({desc})",
        files={f"data/synth/aug/{f}": "HLQC gold (1,925 rows) + synthetic rows (conv_id 'synth:*'); raw/ has pre-verify versions"},
        lineage=lin, domain="by topic mix (see lineage)", modality="synthetic dialogue windows", transcription="generated",
        session_type="synthetic", language="English", scheme="MISC 2.5 (generator-labelled, verifier-checked)",
        unit="thought unit", annotators="Qwen2.5-32B generator + independent verifier", reliability="judge scores in synth docs",
        licence="ours", source="src/synth", role="training augmentation only, never gold",
        known_issues=iss)
    for k, desc, f, lin, iss in [
        ("v1", "v1 one-shot", "Qwen2_5-32B-Instruct-AWQ.csv", "prompts_v1, code-conditioned single shots", ["mode collapse (Self-BLEU 0.199)"]),
        ("v2_proto", "v2 prototype windows", "Qwen2_5-32B-Instruct-AWQ_proto.csv", "prompts_v2 ontology windows, prototype focus", []),
        ("v2_boundary", "v2 boundary windows", "Qwen2_5-32B-Instruct-AWQ_boundary.csv", "prompts_v2, confusable-pair focus", []),
        ("v3_hlqcmix", "v3, HLQC topic mix", "Qwen2_5-32B-Instruct-AWQ_proto_train.csv",
         "generate.py v3 --mix train (HLQC topic proportions); file name says 'proto_train'", []),
        ("v3_mivmix", "v3, MIV6.3A topic mix", "Qwen2_5-32B-Instruct-AWQ_proto_eval.csv",
         "generate.py v3 --mix eval: topic proportions measured on the MIV6.3A TEST set; file name says 'proto_eval' but it is training data",
         ["uses test-set topic statistics (documented ablation)"]),
    ]
]}

SYNTH_FILES = {m.id: next(iter(m.files)) for m in META.values() if m.id.startswith("synth.")}

# One row per file of the real datasets: which file to use for what.
FILES = pd.DataFrame([
    # dataset, file, level, labels, use it for, derived from
    ("misc.miv63a.gold", "data/manual/MIV6.3A_manual.csv", "utterance", "MISC T1/T2 human gold (*_GT) + GPT-4.1 (*_auto)", "TEST set of every MISC model", "pool.miv63a (10 sessions)"),
    ("misc.hlqc.gold", "data/manual/HLQC_balanced_manual.csv", "utterance", "MISC T1/T2 human gold", "TRAIN set; few-shot exemplars", "pool.hlqc (10 sessions)"),
    ("pool.hlqc", "data/HLQC.csv", "turn (volley)", "none (session high/low in id)", "raw turns", "Pérez-Rosas et al. 2019 ASR transcripts"),
    ("pool.hlqc", "data/parsed/HLQC_parsed.csv", "utterance", "none", "unlabelled pool (retrieval, self-training)", "data/HLQC.csv, split by the AutoMISC parser"),
    ("pool.miv63a", "data/MIV6.3A.csv", "turn (volley)", "none", "raw turns", "MIBot v6.3A study logs"),
    ("pool.miv63a", "data/parsed/MIV6.3A_parsed.csv", "utterance", "none", "unlabelled pool (must drop the 10 test sessions)", "data/MIV6.3A.csv, parser"),
    ("pool.miv63a", "data/2024-11-14-MIV6.3A-...merged.csv", "participant", "readiness / importance / confidence, demographics", "session outcomes (join 'Participant id' = conv_id)", "MIBot v6.3A surveys"),
    ("pool.miv63b", "data/MIV6.3B.csv", "turn (volley)", "none", "raw turns", "MIBot v6.3B study logs"),
    ("pool.miv63b", "data/parsed/MIV6.3B_parsed.csv", "utterance", "none", "unlabelled pool (self-training)", "data/MIV6.3B.csv, parser"),
    ("pool.miv63b", "data/2024-11-19-MIV6.1B_...merged.csv", "participant", "survey outcomes", "session outcomes (file name says 6.1B)", "MIBot surveys"),
    ("annomi.gold", "data/external/annomi/AnnoMI-full.csv", "turn x annotator", "AnnoMI attributes per annotator", "agreement analysis (7 ten-rater transcripts)", "official release (Wu et al.)"),
    ("annomi.gold", "data/external/annomi/AnnoMI-simple.csv", "turn", "AnnoMI main behaviour / talk type", "own-scheme train/dev/test", "official release"),
    ("annomi.gold", "data/AnnoMI.csv", "turn", "MISC-mapped AnnoMI labels (AutoMISC)", "source of AnnoMI_eval (different row order)", "AnnoMI-full, mapped by AutoMISC"),
    ("annomi.gold", "data/manual/AnnoMI_eval.csv", "turn", "MISC gold on shared codes only", "cross-scheme TEST of MISC models", "data/AnnoMI.csv (baseline.prep_crossscheme)"),
    ("annomi.gold", "data/parsed/AnnoMI_parsed.csv", "utterance", "none", "utterance-split text", "AnnoMI, parser"),
    ("welivita.gold", "data/external/welivita/MI_Dataset.csv", "sentence", "ann1, ann2, judge stages, final label", "agreement analysis; own-scheme split", "official release (Welivita & Pu)"),
    ("welivita.gold", "data/external/welivita_mi_parsed.csv", "sentence", "final label + MISC map (weak_t2)", "self-training pool", "MI_Dataset.csv (selftrain.ingest)"),
    ("welivita.gold", "data/manual/Welivita_eval.csv", "sentence", "MISC gold on shared codes", "cross-scheme TEST of MISC models", "welivita_mi_parsed.csv (baseline.prep_crossscheme)"),
    ("miti.casaa.gold", "data/external/casaa/CASAA_eval.csv", "utterance", "MITI codes + MISC map where exact", "MITI / MISC TEST (18 clean sessions)", "CASAA PDFs (baseline.prep_casaa)"),
    ("miti.casaa.gold", "data/external/casaa/casaa_turns.csv", "turn", "raw code cell + coder notes", "audit of the parse", "CASAA PDFs"),
    ("miti.casaa.gold", "data/external/casaa/casaa_globals.csv", "session", "MITI global ratings", "session-level analysis (9 sessions)", "CASAA PDFs"),
], columns=["dataset", "file", "level", "labels", "use it for", "derived from"])


# ---------------------------------------------------------------------- loaders
def _std(df, dataset, conv, spk, vol, utt, text, native=None, t1=None, t2=None, extra=()):
    out = pd.DataFrame({
        "dataset": dataset,
        "conv_id": df[conv].astype(str).values,
        "speaker": df[spk].map(_spk).values,
        "vol_idx": df[vol].values if vol else np.arange(len(df)),
        "utt_idx": df[utt].values if utt else np.arange(len(df)),
        "text": df[text].astype(str).values,
        "native": df[native].values if native else None,
        "t1": df[t1].values if t1 else None,
        "t2": df[t2].values if t2 else None,
    })
    for c in extra:
        out[c] = df[c].values
    for c in ("native", "t1", "t2"):
        out[c] = out[c].where(out[c].notna() & (out[c].astype(str).str.strip() != ""), None)
    return out


def _spk(s) -> str:
    s = str(s).strip().lower()
    return "counsellor" if s in {"counsellor", "counselor", "therapist", "listener", "p", "i", "t"} else "client"


def _misc_gold(path, dataset):
    from automisc_ft.data import load_manual
    df = load_manual(path)
    df["native"] = df["t2_label_GT"]
    return _std(df, dataset, "conv_id", "speaker", "conv_vol_idx", "conv_utt_idx", "utt_text",
                "native", "t1_label_GT", "t2_label_GT")


def _pool(parsed, dataset):
    df = pd.read_csv(parsed)
    return _std(df, dataset, "conv_id", "speaker", "conv_vol_idx", "conv_utt_idx", "utt_text")


def load_annomi() -> pd.DataFrame:
    """One row per utterance (simple release), native = main behaviour / talk type,
    plus subtypes (majority over annotators where several), annotator id, quality,
    topic, and the MISC labels our cross-scheme eval uses (t1 client C/S/N,
    t2 counsellor OQ/CQ/SR/CR/GI)."""
    ensure_downloads()
    s = pd.read_csv(EXT / "annomi" / "AnnoMI-simple.csv")
    full = pd.read_csv(EXT / "annomi" / "AnnoMI-full.csv")
    key = ["transcript_id", "utterance_id"]
    sub = (full.groupby(key)
           .agg(n_annotators=("annotator_id", "nunique"),
                annotator_id=("annotator_id", lambda x: int(x.iloc[0]) if x.nunique() == 1 else -1),
                reflection_subtype=("reflection_subtype", _mode),
                question_subtype=("question_subtype", _mode),
                therapist_input_subtype=("therapist_input_subtype", _mode))
           .reset_index())
    s = s.merge(sub, on=key, how="left")
    ev = pd.read_csv(D / "manual" / "AnnoMI_eval.csv")
    a = pd.read_csv(D / "AnnoMI.csv")[["conv_id", "utterance_id"]]
    ev["transcript_id"], ev["utterance_id"] = a["conv_id"].values, a["utterance_id"].values
    s = s.merge(ev[key + ["t1_label_GT", "t2_label_GT"]], on=key, how="left")
    s["topic"] = s["topic"].astype(str).str.strip()   # the release has whitespace variants ("smoking cessation ")
    s["native"] = np.where(s.interlocutor == "therapist", s.main_therapist_behaviour, s.client_talk_type)
    s = s.sort_values(key)
    return _std(s, "annomi.gold", "transcript_id", "interlocutor", "utterance_id", "utterance_id",
                "utterance_text", "native", "t1_label_GT", "t2_label_GT",
                extra=("annotator_id", "n_annotators", "mi_quality", "topic", "video_title",
                       "reflection_subtype", "question_subtype", "therapist_input_subtype"))


def _mode(x):
    x = x.dropna()
    return x.value_counts().index[0] if len(x) else None


def load_welivita() -> pd.DataFrame:
    ensure_downloads()
    raw = pd.read_csv(EXT / "welivita" / "MI_Dataset.csv")
    raw = raw[raw.text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.len() > 0].reset_index(drop=True)
    from selftrain.ingest import MITI_TO_MISC_T2
    raw["native"] = raw["final agreed label"].where(raw["final agreed label"].astype(str) != "-")
    raw["t2"] = raw["native"].map(MITI_TO_MISC_T2)
    raw["source"] = raw.dialog_id.astype(str).str.split("|").str[0].str.strip()
    raw["miti"] = raw["native"].map(lambda x: WELIVITA_TO_MITI.get(x, (None,))[0])
    raw["miti_mapping"] = raw["native"].map(lambda x: WELIVITA_TO_MITI.get(x, (None, None))[1])
    raw["stage1_agreed"] = raw["stage I agreed label"].notna() & (raw["stage I agreed label"].astype(str) != "-")
    return _std(raw, "welivita.gold", "dialog_id", "author", "turn", None, "text", "native", None, "t2",
                extra=("ann1", "ann2", "source", "stage1_agreed", "miti", "miti_mapping"))


def load_casaa() -> pd.DataFrame:
    df = pd.read_csv(D / "external" / "casaa" / "CASAA_eval.csv", keep_default_na=False)
    df = df.replace({"": None})
    return _std(df, "miti.casaa.gold", "conv_id", "speaker", "conv_vol_idx", "conv_utt_idx", "utt_text",
                "miti_codes", "t1_label_GT", "t2_label_GT", extra=("align", "hlqc_overlap"))


def load_synth(dataset_id: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / SYNTH_FILES[dataset_id])
    df = df[df.conv_id.astype(str).str.startswith("synth")].copy()
    df["native"] = df["t2_label_GT"]
    return _std(df, dataset_id, "conv_id", "speaker", "conv_vol_idx", "conv_utt_idx", "utt_text",
                "native", "t1_label_GT", "t2_label_GT")


LOADERS: Dict[str, Callable[[], pd.DataFrame]] = {
    "misc.miv63a.gold": lambda: _misc_gold(D / "manual" / "MIV6.3A_manual.csv", "misc.miv63a.gold"),
    "misc.hlqc.gold": lambda: _misc_gold(D / "manual" / "HLQC_balanced_manual.csv", "misc.hlqc.gold"),
    "pool.hlqc": lambda: _pool(D / "parsed" / "HLQC_parsed.csv", "pool.hlqc"),
    "pool.miv63a": lambda: _pool(D / "parsed" / "MIV6.3A_parsed.csv", "pool.miv63a"),
    "pool.miv63b": lambda: _pool(D / "parsed" / "MIV6.3B_parsed.csv", "pool.miv63b"),
    "annomi.gold": load_annomi,
    "welivita.gold": load_welivita,
    "miti.casaa.gold": load_casaa,
    **{k: (lambda k=k: load_synth(k)) for k in SYNTH_FILES},
}


def load(dataset_id: str) -> pd.DataFrame:
    return LOADERS[dataset_id]()


REAL_IDS = [k for k in LOADERS if not k.startswith("synth.")]
SYNTH_IDS = [k for k in LOADERS if k.startswith("synth.")]


def load_all(include_synth: bool = False) -> Dict[str, pd.DataFrame]:
    """The stable real datasets; pass include_synth=True to add our generated sets
    (documented separately in docs/SYNTHETIC_DATA.md, notebooks/synthetic_eda.ipynb)."""
    return {k: load(k) for k in (REAL_IDS + SYNTH_IDS if include_synth else REAL_IDS)}


def miv_outcomes(run: str = "A") -> pd.DataFrame:
    name = {"A": "2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv",
            "B": "2024-11-19-MIV6.1B_all_data_delta_with_post_keep_high_conf_True_merged.csv"}[run]
    df = pd.read_csv(D / name)
    return df.rename(columns={"Participant id": "conv_id"})


# ---------------------------------------------------------------- code schemes
# Code lists transcribed from the HANDBOOKS, not derived from the data, so that a
# code no dataset happens to contain still shows up (checked 2026-10-05):
#   MISC 2.5  Houck, Moyers, Miller, Glynn & Hallgren (2010), casaa.unm.edu/assets/docs/misc25.pdf
#             counsellor categories p.16; No Code p.14; client categories pp.38-41;
#             MICO/MIIN summary groups pp.47-48; globals p.1
#   MITI 4.2.1 Moyers, Manuel & Ernst (2015), casaa.unm.edu/assets/docs/miti4_21.pdf
#             "Behavior counts" + "Global ratings" sections and the coding summary sheet
#   Welivita  Welivita & Pu (COLING 2022) Table 1 (15 labels adapted from MITI 2.0 / 4.2.1)
#   AnnoMI    Wu et al. (Future Internet 2023) Sec. 4 utterance attributes
HANDBOOK = {
    "MISC 2.5": {
        "counsellor": ["ADP", "ADW", "AF", "CO", "DI", "EC", "FA", "FI", "GI", "OQ", "CQ",
                       "RCP", "RCW", "SR", "CR", "RF", "SU", "ST", "WA"],
        "client": ["FN"] + [f"{c}{v}" for c in ("C", "R", "D", "A", "N", "TS", "O") for v in "+-"],
        "either": ["NC"],
        "globals": ["Acceptance", "Empathy", "Direction", "Autonomy Support", "Collaboration", "Evocation",
                    "Self-Exploration (client)"],
        "notes": "SR/CR require a valence (+/-/0/+-) in the manual; Ask is part of Follow/Neutral (FN).",
    },
    "MITI 4.2.1": {
        "counsellor": ["GI", "Persuade", "Persuade with Permission", "Q", "SR", "CR", "AF", "Seek",
                       "Emphasize", "Confront"],
        "client": [],
        "globals": ["Cultivating Change Talk", "Softening Sustain Talk", "Partnership", "Empathy"],
        "notes": "Clients are not coded; Q is not split into open/closed; uncodable utterances get no code.",
    },
    "Welivita (MITI-derived)": {
        "counsellor": ["Closed Question", "Open Question", "Simple Reflection", "Complex Reflection",
                       "Give Information", "Advise with Permission", "Affirm", "Emphasize Autonomy", "Support",
                       "Advise without Permission", "Confront", "Direct", "Warn", "Self-Disclose", "Other"],
        "client": [],
        "notes": "A MITI variant: labels adapted from MITI 2.0 and 4.2.1 (open/closed questions, Direct, Warn, Support from "
                 "earlier MITI versions) plus Self-Disclose and Other. Maps to MITI 4.2.1 via WELIVITA_TO_MITI. Seekers are not coded.",
    },
    "AnnoMI": {
        "counsellor": ["question:open", "question:closed", "reflection:simple", "reflection:complex",
                       "input:information", "input:advice", "input:options", "input:negotiation/goal-setting",
                       "main:question", "main:input", "main:reflection", "main:other"],
        "client": ["change", "neutral", "sustain"],
        "notes": "Question/Input/Reflection are separate attributes that can co-occur in one utterance; "
                 "a single Main Behaviour is chosen per utterance.",
    },
}

# Welivita's 15 labels (adapted from MITI 2.0 and 4.2.1) -> MITI 4.2.1, with the manual basis.
# exact 57% / approximate 34% / none 8% of listener labels (docs/experiments/2026-10-05-split-review.md section 12).
WELIVITA_TO_MITI = {
    "Closed Question": ("Q", "exact", "MITI 4.2.1 does not split open/closed"),
    "Open Question": ("Q", "exact", "MITI 4.2.1 does not split open/closed"),
    "Simple Reflection": ("SR", "exact", ""),
    "Complex Reflection": ("CR", "exact", ""),
    "Affirm": ("AF", "exact", "MITI 4.2.1 Affirm is stricter than earlier versions (p.26)"),
    "Emphasize Autonomy": ("Emphasize", "exact", ""),
    "Confront": ("Confront", "exact", ""),
    "Advise with Permission": ("PwP", "exact", "E.4.c: permission asked/given or autonomy-supportive preface"),
    "Advise without Permission": ("Persuade", "exact", "E.4.b: advice/suggestions without autonomy emphasis"),
    "Warn": ("Confront", "exact", "E.4.g.2 lists 'warning' under Confront"),
    "Support": ("NC", "exact", "p.26: statements of support are no longer coded ('I know it's really hard to stop smoking')"),
    "Other": ("NC", "exact", "F: greetings and off-topic statements are not coded"),
    "Give Information": ("GI", "approx", "Welivita GI includes opinions; MITI codes unsolicited opinions as Persuade (E.4.b)"),
    "Direct": ("Persuade", "approx", "imperatives are advice (Persuade); with disapproval they are Confront"),
    "Self-Disclose": (None, "none", "Persuade only when used to persuade (E.4.b), otherwise not coded: needs context"),
}

# Our MISC vocabulary vs the MISC 2.5 handbook (AutoMISC naming in brackets).
OUR_MISC_ALIASES = {"N": "FN", **{f"AB{v}": f"A{v}" for v in "+-"}}
MISC_EXTENSIONS = {
    "AC+": "AutoMISC addition (Activation; thesis footnote 1 cites Miller & Rollnick, Motivational Interviewing, 4th ed. 2023, mobilising change talk); not in MISC 2.5 or MISC 2.1. "
           "In the manual, 'offering alternatives' is Commitment (C+).",
    "AC-": "AutoMISC addition (Activation-); not in MISC 2.5.",
}
MISC_MICO = {"AF", "ADP", "EC", "RCP", "SU", "OQ", "SR", "CR"}      # manual p.47 (sMICO incl. OQ + reflections)
MISC_MIIN = {"ADW", "CO", "DI", "RCW", "WA"}                          # manual pp.47-48


def handbook_check(frames: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    """Every handbook code vs (a) our vocabulary and (b) what each dataset contains.

    status: 'ok' (in handbook + our vocabulary), 'not in our vocabulary',
            'extension (not in handbook)'. `datasets_with_examples` shows where it occurs;
            an empty cell is a class no dataset covers, not a class that does not exist.
    """
    from automisc_ft.data import CLIENT_GROUPS, COUNSELLOR_GROUPS
    frames = frames or {}
    ours = {("counsellor", c) for cs in COUNSELLOR_GROUPS.values() for c in cs}
    ours |= {("client", OUR_MISC_ALIASES.get(c, c)) for cs in CLIENT_GROUPS.values() for c in cs}
    observed = defaultdict_set()
    for ds, f in frames.items():
        if ds.startswith(("misc.", "synth.")):
            for spk, c in zip(f.speaker, f.t2):
                if c is not None:
                    observed[(spk, OUR_MISC_ALIASES.get(c, c) if spk == "client" else c)].add(ds)
    rows = []
    hb = HANDBOOK["MISC 2.5"]
    for spk in ("counsellor", "client"):
        for c in hb[spk]:
            grp = "MICO" if c in MISC_MICO else "MIIN" if c in MISC_MIIN else ("" if spk == "client" else "neither")
            rows.append({"scheme": "MISC 2.5", "speaker": spk, "code": c, "manual_group": grp,
                         "status": "ok" if (spk, c) in ours else "not in our vocabulary",
                         "datasets_with_examples": ", ".join(sorted(observed.get((spk, c), [])))})
    rows.append({"scheme": "MISC 2.5", "speaker": "either", "code": "NC", "manual_group": "",
                 "status": "not in our vocabulary", "datasets_with_examples": "(uncodable; rare by design)"})
    for (spk, c) in sorted(ours):
        if c not in hb.get(spk, []):
            rows.append({"scheme": "MISC 2.5", "speaker": spk, "code": c, "manual_group": "",
                         "status": "extension (not in handbook): " + MISC_EXTENSIONS.get(c, ""),
                         "datasets_with_examples": ", ".join(sorted(observed.get((spk, c), [])))})
    for scheme, ds, col in [("MITI 4.2.1", "miti.casaa.gold", "native"),
                            ("Welivita (MITI-derived)", "welivita.gold", "native"),
                            ("AnnoMI", "annomi.gold", None)]:
        seen = set()
        if ds in frames:
            f = frames[ds]
            if scheme == "AnnoMI":
                for k, colname in (("question", "question_subtype"), ("reflection", "reflection_subtype"),
                                   ("input", "therapist_input_subtype")):
                    seen |= {f"{k}:{v}" for v in f[colname].dropna().unique()}
                seen |= {f"main:{'input' if v == 'therapist_input' else v}" for v in
                         f[f.speaker == "counsellor"].native.dropna().unique()}
                seen |= set(f[f.speaker == "client"].native.dropna().unique())
            else:
                for v in f[col].dropna():
                    seen |= set(str(v).split("|"))
        canon = {"PwP": "Persuade with Permission"}
        seen = {canon.get(x, x) for x in seen}
        if scheme == "AnnoMI":
            seen = {x.replace("input:negotiation", "input:negotiation/goal-setting") for x in seen}
        hb = HANDBOOK[scheme]
        for spk in ("counsellor", "client"):
            for c in hb[spk]:
                rows.append({"scheme": scheme, "speaker": spk, "code": c, "manual_group": "", "status": "ok",
                             "datasets_with_examples": ds if c in seen else ""})
        extra = seen - set(hb["counsellor"]) - set(hb["client"])
        for c in sorted(extra):
            rows.append({"scheme": scheme, "speaker": "", "code": c, "manual_group": "",
                         "status": "in data, not in handbook" + (" (CASAA transcript convention)" if c in ("NC", "SAME") else ""),
                         "datasets_with_examples": ds})
    return pd.DataFrame(rows)


def defaultdict_set():
    from collections import defaultdict
    return defaultdict(set)


def schemes() -> Dict[str, pd.DataFrame]:
    """Every handbook code of every scheme in use, with its MISC 2.5 crosswalk."""
    from automisc_ft.data import CLIENT_GROUPS, COUNSELLOR_GROUPS
    from baseline.prep_casaa import MITI_TO_MISC
    from selftrain.ingest import MITI_TO_MISC_T2

    t1_of = {c: g for g, cs in COUNSELLOR_GROUPS.items() for c in cs}
    misc = [{"speaker": "counsellor", "code": c, "our_t1 (AutoMISC grouping)": t1_of.get(c),
             "manual_group": "MICO" if c in MISC_MICO else "MIIN" if c in MISC_MIIN else "neither",
             "source": "MISC 2.5 p.16"} for c in HANDBOOK["MISC 2.5"]["counsellor"]]
    our_client = {OUR_MISC_ALIASES.get(c, c): (g, c) for g, cs in CLIENT_GROUPS.items() for c in cs}
    for c in HANDBOOK["MISC 2.5"]["client"]:
        g, ours = our_client.get(c, (None, None))
        misc.append({"speaker": "client", "code": c + (f" (ours: {ours})" if ours and ours != c else ""),
                     "our_t1 (AutoMISC grouping)": g, "manual_group": "CT" if c.endswith("+") else "ST" if c.endswith("-") else "",
                     "source": "MISC 2.5 pp.38-41"})
    for c in ("AC+", "AC-"):
        misc.append({"speaker": "client", "code": c, "our_t1 (AutoMISC grouping)": "C" if c.endswith("+") else "S",
                     "manual_group": "-", "source": MISC_EXTENSIONS[c]})
    misc.append({"speaker": "either", "code": "NC (No Code)", "our_t1 (AutoMISC grouping)": None,
                 "manual_group": "-", "source": "MISC 2.5 p.14; not in our vocabulary"})

    miti = []
    meaning = {"GI": "Giving information", "Persuade": "Persuade", "Persuade with Permission": "Persuade with permission",
               "Q": "Question (open/closed not split)", "SR": "Simple reflection", "CR": "Complex reflection",
               "AF": "Affirm", "Seek": "Seeking collaboration", "Emphasize": "Emphasizing autonomy", "Confront": "Confront"}
    for code in HANDBOOK["MITI 4.2.1"]["counsellor"]:
        t1, t2 = MITI_TO_MISC.get("PwP" if code == "Persuade with Permission" else code, (None, None))
        miti.append({"code": code, "meaning": meaning[code], "misc_t1": t1, "misc_t2": t2,
                     "mapping": "exact" if t2 else ("T1 only" if t1 else "none"), "source": "MITI 4.2.1"})
    for code, desc in [("NC", "CASAA transcript convention: not coded (structure, greeting, facilitate)"),
                       ("SAME", "CASAA transcript convention: continues the previous coded utterance")]:
        t1, t2 = MITI_TO_MISC.get(code, (None, None))
        miti.append({"code": code, "meaning": desc, "misc_t1": t1, "misc_t2": t2,
                     "mapping": "T1 only" if t1 else "none", "source": "not a MITI code"})

    wel = []
    for code in HANDBOOK["Welivita (MITI-derived)"]["counsellor"]:
        t2 = MITI_TO_MISC_T2.get(code)
        w_miti, mq, basis = WELIVITA_TO_MITI[code]
        wel.append({"code": code, "MITI 4.2.1": w_miti, "MITI mapping": mq, "MITI basis": basis,
                    "misc_t2": t2, "misc_t1": t1_of.get(t2), "MISC mapping": "exact" if t2 else "none",
                    "source": "Welivita & Pu 2022 Table 1"})

    annomi = [
        ("therapist", "question", "open", "OQ", "exact"), ("therapist", "question", "closed", "CQ", "exact"),
        ("therapist", "reflection", "simple", "SR", "exact"), ("therapist", "reflection", "complex", "CR", "exact"),
        ("therapist", "input", "information", "GI", "exact"),
        ("therapist", "input", "advice", "ADP/ADW", "permission unknown"),
        ("therapist", "input", "negotiation/goal-setting", None, "none"),
        ("therapist", "input", "options", None, "none"),
        ("therapist", "other (main behaviour)", None, None, "none"),
        ("client", "change", None, "C (T1)", "T1 only"), ("client", "sustain", None, "S (T1)", "T1 only"),
        ("client", "neutral", None, "N (= FN)", "exact"),
    ]
    annomi = [dict(zip(["speaker", "attribute", "subtype", "misc", "mapping"], r)) for r in annomi]
    return {"MISC 2.5": pd.DataFrame(misc), "MITI 4.2.1 (CASAA)": pd.DataFrame(miti),
            "Welivita MITI-derived": pd.DataFrame(wel), "AnnoMI": pd.DataFrame(annomi)}


# ----------------------------------------------------------- agreement helpers
def fleiss(df: pd.DataFrame, item: str, label: str):
    t = pd.crosstab(df[item], df[label])
    t = t[t.sum(1) == t.sum(1).max()].values.astype(float)
    n, N = t.sum(1)[0], len(t)
    p = t.sum(0) / (N * n)
    P = ((t ** 2).sum(1) - n) / (n * (n - 1))
    return float((P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())), N


def annomi_agreement() -> Dict[str, object]:
    """Agreement on the 7 transcripts coded by all 10 AnnoMI annotators."""
    from sklearn.metrics import cohen_kappa_score
    ensure_downloads()
    f = pd.read_csv(EXT / "annomi" / "AnnoMI-full.csv")
    key = ["transcript_id", "utterance_id"]
    m = f.groupby(key).filter(lambda x: x.annotator_id.nunique() > 1).copy()
    m["item"] = m.transcript_id.astype(str) + "_" + m.utterance_id.astype(str)
    th, cl = m[m.interlocutor == "therapist"].copy(), m[m.interlocutor == "client"].copy()
    out = {"transcripts": sorted(m.transcript_id.unique().tolist()),
           "therapist_items": th.item.nunique(), "client_items": cl.item.nunique(),
           "fleiss_main": fleiss(th, "item", "main_therapist_behaviour")[0],
           "fleiss_client": fleiss(cl, "item", "client_talk_type")[0]}
    for c in ["reflection_subtype", "question_subtype", "therapist_input_subtype"]:
        th[c + "_f"] = th[c].fillna("none")
        out["fleiss_" + c] = fleiss(th, "item", c + "_f")[0]

    def loo(df, label):
        res = {}
        for a in sorted(df.annotator_id.unique()):
            me = df[df.annotator_id == a].set_index("item")[label]
            maj = df[df.annotator_id != a].groupby("item")[label].agg(lambda s: s.value_counts().index[0])
            res[int(a)] = round(cohen_kappa_score(me, maj.loc[me.index]), 3)
        return res
    out["loo_kappa_main"] = loo(th, "main_therapist_behaviour")
    out["loo_kappa_client"] = loo(cl, "client_talk_type")
    out["complex_share_same_items"] = th.groupby("annotator_id").reflection_subtype.apply(
        lambda s: (s == "complex").sum() / max(1, s.notna().sum())).round(3).to_dict()
    out["change_share_same_items"] = cl.groupby("annotator_id").client_talk_type.apply(
        lambda s: (s == "change").mean()).round(3).to_dict()
    return out


def welivita_agreement() -> Dict[str, float]:
    from sklearn.metrics import cohen_kappa_score
    ensure_downloads()
    w = pd.read_csv(EXT / "welivita" / "MI_Dataset.csv")
    lis = w[w.author == "listener"].dropna(subset=["ann1", "ann2"])
    return {"n": len(lis), "raw_agreement": float(np.mean(lis.ann1 == lis.ann2)),
            "kappa": float(cohen_kappa_score(lis.ann1, lis.ann2)),
            "stage1_agreed": int((lis.ann1 == lis.ann2).sum())}
