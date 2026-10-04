"""One canonical ID, one loader and one metadata record per available dataset.

Why this exists: the same corpus lives in several files under unrelated names
(`HLQC.csv`, `parsed/HLQC_parsed.csv`, `manual/HLQC_balanced_manual.csv` are a
volley-level pool, an utterance-level pool and a 10-session gold subset), and
`synth/aug/*_proto_eval.csv` is synthetic TRAINING data, not an eval set. Every
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
        scheme="MISC 2.5, two-tier (T1 group -> T2 code), both speakers", unit="utterance (thought unit), single label",
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
        language="English", scheme="MISC 2.5, two-tier, both speakers", unit="utterance, single label",
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
        scheme="MITI-derived, 15 listener codes; seekers uncoded", unit="sentence-level listener segment",
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
    raw["stage1_agreed"] = raw["stage I agreed label"].notna() & (raw["stage I agreed label"].astype(str) != "-")
    return _std(raw, "welivita.gold", "dialog_id", "author", "turn", None, "text", "native", None, "t2",
                extra=("ann1", "ann2", "source", "stage1_agreed"))


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


def load_all() -> Dict[str, pd.DataFrame]:
    return {k: load(k) for k in LOADERS}


def miv_outcomes(run: str = "A") -> pd.DataFrame:
    name = {"A": "2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv",
            "B": "2024-11-19-MIV6.1B_all_data_delta_with_post_keep_high_conf_True_merged.csv"}[run]
    df = pd.read_csv(D / name)
    return df.rename(columns={"Participant id": "conv_id"})


# ---------------------------------------------------------------- code schemes
def schemes() -> Dict[str, pd.DataFrame]:
    """Every code of every scheme in use, with its MISC 2.5 crosswalk."""
    from automisc_ft.data import CLIENT_GROUPS, COUNSELLOR_GROUPS
    from baseline.prep_casaa import MITI_TO_MISC
    from selftrain.ingest import MITI_TO_MISC_T2

    misc = [{"speaker": "counsellor", "t1": g, "t2": c} for g, cs in COUNSELLOR_GROUPS.items() for c in cs]
    misc += [{"speaker": "client", "t1": g, "t2": c} for g, cs in CLIENT_GROUPS.items() for c in cs]
    t1_of = {r["t2"]: r["t1"] for r in misc if r["speaker"] == "counsellor"}

    miti = []
    for code, desc in [("GI", "Giving information"), ("Persuade", "Persuade (incl. unsolicited advice)"),
                       ("PwP", "Persuade with permission"), ("Q", "Question (open/closed not split)"),
                       ("SR", "Simple reflection"), ("CR", "Complex reflection"), ("AF", "Affirm"),
                       ("Seek", "Seeking collaboration"), ("Emphasize", "Emphasizing autonomy"),
                       ("Confront", "Confront"), ("NC", "Not coded (structure, greeting, facilitate)"),
                       ("SAME", "CASAA convention: continues the previous coded utterance")]:
        t1, t2 = MITI_TO_MISC.get(code, (None, None))
        miti.append({"code": code, "meaning": desc, "misc_t1": t1, "misc_t2": t2,
                     "mapping": "exact" if t2 else ("T1 only" if t1 else "none")})

    wel = [{"code": k, "misc_t2": v, "misc_t1": t1_of.get(v), "mapping": "exact"} for k, v in MITI_TO_MISC_T2.items()]
    wel.append({"code": "Self-Disclose", "misc_t2": None, "misc_t1": None, "mapping": "none"})
    wel.append({"code": "Other", "misc_t2": None, "misc_t1": None, "mapping": "none"})

    annomi = [
        ("therapist", "question", "open", "OQ", "exact"), ("therapist", "question", "closed", "CQ", "exact"),
        ("therapist", "reflection", "simple", "SR", "exact"), ("therapist", "reflection", "complex", "CR", "exact"),
        ("therapist", "therapist_input", "information", "GI", "exact"),
        ("therapist", "therapist_input", "advice", "ADP/ADW", "permission unknown"),
        ("therapist", "therapist_input", "negotiation", None, "none"),
        ("therapist", "therapist_input", "options", None, "none"),
        ("therapist", "other", None, None, "none"),
        ("client", "change", None, "C (T1)", "T1 only"), ("client", "sustain", None, "S (T1)", "T1 only"),
        ("client", "neutral", None, "N", "exact"),
    ]
    annomi = [dict(zip(["speaker", "main", "subtype", "misc", "mapping"], r)) for r in annomi]
    return {"MISC 2.5": pd.DataFrame(misc), "MITI 4 (CASAA)": pd.DataFrame(miti),
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
