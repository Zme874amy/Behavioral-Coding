"""Annotation-quality audit of the gold MISC sets, aimed at what can hurt training.

Five lenses, each returning plain DataFrames for the notebook (§5) and DATASETS.md:

  codebook_audit     rule checks taken from the MISC 2.5 manual (Houck et al. 2010)
                     that can be tested on text alone (FA vs FI, the 5% FI ceiling,
                     CQ vs OQ grammar rules, spoiled reflections, ruler answers,
                     replies to client questions). Heuristics: they flag candidate
                     violations, not proven errors.
  identical_text     the same (normalised) utterance labelled differently, within a
                     dataset and between the train and test sets.
  cross_annotation   the same SESSION coded independently by someone else: HLQC gold
                     vs AnnoMI experts (4 shared sessions) and vs the CASAA MITI
                     reference coding (Emmy = high_121). Utterances are aligned by
                     word-level sequence matching, which also yields the ASR word
                     error rate of HLQC against the manual transcripts.
  shift              train (HLQC) vs test (MIV6.3A) label priors, codes present in only
                     one of them, and the training mass spent on codes the test never has.
  concentration      how many sessions each training code comes from, and whether the
                     MI-inconsistent codes come only from low-quality sessions.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from eda import registry

_N = re.compile(r"[^a-z0-9' ]")
ACK = {"ok", "okay", "right", "mmhmm", "mmhm", "mhm", "mm", "hmm", "uhhuh", "uh huh", "mm hmm", "yeah", "yep",
       "yup", "yes", "got it", "i see", "good", "alright", "all right", "sure", "oh", "wow", "yeah yeah",
       "okay okay", "right right", "yeah okay", "okay yeah", "uh", "um", "great", "okay great"}
AUX = re.compile(r"^(?:so |and |but |well |okay |ok |now )*(?:do|does|did|is|are|was|were|can|could|would|will|"
                 r"have|has|had|should|shall|may|might|am)\b")
WH_OPEN = re.compile(r"^(?:so |and |but |well |okay |ok |now )*(?:what|how|why|tell me|i'm wondering|i wonder|"
                     r"i'd like to know)\b")


PERMISSION = (r"\b(?:would it be (?:ok|okay|alright|all right)|is it (?:ok|okay|alright|all right) if|do you mind if|"
              r"would you mind if|can i (?:share|tell|ask|offer|give|suggest)|may i|would you like (?:me )?to "
              r"(?:hear|talk|explore|go over|discuss)|would you be (?:open|interested|willing))")


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", _N.sub("", str(t).lower().replace("-", ""))).strip()


def _gold(ds: str) -> pd.DataFrame:
    d = registry.load(ds).copy()
    d["k"] = d.text.map(norm)
    d["prev_speaker"] = d.groupby("conv_id").speaker.shift()
    d["prev_text"] = d.groupby("conv_id").text.shift()
    return d


# ------------------------------------------------------------------ codebook
def codebook_audit(datasets=("misc.hlqc.gold", "misc.miv63a.gold")) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        d = _gold(ds)
        c = d[d.speaker == "counsellor"]
        cl = d[d.speaker == "client"]

        def add(rule, manual, subset, ok_mask, note=""):
            n = len(subset)
            bad = subset[~ok_mask]
            rows.append({"dataset": ds, "rule": rule, "manual says": manual, "applicable": n,
                         "consistent": int(ok_mask.sum()), "flagged": len(bad),
                         "flagged %": round(100 * len(bad) / n, 1) if n else np.nan,
                         "flagged label mix": dict(Counter(bad.t2).most_common(5)), "note": note})

        ack = c[c.k.isin(ACK)]
        after_q = ack.prev_speaker.eq("client") & ack.prev_text.astype(str).str.strip().str.endswith("?")
        add("standalone acknowledgement ('okay', 'yeah', 'mm-hmm', 'right' ...)",
            "FA (p.22: 'keep-going acknowledgments ... OK, Right, Mm Hmm'); GI if answering a client question",
            ack, ack.t2.isin(["FA"]) | (after_q & ack.t2.eq("GI")) | ack.t2.isin(["AF", "SU"]) & ack.k.isin({"good", "great", "wow"}))
        n_fi = int((c.t2 == "FI").sum())
        rows.append({"dataset": ds, "rule": "share of counsellor utterances coded FI",
                     "manual says": "FI is rare: 'if these exceed 5% of Counselor responses, they probably are being over-coded' (p.24)",
                     "applicable": len(c), "consistent": np.nan, "flagged": n_fi,
                     "flagged %": round(100 * n_fi / len(c), 1), "flagged label mix": {}, "note": "ceiling = 5%"})
        q = c[c.t2.isin(["OQ", "CQ"])]
        aux = q[q.k.str.match(AUX)]
        add("question starting with an auxiliary verb ('do you', 'could you', 'is it' ...)",
            "CQ: 'if the question can be answered by yes/no, it is a closed question'; 'Could you tell me...' = CQ (p.28)",
            aux, aux.t2.eq("CQ"))
        tell = q[q.k.str.match(r"^(?:so |and |well |okay )*tell me\b")]
        add("'tell me ...' requests", "OQ ('Tell me more.' / 'Tell me about your family.' = OQ, p.26)", tell, tell.t2.eq("OQ"),
            "exception: 'tell me how old you are' = CQ")
        scale = q[q.k.str.contains(r"scale|one to ten|1 to 10|zero to ten|0 to 10")]
        add("scaling / ruler questions", "CQ ('On a scale of 1 to 10, how much ...?' = CQ, p.28)", scale, scale.t2.eq("CQ"))
        refl_q = c[c.text.astype(str).str.strip().str.endswith("?") & c.t2.isin(["SR", "CR"])]
        allq = c[c.text.astype(str).str.strip().str.endswith("?")]
        add("utterance ending in '?' coded as a reflection",
            "a reflection with upward inflection is a (closed) question ('spoiled reflection' = CQ, p.27)",
            allq, ~allq.t2.isin(["SR", "CR"]), f"{len(refl_q)} of {len(allq)} '?'-utterances coded SR/CR")
        reply = c[c.prev_speaker.eq("client") & c.prev_text.astype(str).str.strip().str.endswith("?")
                  & c.k.str.split().str.len().le(3)]
        add("short counsellor reply right after a client question",
            "GI ('Responses to client questions are typically coded as Giving Information, even if brief', p.23)",
            reply, reply.t2.eq("GI"), "only checkable where the client question has a '?'")
        nums = cl[cl.k.str.fullmatch(r"(?:(?:about|maybe|probably|i'd say|i would say|like|around|uh|um) )*"
                                     r"(?:a |an )?(zero|one|two|three|four|five|six|seven|eight|nine|ten|\d{1,2})")]
        val = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}

        def expect(t):
            w = t.split()[-1]
            v = int(w) if w.isdigit() else val[w]
            return "S" if v <= 4 else ("N" if v == 5 else "C")
        exp = nums.k.map(expect)
        add("client ruler answer that is just a number", "0-4 = sustain talk, 5 = FN, 6-10 = change talk (p.40)",
            nums, nums.t1.astype(str).values == exp.values)
        perm = c[c.k.str.contains(PERMISSION)]
        add("permission-seeking ('would it be okay if...', 'do you mind if...', 'can I share...')",
            "EC: 'permission-seeking utterances are coded as a separate utterance of Emphasize Control' (p.16)",
            perm, perm.t2.eq("EC"))
        q_no_mark = q[~q.text.astype(str).str.contains(r"\?")]
        rows.append({"dataset": ds, "rule": "questions (OQ/CQ gold) with no '?' in the text",
                     "manual says": "(transcript property, not a rule) - the model cannot see the question mark",
                     "applicable": len(q), "consistent": len(q) - len(q_no_mark), "flagged": len(q_no_mark),
                     "flagged %": round(100 * len(q_no_mark) / len(q), 1) if len(q) else np.nan,
                     "flagged label mix": dict(Counter(q_no_mark.t2)), "note": "ASR transcripts drop punctuation"})
    return pd.DataFrame(rows)


def fi_content(datasets=("misc.hlqc.gold", "misc.miv63a.gold")) -> pd.DataFrame:
    """What FI actually contains in each set: acknowledgement / pleasantry / fragment / other."""
    pleas = re.compile(r"\b(?:hello|hi|hey|good (?:morning|afternoon|evening)|nice to meet|thank|thanks|bye|goodbye|"
                       r"take care|welcome|how are you|glad|pleasure|have a (?:good|great|nice))\b")
    out = []
    for ds in datasets:
        d = _gold(ds)
        fi = d[(d.speaker == "counsellor") & (d.t2 == "FI")]
        kind = np.select([fi.k.isin(ACK), fi.k.str.contains(pleas), fi.k.str.split().str.len() <= 3],
                         ["acknowledgement (codebook: FA)", "pleasantry (codebook: FI)", "short fragment (<=3 words)"],
                         "other / longer statement")
        vc = pd.Series(kind).value_counts()
        out.append((vc / vc.sum()).round(3).rename(f"{ds} (n={len(fi)})"))
    return pd.DataFrame(out).fillna(0).T


# ------------------------------------------------------------- identical text
def identical_text(ds: str = "misc.hlqc.gold", min_n: int = 3) -> pd.DataFrame:
    d = _gold(ds)
    g = d.groupby(["speaker", "k"]).t2.agg(n="size", labels=lambda s: dict(Counter(s).most_common()),
                                           n_labels="nunique").reset_index()
    g = g[(g.n >= min_n)]
    g["majority share"] = g.labels.map(lambda m: round(max(m.values()) / sum(m.values()), 2))
    return g.sort_values(["n_labels", "n"], ascending=[False, False]).reset_index(drop=True)


def train_test_convention(train="misc.hlqc.gold", test="misc.miv63a.gold", min_n=2) -> pd.DataFrame:
    """Utterances whose normalised text occurs in BOTH sets, with each set's labels."""
    a, b = _gold(train), _gold(test)
    ga = a.groupby(["speaker", "k"]).t2.agg(lambda s: dict(Counter(s).most_common()))
    gb = b.groupby(["speaker", "k"]).t2.agg(lambda s: dict(Counter(s).most_common()))
    j = pd.concat([ga.rename("train labels"), gb.rename("test labels")], axis=1, join="inner").reset_index()
    j["n"] = j["train labels"].map(lambda m: sum(m.values())) + j["test labels"].map(lambda m: sum(m.values()))
    j["same majority"] = [max(x, key=x.get) == max(y, key=y.get) for x, y in zip(j["train labels"], j["test labels"])]
    return j[j.n >= min_n].sort_values("n", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------- cross-annotation
def _tokens(df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    toks, owner = [], []
    for i, t in enumerate(df.text.astype(str)):
        for w in norm(t).split():
            toks.append(w)
            owner.append(i)
    return toks, owner


def align(a: pd.DataFrame, b: pd.DataFrame, min_cover: float = 0.5) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Map each utterance of `a` to the utterance of `b` holding most of its matched words.
    Returns the pairs and word-level stats (WER of a against b as reference)."""
    ta, oa = _tokens(a)
    tb, ob = _tokens(b)
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    match = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            match[blk.a + k] = blk.b + k
    S = D = I = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace":
            S += min(i2 - i1, j2 - j1); D += max(0, (j2 - j1) - (i2 - i1)); I += max(0, (i2 - i1) - (j2 - j1))
        elif tag == "delete":
            I += i2 - i1          # words in `a` (ASR) absent from the reference
        elif tag == "insert":
            D += j2 - j1          # reference words the ASR missed
    stats = {"words_hyp": len(ta), "words_ref": len(tb), "matched": len(match),
             "WER": round((S + D + I) / max(1, len(tb)), 3)}
    per = {}
    for ia, ib in match.items():
        per.setdefault(oa[ia], Counter())[ob[ib]] += 1
    n_words = Counter(oa)
    rows = []
    for ua, cnt in per.items():
        ub, hit = cnt.most_common(1)[0]
        if hit / n_words[ua] >= min_cover:
            rows.append({"a_idx": ua, "b_idx": ub, "cover": round(hit / n_words[ua], 2)})
    return pd.DataFrame(rows), stats


from schemes.mappings import MISC_TO_ANNOMI, MISC_TO_MITI  # noqa: E402


def cross_annotation() -> Dict[str, object]:
    """HLQC gold labels vs independent coders on the same sessions."""
    from sklearn.metrics import cohen_kappa_score
    frames = {k: registry.load(k) for k in ("misc.hlqc.gold", "annomi.gold", "miti.casaa.gold")}
    h = frames["misc.hlqc.gold"]
    pairs = [("low_001", "53"), ("low_080", "15"), ("high_099", "21"), ("low_033", "44")]
    rows, wer = [], []
    for hid, aid in pairs:
        a = h[h.conv_id == hid].reset_index(drop=True)
        b = frames["annomi.gold"][frames["annomi.gold"].conv_id == aid].reset_index(drop=True)
        m, st = align(a, b)
        wer.append({"hlqc": hid, "reference": f"AnnoMI {aid}", **st})
        for r in m.itertuples():
            x, y = a.loc[r.a_idx], b.loc[r.b_idx]
            if x.speaker != y.speaker:
                continue
            rows.append({"session": hid, "ref_unit": f"{aid}:{r.b_idx}", "n_words": len(norm(x.text).split()),
                         "speaker": x.speaker, "hlqc_t2": x.t2, "hlqc_t1": x.t1,
                         "annomi": y.native, "annomi_sub": (y.question_subtype or y.reflection_subtype
                                                            or y.therapist_input_subtype) if x.speaker == "counsellor" else None,
                         "annomi_annotator": y.annotator_id, "hlqc_text": x.text, "annomi_text": y.text})
    am = pd.DataFrame(rows)
    c = am[am.speaker == "counsellor"].copy()
    c["hlqc_as_annomi"] = c.hlqc_t2.map(MISC_TO_ANNOMI).fillna("other")
    cl = am[am.speaker == "client"].copy()
    cl["hlqc_as_annomi"] = cl.hlqc_t1.map({"C": "change", "S": "sustain", "N": "neutral"})
    res = {"wer": pd.DataFrame(wer), "annomi_pairs": am,
           "annomi_counsellor_utterance": _agree(c.hlqc_as_annomi, c.annomi),
           "annomi_counsellor": turn_level(c, "annomi", "hlqc_as_annomi"),
           "annomi_client": turn_level(cl, "annomi", "hlqc_as_annomi", ignore=())}
    sub = c[c.hlqc_t2.isin(["OQ", "CQ"]) & c.annomi_sub.isin(["open", "closed"])]
    res["annomi_open_closed"] = _agree(sub.hlqc_t2.map({"OQ": "open", "CQ": "closed"}), sub.annomi_sub)
    sub = c[c.hlqc_t2.isin(["SR", "CR"]) & c.annomi_sub.isin(["simple", "complex"])]
    res["annomi_simple_complex"] = _agree(sub.hlqc_t2.map({"SR": "simple", "CR": "complex"}), sub.annomi_sub)

    a = h[h.conv_id == "high_121"].reset_index(drop=True)
    b = frames["miti.casaa.gold"][frames["miti.casaa.gold"].conv_id == "casaa_emmys-first-encounter"].reset_index(drop=True)
    m, st = align(a, b)
    wer.append({"hlqc": "high_121", "reference": "CASAA Emmy", **st})
    res["wer"] = pd.DataFrame(wer)
    rows = []
    for r in m.itertuples():
        x, y = a.loc[r.a_idx], b.loc[r.b_idx]
        if x.speaker == "counsellor" and y.speaker == "counsellor" and y.native and "|" not in str(y.native) \
                and y.native != "SAME":
            rows.append({"ref_unit": r.b_idx, "n_words": len(norm(x.text).split()), "hlqc_t2": x.t2,
                         "hlqc_as_miti": MISC_TO_MITI.get(x.t2, "other"), "casaa_miti": y.native,
                         "hlqc_text": x.text})
    cm = pd.DataFrame(rows)
    res["casaa_pairs"] = cm
    res["casaa_counsellor"] = turn_level(cm, "casaa_miti", "hlqc_as_miti")
    return res


def turn_level(pairs: pd.DataFrame, ref_col: str, hlqc_col: str, ignore=("other", "NC")) -> Dict[str, object]:
    """Compare at the reference coder's unit (a turn). AnnoMI / CASAA code a whole turn
    once; HLQC codes each utterance, so 'okay' + question is FI + CQ in HLQC and a single
    'question' in AnnoMI. A turn agrees if the reference label is among HLQC's labels
    for the utterances aligned into it; HLQC's 'main' label is its longest utterance
    that is not a filler/other code."""
    rows = []
    for _, g in pairs.groupby("ref_unit"):
        ref = g[ref_col].iloc[0]
        labs = list(g[hlqc_col])
        subst = g[~g[hlqc_col].isin(ignore)]
        main = subst.loc[subst.n_words.idxmax(), hlqc_col] if len(subst) else g[hlqc_col].iloc[0]
        rows.append({"ref": ref, "hlqc_main": main, "contains": ref in labs, "n_hlqc_utts": len(g)})
    t = pd.DataFrame(rows)
    out = _agree(t.hlqc_main, t.ref)
    out["turns"] = len(t)
    out["ref label found among HLQC labels in the turn"] = round(float(t.contains.mean()), 3)
    out["HLQC utterances per reference turn"] = round(float(t.n_hlqc_utts.mean()), 2)
    return out


def _agree(a: pd.Series, b: pd.Series) -> Dict[str, object]:
    from sklearn.metrics import cohen_kappa_score
    a, b = a.astype(str), b.astype(str)
    if len(a) == 0:
        return {"n": 0}
    return {"n": len(a), "agreement": round(float((a.values == b.values).mean()), 3),
            "kappa": round(float(cohen_kappa_score(a, b)), 3),
            "confusion": pd.crosstab(a.rename("HLQC gold"), b.rename("independent coder"))}


# -------------------------------------------------------------- segmentation
def segmentation(datasets=("misc.hlqc.gold", "misc.miv63a.gold")) -> pd.DataFrame:
    """How the transcripts were cut into coded utterances. Over-splitting turns one
    complex reflection into several 'simple' fragments and makes fillers out of pieces."""
    rows = []
    for ds in datasets:
        d = _gold(ds)
        c = d[d.speaker == "counsellor"].copy()
        w = c.k.str.split().str.len()
        per_turn = c.groupby(["conv_id", "vol_idx"]).size()
        same_turn_prev = c.conv_id.eq(c.conv_id.shift()) & c.vol_idx.eq(c.vol_idx.shift())
        refl = c.t2.isin(["SR", "CR"])
        rows.append({
            "dataset": ds,
            "counsellor utterances": len(c),
            "utterances per counsellor turn (mean)": round(per_turn.mean(), 2),
            "turns split into >= 3 utterances %": round(100 * (per_turn >= 3).mean(), 1),
            "utterances <= 3 words %": round(100 * (w <= 3).mean(), 1),
            "utterances starting with and/but/or/so/because %": round(
                100 * c.k.str.match(r"^(and|but|or|so|because)\b").mean(), 1),
            "reflections that continue a reflection in the same turn %": round(
                100 * (refl & same_turn_prev & c.t2.shift().isin(["SR", "CR"])).sum() / max(1, refl.sum()), 1),
            "SR : CR": f"{int((c.t2 == 'SR').sum())} : {int((c.t2 == 'CR').sum())}",
        })
    return pd.DataFrame(rows).set_index("dataset").T


def reflection_style(datasets=("misc.hlqc.gold", "misc.miv63a.gold")) -> pd.DataFrame:
    """Length of simple vs complex reflections in each set (the SR/CR boundary)."""
    rows = []
    for ds in datasets:
        d = _gold(ds)
        c = d[d.speaker == "counsellor"]
        w = c.k.str.split().str.len()
        for code in ("SR", "CR"):
            x = w[c.t2 == code]
            rows.append({"dataset": ds, "code": code, "n": len(x), "median words": float(x.median()),
                         ">= 15 words %": round(100 * (x >= 15).mean(), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- shift
def shift(train="misc.hlqc.gold", test="misc.miv63a.gold") -> pd.DataFrame:
    a, b = registry.load(train), registry.load(test)
    rows = []
    for spk in ("counsellor", "client"):
        pa = a[a.speaker == spk].t2.value_counts(normalize=True)
        pb = b[b.speaker == spk].t2.value_counts(normalize=True)
        na = a[a.speaker == spk].t2.value_counts()
        nb = b[b.speaker == spk].t2.value_counts()
        for code in pa.index.union(pb.index):
            rows.append({"speaker": spk, "code": code, "train n": int(na.get(code, 0)), "test n": int(nb.get(code, 0)),
                         "train %": round(100 * pa.get(code, 0), 1), "test %": round(100 * pb.get(code, 0), 1),
                         "ratio train/test": round(pa.get(code, 0) / pb.get(code, np.nan), 2) if pb.get(code, 0) else np.inf,
                         "only in": "train" if code not in pb else ("test" if code not in pa else "")})
    return pd.DataFrame(rows).sort_values(["speaker", "train n"], ascending=[False, False]).reset_index(drop=True)


# -------------------------------------------------------------- concentration
def concentration(ds="misc.hlqc.gold") -> pd.DataFrame:
    d = registry.load(ds)
    rows = []
    for (spk, code), g in d.groupby(["speaker", "t2"]):
        per = g.conv_id.value_counts()
        rows.append({"speaker": spk, "code": code, "n": len(g), "sessions": len(per),
                     "top session": per.index[0], "top session share": round(per.iloc[0] / len(g), 2),
                     "from low-quality sessions %": round(100 * g.conv_id.str.startswith("low").mean(), 0)})
    return pd.DataFrame(rows).sort_values(["speaker", "n"], ascending=[False, False]).reset_index(drop=True)


# ------------------------------------------------------------------ impact
MAIN_PREDS = {  # ft1mix_bare (main setting), MIV6.3A test predictions, three training seeds
    "seed42": "data/annotated/baseline/qwen_ft1mix_bare_inf_bare_ctx5.csv",
    "seed1": "data/annotated/base_1a_s1/qwen_ft1mix_bare_inf_bare_ctx5.csv",
    "seed2": "data/annotated/base_1a_s2/qwen_ft1mix_bare_inf_bare_ctx5.csv",
}


def impact(preds: Dict[str, str] = None) -> Dict[str, pd.DataFrame]:
    """Do the HLQC quality issues show up as errors of the main model on the MIV6.3A test?"""
    preds = preds or MAIN_PREDS
    gold = _gold("misc.miv63a.gold")
    pleas = re.compile(r"\b(?:hello|hi|hey|good (?:morning|afternoon|evening)|nice to meet|thank|thanks|bye|goodbye|"
                       r"take care|welcome|how are you|glad|pleasure)\b")
    frames = []
    for name, path in preds.items():
        p = pd.read_csv(registry.REPO / path)[["conv_id", "corp_utt_idx", "t2_label_auto"]]
        frames.append(p.rename(columns={"t2_label_auto": "pred"}).assign(seed=name))
    P = pd.concat(frames)
    g = pd.read_csv(registry.D / "manual" / "MIV6.3A_manual.csv")[["conv_id", "corp_utt_idx", "utt_text", "speaker", "t2_label_GT"]]
    m = P.merge(g, on=["conv_id", "corp_utt_idx"], how="inner")
    m["speaker"] = m.speaker.map(lambda s: "counsellor" if str(s).lower().startswith("couns") else "client")
    m["k"] = m.utt_text.map(norm)
    out = {}
    rec = (m.assign(ok=m.pred == m.t2_label_GT).groupby(["speaker", "t2_label_GT"])
           .agg(n_per_seed=("ok", lambda s: len(s) // len(preds)), recall=("ok", "mean")).round(3).reset_index())
    out["recall_by_code"] = rec.sort_values(["speaker", "n_per_seed"], ascending=[False, False])
    fi = m[(m.t2_label_GT == "FI")]
    fi = fi.assign(kind=np.where(fi.k.str.contains(pleas), "pleasantry", "other FI"))
    out["gold_FI_predicted_as"] = pd.crosstab(fi.kind, fi.pred, normalize="index").round(2)
    ct = m[(m.speaker == "client") & (m.t2_label_GT != "N")]
    out["gold_change_sustain_predicted_as"] = (ct.pred.value_counts(normalize=True).round(3)
                                               .rename("share of gold change/sustain-talk predictions").to_frame())
    train_only = {"ADW", "RCW", "WA", "CO", "O-", "N+", "C-"}
    fp = m[m.pred.isin(train_only)]
    out["predictions_of_train_only_codes"] = (fp.groupby(["pred", "t2_label_GT"]).size()
                                              .rename("count over 3 seeds").reset_index())
    out["summary"] = pd.DataFrame([{
        "test utterances x seeds": len(m),
        "gold FI pleasantries predicted FI": round(float((fi[fi.kind == 'pleasantry'].pred == 'FI').mean()), 3),
        "gold client change/sustain predicted N": round(float((ct.pred == "N").mean()), 3),
        "gold SU recall": round(float((m[m.t2_label_GT == 'SU'].pred == 'SU').mean()), 3),
        "gold EC recall": round(float((m[m.t2_label_GT == 'EC'].pred == 'EC').mean()), 3),
        "predictions of train-only codes (per seed)": round(len(fp) / len(preds), 1),
    }]).T.rename(columns={0: "value"})
    return out
