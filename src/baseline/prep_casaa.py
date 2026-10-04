"""Parse the CASAA MITI 4 coded training transcripts into a `load_manual` eval CSV.

Source: UNM Center on Alcohol, Substance use, And Addictions (CASAA), MITI coder
training materials, https://casaa.unm.edu/tools/miti.html. Each transcript is a PDF
table with columns turn #, speaker (P/I provider or C client), utterance text, MITI
4 code(s) and sometimes a free-text notes column. The codes are the reference
coding by the MITI developers' own lab, so this is the cleanest gold that exists,
but it is small (20 transcripts) and counsellor-only (MITI does not code clients).

Parsing works on word coordinates (pdfplumber), not on extracted text, because the
code and notes columns interleave with wrapped utterance lines in plain text:
  * the code column starts at the left edge of the right-hand cluster of code
    tokens; the notes column, when the table has one, at its header word;
  * a row starts on a line whose first word is the turn number and second the
    speaker; everything up to the next row (across page breaks) belongs to it;
  * repeated page headers/footers are dropped;
  * in a code cell, parenthesised remarks are removed and lines are read top-down
    until the first line that is not a pure code; the rest is a coder note
    ("Persuade ruled out because ..." is a note, not a Persuade code).

MITI -> MISC 2.5 mapping, exact concepts only (as in `baseline.prep_crossscheme`):
  SR->SRL/SR, CR->CRL/CR, AF->CRL/AF, Emphasize->CRL/EC, GI->IMC/GI,
  Confront->IMI/CO, Q->Q (T1 only: MITI 4 does not split open/closed),
  NC->O (T1 only: MITI's uncoded structure/greeting/facilitate = MISC FA/FI/ST).
  Persuade, Persuade-with-permission and Seek have no exact MISC counterpart:
  they keep `miti_code` but get no MISC gold.

Two transcripts are HLQC sessions (`hlqc_overlap`): Emmy's First Encounter is
HLQC high_121, which is in our HLQC_balanced_manual training set; the Rounder is
high_072. Three PDFs skip turn 25 in the source itself (not a parse loss).

A turn with one code is one utterance. A turn with several codes is split into
sentences only when the sentence count equals the code count (`align=sentence`);
otherwise it stays one row with all codes in `miti_codes` and no gold.

Outputs (gitignored; the transcripts are CASAA's, not ours to redistribute):
  data/external/casaa/CASAA_eval.csv     load_manual schema + miti columns
  data/external/casaa/casaa_turns.csv    one row per turn, with notes, for audit
  data/external/casaa/casaa_globals.csv  MITI global ratings where the transcript has them

Usage:
    PYTHONPATH=src python -m baseline.prep_casaa            # download + parse
    PYTHONPATH=src python -m baseline.prep_casaa --no-download
"""
from __future__ import annotations

import argparse
import re
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

from baseline.local_arm import REPO_ROOT
from baseline.prep_crossscheme import SCHEMA

OUT_DIR = REPO_ROOT / "data" / "external" / "casaa"
PDF_DIR = OUT_DIR / "pdf"
BASE_URL = "https://casaa.unm.edu/assets/docs/"

# The 20 MITI 4 coded transcripts on the CASAA MITI page (2026-10-04).
TRANSCRIPTS = [
    "emmys-first-encounter-coded-transcript", "have-you-ever-been-a-smoker-coded-transcript",
    "hawaii-every-month-coded-transcript", "i-guess-i-dont-have-a-choice-coded-transcript",
    "living-with-diabetes-coded-transcript", "maybe-hell-take-my-kids-coded-transcript",
    "my-father-also-hit-me-coded-transcript", "now-i-only-wear-yoga-pants-coded-transcript",
    "orientation-to-a-parenting-class-coded-transcript", "overuse-of-directing-coded-transcript",
    "overuse-of-following-coded-transcript", "rounder-miti-4-coded-transcript",
    "should-i-really-be-honest-coded-transcript", "the-confirmed-smoker-coded-transcript",
    "the-suspiscious-smoker-coded-transcript",
    "session1miti", "session2miti", "session3miti", "session4miti", "session5miti",
]

# Surface form (lowercased letters only) -> canonical MITI 4 code. "SAME" marks a
# provider turn that continues the previous, already-coded utterance.
CODE_FORMS = {
    "q": "Q", "question": "Q",
    "sr": "SR", "cr": "CR",
    "gi": "GI",
    "af": "AF", "aff": "AF", "affirm": "AF",
    "seek": "Seek",
    "emphasize": "Emphasize", "emph": "Emphasize", "emp": "Emphasize", "emphasise": "Emphasize",
    "persuade": "Persuade", "pers": "Persuade",
    "pwp": "PwP",
    "confront": "Confront", "con": "Confront",
    "nc": "NC", "structure": "NC",
    "same": "SAME",
}
MULTI_FORMS = {("persuade", "with", "permission"): "PwP", ("persuasion", "with", "permission"): "PwP",
               ("not", "coded"): "NC"}
CANCEL = {("ruled", "out"), ("rule", "out"), ("not", "coded")}
MITI_TO_MISC = {  # canonical MITI code -> (T1, T2 or None)
    "SR": ("SRL", "SR"), "CR": ("CRL", "CR"), "AF": ("CRL", "AF"),
    "Emphasize": ("CRL", "EC"), "GI": ("IMC", "GI"), "Confront": ("IMI", "CO"),
    "Q": ("Q", None), "NC": ("O", None),
}
# Same session as an HLQC transcript (5-gram overlap with the HLQC ASR text, checked
# 2026-10-04). Drop these from evaluation of anything trained on HLQC.
HLQC_OVERLAP = {
    "casaa_emmys-first-encounter": "high_121",  # 0.55 5-gram hit rate; in HLQC_balanced_manual (train)
    "casaa_rounder-miti-4": "high_072",         # HLQC holds only the first ~900 words
}
SPEAKERS = {"P": "counsellor", "I": "counsellor", "T": "counsellor", "C": "client"}
NOTES_HEADERS = {"explanation", "notes", "note", "comments", "rationale"}


def download(names=TRANSCRIPTS) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for n in names:
        dst = PDF_DIR / f"{n}.pdf"
        if not dst.exists():
            urllib.request.urlretrieve(BASE_URL + f"{n}.pdf", dst)


def _norm_code(s: str):
    return CODE_FORMS.get(re.sub(r"[^a-z]", "", s.lower()))


def _lines(words, tol=3.0):
    """Group words into visual lines by `top`, tolerant to 1-2pt baseline jitter."""
    out = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if out and abs(w["top"] - out[-1]["top"]) <= tol:
            out[-1]["words"].append(w)
        else:
            out.append({"top": w["top"], "words": [w]})
    for ln in out:
        ln["words"].sort(key=lambda w: w["x0"])
        ln["text"] = " ".join(w["text"] for w in ln["words"])
    return out


def _columns(pages):
    """(text_right_edge, notes_left_edge) shared by every page of one transcript.

    Code column: code tokens that sit after a column gap (>= 12pt from the previous
    word on their line, or alone on it), left edge of their modal cluster. Header
    words are not used for this: several tables right-align "Code" past the codes.
    Notes column: its header word ("Notes", "Explanation", ...), when there is one.
    """
    xs = []
    for p in pages:
        for ln in p:
            ws = ln["words"]
            for i, w in enumerate(ws):
                gap = w["x0"] - ws[i - 1]["x1"] if i else w["x0"]
                if _norm_code(w["text"]) and w["x0"] > 250 and gap >= 12:
                    xs.append(w["x0"])
    if not xs:
        raise ValueError("no code column found")
    mode = Counter(round(x) for x in xs).most_common(1)[0][0]
    code_x = min(x for x in xs if abs(x - mode) < 25)
    notes_x = None
    for p in pages[:1]:
        for ln in p:
            for w in ln["words"]:
                if w["text"].strip(":").lower() in NOTES_HEADERS and w["x0"] > code_x + 15:
                    notes_x = w["x0"] if notes_x is None else min(notes_x, w["x0"])
    return code_x - 4, (notes_x - 4) if notes_x else None


def _speaker_x(pages, text_right):
    """Modal x of the speaker column (P/I/T/C as the 2nd word after a turn number).

    The column drifts a few points between pages in some PDFs, so rows are matched
    with a +-10pt tolerance around this mode."""
    xs = [ln["words"][1]["x0"] for p in pages for ln in p
          if len(ln["words"]) > 1 and re.fullmatch(r"\d+", ln["words"][0]["text"])
          and ln["words"][1]["text"] in SPEAKERS and ln["words"][1]["x0"] < text_right]
    return Counter(round(x) for x in xs).most_common(1)[0][0]


def _parse_codes(cell_lines):
    """Leading code tokens of a code cell; whatever follows is a coder note.

    "GI / Persuade / ruled out because ..." -> [GI]  (the Persuade is ruled out)
    "NC- Affirm not coded because ..."      -> [NC]
    "Empha / size"                          -> [Emphasize]  (word wrapped in the cell)
    """
    lines = [l.strip() for l in cell_lines if l.strip()]
    merged = []
    for l in lines:  # re-join a code word that wrapped across cell lines
        if merged and not _norm_code(merged[-1]) and _norm_code(merged[-1] + l):
            merged[-1] += l
        else:
            merged.append(l)
    text = "\n".join(merged)
    text = re.sub(r"\([^)]*\)?", " ", text, flags=re.S)  # drop remarks, even unclosed
    toks = re.findall(r"[A-Za-z]+|[“\"]", text)
    codes, i = [], 0
    while i < len(toks):
        low = [t.lower() for t in toks[i:i + 3]]
        if tuple(low[:2]) in CANCEL:
            if codes and codes[-1] != "NC" and i > 0:
                codes.pop()  # "X ruled out" / "X not coded": X is not the code
            elif not codes and tuple(low[:2]) == ("not", "coded"):
                codes.append("NC")
            i += 2
            break
        multi = next((c for k, c in MULTI_FORMS.items() if tuple(low[:len(k)]) == k), None)
        if multi:
            codes.append(multi)
            i += 3 if multi == "PwP" else 2
            continue
        c = _norm_code(toks[i])
        if not c:
            break
        codes.append(c)
        i += 1
    codes = [c for j, c in enumerate(codes) if j == 0 or c != codes[j - 1]]
    note = " ".join(toks[i:]) if i < len(toks) else ""
    return codes, note


def parse_pdf(path: Path):
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        pages = [_lines(p.extract_words()) for p in pdf.pages]
        heights = [p.height for p in pdf.pages]
    text_right, notes_left = _columns(pages)
    spk_x = _speaker_x(pages, text_right)

    # Repeated header/footer lines: same text on 2+ pages.
    seen = Counter(ln["text"] for p in pages for ln in {l["text"]: l for l in p}.values())
    repeated = {t for t, n in seen.items() if n >= 2 and len(pages) > 1}

    turns, cur = [], None
    for p, h in zip(pages, heights):
        for ln in p:
            if ln["text"] in repeated or (ln["top"] > h - 50 and re.fullmatch(r"\d+", ln["text"])):
                continue
            ws = ln["words"]
            left = [w for w in ws if w["x0"] < text_right]
            # A row starts with the speaker letter in the speaker column, preceded by
            # the turn number (one transcript has a typo, "Q" for "2"; infer it).
            if (len(left) >= 2 and left[1]["text"] in SPEAKERS
                    and abs(left[1]["x0"] - spk_x) <= 10 and left[0]["x0"] < left[1]["x0"] - 10
                    and (left[0]["text"].isdigit() or (cur is not None and len(left[0]["text"]) <= 2))):
                num = left[0]["text"]
                turn = int(num) if num.isdigit() else (cur["turn"] + 1 if cur else 1)
                cur = {"turn": turn, "spk": left[1]["text"], "text": [], "code": [], "note": []}
                turns.append(cur)
                left = left[2:]
            if cur is None:
                continue  # preamble before the first row
            if left:
                cur["text"].append(" ".join(w["text"] for w in left))
            code_ws = [w for w in ws if w["x0"] >= text_right and (notes_left is None or w["x0"] < notes_left)]
            note_ws = [w for w in ws if notes_left is not None and w["x0"] >= notes_left]
            if code_ws:
                cur["code"].append(" ".join(w["text"] for w in code_ws))
            if note_ws:
                cur["note"].append(" ".join(w["text"] for w in note_ws))
    # Rows whose speaker line was split from its number (number alone on a line,
    # speaker + text on the next) are rare; the turn-number sequence check in
    # `main` reports any gaps.
    rows = []
    for t in turns:
        codes, code_note = _parse_codes(t["code"])
        text = re.sub(r"\s+", " ", " ".join(t["text"])).strip()
        text = re.sub(r"\(\s*did not code this\s*\)", "", text, flags=re.I).strip()
        rows.append({
            "turn": t["turn"], "speaker": SPEAKERS[t["spk"]], "text": text,
            "miti_codes": "|".join(codes),
            "note": " ".join(x for x in [code_note, " ".join(t["note"])] if x).strip(),
            "raw_code_cell": " / ".join(t["code"]),
        })
    return rows


# Session-level MITI global ratings, typed after the last turn in some transcripts:
# "Global Rating: CCT 1 SST 1 PAR 1 EMP 1" or "CCT: 1 / SST: 4 (note) / PAR: 1 / EMP: 1".
_GLOBALS = re.compile(r"(?:Global Ratings?:?\s*)?\bCCT:?\s*(\d)\b.*?\bSST:?\s*(\d)\b.*?"
                      r"\bPAR:?\s*(\d)\b.*?\bEMP:?\s*(\d)\b.*$", re.S)


def split_globals(text: str):
    m = _GLOBALS.search(text)
    if not m:
        return text, None
    g = dict(zip(["CCT", "SST", "PAR", "EMP"], map(int, m.groups())))
    return text[:m.start()].strip(), g


_SENT = re.compile(r"(?<=[.?!…])[\"”’)]*\s+(?=[A-Z“\"‘(])")


def to_eval(turn_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for conv_id, grp in turn_df.groupby("conv_id", sort=False):
        for vol_i, r in enumerate(grp.itertuples(index=False)):
            codes = [c for c in str(r.miti_codes).split("|") if c and c != "nan"]
            base = {"conv_id": conv_id, "speaker": r.speaker, "conv_vol_idx": vol_i,
                    "vol_text": r.text, "miti_turn": r.turn}
            segs, align = [(r.text, codes)], "turn"
            if len(codes) > 1:
                sents = [s.strip() for s in _SENT.split(r.text) if s.strip()]
                if len(sents) == len(codes):
                    segs, align = [(s, [c]) for s, c in zip(sents, codes)], "sentence"
                else:
                    align = "multi"
            for text, cs in segs:
                t1 = t2 = None
                if r.speaker == "counsellor" and len(cs) == 1 and cs[0] in MITI_TO_MISC:
                    t1, t2 = MITI_TO_MISC[cs[0]]
                out.append({**base, "utt_text": text, "miti_codes": "|".join(cs),
                            "align": align if cs else "", "t1_label_GT": t1, "t2_label_GT": t2})
    df = pd.DataFrame(out)
    df["corp_conv_idx"] = pd.factorize(df["conv_id"])[0]
    df["conv_utt_idx"] = df.groupby("conv_id", sort=False).cumcount()
    df["corp_utt_idx"] = range(len(df))
    df["corp_vol_idx"] = pd.factorize(df["conv_id"] + "#" + df["conv_vol_idx"].astype(str))[0]
    df["hlqc_overlap"] = df["conv_id"].map(HLQC_OVERLAP).fillna("")
    return df[SCHEMA + ["miti_codes", "align", "miti_turn", "hlqc_overlap"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--pdf-dir", default=str(PDF_DIR))
    args = ap.parse_args(argv)

    if not args.no_download:
        download()
    rows = []
    for n in TRANSCRIPTS:
        for r in parse_pdf(Path(args.pdf_dir) / f"{n}.pdf"):
            rows.append({"conv_id": f"casaa_{n.replace('-coded-transcript', '')}", **r})
    turns = pd.DataFrame(rows)
    glob = []
    for i, r in turns.iterrows():
        text, g = split_globals(r.text)
        if g:
            turns.at[i, "text"] = text
            glob.append({"conv_id": r.conv_id, **g})

    # Sanity: turn numbers should run 1..N without gaps in every transcript.
    for conv, g in turns.groupby("conv_id", sort=False):
        missing = sorted(set(range(1, g.turn.max() + 1)) - set(g.turn))
        if missing:
            print(f"WARNING {conv}: missing turn numbers {missing[:10]}")
    unknown = turns[(turns.speaker == "counsellor") & (turns.miti_codes == "") & (turns.raw_code_cell != "")]
    for r in unknown.itertuples():
        print(f"UNPARSED code cell {r.conv_id} turn {r.turn}: {r.raw_code_cell!r}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    turns.to_csv(OUT_DIR / "casaa_turns.csv", index=False)
    pd.DataFrame(glob).to_csv(OUT_DIR / "casaa_globals.csv", index=False)
    print(f"MITI global ratings found for {len(glob)} transcripts")
    ev = to_eval(turns)
    ev.to_csv(OUT_DIR / "CASAA_eval.csv", index=False)

    c = ev[ev.speaker == "counsellor"]
    coded = c[c.miti_codes != ""]
    print(f"{turns.conv_id.nunique()} transcripts, {len(turns)} turns "
          f"({(turns.speaker == 'counsellor').sum()} counsellor) -> {len(ev)} eval rows")
    print("MITI codes on single-code counsellor rows:",
          dict(Counter(coded[~coded.miti_codes.str.contains('|', regex=False)].miti_codes).most_common()))
    print("alignment:", dict(Counter(coded["align"])))
    print(f"T1 gold: {c.t1_label_GT.notna().sum()}  T2 gold: {c.t2_label_GT.notna().sum()}",
          dict(Counter(c.t2_label_GT.dropna())))
    clean = c[c.hlqc_overlap == ""]
    print(f"Excluding HLQC-overlap sessions: T1 gold {clean.t1_label_GT.notna().sum()}, "
          f"T2 gold {clean.t2_label_GT.notna().sum()}", dict(Counter(clean.t2_label_GT.dropna())))


if __name__ == "__main__":
    main()
