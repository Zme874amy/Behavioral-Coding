"""Score every data-expansion arm against the 1-adapter baseline, across seeds.

Each expansion arm (self-training, the synthesis variants, their ablations, and
retrieval on the same backbone) is a retrain of `ft1mix_bare` -- one shared
adapter, two calls -- on a different training set, predicted on the same 821
MIV6.3A gold utterances. Their result CSVs live outside
`data/annotated/baseline/`, so `baseline.eval` never sees them; this module is
where they are compared.

What it reports, per arm:
  * T2 accuracy and Macro-F1 (gold), mean +/- SD over the seeds present;
  * the paired difference against its reference at MATCHED seeds (42<->42,
    1<->1, 2<->2), with a conversation-clustered 95% CI on the seed-averaged
    difference: every bootstrap draw resamples the evaluation conversations
    once and applies that same resample to every seed of both arms;
  * a verdict. `real` needs all three of: the difference has the same sign in
    every matched seed, its CI excludes zero, and it exceeds the seed-noise band
    1.96*sqrt((sd_ref^2 + sd_arm^2)/k) for k matched seeds. An arm with a single
    matched seed is `single-seed` -- its CI is shown but no seed claim is made.

Plus cross-scheme rows for any arm evaluated on AnnoMI / Welivita, and the
generalization ladder (HLQC CV -> MIV6.3A -> AnnoMI -> Welivita) on the
1-adapter model.

Outputs:
    outputs/expansion/expansion_table.csv
    docs/EXPANSION_RESULTS.md        (generated -- edit this module, not the doc)

Usage:
    PYTHONPATH=src python -m selftrain.expansion_table
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from baseline.eval import (
    BOOT_SEED,
    CI_LEVEL,
    CROSS_SCHEME,
    N_BOOT,
    _crossscheme_metrics,
    _macro_over_supported,
    score,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ANN = REPO_ROOT / "data" / "annotated"
OUT_CSV = REPO_ROOT / "outputs" / "expansion" / "expansion_table.csv"
DOC = REPO_ROOT / "docs" / "EXPANSION_RESULTS.md"
CV_REPORT = REPO_ROOT / "outputs" / "cv_hlqc" / "ft1mix_bare_ctx5_report.json"

PRED = "qwen_ft1mix_bare_inf_bare_ctx5.csv"
SEEDS = (42, 1, 2)


def _seeded(prefix: str, fname: str = PRED) -> Dict[int, Path]:
    """Result files for a retrain arm: `<prefix>/` holds seed 42, `<prefix>_s<N>/` seed N."""
    return {s: ANN / (prefix if s == 42 else f"{prefix}_s{s}") / fname for s in SEEDS}


@dataclass
class Arm:
    key: str
    label: str
    group: str                      # main | ablation | retrieval
    files: Dict[int, Path]
    style: str = "bare"
    ref: str = "baseline"           # key of the arm it is compared against
    xs_dir: Optional[Path] = None   # dir holding its `_ds<corpus>` cross-scheme files
    xs_skip: tuple = ()             # corpora it must not be scored on (leakage)
    note: str = ""
    dropped: str = ""               # why a planned arm was abandoned (never run)
    seeds: Dict[int, Path] = field(init=False)

    def __post_init__(self):
        self.seeds = {s: p for s, p in self.files.items() if p.exists()}


BASELINES = {
    "bare": Arm("baseline", "Baseline FT1-Mix (human HLQC only)", "reference", {
        42: ANN / "baseline" / PRED,
        1: ANN / "base_1a_s1" / PRED,
        2: ANN / "base_1a_s2" / PRED,
    }, xs_dir=ANN / "baseline"),
    "cot": Arm("baseline_cot", "Baseline FT1-Mix, Inf-CoT", "reference", {
        42: ANN / "baseline" / "qwen_ft1mix_bare_inf_cot_ctx5.csv",
    }, style="cot"),
}


# docs/DATASETS.md overlap map (2026-10-04): HLQC gold train sessions low_001,
# high_099, low_033 and low_080 are AnnoMI transcripts 53, 21, 44 and 15.
ANNOMI_CAVEAT = ("**Caveat (2026-10-04):** every AnnoMI number here includes the four "
                 "AnnoMI transcripts (15, 21, 44, 53) that are also HLQC training sessions, "
                 "so it is optimistic; the re-run (docs/RERUN_PLAN.md) scores AnnoMI with "
                 "the `annomi_own`/MISC-transfer exclusions applied.")

NOGO = ("dropped at the 32B-teacher NO-GO gate (HLQC few-shot macro-F1 0.324 vs "
        "student 0.399, 2026-09-30); never trained")


def _registry() -> List[Arm]:
    s = ANN
    return [
        Arm("selftrain", "Self-training (Welivita pseudo-labels)", "main",
            _seeded("selftrain_1a"), xs_dir=s / "selftrain_1a", xs_skip=("welivita",),
            note="trained on Welivita pseudo-labels, so never scored on Welivita"),
        Arm("synth_v1", "Synthesis v1 (one-shot utterances)", "main", _seeded("synth_v1_1a"),
            xs_dir=s / "synth_v1_1a"),
        Arm("synth_v2_proto", "Synthesis v2 proto (dialogue windows)", "main",
            _seeded("synth_v2_proto_1a"), xs_dir=s / "synth_v2_proto_1a"),
        Arm("synth_v2_bound", "Synthesis v2 boundary", "main", _seeded("synth_v2_bound_1a"),
            xs_dir=s / "synth_v2_bound_1a"),
        Arm("synth_v3_train", "Synthesis v3, HLQC topic mix", "main",
            _seeded("synth_v3_train_1a"), xs_dir=s / "synth_v3_train_1a"),
        Arm("synth_v3_eval", "Synthesis v3, MIV6.3A topic mix", "main",
            _seeded("synth_v3_eval_1a"), xs_dir=s / "synth_v3_eval_1a"),
        # Ablations -- each is compared against the arm it differs from by ONE factor.
        Arm("synth_v3_train_rule", "v3 HLQC mix + SocialDial rule", "ablation",
            _seeded("synth_v3_train_rule_1a"), ref="synth_v3_train",
            note="isolates the rule: same generator and topic mix, rule on"),
        Arm("synth_v2_proto_synthonly", "v2 proto, synthetic rows only (no human)", "ablation",
            {42: s / "synth_v2_proto_1a_SYNTHONLY" / PRED},
            note="SocialDial's human / synthetic / both split; read with baseline (human) "
                 "and v2 proto (both)"),
        Arm("synth_v2_bound_kh", "v2 boundary, keep-hard (verifier off)", "ablation",
            {42: s / "synth_v2_bound_kh_1a" / PRED}, ref="synth_v2_bound",
            note="isolates the verifier filter: same windows, unverified labels kept"),
        # Self-training v2 (calibrated likelihood, self-ensemble, prior-aligned,
        # budgeted, counsellor-only) and the 32B-teacher comparison on the SAME
        # pipeline. Teacher rows are also compared directly against the self arm
        # on the same pool -- the controlled "who labels" contrast.
        Arm("selftrain_v2", "Self-training v2 (Welivita)", "st2", _seeded("selftrain_v2_1a"),
            xs_dir=s / "selftrain_v2_1a", xs_skip=("welivita",)),
        Arm("selftrain_v2r2", "Self-training v2, round 2 (Welivita)", "st2",
            _seeded("selftrain_v2r2_1a"), xs_dir=s / "selftrain_v2r2_1a", xs_skip=("welivita",),
            dropped="round 1 lowered macro-F1 (-0.047, real), so round 2 was not run"),
        Arm("selftrain_v2b", "Self-training v2 (MIV6.3B pool)", "st2", _seeded("selftrain_v2b_1a"),
            xs_dir=s / "selftrain_v2b_1a"),
        Arm("distill_v2", "32B-teacher labels (Welivita)", "st2", _seeded("distill_v2_1a"),
            xs_dir=s / "distill_v2_1a", xs_skip=("welivita",), dropped=NOGO),
        Arm("distill_v2b", "32B-teacher labels (MIV6.3B pool)", "st2", _seeded("distill_v2b_1a"),
            xs_dir=s / "distill_v2b_1a", dropped=NOGO),
        Arm("distill_v2_vs_self", "Teacher vs self (Welivita)", "self_vs_teacher",
            _seeded("distill_v2_1a"), ref="selftrain_v2", dropped=NOGO),
        Arm("distill_v2b_vs_self", "Teacher vs self (MIV6.3B)", "self_vs_teacher",
            _seeded("distill_v2b_1a"), ref="selftrain_v2b", dropped=NOGO),
        # Retrieval: the same 1-adapter model, with retrieved exemplars in the prompt.
        Arm("ag_bare", "Retrieval few-shot on FT1-Mix, Inf-Bare", "retrieval",
            {42: s / "baseline" / "qwen_ag_qwen_ft1mix_bare_inf_bare_ctx5.csv"}),
        Arm("ag_cot", "Retrieval few-shot on FT1-Mix, Inf-CoT", "retrieval",
            {42: s / "baseline" / "qwen_ag_qwen_ft1mix_bare_inf_cot_ctx5.csv"},
            style="cot", ref="baseline_cot"),
    ]


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def _load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.drop_duplicates("corp_utt_idx", keep="last")
    return d.set_index("corp_utt_idx")


def _subset(d: pd.DataFrame, level: str, scope: str) -> pd.DataFrame:
    gt, pr = f"{level}_label_GT", f"{level}_label_auto"
    m = d[gt].notna() & d[pr].notna()
    if scope != "all":
        m &= d["speaker"] == scope
    return d.loc[m, ["conv_id", gt, pr]].rename(columns={gt: "gt", pr: "pred"})


def _point(d: pd.DataFrame, level: str, scope: str) -> dict:
    x = _subset(d, level, scope)
    return score(x["gt"], x["pred"], level.lower())


def _paired_boot(pairs: List[tuple], level: str, scope: str) -> dict:
    """Seed-averaged paired deltas (accuracy, Macro-F1 gold) with a clustered CI.

    `pairs` is a list of (ref_df, arm_df) at matched seeds. One multinomial draw
    over the evaluation conversations is shared by every seed of both arms, so
    the interval is on the difference of the seed means, conditional on the
    shared item difficulty.
    """
    aligned = []
    for ref, arm in pairs:
        r, a = _subset(ref, level, scope), _subset(arm, level, scope)
        idx = r.index.intersection(a.index)
        aligned.append((r.loc[idx], a.loc[idx]))
    groups = pd.unique(np.concatenate([r["conv_id"].to_numpy() for r, _ in aligned]))
    g_of = {g: i for i, g in enumerate(groups)}
    G = len(groups)
    labels = sorted(set().union(*[set(r["gt"]) | set(r["pred"]) | set(a["pred"])
                                  for r, a in aligned]))
    li = {c: i for i, c in enumerate(labels)}
    K = len(labels)

    rng = np.random.default_rng(BOOT_SEED)
    counts = rng.multinomial(G, np.full(G, 1.0 / G), size=N_BOOT).astype(np.float32)

    def draws(gt, pred, conv):
        """Per-draw accuracy and Macro-F1 (gold, supported classes) for one model."""
        gi = np.array([g_of[c] for c in conv])
        ti = np.array([li[c] for c in gt])
        pi = np.array([li[c] for c in pred])
        conf = np.zeros((G, K * K), dtype=np.float32)
        np.add.at(conf, (gi, ti * K + pi), 1.0)
        C = (counts @ conf).reshape(N_BOOT, K, K)
        diag = np.diagonal(C, axis1=1, axis2=2)
        rowsum, colsum = C.sum(2), C.sum(1)
        total = C.sum((1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = np.where(total > 0, diag.sum(1) / total, 0.0)
            prec = np.where(colsum > 0, diag / colsum, 0.0)
            rec = np.where(rowsum > 0, diag / rowsum, 0.0)
            den = prec + rec
            f1 = np.where(den > 0, 2 * prec * rec / den, 0.0)
        gold_idx = np.array(sorted({li[c] for c in gt}), dtype=int)
        return acc, _macro_over_supported(f1, rowsum, gold_idx)

    d_acc, d_f1 = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for r, a in aligned:
        ra, rf = draws(r["gt"], r["pred"], r["conv_id"])
        aa, af = draws(a["gt"], a["pred"], a["conv_id"])
        d_acc += (aa - ra) / len(aligned)
        d_f1 += (af - rf) / len(aligned)
    q = [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2]
    return {"acc_ci": tuple(np.quantile(d_acc, q)), "f1_ci": tuple(np.quantile(d_f1, q))}


def _verdict(deltas: List[float], ci: tuple, sd_ref: float, sd_arm: float) -> str:
    k = len(deltas)
    if k == 0:
        return "pending"
    if ci[0] <= 0 <= ci[1]:
        return "ns"
    if k == 1:
        return "single-seed"
    if len({np.sign(d) for d in deltas}) > 1:
        return "sign-flips"
    band = 1.96 * np.sqrt((sd_ref ** 2 + sd_arm ** 2) / k)
    return "real" if abs(np.mean(deltas)) > band else "<floor"


def compare(arm: Arm, ref: Arm, level: str = "T2", scope: str = "all") -> dict:
    """One arm vs its reference at matched seeds, on one level/scope."""
    matched = sorted(set(arm.seeds) & set(ref.seeds))
    row = {"key": arm.key, "label": arm.label, "group": arm.group, "ref": ref.key,
           "level": level, "scope": scope, "seeds_run": len(arm.seeds),
           "seeds_matched": len(matched)}
    if not arm.seeds:
        v = "dropped" if arm.dropped else "pending"
        return {**row, "verdict_acc": v, "verdict_f1": v}
    lvl = level.lower()
    arm_pts = {s: _point(_load(p), lvl, scope) for s, p in arm.seeds.items()}
    ref_pts = {s: _point(_load(p), lvl, scope) for s, p in ref.seeds.items()}
    for name, key in (("acc", "accuracy"), ("f1", "f1_macro_gold")):
        av = [v[key] for v in arm_pts.values()]
        rv = [v[key] for v in ref_pts.values()]
        row[f"{name}_mean"], row[f"{name}_sd"] = float(np.mean(av)), (
            float(np.std(av, ddof=1)) if len(av) > 1 else float("nan"))
        row[f"ref_{name}_mean"], row[f"ref_{name}_sd"] = float(np.mean(rv)), (
            float(np.std(rv, ddof=1)) if len(rv) > 1 else float("nan"))
    if not matched:
        return {**row, "verdict_acc": "no matched seed", "verdict_f1": "no matched seed"}
    d_acc = [arm_pts[s]["accuracy"] - ref_pts[s]["accuracy"] for s in matched]
    d_f1 = [arm_pts[s]["f1_macro_gold"] - ref_pts[s]["f1_macro_gold"] for s in matched]
    boot = _paired_boot([(_load(ref.seeds[s]), _load(arm.seeds[s])) for s in matched],
                        lvl, scope)
    # The SD fed to the seed band: an arm without replicates borrows its reference's.
    sd = lambda x, fb: x if not np.isnan(x) else fb
    ref_sd_acc = sd(row["ref_acc_sd"], 0.0)
    ref_sd_f1 = sd(row["ref_f1_sd"], 0.0)
    row.update({
        "d_acc": float(np.mean(d_acc)), "d_acc_lo": boot["acc_ci"][0], "d_acc_hi": boot["acc_ci"][1],
        "d_f1": float(np.mean(d_f1)), "d_f1_lo": boot["f1_ci"][0], "d_f1_hi": boot["f1_ci"][1],
        # Keyed by seed so the CSV cannot be misread by position.
        "d_acc_per_seed": {int(s): round(v, 4) for s, v in zip(matched, d_acc)},
        "d_f1_per_seed": {int(s): round(v, 4) for s, v in zip(matched, d_f1)},
        "verdict_acc": _verdict(d_acc, boot["acc_ci"], ref_sd_acc, sd(row["acc_sd"], ref_sd_acc)),
        "verdict_f1": _verdict(d_f1, boot["f1_ci"], ref_sd_f1, sd(row["f1_sd"], ref_sd_f1)),
    })
    return row


# -----------------------------------------------------------------------------
# Cross-scheme and the ladder
# -----------------------------------------------------------------------------
def crossscheme_rows(arms: List[Arm]) -> List[dict]:
    """Shared-vocabulary scores on AnnoMI / Welivita for any arm evaluated there."""
    subjects = [("zs", "Zero-shot (no adapter)", ANN / "baseline", "qwen_zs_inf_bare_ctx5", ()),
                ("baseline", BASELINES["bare"].label, ANN / "baseline", "qwen_ft1mix_bare_inf_bare_ctx5", ())]
    subjects += [(a.key, a.label, a.xs_dir, "qwen_ft1mix_bare_inf_bare_ctx5", a.xs_skip)
                 for a in arms if a.xs_dir is not None]
    rows = []
    for key, label, d, stem, skip in subjects:
        for ds, spec in CROSS_SCHEME.items():
            f = d / f"{stem}_ds{ds}.csv"
            if ds in skip or not f.exists():
                continue
            df = pd.read_csv(f)
            for level, speakers in spec["levels"].items():
                for speaker, allowed in speakers.items():
                    gt, pr = f"{level}_label_GT", f"{level}_label_auto"
                    x = df[df[gt].notna() & df[pr].notna() & (df["speaker"] == speaker)]
                    x = x[x[gt].isin(allowed)]
                    if x.empty:
                        continue
                    m = _crossscheme_metrics(x[gt], x[pr], x["conv_id"], allowed)
                    rows.append({"key": key, "label": label, "dataset": ds,
                                 "level": level.upper(), "speaker": speaker, **m})
    return rows


def ladder(xs: List[dict]) -> List[dict]:
    """Counsellor T2 accuracy for the 1-adapter baseline, rung by rung."""
    rungs = []
    if CV_REPORT.exists():
        rep = json.loads(CV_REPORT.read_text())
        for s in rep["scores"]:
            if s["level"] == "T2" and s["scope"] == "counsellor":
                rungs.append({"rung": "HLQC, in-distribution 5-fold CV", "vocab": "full MISC",
                              "acc": s["accuracy"], "lo": s.get("accuracy_lo"),
                              "hi": s.get("accuracy_hi")})
    base = BASELINES["bare"]
    if 42 in base.seeds:
        x = _subset(_load(base.seeds[42]), "t2", "counsellor")
        s = score(x["gt"], x["pred"], "t2", conv=x["conv_id"])
        rungs.append({"rung": "MIV6.3A, cross-corpus (seed 42)", "vocab": "full MISC",
                      "acc": s["accuracy"], "lo": s.get("accuracy_lo"), "hi": s.get("accuracy_hi")})
    for ds, name in (("annomi", "AnnoMI, cross-scheme"), ("welivita", "Welivita/MITI, cross-scheme (weak gold)")):
        for r in xs:
            if r["key"] == "baseline" and r["dataset"] == ds and r["level"] == "T2":
                rungs.append({"rung": name, "vocab": "shared codes", "acc": r["accuracy"],
                              "lo": r.get("accuracy_lo"), "hi": r.get("accuracy_hi")})
    return rungs


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
def _f(x, spec=".3f"):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def _ms(row, name):
    m, sd = row.get(f"{name}_mean"), row.get(f"{name}_sd")
    if m is None:
        return "—"
    return _f(m) if sd is None or np.isnan(sd) else f"{_f(m)} ± {_f(sd)}"


def _delta(row, name):
    if row.get(f"d_{name}") is None:
        return "—"
    return f"{_f(row[f'd_{name}'], '+.3f')} [{_f(row[f'd_{name}_lo'], '+.3f')}, {_f(row[f'd_{name}_hi'], '+.3f')}]"


def _table(rows: List[dict]) -> List[str]:
    out = ["| Arm | vs | Seeds | T2 acc | Δ acc [95% CI] | Verdict | "
           "T2 Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |",
           "|---|---|---:|---:|---:|---|---:|---:|---|"]
    for r in rows:
        out.append(
            f"| {r['label']} | {r['ref']} | {r['seeds_matched']}/{r['seeds_run']} "
            f"| {_ms(r, 'acc')} | {_delta(r, 'acc')} | {r['verdict_acc']} "
            f"| {_ms(r, 'f1')} | {_delta(r, 'f1')} | {r['verdict_f1']} |")
    return out


def main() -> None:
    arms = _registry()
    by_key = {a.key: a for a in arms} | {b.key: b for b in BASELINES.values()}
    ref_of = lambda a: by_key[a.ref]

    rows_t2 = [compare(a, ref_of(a), "T2", "all") for a in arms]
    rows_spk = [compare(a, ref_of(a), "T2", spk) for a in arms if a.seeds
                for spk in ("counsellor", "client")]
    rows_t1 = [compare(a, ref_of(a), "T1", "all") for a in arms if a.seeds]
    xs = crossscheme_rows(arms)
    rungs = ladder(xs)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_t2 + rows_spk + rows_t1).to_csv(OUT_CSV, index=False)

    base = BASELINES["bare"]
    L = [
        "# Data-expansion results (1-adapter backbone)",
        "",
        f"*Generated {datetime.now(timezone.utc):%Y-%m-%d} by `selftrain.expansion_table` — "
        "edit that module, not this file.*",
        "",
        "Every arm below is a retrain of **FT1-Mix-Bare** (one shared adapter, two calls), "
        "differing only in its training set, and is scored on the same 821 MIV6.3A gold "
        "utterances. The baseline is the same recipe on human HLQC data alone, run at "
        f"{len(base.seeds)} seeds.",
        "",
        "**How to read.** `Seeds` is matched/run: the difference is taken at matched training "
        "seeds (42↔42, 1↔1, 2↔2) and averaged. The bracket is a 95% CI from resampling the "
        "evaluation *conversations* (the same resample applied to every seed of both arms). "
        "`real` needs all three: the same sign in every matched seed, a CI that excludes 0, "
        "and a mean difference larger than the seed-noise band. `single-seed` means the CI "
        "excludes 0 but there is only one matched seed — not yet a claim. `ns` means the CI "
        "includes 0. `sign-flips` means seeds disagree on the direction.",
        "",
        "## Main arms — T2, all speakers", "",
        *_table([r for r in rows_t2 if r["group"] == "main"]), "",
        "## Ablations — each against the arm it differs from by one factor", "",
        *_table([r for r in rows_t2 if r["group"] == "ablation"]), "",
    ]
    L += [f"- **{a.label}** — {a.note}" for a in arms if a.group == "ablation" and a.note]
    if any(a.seeds for a in arms if a.group in ("st2", "self_vs_teacher")):
        L += ["", "## Self-training v2 and the 32B-teacher comparison", "",
              "Same pipeline for every row: calibrated per-code thresholds, prior-aligned "
              "quotas, at most one pseudo label per human label, counsellor rows only. Only "
              "the labeller differs (the student's 3-seed self-ensemble vs the 32B teacher).", "",
              *_table([r for r in rows_t2 if r["group"] == "st2"]), "",
              "### Teacher vs self on the same pool", "",
              *_table([r for r in rows_t2 if r["group"] == "self_vs_teacher"]), ""]
    L += ["", "## Retrieval on the same backbone", "",
          *_table([r for r in rows_t2 if r["group"] == "retrieval"]), ""]
    L += ["## Where the T2 change comes from — by speaker", "",
          "| Arm | Speaker | Seeds | Macro-F1 | Δ Macro-F1 [95% CI] | Verdict |",
          "|---|---|---:|---:|---:|---|"]
    for r in rows_spk:
        L.append(f"| {r['label']} | {r['scope']} | {r['seeds_matched']}/{r['seeds_run']} "
                 f"| {_ms(r, 'f1')} | {_delta(r, 'f1')} | {r['verdict_f1']} |")
    L += ["", "## T1 check — does the added data cost Tier-1?", "",
          "| Arm | Seeds | T1 acc | Δ acc [95% CI] | Verdict |", "|---|---:|---:|---:|---|"]
    for r in rows_t1:
        L.append(f"| {r['label']} | {r['seeds_matched']}/{r['seeds_run']} | {_ms(r, 'acc')} "
                 f"| {_delta(r, 'acc')} | {r['verdict_acc']} |")
    if xs:
        L += ["", "## Cross-scheme generalization (shared codes only)", "",
              ANNOMI_CAVEAT, "",
              "| Model | Corpus | Level | Speaker | n | Accuracy [95% CI] | Macro-F1 shared [95% CI] | OOV pred |",
              "|---|---|---|---|---:|---:|---:|---:|"]
        for r in xs:
            L.append(
                f"| {r['label']} | {r['dataset']} | {r['level']} | {r['speaker']} | {r['n']} "
                f"| {_f(r['accuracy'])} [{_f(r.get('accuracy_lo'))}–{_f(r.get('accuracy_hi'))}] "
                f"| {_f(r['f1_macro_shared'])} [{_f(r.get('f1_macro_shared_lo'))}–{_f(r.get('f1_macro_shared_hi'))}] "
                f"| {r['oov_pred_rate']:.1%} |")
    if rungs:
        L += ["", "## Generalization ladder — FT1-Mix baseline, counsellor T2 accuracy", "",
              "Read top to bottom: each rung adds one kind of shift. The cross-scheme rungs are "
              "scored on the shared codes only, so they are not on exactly the same vocabulary "
              "as the first two. " + ANNOMI_CAVEAT, "",
              "| Rung | Vocabulary | Accuracy [95% CI] |", "|---|---|---:|"]
        for g in rungs:
            L.append(f"| {g['rung']} | {g['vocab']} | {_f(g['acc'])} [{_f(g['lo'])}–{_f(g['hi'])}] |")
    pending = [(a.label, s) for a in arms for s in (SEEDS if a.group == "main" else sorted(a.files))
               if s in a.files and s not in a.seeds and not a.dropped]
    dropped = [a for a in arms if a.dropped and not a.seeds]
    if dropped:
        L += ["", "## Dropped (planned, never run)", ""] + [f"- {a.label} — {a.dropped}"
                                                            for a in dropped]
    if pending:
        L += ["", "## Not yet available", ""] + [f"- {lab} — seed {s}" for lab, s in pending]
    L.append("")
    DOC.write_text("\n".join(L))
    print(f"wrote {OUT_CSV} and {DOC}")
    for r in rows_t2:
        print(f"{r['label'][:44]:44s} seeds {r['seeds_matched']}/{r['seeds_run']}  "
              f"Δacc {_f(r.get('d_acc'), '+.3f'):>7s} {r['verdict_acc']:<12s} "
              f"ΔF1 {_f(r.get('d_f1'), '+.3f'):>7s} {r['verdict_f1']}")


if __name__ == "__main__":
    main()
