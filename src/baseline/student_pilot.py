"""P5 student pilot: pick the third student family on data that is not the test set.

Two cheap checks per candidate (docs/experiments/2026-10-06-student-pilot.md):

  zs     zero-shot, two-call label-only prompts (v2), on the HLQC validation
         fold (val_folds 7, val_fold 0: the fold GRPO selects on; 369 rows).
         MIV6.3A is never read, so the choice does not touch the test set.
  lora   a short LoRA run (1-adapter mixed regime, the main setting's recipe)
         on the HLQC rows OUTSIDE that fold, capped at --rows rows: does it fit
         a 40 GB A100, does the loss fall, are the LoRA layers on the language
         model only?
  report one table over every candidate, written into the campaign doc.

The selection rule is fixed in the doc BEFORE the runs. Briefly: a candidate
must reach >= 99% parseable answers zero-shot and train cleanly; among those,
the highest robust learnable macro-F1 (mean of the two speakers) wins.

    PYTHONPATH=src python -m baseline.student_pilot zs   --model allenai/Olmo-3-7B-Instruct
    PYTHONPATH=src python -m baseline.student_pilot lora --model allenai/Olmo-3-7B-Instruct
    PYTHONPATH=src python -m baseline.student_pilot report
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from baseline import rerun

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "student_pilot"
HLQC = REPO / "data" / "manual" / "HLQC_balanced_manual.csv"
VAL_FOLDS, VAL_FOLD, FOLD_SEED = 7, 0, 42
CTX = 5
CANDIDATES = {  # the comparison set; Qwen2.5-7B is the zero-cost reference (already staged)
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "Mistral (2025-12, Apache-2.0)",
    "allenai/Olmo-3-7B-Instruct": "AI2 OLMo 3 (2025-11, Apache-2.0, fully open data)",
    "microsoft/phi-4": "Microsoft Phi-4 14B (2024-12, MIT)",
    "Qwen/Qwen2.5-7B-Instruct": "reference: the previous student",
}


def split():
    from automisc_ft.data import load_manual
    from baseline.grpo import split_by_conversation
    df = load_manual(HLQC)
    df["dataset"], df["uid"] = "misc.hlqc.gold", "misc.hlqc.gold:" + df["corp_utt_idx"].astype(str)
    train, val = split_by_conversation(df, VAL_FOLDS, VAL_FOLD, FOLD_SEED)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def _cfg(model: str, extra):
    from baseline.local_arm import load_config
    return load_config([f"model.base_model={model}", f"annotator.num_context_turns={CTX}",
                        "training.save_strategy=no", *extra])


def cmd_zs(a) -> None:
    from automisc_ft.infer import TieredAnnotator
    os.environ["PROMPT_VERSION"] = "v2"
    cfg = _cfg(a.model, a.overrides)
    _, val = split()
    slug = rerun.student_slug(a.model)
    out = OUT / f"{slug}_zs_hlqcval.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    ann = TieredAnnotator(base_model=a.model, force_cpu=bool(cfg.inference.force_cpu),
                          trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
                          max_new_tokens=int(cfg.inference.max_new_tokens),
                          max_input_len=int(cfg.inference.max_input_len["zs"]), structure_suffix="_bare")
    try:
        for pos in range(len(val))[: a.limit or None]:
            p = ann.predict_row(val, pos, cfg.annotator.context_mode, CTX, False)
            rows.append({**val.iloc[pos].to_dict(), "t1_label_auto": p["t1_pred"], "t2_label_auto": p["t2_pred"],
                         "t1_raw": p["t1_raw"], "t2_raw": p["t2_raw"],
                         "t2_n_gen_tokens": p["t2_n_gen_tokens"]})
    finally:
        ann.close()
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    from components.prompts.loader import prompt_fingerprint
    rerun.write_meta(out, {"model": a.model, "rows": len(df), "seconds": round(time.time() - t0, 1),
                           "split": f"HLQC val fold {VAL_FOLD}/{VAL_FOLDS} (seed {FOLD_SEED})",
                           "prompts": prompt_fingerprint("v2")})
    print(f"wrote {out} ({len(df)} rows, {time.time() - t0:.0f}s)")


def cmd_lora(a) -> None:
    from automisc_ft.train import train_single_adapter
    os.environ["PROMPT_VERSION"] = "v2"
    cfg = _cfg(a.model, ["training.num_train_epochs=1", *a.overrides])
    train, _ = split()
    slug = rerun.student_slug(a.model)
    out_dir = REPO / "data" / "fine_tuning" / "student_pilot" / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = list(range(min(a.rows, len(train))))
    t0, status, err = time.time(), "ok", None
    try:
        adapter = train_single_adapter(cfg, train, positions, out_dir, "_bare", None, "mixed")
        log = json.loads((Path(adapter) / "train_log.json").read_text())
    except Exception as e:  # an OOM or a loading failure is a pilot RESULT, not a crash
        status, err, log = "failed", f"{type(e).__name__}: {e}"[:500], {}
    rec = {"model": a.model, "status": status, "error": err, "rows": len(positions),
           "seconds": round(time.time() - t0, 1), "overrides": list(a.overrides), **log}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{slug}_lora.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({k: v for k, v in rec.items() if k != "log_history"}, indent=2, default=str))


def _loss_drop(hist) -> tuple:
    losses = [h["loss"] for h in hist if "loss" in h]
    if len(losses) < 2:
        return None, None
    k = max(1, len(losses) // 4)
    first, last = sum(losses[:k]) / k, sum(losses[-k:]) / k
    return round(first, 3), round(last, 3)


def reparse(d: pd.DataFrame) -> pd.DataFrame:
    """Re-parse the saved raw generations with the current parser, using the same
    allowed sets the run used (T1: speaker's groups; T2: speaker's whole vocabulary,
    as predict_row does with restrict_t2_to_group=False)."""
    from automisc_ft.infer import parse_label
    from schemes.misc import t1_codes, t2_codes
    d = d.copy()
    d["t1_label_auto"] = [parse_label(str(r), t1_codes(s)) for r, s in zip(d.t1_raw.fillna(""), d.speaker)]
    d["t2_label_auto"] = [parse_label(str(r), t2_codes(s)) for r, s in zip(d.t2_raw.fillna(""), d.speaker)]
    return d


def summarise(slug: str) -> dict:
    from baseline import fold_stats
    from baseline.rerun_eval import ROBUST, score_block
    row = {"student": slug}
    zs = OUT / f"{slug}_zs_hlqcval.csv"
    if zs.exists():
        d = pd.read_csv(zs)
        train, _ = split()
        learn = fold_stats.learnable_codes(train)
        row["n"] = len(d)
        bad = lambda x: (x[["t1_label_auto", "t2_label_auto"]] == "UNKNOWN").any(axis=1).mean()
        row["parseable % (strict)"] = round(100 * (1 - bad(d)), 1)
        d = reparse(d)          # current parser: also resolves full code names (deviation 1)
        row["parseable %"] = round(100 * (1 - bad(d)), 1)
        f1s, rob = [], []
        for spk in ("counsellor", "client"):
            s = d[(d.speaker == spk) & d.t2_label_GT.notna()]
            L = [c for c in learn[spk] if c in set(s.t2_label_GT)]
            r = score_block(s.t2_label_GT, s.t2_label_auto.fillna("UNKNOWN"), s.conv_id, {"l": L}, boot=False)
            f1s.append(r["f1_l"])
            m = {**ROBUST["fa_fi_merged"], **(ROBUST["reflection_t1"] if spk == "counsellor" else {})}
            rr = score_block(s.t2_label_GT.replace(m), s.t2_label_auto.fillna("UNKNOWN").replace(m), s.conv_id,
                             {"l": sorted({m.get(c, c) for c in L})}, boot=False)
            rob.append(rr["f1_l"])
            row[f"{spk} T2 F1"] = round(r["f1_l"], 3)
        row["T2 F1 (mean)"] = round(sum(f1s) / 2, 3)
        row["robust T2 F1 (mean)"] = round(sum(rob) / 2, 3)
        row["T1 acc"] = round((d.t1_label_GT == d.t1_label_auto).mean(), 3)
        meta = json.loads(zs.with_suffix(".meta.json").read_text())
        row["s/utt"] = round(meta["seconds"] / max(len(d), 1), 2)
    lj = OUT / f"{slug}_lora.json"
    if lj.exists():
        L = json.loads(lj.read_text())
        first, last = _loss_drop(L.get("log_history", []))
        row.update({"LoRA": L["status"], "loss first→last": f"{first}→{last}" if first else "-",
                    "peak GB": L.get("peak_gpu_mem_gb"), "LoRA outside LM": L.get("lora_modules_outside_lm"),
                    "train s": L.get("seconds")})
        if L["status"] != "ok":
            row["error"] = L.get("error")
    return row


def cmd_report(a) -> None:
    rows = [summarise(rerun.student_slug(m)) for m in CANDIDATES]
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    (OUT / "report.md").write_text(t.to_markdown(index=False))
    print(f"wrote {OUT / 'report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("zs", "lora"):
        p = sub.add_parser(name)
        p.add_argument("--model", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--rows", type=int, default=300, help="lora: training rows (x2 tiers, 1 epoch)")
        p.add_argument("overrides", nargs="*")
    sub.add_parser("report")
    a = ap.parse_args()
    {"zs": cmd_zs, "lora": cmd_lora, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
