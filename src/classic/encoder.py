"""Encoder fine-tune for MISC coding (the deep-learning arm of the classic pathway).

Fine-tunes a sentence-classification encoder (default roberta-base, JMIR's
BERTje approach for English) as a T1 model and a T2 model, on HLQC, predicting
MIV6.3A. The speaker is prepended to the text ("counsellor: <utterance>") so one
model per tier covers both speakers; T2 logits are masked to the predicted-T1's
children at inference (hierarchy-consistent). Optional class-weighted loss and
context.

Runs on a GPU (MLeRP) in minutes; a 1-epoch --limit smoke runs on CPU. The model
must be present in the HF cache (staged offline on MLeRP): load with
HF_HUB_OFFLINE=1.

Usage:
    PYTHONPATH=src python -m classic.encoder --model roberta-base --ctx 5 --balanced
    PYTHONPATH=src python -m classic.encoder --model roberta-base --ctx 5 --epochs 1 --limit 32
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from automisc_ft.data import load_manual, t2_codes_for_group
from baseline.local_arm import REPO_ROOT, load_config

TAIL = ["SU", "EC", "AF", "GI"]
OUT_DIR = REPO_ROOT / "data" / "annotated" / "classic"


def _texts(df: pd.DataFrame, use_context: bool, mode: str, ctx: int) -> List[str]:
    from components.context import build_context_excerpt
    if not use_context:
        return [f"{r['speaker']}: {r['utt_text']}" for _, r in df.iterrows()]
    out = []
    for pos in range(len(df)):
        exc = build_context_excerpt(df, pos, mode, ctx)
        out.append(f"{df.iloc[pos]['speaker']}: {exc}\n{df.iloc[pos]['utt_text']}")
    return out


class _DS:
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def _train_tier(model_id, texts, gold, epochs, lr, balanced, max_len, device_cpu):
    """Fine-tune one classifier; return (model, tokenizer, id2label)."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    labels = sorted(pd.Series(gold).unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    y = np.array([lab2id[g] for g in gold])

    tok = AutoTokenizer.from_pretrained(model_id)
    enc = tok(list(texts), truncation=True, padding=True, max_length=max_len,
              return_tensors="pt")
    ds = _DS(enc, y)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(labels),
        id2label={i: l for l, i in lab2id.items()}, label2id=lab2id)

    class_weights = None
    if balanced:
        counts = np.bincount(y, minlength=len(labels))
        w = len(y) / (len(labels) * np.clip(counts, 1, None))
        class_weights = torch.tensor(w, dtype=torch.float32)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            wt = class_weights.to(out.logits.device) if class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=wt)
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(REPO_ROOT / "outputs" / "classic" / "hf"),
        num_train_epochs=epochs, per_device_train_batch_size=16,
        learning_rate=lr, logging_steps=50, save_strategy="no",
        report_to=[], use_cpu=device_cpu)
    WTrainer(model=model, args=args, train_dataset=ds).train()
    return model, tok, {i: l for l, i in lab2id.items()}


def _predict_logits(model, tok, texts, max_len) -> Tuple[np.ndarray, Dict[int, str]]:
    import torch
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(texts), 32):
        enc = tok(list(texts[i:i + 32]), truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            out.append(model(**enc).logits.cpu().numpy())
    return np.vstack(out), model.config.id2label


def run(args) -> None:
    cfg = load_config(args.overrides)
    ctx, mode = args.ctx, cfg.annotator.context_mode

    train = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    ev = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    ev = ev.drop(columns=[c for c in ev.columns if c.endswith("_auto")]).reset_index(drop=True)
    if args.limit:
        train = train.iloc[: int(args.limit) * 4].copy()
        ev = ev.iloc[: int(args.limit)].copy()

    tr_txt = _texts(train, args.context, mode, ctx)
    ev_txt = _texts(ev, args.context, mode, ctx)

    print("Fine-tuning T1 encoder...")
    m1, tok1, _ = _train_tier(args.model, tr_txt, train["t1_label_GT"].tolist(),
                              args.epochs, args.lr, args.balanced, args.max_len, args.cpu)
    print("Fine-tuning T2 encoder...")
    m2, tok2, _ = _train_tier(args.model, tr_txt, train["t2_label_GT"].tolist(),
                              args.epochs, args.lr, args.balanced, args.max_len, args.cpu)

    L1, id2l1 = _predict_logits(m1, tok1, ev_txt, args.max_len)
    L2, id2l2 = _predict_logits(m2, tok2, ev_txt, args.max_len)
    t2_col = {l: i for i, l in id2l2.items()}

    pred_t1 = [id2l1[int(i)] for i in L1.argmax(1)]
    pred_t2 = []
    for j, t1 in enumerate(pred_t1):
        spk = ev.iloc[j]["speaker"]
        kids = [c for c in t2_codes_for_group(spk, t1) if c in t2_col]
        if kids:
            pred_t2.append(kids[int(np.argmax([L2[j, t2_col[c]] for c in kids]))])
        else:
            pred_t2.append(id2l2[int(L2[j].argmax())])

    ev["t1_label_auto"] = pred_t1
    ev["t2_label_auto"] = pred_t2

    tag = f"encoder_{args.model.replace('/', '-')}" + ("_bal" if args.balanced else "") + \
          ("_ctx" if args.context else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"classic_{tag}_ctx{ctx}.csv"
    ev.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    _report(ev, tag)


def _report(ev: pd.DataFrame, tag: str) -> None:
    yt2, yp2 = ev["t2_label_GT"], ev["t2_label_auto"]
    labs = sorted(yt2.dropna().unique())
    macro = f1_score(yt2, yp2, labels=labs, average="macro", zero_division=0)
    print(f"\n=== classic:{tag} on {len(ev)} rows ===")
    print(f"T1 acc={accuracy_score(ev['t1_label_GT'], ev['t1_label_auto']):.3f}  "
          f"T2 acc={accuracy_score(yt2, yp2):.3f}  T2 macro-F1(gold)={macro:.3f}")
    for c in TAIL:
        print(f"  F1[{c}]={f1_score(yt2, yp2, labels=[c], average='macro', zero_division=0):.3f} "
              f"(support {int((yt2 == c).sum())})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--context", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", dest="max_len", type=int, default=256)
    p.add_argument("--ctx", type=int, default=5)
    p.add_argument("--cpu", action="store_true", help="force CPU (smoke)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("overrides", nargs="*")
    run(p.parse_args())


if __name__ == "__main__":
    main()
