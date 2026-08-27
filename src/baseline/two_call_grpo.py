"""Joint-trajectory GRPO for the two-call cells.

The decoupled trainer in `baseline.grpo_tc` freezes the T1 conditioning so each
call is a stationary single-turn GRPO problem TRL can drive. The joint arm does
the opposite on purpose: it samples the WHOLE trajectory -- T1, then T2 given the
sampled T1 -- and gives the pair one reward, so the two calls are coupled exactly
as they are at inference. That coupling is what single-call sidestepped, and
comparing joint against decoupled (and against `sc_grpo`) is what prices it.

A two-turn trajectory does not fit TRL's one-prompt-one-reward `GRPOTrainer`, so
this is a self-contained GRPO loop over plain `transformers` + `peft`:

  * For each prompt, sample G T1 completions; for each, build the T2 prompt from
    its own sampled T1 and sample one T2 completion.
  * Score each (T1, T2) pair with `two_call.score_joint` (the single-call
    four-part reward, max 1.0), times the rare-class weight.
  * Group-centre the reward per prompt for the advantage
    (`A = r - mean_g r`; no std division, matching `scale_rewards=none` /
    `dr_grpo`), and apply that ONE advantage to BOTH turns' tokens. For the pair
    regime the T1 tokens' log-probs come from adapter `t1` and the T2 tokens'
    from adapter `t2`, so the shared advantage updates both adapters at once.
  * Penalise KL to the adapter-disabled base (the k3 estimator), the same
    reference cell F uses at no extra memory.

Loss (dr_grpo normalisation, constant `max_completion_length` denominator so
longer completions are not down-weighted):

    L = mean_traj [ sum_t ( -A * logp_t + beta * kl_t ) / max_completion_length ]

`train(args)` is called by `baseline.grpo_tc` on `--credit joint`.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


# -----------------------------------------------------------------------------
# Model + adapters
# -----------------------------------------------------------------------------
def _load_base(cfg: DictConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from components.hf_load import resolve_hf_token

    token = resolve_hf_token()
    tok = AutoTokenizer.from_pretrained(cfg.model.base_model, token=token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, token=token,
        dtype=torch.bfloat16 if cfg.model.bf16 else torch.float32,
        attn_implementation=cfg.model.get("attn_implementation", "sdpa"),
    )
    return model, tok


def _fresh_lora(cfg):
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg.model.peft_r), lora_alpha=int(cfg.model.peft_alpha),
        lora_dropout=float(cfg.model.peft_dropout),
        target_modules=list(cfg.model.target_modules), bias="none", task_type="CAUSAL_LM",
    )


def build_policy(cfg: DictConfig, regime: str, ctx: int, cold_start: bool):
    """Return `(model, tokenizer, adapter_names)`.

    pair -> two trainable adapters {"t1","t2"} (warm-started from the `ft_bare`
    pair, or fresh on cold start); mix -> one adapter {"shared"} from
    `ft1mix_bare`. All returned adapters are trainable; the active one is switched
    per turn with `model.set_adapter`.
    """
    from peft import PeftModel, get_peft_model

    from baseline.grpo_tc import warmstart_dirs

    model, tok = _load_base(cfg)
    warm_t1, warm_t2 = warmstart_dirs(ctx, regime)

    if regime == "pair":
        if cold_start:
            model = get_peft_model(model, _fresh_lora(cfg), adapter_name="t1")
            model.add_adapter("t2", _fresh_lora(cfg))
        else:
            for path in (warm_t1, warm_t2):
                if path is None or not Path(path).exists():
                    raise SystemExit(f"Missing warm-start adapter {path}; train ft_bare first.")
            model = PeftModel.from_pretrained(model, str(warm_t1), adapter_name="t1",
                                              is_trainable=True)
            model.load_adapter(str(warm_t2), adapter_name="t2", is_trainable=True)
        return model, tok, ("t1", "t2")

    # mix: single shared adapter
    if cold_start:
        model = get_peft_model(model, _fresh_lora(cfg), adapter_name="shared")
    else:
        if warm_t1 is None or not Path(warm_t1).exists():
            raise SystemExit(f"Missing warm-start adapter {warm_t1}; train ft1mix_bare first.")
        model = PeftModel.from_pretrained(model, str(warm_t1), adapter_name="shared",
                                          is_trainable=True)
    return model, tok, ("shared",)


def _adapter_for_turn(adapter_names: Tuple[str, ...], turn: str) -> str:
    """Active adapter for a turn: pair routes t1/t2, mix uses its one adapter."""
    if len(adapter_names) == 1:
        return adapter_names[0]
    return "t1" if turn == "t1" else "t2"


# -----------------------------------------------------------------------------
# Generation + log-probs
# -----------------------------------------------------------------------------
def _render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _sample(model, tok, messages, adapter: str, n: int, cfg, device) -> List["torch.Tensor"]:
    """Sample `n` completions; return their completion-token id tensors (no prompt)."""
    import torch

    model.set_adapter(adapter)
    text = _render(tok, messages)
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=int(cfg.vllm.max_model_length), add_special_tokens=False).to(device)
    n_prompt = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=True, num_return_sequences=n,
            temperature=float(cfg.grpo.temperature), top_p=float(cfg.grpo.top_p),
            max_new_tokens=int(cfg.grpo.max_completion_length),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    return [out[i, n_prompt:] for i in range(out.shape[0])]


def _token_logprobs(model, tok, messages, completion_ids, adapter: str, device,
                    with_grad: bool):
    """Per-token log-probs of `completion_ids` continuing `messages`, under
    `adapter`. `with_grad=False` (and adapter=None) gives the reference pass."""
    import torch

    prompt_ids = tok(_render(tok, messages), return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)
    comp = completion_ids.unsqueeze(0).to(device)
    input_ids = torch.cat([prompt_ids, comp], dim=1)
    n_prompt = prompt_ids.shape[1]

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        if adapter is None:
            with model.disable_adapter():
                logits = model(input_ids).logits
        else:
            model.set_adapter(adapter)
            logits = model(input_ids).logits
    # logits at position t predict token t+1; align to the completion tokens.
    logits = logits[:, n_prompt - 1 : -1, :]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, comp.unsqueeze(-1)).squeeze(-1).squeeze(0)


# -----------------------------------------------------------------------------
# One optimizer step
# -----------------------------------------------------------------------------
def _fallback_t1(speaker: str) -> str:
    from automisc_ft.data import t1_codes_for_speaker

    return t1_codes_for_speaker(speaker)[0]


def train_step(model, tok, batch, cfg, adapter_names, optimizer, device, stats):
    """One GRPO step over a batch of prompt records. Returns the scalar loss."""
    import torch

    from automisc_ft.data import t1_codes_for_speaker
    from automisc_ft.infer import parse_label
    from baseline import two_call

    G = int(cfg.grpo.num_generations)
    beta = float(cfg.grpo.beta)
    norm = float(cfg.grpo.max_completion_length)
    weighted = bool(cfg.grpo.rare_class_weighting)

    trajectories = []  # (t1_msgs, t1_ids, t2_msgs, t2_ids, advantage)
    for rec in batch:
        speaker = rec["speaker"]
        full_df, pos = rec["full_df"], rec["pos"]
        t1_msgs = two_call.t1_messages(full_df, pos, rec["ctx_mode"], rec["ctx"])
        t1_samples = _sample(model, tok, t1_msgs, _adapter_for_turn(adapter_names, "t1"),
                             G, cfg, device)
        group = []
        rewards = []
        for t1_ids in t1_samples:
            t1_text = tok.decode(t1_ids, skip_special_tokens=True)
            t1_parsed = parse_label(t1_text, t1_codes_for_speaker(speaker))
            t1_cond = t1_parsed if t1_parsed != two_call.UNKNOWN else _fallback_t1(speaker)
            t2_msgs = two_call.t2_messages(full_df, pos, t1_cond, rec["ctx_mode"], rec["ctx"])
            t2_ids = _sample(model, tok, t2_msgs,
                            _adapter_for_turn(adapter_names, "t2"), 1, cfg, device)[0]
            t2_text = tok.decode(t2_ids, skip_special_tokens=True)
            scored = two_call.score_joint(t1_text, t2_text, speaker,
                                          rec["t1_gold"], rec["t2_gold"])
            r = scored["reward"] * (float(rec["weight"]) if weighted else 1.0)
            rewards.append(r)
            n_words = len([w for w in (t1_text + " " + t2_text).split()
                           if any(c.isalpha() for c in w)])
            stats.push(scored, n_words)
            group.append((t1_msgs, t1_ids, t2_msgs, t2_ids))
        baseline = statistics.fmean(rewards)
        for (t1_msgs, t1_ids, t2_msgs, t2_ids), r in zip(group, rewards):
            trajectories.append((t1_msgs, t1_ids, t2_msgs, t2_ids, r - baseline))

    # Loss over all trajectories' tokens: -A*logp + beta*KL(policy||ref), dr_grpo norm.
    #
    # Backprop is done PER TURN, immediately, so only one sequence's forward graph
    # is ever alive -- summing the loss across all G x prompts x 2 turns into one
    # graph before a single backward() piles up every activation at once and OOMs
    # the 40GB card. Gradients accumulate in `.grad` across turns; one step() at
    # the end. Constant `norm * n_traj` denominator keeps it dr_grpo-normalised and
    # matched to the old averaged loss.
    optimizer.zero_grad()
    n_traj = max(1, len(trajectories))
    running = 0.0
    for t1_msgs, t1_ids, t2_msgs, t2_ids, adv in trajectories:
        for turn, msgs, ids in (("t1", t1_msgs, t1_ids), ("t2", t2_msgs, t2_ids)):
            if ids.numel() == 0:
                continue
            adapter = _adapter_for_turn(adapter_names, turn)
            lp = _token_logprobs(model, tok, msgs, ids, adapter, device, with_grad=True)
            ref_lp = _token_logprobs(model, tok, msgs, ids, None, device, with_grad=False)
            kl = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1.0  # k3, per token
            loss_i = (-float(adv) * lp.sum() + beta * kl.sum()) / (norm * n_traj)
            loss_i.backward()  # frees this turn's graph before the next forward
            running += float(loss_i.detach())
            del lp, ref_lp, kl, loss_i
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], float(cfg.grpo.max_grad_norm)
    )
    optimizer.step()
    return running


# -----------------------------------------------------------------------------
# Records + validation
# -----------------------------------------------------------------------------
def _records(df, full_df, cfg, ctx, weights) -> List[Dict]:
    pos_of = {int(v): i for i, v in enumerate(full_df["corp_utt_idx"].tolist())}
    ctx_mode = cfg.annotator.context_mode
    out = []
    for _, row in df.iterrows():
        speaker = row["speaker"]
        t1g, t2g = row.get("t1_label_GT"), row.get("t2_label_GT")
        if speaker not in {"counsellor", "client"} or not (isinstance(t1g, str) and isinstance(t2g, str)):
            continue
        weight = float((weights or {}).get(speaker, {}).get(t2g, 1.0)) if weights else 1.0
        out.append({
            "full_df": full_df, "pos": pos_of[int(row["corp_utt_idx"])], "ctx": ctx,
            "ctx_mode": ctx_mode, "speaker": speaker, "t1_gold": t1g, "t2_gold": t2g,
            "weight": weight, "corp_utt_idx": int(row["corp_utt_idx"]),
        })
    return out


def _macro_f1(golds: List[str], preds: List[str]) -> float:
    """Macro-F1 over gold classes, without sklearn (kept importable locally)."""
    classes = sorted(set(golds))
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = sum(1 for g, p in zip(golds, preds) if g == c and p == c)
        fp = sum(1 for g, p in zip(golds, preds) if g != c and p == c)
        fn = sum(1 for g, p in zip(golds, preds) if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return statistics.fmean(f1s)


def validate(model, tok, records, cfg, adapter_names, device) -> Dict[str, float]:
    """Greedy two-call decode over held-out rows; mean macro-F1 across tiers."""
    import torch

    from automisc_ft.data import t1_codes_for_speaker, t2_codes_for_speaker
    from automisc_ft.infer import parse_label
    from baseline import two_call

    was_training = model.training
    model.eval()
    golds = {"t1": [], "t2": []}
    preds = {"t1": [], "t2": []}
    try:
        for rec in records:
            speaker = rec["speaker"]
            t1_msgs = two_call.t1_messages(rec["full_df"], rec["pos"], rec["ctx_mode"], rec["ctx"])
            model.set_adapter(_adapter_for_turn(adapter_names, "t1"))
            enc = tok(_render(tok, t1_msgs), return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out = model.generate(**enc, do_sample=False,
                                     max_new_tokens=int(cfg.grpo.max_completion_length),
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            t1_text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            t1_pred = parse_label(t1_text, t1_codes_for_speaker(speaker))
            cond = t1_pred if t1_pred != two_call.UNKNOWN else _fallback_t1(speaker)
            t2_msgs = two_call.t2_messages(rec["full_df"], rec["pos"], cond, rec["ctx_mode"], rec["ctx"])
            model.set_adapter(_adapter_for_turn(adapter_names, "t2"))
            enc2 = tok(_render(tok, t2_msgs), return_tensors="pt", add_special_tokens=False).to(device)
            with torch.no_grad():
                out2 = model.generate(**enc2, do_sample=False,
                                      max_new_tokens=int(cfg.grpo.max_completion_length),
                                      pad_token_id=tok.pad_token_id or tok.eos_token_id)
            t2_text = tok.decode(out2[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)
            preds["t1"].append(t1_pred)
            preds["t2"].append(parse_label(t2_text, t2_codes_for_speaker(speaker)))
            golds["t1"].append(rec["t1_gold"])
            golds["t2"].append(rec["t2_gold"])
    finally:
        if was_training:
            model.train()
    f1 = {t: _macro_f1(golds[t], preds[t]) for t in ("t1", "t2")}
    return {"t1_macro_f1": f1["t1"], "t2_macro_f1": f1["t2"],
            "macro_f1_mean": statistics.fmean(f1.values())}


class _Stats:
    def __init__(self, window=512):
        self.window = window
        self.words: List[float] = []
        self.comp: Dict[str, List[float]] = {}

    def push(self, scored, n_words):
        self.words.append(float(n_words))
        for k in ("reward", "format", "hierarchy", "t1_hit", "t2_hit"):
            if k in scored:
                self.comp.setdefault(k, []).append(float(scored[k]))
        for s in (self.words, *self.comp.values()):
            if len(s) > self.window:
                del s[: len(s) - self.window]

    def mean_words(self):
        return statistics.fmean(self.words) if self.words else 0.0

    def summary(self):
        return {**{k: statistics.fmean(v) for k, v in self.comp.items() if v},
                "rationale_words": self.mean_words()}


# -----------------------------------------------------------------------------
# train entrypoint (called from baseline.grpo_tc on --credit joint)
# -----------------------------------------------------------------------------
def train(args) -> None:
    import random

    import torch

    from automisc_ft.data import load_manual
    from baseline.grpo_tc import (
        ARM_NAME, grpo_tc_adapter_dir, load_config, split_by_conversation, warmstart_dirs,
    )
    from baseline.single_call import speaker_class_weights

    cfg = load_config(args.overrides)
    ctx, seed, regime = args.ctx, args.seed, args.regime
    weighted = bool(cfg.grpo.rare_class_weighting) and not args.no_rare_weighting
    variant = "coldstart" if args.cold_start else ("weighted" if weighted else "unweighted")
    arm = ARM_NAME[(regime, "joint")]
    out_dir = grpo_tc_adapter_dir(cfg, ctx, regime, "joint", seed, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    full_df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    train_df, val_df = split_by_conversation(
        full_df, int(cfg.grpo.val_folds), int(cfg.grpo.val_fold), seed=42
    )
    weights = None
    if weighted:
        weights = speaker_class_weights(
            train_df, power=float(cfg.grpo.weight_power),
            floor=float(cfg.grpo.weight_floor), ceiling=float(cfg.grpo.weight_ceiling),
        )
    train_recs = _records(train_df, full_df, cfg, ctx, weights)
    val_recs = _records(val_df, full_df, cfg, ctx, None)
    if args.limit:
        train_recs = train_recs[: int(args.limit)]
        val_recs = val_recs[: max(4, int(args.limit) // 4)]
    random.Random(seed).shuffle(train_recs)

    model, tok, adapter_names = build_policy(cfg, regime, ctx, args.cold_start)
    if bool(cfg.inference.get("force_cpu", False)):
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=float(cfg.grpo.learning_rate)
    )

    stats = _Stats()
    bs = int(cfg.grpo.get("joint_prompts_per_step", 2))
    epochs = float(cfg.grpo.num_train_epochs)
    n_steps = max(1, int(len(train_recs) * epochs / bs))
    val_every = int(cfg.grpo.val_every_steps)
    val_cap = int(cfg.grpo.get("val_max_examples", 200))
    floor = float(cfg.grpo.min_mean_rationale_words)

    best = -1.0
    history: List[Dict[str, float]] = []
    started = time.time()
    print(f"Joint GRPO {arm} seed={seed} ctx={ctx}: {n_steps} steps, "
          f"{bs} prompts/step, adapters={adapter_names} -> {out_dir}")
    for step in range(1, n_steps + 1):
        batch = [train_recs[(step * bs + i) % len(train_recs)] for i in range(bs)]
        loss = train_step(model, tok, batch, cfg, adapter_names, optimizer, device, stats)
        if step % int(cfg.grpo.logging_steps) == 0:
            s = stats.summary()
            print(f"  step {step}/{n_steps} loss={loss:.4f} reward={s.get('reward',0):.3f} "
                  f"words={s['rationale_words']:.1f}")
        if step >= 25 and step % 25 == 0 and stats.mean_words() < floor:
            print(f"ABORT step {step}: rationale collapsed to {stats.mean_words():.1f} words.")
            break
        if val_every and step % val_every == 0:
            m = validate(model, tok, val_recs[:val_cap], cfg, adapter_names, device)
            m["step"] = step
            history.append(m)
            marker = ""
            if m["macro_f1_mean"] > best:
                best = m["macro_f1_mean"]
                model.save_pretrained(str(out_dir))
                marker = "  <- best, saved"
            print(f"  [val {step}] mean_macroF1={m['macro_f1_mean']:.3f}{marker}")

    if not (out_dir / "adapter_config.json").exists() and not any(
        (out_dir / n).exists() for n in ("t1", "t2")
    ):
        model.save_pretrained(str(out_dir))

    meta = {
        "format": "two_call", "arm": arm, "regime": regime, "credit": "joint",
        "variant": variant, "seed": seed, "num_context_turns": ctx,
        "rare_class_weighting": weighted, "cold_start": bool(args.cold_start),
        "base_model": cfg.model.base_model, "adapters": list(adapter_names),
        "warmstart": [str(p) for p in warmstart_dirs(ctx, regime)],
        "grpo": OmegaConf.to_container(cfg.grpo, resolve=True),
        "n_train_prompts": len(train_recs), "n_steps": n_steps,
        "wall_clock_hours": round((time.time() - started) / 3600, 3),
        "best_val_macro_f1_mean": best, "val_history": history,
        "final_reward_components": stats.summary(),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nDone. Best val mean macro-F1 {best:.3f}. Metadata -> {out_dir}")
