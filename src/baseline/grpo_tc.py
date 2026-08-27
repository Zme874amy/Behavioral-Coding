"""GRPO on the two-call format: extend the RL ladder into cells A and B.

`baseline.grpo` trains the single-call cell (F): one generation, one adapter,
one scalar reward. This module trains the two remaining structural cells under
RL, so the GRPO grid mirrors the SFT grid in `baseline.local_arm`:

    regime   cell   adapters   arm names
      pair     A        2       grpo_pair_dec / grpo_pair_joint
      mix      B        1       grpo_mix_dec  / grpo_mix_joint

Each cell is run under two credit-assignment schemes:

  * decoupled (this module) -- each call is its own stationary GRPO problem.
    T1 is optimised against the T1 reward; T2 is optimised against the T2 reward
    while conditioned on a FROZEN predicted T1 (the out-of-fold store the SFT
    teacher-forcing arms already use), so the conditioning distribution does not
    move under it. The pair regime runs two stages (T1 adapter, then T2 adapter);
    the mix regime runs one stage over an interleaved T1+T2 dataset updating one
    shared adapter, each prompt scored by its own tier's reward.

  * joint (`baseline.two_call_grpo`) -- the two calls are one trajectory with one
    shared advantage. Delegated to that module; `--credit joint` dispatches there.

The three things `baseline.grpo` documents as easy to get wrong all still apply,
per stage: rare-class weighting only bites with `scale_rewards=none`; the policy
must start from an adapter that can already sometimes be right (warm-start from
the SFT arm, cold-start is the ablation); and the rationale can be farmed, so the
collapse guard runs on every call.

Usage:
    PYTHONPATH=src python -m baseline.grpo_tc train --regime pair --credit dec --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.grpo_tc train --regime mix  --credit dec --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.grpo_tc train --regime pair --credit joint --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.grpo_tc calibrate --regime mix --ctx 5 --limit 50
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "conf" / "grpo_tc_config.yaml"

REGIMES = ("pair", "mix")
CREDITS = ("dec", "joint")

# regime x credit -> arm name and structural cell tag (must match eval.py).
ARM_NAME = {
    ("pair", "dec"): "grpo_pair_dec",
    ("pair", "joint"): "grpo_pair_joint",
    ("mix", "dec"): "grpo_mix_dec",
    ("mix", "joint"): "grpo_mix_joint",
}
ARM_TO_RC = {name: rc for rc, name in ARM_NAME.items()}
ARMS = tuple(ARM_NAME.values())
# Variant -> arm-name suffix, mirroring cell F's sc_grpo / sc_grpo_unw /
# sc_grpo_cold. The suffix enters the result filename so an ablation can never
# overwrite the Phase-1 arm it is compared against.
VARIANT_SUFFIX = {"weighted": "", "unweighted": "_unw", "coldstart": "_cold"}
# The SFT arm each cell warm-starts from, resolved through the two-call config.
WARMSTART_SFT = {"pair": ("bare", "pair"), "mix": ("bare", "mixed")}


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


# -----------------------------------------------------------------------------
# Adapter directories. Kept strictly disjoint from cell F (`grpo`) and the SFT
# arms (`local_arm`): a shared path would let one run overwrite another and end
# with an arm compared against itself.
# -----------------------------------------------------------------------------
def grpo_tc_adapter_dir(
    cfg: DictConfig, ctx: int, regime: str, credit: str, seed: int,
    variant: str = "weighted", tier: Optional[str] = None,
) -> Path:
    base = (
        REPO_ROOT / cfg.paths.grpo_tc_adapter_dir / f"ctx{ctx}"
        / f"{regime}_{credit}" / variant / f"seed{seed}"
    )
    # Pair keeps one subdir per tier adapter; mix has a single adapter.
    return base / tier if tier else base


def _warmstart_config():
    """The two-call SFT config, for resolving warm-start adapter paths."""
    from baseline.local_arm import load_config as load_sft_config

    return load_sft_config()


def warmstart_dirs(ctx: int, regime: str):
    """Resolve the SFT adapter(s) a cell initialises from.

    pair -> (t1_dir, t2_dir) from the `ft_bare` pair; mix -> (shared_dir, None)
    from the `ft1mix_bare` adapter. These are trained by `baseline.local_arm`.
    """
    from baseline.local_arm import adapter_dirs, shared_adapter_dir

    sft_cfg = _warmstart_config()
    target, sft_regime = WARMSTART_SFT[regime]
    if regime == "pair":
        return adapter_dirs(sft_cfg, target, ctx)
    return shared_adapter_dir(sft_cfg, target, ctx, sft_regime), None


# -----------------------------------------------------------------------------
# Dataset. One record per (utterance, tier). For T2 the conditioning T1 is baked
# into the prompt here, so the trainer never has to reason about it again.
# -----------------------------------------------------------------------------
def split_by_conversation(df: pd.DataFrame, n_folds: int, val_fold: int, seed: int):
    """Conversation-level train/val split of the TRAINING corpus (reused as-is
    from the cell-F trainer's rationale: row-level would leak context windows)."""
    from automisc_ft.data import assign_folds

    fold_of = assign_folds(df, n_folds, seed)
    is_val = df["conv_id"].map(fold_of) == val_fold
    return df[~is_val].copy(), df[is_val].copy()


def build_tier_records(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: DictConfig,
    ctx: int,
    tier: str,
    weights: Optional[Dict[str, Dict[str, float]]] = None,
    predicted_t1: Optional[Dict[str, dict]] = None,
) -> List[Dict]:
    """Records for one tier: chat prompt, gold, class weight, and (for T2) the
    conditioning T1 that its hierarchy reward is scored against.

    T2 conditions on the frozen out-of-fold predicted T1 when `predicted_t1` is
    given (the `tfzero` shape the pipeline actually meets at inference); it falls
    back to the gold T1 only when a row has no usable prediction on file, which
    is counted and warned so a gappy store cannot silently train on gold.
    """
    from automisc_ft.data import t1_codes_for_speaker
    from baseline import two_call

    pos_of = {int(v): i for i, v in enumerate(full_df["corp_utt_idx"].tolist())}
    ctx_mode = cfg.annotator.context_mode
    records: List[Dict] = []
    n_missing_pred = 0
    for _, row in df.iterrows():
        speaker = row["speaker"]
        t1_gold, t2_gold = row.get("t1_label_GT"), row.get("t2_label_GT")
        if speaker not in {"counsellor", "client"}:
            continue
        if not isinstance(t1_gold, str) or not isinstance(t2_gold, str):
            continue
        pos = pos_of[int(row["corp_utt_idx"])]

        if tier == "t1":
            prompt = two_call.t1_messages(full_df, pos, ctx_mode, ctx)
            gold, t1_cond, weight_code = t1_gold, None, None
        else:
            t1_cond = t1_gold
            if predicted_t1 is not None:
                entry = predicted_t1.get(str(row.get("corp_utt_idx"))) or {}
                pred = entry.get("t1_pred")
                if isinstance(pred, str) and pred in t1_codes_for_speaker(speaker):
                    t1_cond = pred
                else:
                    n_missing_pred += 1
            prompt = two_call.t2_messages(full_df, pos, t1_cond, ctx_mode, ctx)
            gold, weight_code = t2_gold, t2_gold

        weight = 1.0
        if weights and weight_code is not None:
            weight = float(weights.get(speaker, {}).get(weight_code, 1.0))
        records.append(
            {
                "prompt": prompt,
                "tier": tier,
                "speaker": speaker,
                "t1_gold": t1_gold,
                "t2_gold": t2_gold,
                "t1_cond": t1_cond if t1_cond is not None else "",
                "gold": gold,
                "weight": weight,
                "corp_utt_idx": int(row["corp_utt_idx"]),
            }
        )
    if tier == "t2" and predicted_t1 is not None and n_missing_pred:
        print(
            f"WARNING: {n_missing_pred} T2 rows had no usable out-of-fold T1 and "
            "fell back to gold conditioning; regenerate the oof_t1 store for this ctx."
        )
    return records


def oversample(records: List[Dict], counts: Dict[str, Dict[str, int]]) -> List[Dict]:
    """Repeat rare-gold T2 prompts (keyed on the gold T2 code, per speaker)."""
    out: List[Dict] = []
    for rec in records:
        n = counts.get(rec["speaker"], {}).get(rec["t2_gold"], 1)
        out.extend([rec] * max(1, n))
    return out


# -----------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------
class RewardStats:
    """Rolling window of two-call reward components, per tier, for the collapse
    guard and logging. Keys present depend on which tiers are in the stream."""

    def __init__(self, window: int = 512):
        self.window = window
        self.rationale_words: List[float] = []
        self.components: Dict[str, List[float]] = {}

    def _push_series(self, key: str, value: float) -> None:
        series = self.components.setdefault(key, [])
        series.append(float(value))
        if len(series) > self.window:
            del series[: len(series) - self.window]

    def push(self, scored: Dict[str, object], tier: str, n_words: int) -> None:
        self.rationale_words.append(float(n_words))
        if len(self.rationale_words) > self.window:
            del self.rationale_words[: len(self.rationale_words) - self.window]
        self._push_series("reward", scored["reward"])
        for key in ("format", "hierarchy", "t1_hit", "t2_hit"):
            if key in scored:
                self._push_series(f"{tier}/{key}", scored[key])

    def mean_rationale_words(self) -> float:
        return statistics.fmean(self.rationale_words) if self.rationale_words else 0.0

    def summary(self) -> Dict[str, float]:
        out = {k: (statistics.fmean(v) if v else 0.0) for k, v in self.components.items()}
        out["rationale_words"] = self.mean_rationale_words()
        return out


def make_reward_fn(stats: RewardStats, weighted: bool, expect_rationale: bool = True):
    """TRL reward callable that dispatches on each record's tier.

    A single reward function serves both the per-tier `pair` stages and the
    interleaved `mix` dataset: the `tier` column TRL passes through selects
    `score_t1` or `score_t2`, and the class weight survives into the gradient
    only because `scale_rewards` is off (same invariant as cell F).
    """
    from automisc_ft.infer import emitted_rationale
    from baseline import two_call

    def reward_fn(completions, tier, speaker, t1_gold, t2_gold, t1_cond, weight,
                 log_metric=None, **kwargs):
        rewards: List[float] = []
        for comp, tr, spk, g1, g2, cond, w in zip(
            completions, tier, speaker, t1_gold, t2_gold, t1_cond, weight
        ):
            text = _completion_text(comp)
            if tr == "t1":
                scored = two_call.score_t1(text, spk, g1, expect_rationale)
            else:
                scored = two_call.score_t2(text, spk, cond, g2, expect_rationale)
            n_words = len(
                [w2 for w2 in text.split() if any(c.isalpha() for c in w2)]
            )
            stats.push(scored, tr, n_words)
            rewards.append(scored["reward"] * (float(w) if weighted else 1.0))
        if log_metric is not None:
            for name, value in stats.summary().items():
                log_metric(f"reward/{name}", value)
        return rewards

    reward_fn.__name__ = "two_call_reward"
    return reward_fn


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        return last.get("content", "") if isinstance(last, dict) else str(last)
    return ""


def _check_weighting_is_live(scale_rewards, weighted: bool) -> None:
    if weighted and scale_rewards not in (False, "none"):
        raise SystemExit(
            f"grpo.scale_rewards={scale_rewards!r} cancels the rare-class weight "
            "(every rollout in a group shares one gold label). Set it to none, or "
            "run with --no-rare-weighting."
        )


# -----------------------------------------------------------------------------
# Validation: greedy-decode the tier prompts and score that tier.
# -----------------------------------------------------------------------------
def evaluate_records(
    model, tokenizer, records: Sequence[Dict], max_new_tokens: int = 256,
    batch_size: int = 8, expect_rationale: bool = True, desc: str = "eval",
) -> Dict[str, float]:
    """Per-tier accuracy and macro-F1; selection uses the mean macro-F1 across
    whatever tiers are present, matching the cell-F selector's rationale."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    from automisc_ft.data import t1_codes_for_speaker, t2_codes_for_speaker
    from automisc_ft.infer import parse_label

    preds: Dict[str, List[str]] = {"t1": [], "t2": []}
    golds: Dict[str, List[str]] = {"t1": [], "t2": []}
    device = next(model.parameters()).device
    for start in tqdm(range(0, len(records), batch_size), desc=desc, leave=False):
        batch = records[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(r["prompt"], tokenize=False,
                                          add_generation_prompt=True)
            for r in batch
        ]
        prev = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        tokenizer.padding_side = prev
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        for rec, ids in zip(batch, gen):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            tier = rec["tier"]
            allowed = (
                t1_codes_for_speaker(rec["speaker"]) if tier == "t1"
                else t2_codes_for_speaker(rec["speaker"])
            )
            preds[tier].append(parse_label(text, allowed))
            golds[tier].append(rec["gold"])

    metrics: Dict[str, float] = {}
    present = [t for t in ("t1", "t2") if golds[t]]
    for tier in present:
        metrics[f"{tier}_acc"] = float(accuracy_score(golds[tier], preds[tier]))
        classes = sorted(set(golds[tier]))
        metrics[f"{tier}_macro_f1"] = float(
            f1_score(golds[tier], preds[tier], labels=classes,
                     average="macro", zero_division=0)
        )
    metrics["macro_f1_mean"] = (
        statistics.fmean(metrics[f"{t}_macro_f1"] for t in present) if present else 0.0
    )
    return metrics


def _make_callbacks(cfg, stats, val_records, out_dir, tokenizer, expect_rationale):
    from transformers import TrainerCallback

    class RationaleCollapseGuard(TrainerCallback):
        def __init__(self, floor: float, check_every: int = 25):
            self.floor, self.check_every = float(floor), int(check_every)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step < self.check_every or state.global_step % self.check_every:
                return control
            if stats.mean_rationale_words() < self.floor:
                print(f"\nABORT at step {state.global_step}: mean rationale "
                      f"{stats.mean_rationale_words():.1f} words below floor "
                      f"{self.floor}; the policy is farming label reward.")
                control.should_training_stop = True
            return control

    class ValidationSelector(TrainerCallback):
        def __init__(self, every: int, max_examples: int):
            self.every, self.max_examples = int(every), int(max_examples)
            self.best = -1.0
            self.history: List[Dict[str, float]] = []

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not state.global_step or state.global_step % self.every:
                return control
            was_training = model.training
            model.eval()
            try:
                metrics = evaluate_records(
                    model, tokenizer, val_records[: self.max_examples],
                    max_new_tokens=int(cfg.grpo.max_completion_length),
                    expect_rationale=expect_rationale, desc=f"val@{state.global_step}",
                )
            finally:
                if was_training:
                    model.train()
            metrics["step"] = state.global_step
            self.history.append(metrics)
            score = metrics["macro_f1_mean"]
            marker = ""
            if score > self.best:
                self.best = score
                model.save_pretrained(str(out_dir))
                marker = "  <- best, saved"
            print(f"\n[val step {state.global_step}] mean_macroF1={score:.3f}{marker}")
            return control

    guard = RationaleCollapseGuard(cfg.grpo.min_mean_rationale_words)
    selector = ValidationSelector(
        int(cfg.grpo.val_every_steps), int(cfg.grpo.get("val_max_examples", 200))
    )
    return guard, selector


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def load_policy_from(cfg: DictConfig, adapter: Optional[Path], cold_start: bool):
    """Load Qwen and either continue `adapter` (warm start) or attach fresh LoRA.

    Returns `(model, tokenizer, peft_config)`; `peft_config` is None when an
    adapter is continued, so TRL uses the adapter-disabled base as the KL
    reference at no extra memory -- the same trick cell F relies on.
    """
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from components.hf_load import resolve_hf_token

    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model, token=token, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model, token=token,
        dtype=torch.bfloat16 if cfg.model.bf16 else torch.float32,
        attn_implementation=cfg.model.get("attn_implementation", "sdpa"),
    )
    if cold_start:
        print("Cold start: fresh LoRA on the base model")
        peft_config = LoraConfig(
            r=int(cfg.model.peft_r), lora_alpha=int(cfg.model.peft_alpha),
            lora_dropout=float(cfg.model.peft_dropout),
            target_modules=list(cfg.model.target_modules), bias="none",
            task_type="CAUSAL_LM",
        )
        return model, tokenizer, peft_config
    if adapter is None or not Path(adapter).exists():
        raise SystemExit(
            f"Missing warm-start adapter {adapter}. Train the SFT arm first with "
            "`baseline.local_arm train` (see docs/EXPERIMENTS.md cells A/B)."
        )
    print(f"Initialising policy from {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    return model, tokenizer, None


# -----------------------------------------------------------------------------
# One decoupled GRPO stage (reused by both pair stages and the mix run)
# -----------------------------------------------------------------------------
def _grpo_config(cfg, out_dir, seed):
    from trl import GRPOConfig

    g = cfg.grpo
    return GRPOConfig(
        output_dir=str(out_dir / "trainer"), seed=seed,
        num_generations=int(g.num_generations), num_train_epochs=float(g.num_train_epochs),
        learning_rate=float(g.learning_rate), temperature=float(g.temperature),
        top_p=float(g.top_p), max_completion_length=int(g.max_completion_length),
        beta=float(g.beta), epsilon=float(g.epsilon), num_iterations=int(g.num_iterations),
        scale_rewards=g.scale_rewards, loss_type=str(g.loss_type),
        per_device_train_batch_size=int(g.per_device_train_batch_size),
        gradient_accumulation_steps=int(g.gradient_accumulation_steps),
        gradient_checkpointing=bool(g.gradient_checkpointing),
        max_grad_norm=float(g.max_grad_norm), warmup_ratio=float(g.warmup_ratio),
        lr_scheduler_type=str(g.lr_scheduler_type), bf16=bool(cfg.model.bf16),
        use_vllm=bool(cfg.vllm.enabled), vllm_mode=str(cfg.vllm.mode),
        vllm_gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        vllm_max_model_length=int(cfg.vllm.max_model_length),
        vllm_enable_sleep_mode=bool(cfg.vllm.get("enable_sleep_mode", False)),
        log_completions=bool(g.log_completions),
        num_completions_to_print=int(g.num_completions_to_print),
        logging_steps=int(g.logging_steps), save_strategy="no", report_to=[],
    )


def _run_stage(cfg, train_records, val_records, warmstart, out_dir, seed,
               weighted, cold_start, expect_rationale) -> Dict[str, object]:
    """Train one decoupled GRPO stage and return its metadata dict."""
    from datasets import Dataset
    from trl import GRPOTrainer

    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = load_policy_from(cfg, warmstart, cold_start)
    stats = RewardStats()
    reward_fn = make_reward_fn(stats, weighted, expect_rationale)
    guard, selector = _make_callbacks(cfg, stats, val_records, out_dir, tokenizer,
                                      expect_rationale)
    trainer = GRPOTrainer(
        model=model, args=_grpo_config(cfg, out_dir, seed), reward_funcs=reward_fn,
        train_dataset=Dataset.from_list(train_records), peft_config=peft_config,
        processing_class=tokenizer, callbacks=[guard, selector],
    )
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    if not (out_dir / "adapter_config.json").exists():
        print("No validation improvement recorded; saving the final policy.")
        trainer.model.save_pretrained(str(out_dir))
    return {
        "wall_clock_hours": round(elapsed / 3600, 3),
        "best_val_macro_f1_mean": selector.best,
        "val_history": selector.history,
        "final_reward_components": stats.summary(),
        "n_train_prompts": len(train_records),
        "n_val_prompts": len(val_records),
    }


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def _load_oof(ctx: int, train_df: pd.DataFrame) -> Dict[str, dict]:
    """Frozen out-of-fold predicted T1, with the same full-coverage assertion the
    SFT teacher-forcing arms make: an uncovered row would silently train on gold."""
    from baseline.oof_t1 import load_oof_t1, oof_t1_path

    store = load_oof_t1(ctx)
    if not store:
        raise SystemExit(
            f"No out-of-fold predicted T1 for ctx={ctx} at {oof_t1_path(ctx)}. "
            "Generate it (one job per fold) before decoupled T2:\n"
            f"  PYTHONPATH=src python -m baseline.oof_t1 train-fold   --ctx {ctx} --fold K\n"
            f"  PYTHONPATH=src python -m baseline.oof_t1 predict-fold --ctx {ctx} --fold K"
        )
    covered = sum(
        1 for _, r in train_df.iterrows() if str(r.get("corp_utt_idx")) in store
    )
    if covered < len(train_df):
        raise SystemExit(
            f"Out-of-fold T1 for ctx={ctx} covers only {covered}/{len(train_df)} "
            "training rows; the rest would fall back to gold conditioning. Finish "
            "every fold and re-run `baseline.oof_t1 report`."
        )
    return store


def cmd_train(args) -> None:
    if args.credit == "joint":
        from baseline import two_call_grpo

        two_call_grpo.train(args)
        return

    from automisc_ft.data import load_manual
    from baseline.single_call import oversample_counts, speaker_class_weights

    cfg = load_config(args.overrides)
    ctx, seed, regime = args.ctx, args.seed, args.regime
    weighted = bool(cfg.grpo.rare_class_weighting) and not args.no_rare_weighting
    _check_weighting_is_live(cfg.grpo.scale_rewards, weighted)
    variant = "coldstart" if args.cold_start else ("weighted" if weighted else "unweighted")
    arm = ARM_NAME[(regime, args.credit)]

    full_df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    train_df, val_df = split_by_conversation(
        full_df, int(cfg.grpo.val_folds), int(cfg.grpo.val_fold), seed=42
    )
    print(f"HLQC split by conversation: train={len(train_df)} rows, val={len(val_df)} rows")

    weights = counts = None
    if weighted:
        weights = speaker_class_weights(
            train_df, power=float(cfg.grpo.weight_power),
            floor=float(cfg.grpo.weight_floor), ceiling=float(cfg.grpo.weight_ceiling),
        )
        counts = {
            spk: oversample_counts([], w, max_copies=int(cfg.grpo.oversample_max_copies))
            for spk, w in weights.items()
        }

    predicted_t1 = _load_oof(ctx, train_df)
    warm_t1, warm_t2 = warmstart_dirs(ctx, regime)

    def records_for(tier, df):
        return build_tier_records(df, full_df, cfg, ctx, tier, weights,
                                  predicted_t1 if tier == "t2" else None)

    # Pair runs two GRPO stages. They are INDEPENDENT -- T2 conditions on the
    # frozen oof T1, not the RL'd T1 -- so `--stage t1|t2` can split them into
    # separate (even parallel) jobs, each fitting the wall; `both` runs them in
    # one process. Mix ignores this (one shared run).
    stages = ("t1", "t2") if args.stage == "both" else (args.stage,)
    meta_stages: Dict[str, object] = {}
    if regime == "pair":
        # Stage 1: T1 adapter. Stage 2: T2 adapter on frozen predicted-T1.
        for tier, warm in (("t1", warm_t1), ("t2", warm_t2)):
            if tier not in stages:
                continue
            train_recs = records_for(tier, train_df)
            val_recs = records_for(tier, val_df)
            if args.limit:
                train_recs = train_recs[: int(args.limit)]
                val_recs = val_recs[: max(8, int(args.limit) // 4)]
            if weighted and tier == "t2":
                before = len(train_recs)
                train_recs = oversample(train_recs, counts)
                print(f"Oversampled rare T2: {before} -> {len(train_recs)} prompts")
            out_dir = grpo_tc_adapter_dir(cfg, ctx, regime, args.credit, seed, variant, tier)
            print(f"\n=== {arm} stage {tier} -> {out_dir} ===")
            meta_stages[tier] = _run_stage(
                cfg, train_recs, val_recs, warm, out_dir, seed, weighted,
                args.cold_start, expect_rationale=True,
            )
    else:
        # Mix: one shared adapter over interleaved T1 + T2 records.
        import random as _random

        train_recs = records_for("t1", train_df) + records_for("t2", train_df)
        val_recs = records_for("t1", val_df) + records_for("t2", val_df)
        _random.Random(seed).shuffle(train_recs)
        if args.limit:
            train_recs = train_recs[: int(args.limit)]
            val_recs = val_recs[: max(8, int(args.limit) // 4)]
        if weighted:
            before = len(train_recs)
            train_recs = oversample(train_recs, counts)
            print(f"Oversampled rare T2: {before} -> {len(train_recs)} prompts")
        out_dir = grpo_tc_adapter_dir(cfg, ctx, regime, args.credit, seed, variant)
        print(f"\n=== {arm} mixed -> {out_dir} ===")
        meta_stages["shared"] = _run_stage(
            cfg, train_recs, val_recs, warm_t1, out_dir, seed, weighted,
            args.cold_start, expect_rationale=True,
        )

    root = grpo_tc_adapter_dir(cfg, ctx, regime, args.credit, seed, variant)
    root.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "two_call", "arm": arm, "regime": regime, "credit": args.credit,
        "variant": variant, "seed": seed, "num_context_turns": ctx, "stage": args.stage,
        "rare_class_weighting": weighted, "cold_start": bool(args.cold_start),
        "base_model": cfg.model.base_model,
        "warmstart": {"t1": str(warm_t1), "t2": str(warm_t2)},
        "t2_conditioning": "oof_predicted_t1",
        "grpo": OmegaConf.to_container(cfg.grpo, resolve=True),
        "stages": meta_stages,
    }
    # Split pair stages run as separate jobs, so each writes its own metadata file
    # rather than racing on one shared name.
    meta_name = ("train_metadata.json" if (regime == "mix" or args.stage == "both")
                 else f"train_metadata_{args.stage}.json")
    (root / meta_name).write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nDone. Metadata -> {root / meta_name}")


# -----------------------------------------------------------------------------
# calibrate (throughput, mirrors baseline.grpo.calibrate for the two-call shape)
# -----------------------------------------------------------------------------
def cmd_calibrate(args) -> None:
    from automisc_ft.data import load_manual

    cfg = load_config(args.overrides)
    ctx = args.ctx
    full_df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    train_df, _ = split_by_conversation(
        full_df, int(cfg.grpo.val_folds), int(cfg.grpo.val_fold), seed=42
    )
    # Time the T2 stage: it is the heavier of the two (longer conditioned prompt).
    predicted_t1 = _load_oof(ctx, train_df)
    records = build_tier_records(train_df, full_df, cfg, ctx, "t2", None, predicted_t1)
    records = records[: int(args.limit)]

    warm_t1, warm_t2 = warmstart_dirs(ctx, args.regime)
    warm = warm_t2 if args.regime == "pair" else warm_t1
    tokenizer_probe = None
    lengths: List[int] = []
    from transformers import AutoTokenizer

    from components.hf_load import resolve_hf_token

    tok = AutoTokenizer.from_pretrained(cfg.model.base_model, token=resolve_hf_token())
    for r in records:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
    print(f"T2 prompt tokens (ctx={ctx}): mean={statistics.fmean(lengths):.0f} "
          f"max={max(lengths)}  n={len(lengths)}")

    from datasets import Dataset
    from trl import GRPOTrainer

    out = REPO_ROOT / "outputs" / "grpo" / "calibrate_tc"
    model, tokenizer, peft_config = load_policy_from(cfg, warm, args.cold_start)
    stats = RewardStats()
    gcfg = _grpo_config(cfg, out, 0)
    gcfg.max_steps = int(args.steps)
    trainer = GRPOTrainer(
        model=model, args=gcfg, reward_funcs=make_reward_fn(stats, False, True),
        train_dataset=Dataset.from_list(records), peft_config=peft_config,
        processing_class=tokenizer,
    )
    started = time.time()
    trainer.train()
    per_step = (time.time() - started) / max(1, int(args.steps))
    prompts_per_step = (
        int(cfg.grpo.per_device_train_batch_size)
        * int(cfg.grpo.gradient_accumulation_steps)
        // int(cfg.grpo.num_generations)
    )
    report = {
        "regime": args.regime, "steps_timed": int(args.steps),
        "s_per_step": round(per_step, 2),
        "s_per_prompt": round(per_step / max(1, prompts_per_step), 3),
        "t2_prompt_tokens": {"mean": round(statistics.fmean(lengths)), "max": max(lengths)},
        "reward_components": stats.summary(),
    }
    out_path = REPO_ROOT / "outputs" / "grpo" / f"calibration_tc_{args.regime}_ctx{ctx}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nCalibration -> {out_path}\n{json.dumps(report, indent=2)}")


# -----------------------------------------------------------------------------
# predict: run the trained adapters through the production two-call annotator.
# -----------------------------------------------------------------------------
def _resolve_pred_adapters(cfg, arm, ctx, seed, variant):
    """(t1_dir, t2_dir, shared_dir) for a trained arm, tolerant of PEFT's two
    save layouts (a named adapter can land in a subdir or at the root)."""
    regime, credit = ARM_TO_RC[arm]
    root = grpo_tc_adapter_dir(cfg, ctx, regime, credit, seed, variant)

    def pick(*cands):
        for c in cands:
            if (c / "adapter_config.json").exists():
                return c
        return cands[0]  # report the primary candidate in the missing-adapter error

    if regime == "pair":
        return pick(root / "t1"), pick(root / "t2"), None
    return None, None, pick(root, root / "shared")


def cmd_predict(args) -> None:
    from automisc_ft.data import load_manual
    from automisc_ft.infer import TieredAnnotator
    from baseline.local_arm import _report, _write_run_meta, _utc_now

    cfg = load_config(args.overrides)
    ctx, arm, seed, variant = args.ctx, args.arm, args.seed, args.variant
    started = _utc_now()

    t1_dir, t2_dir, shared_dir = _resolve_pred_adapters(cfg, arm, ctx, seed, variant)
    for path in (t1_dir, t2_dir, shared_dir):
        if path is not None and not path.exists():
            raise SystemExit(
                f"Missing adapter {path}. Train it first:\n"
                f"  PYTHONPATH=src python -m baseline.grpo_tc train "
                f"--regime {ARM_TO_RC[arm][0]} --credit {ARM_TO_RC[arm][1]} "
                f"--seed {seed} --ctx {ctx}"
            )

    df = load_manual(REPO_ROOT / cfg.dataset.eval_csv)
    df = df.drop(columns=[c for c in df.columns if c.endswith("_auto")])
    limit = args.limit if args.limit is not None else cfg.limit
    n_total = len(df) if limit in (None, "null") else min(int(limit), len(df))

    # cot style (these arms emit rationales); the variant suffix keeps the two
    # Phase-2 ablations from overwriting the Phase-1 arm. Seeded, same scheme as sc_arm.
    arm_out = arm + VARIANT_SUFFIX[variant]
    save_path = (
        REPO_ROOT / cfg.paths.output_dir
        / f"{cfg.tier}_{arm_out}_inf_cot_ctx{ctx}_seed{seed}.csv"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    utt_checkpoint = -1
    existing_df = None
    if save_path.exists():
        existing_df = pd.read_csv(save_path)
        if not existing_df.empty:
            utt_checkpoint = existing_df["corp_utt_idx"].max()
            print(f"Resuming: {len(existing_df)} rows done (to corp_utt_idx {utt_checkpoint})")
    positions = [i for i in range(n_total) if df.iloc[i]["corp_utt_idx"] > utt_checkpoint]
    if not positions:
        print(f"Nothing to do; {save_path} is already complete.")
        return

    max_input_len = int(cfg.inference.max_input_len)
    cond = f"{cfg.tier}_{arm_out}_inf_cot"
    print(f"Condition={cond} ctx={ctx} seed={seed} t1={t1_dir} t2={t2_dir} "
          f"shared={shared_dir} n={len(positions)} -> {save_path}")

    annotator = TieredAnnotator(
        base_model=cfg.model.base_model,
        t1_adapter_dir=str(t1_dir) if t1_dir else None,
        t2_adapter_dir=str(t2_dir) if t2_dir else None,
        shared_adapter_dir=str(shared_dir) if shared_dir else None,
        force_cpu=bool(cfg.inference.force_cpu),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
        max_new_tokens=int(cfg.inference.max_new_tokens),
        max_input_len=max_input_len, structure_suffix="",  # cot prompts
    )
    context_mode = cfg.annotator.context_mode
    output_rows: List[Dict] = []

    def save():
        nonlocal existing_df, output_rows
        if not output_rows:
            return
        out = pd.DataFrame(output_rows)
        if existing_df is not None:
            out = pd.concat([existing_df, out], ignore_index=True)
        out.to_csv(save_path, index=False)
        existing_df = out
        output_rows = []

    from tqdm import tqdm

    try:
        for pos in tqdm(positions, desc=cond, unit="utt"):
            pred = annotator.predict_row(df, pos, context_mode, ctx)
            output_rows.append({
                **df.iloc[pos].to_dict(),
                "t1_label_auto": pred["t1_pred"], "t2_label_auto": pred["t2_pred"],
                "t1_raw": pred["t1_raw"], "t2_raw": pred["t2_raw"],
                "t1_emitted_rationale": pred["t1_emitted_rationale"],
                "t2_emitted_rationale": pred["t2_emitted_rationale"],
                "t1_n_prompt_tokens": pred["t1_n_prompt_tokens"],
                "t2_n_prompt_tokens": pred["t2_n_prompt_tokens"],
                "t1_n_gen_tokens": pred["t1_n_gen_tokens"],
                "t2_n_gen_tokens": pred["t2_n_gen_tokens"],
            })
            if len(output_rows) >= int(cfg.checkpoint_every):
                save()
    finally:
        save()
        annotator.close()

    _report(existing_df, cond, save_path, int(cfg.inference.max_new_tokens), max_input_len)
    _write_run_meta(save_path, arm_out, "cot", ctx, existing_df, started)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one two-call GRPO arm")
    p_train.add_argument("--regime", choices=REGIMES, required=True)
    p_train.add_argument("--credit", choices=CREDITS, required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--stage", choices=("t1", "t2", "both"), default="both",
                         help="pair regime only: run one decoupled stage per job "
                              "(the stages are independent) or both in one process")
    p_train.add_argument("--no-rare-weighting", action="store_true",
                         help="Phase 2 ablation: uniform weights, no oversampling")
    p_train.add_argument("--limit", type=int, default=None, help="cap prompts (smoke test)")

    p_cal = sub.add_parser("calibrate", help="measure two-call throughput")
    p_cal.add_argument("--regime", choices=REGIMES, default="pair")
    p_cal.add_argument("--steps", type=int, default=5)
    p_cal.add_argument("--limit", type=int, default=50)

    p_pred = sub.add_parser("predict", help="annotate the evaluation set with a trained arm")
    p_pred.add_argument("--arm", choices=ARMS, required=True)
    p_pred.add_argument("--seed", type=int, required=True)
    p_pred.add_argument("--variant", default="weighted",
                        choices=("weighted", "unweighted", "coldstart"),
                        help="which trained variant to load (default: the Phase-1 weighted arm)")
    p_pred.add_argument("--limit", type=int, default=None)

    for p in (p_train, p_cal):
        p.add_argument("--cold-start", action="store_true",
                       help="Phase 2 ablation: start from the base model, not the SFT arm")
    for p in (p_train, p_cal, p_pred):
        p.add_argument("--ctx", type=int, default=None)
        p.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"train": cmd_train, "calibrate": cmd_calibrate, "predict": cmd_predict}[args.cmd](args)


if __name__ == "__main__":
    main()
