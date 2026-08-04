"""GRPO on the single-call format: the model discovers its own reasoning.

The headline arm of the ladder. `sc_ft_bare` learns labels with no reasoning
and `sc_ft_rat` imitates gpt-4o rationales generated post-hoc from the gold
label; this arm instead samples G rollouts per utterance and reinforces the
ones that land the right codes, so whatever reasoning survives is reasoning
that actually helped rather than reasoning that was copied.

Three things here are easy to get wrong and are therefore enforced rather than
documented:

Rare-class weighting only works without std scaling. Every rollout in a group
answers the same prompt, so they share one gold label and one class weight.
Under GRPO's default `scale_rewards="group"` the advantage is
`(w*r - mean(w*r)) / std(w*r)`, and `w` cancels top and bottom: the weighting
becomes a silent no-op that still looks configured. `_check_weighting_is_live`
refuses to start unless scaling is off.

Initialisation has to be a model that can already sometimes be right. GRPO
learns from within-group reward *variance*; if all G rollouts are wrong the
advantage is zero and so is the gradient. A cold Qwen2.5-7B sits at 0.38 T2
accuracy, so most groups carry no signal. The Phase 2 cold-start run exists to
show this rather than assume it.

The reward can be farmed. Only format (0.1) and hierarchy (0.1) constrain the
shape; the labels carry 0.8. A policy can therefore trade the format point away
and emit a stub rationale. `RationaleCollapseGuard` aborts the run if mean
rationale length falls through the floor, and the rationale-swap probe in
`baseline.faithfulness` is what finally decides whether the rationale is
load-bearing.

Usage:
    PYTHONPATH=src python -m baseline.grpo train --seed 0 --ctx 5
    PYTHONPATH=src python -m baseline.grpo train --seed 0 --ctx 5 --no-rare-weighting
    PYTHONPATH=src python -m baseline.grpo train --seed 0 --ctx 5 --cold-start
    PYTHONPATH=src python -m baseline.grpo calibrate --ctx 5 --limit 50
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
CONFIG_PATH = REPO_ROOT / "conf" / "grpo_config.yaml"


def load_config(overrides: Optional[List[str]] = None) -> DictConfig:
    cfg = OmegaConf.load(CONFIG_PATH)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
def split_by_conversation(df: pd.DataFrame, n_folds: int, val_fold: int, seed: int):
    """Conversation-level train/validation split of the TRAINING corpus.

    Conversation-level, not row-level: utterances from one session share
    context windows, so a row-level split would leak the validation
    conversations into the prompts of training rows.

    The evaluation corpus (MIV6.3A) is not touched here at all. Checkpoint
    selection reads only this held-out slice of HLQC, so the reported
    MIV6.3A numbers stay out-of-sample.
    """
    from automisc_ft.data import assign_folds

    fold_of = assign_folds(df, n_folds, seed)
    is_val = df["conv_id"].map(fold_of) == val_fold
    return df[~is_val].copy(), df[is_val].copy()


def build_records(
    df: pd.DataFrame,
    full_df: pd.DataFrame,
    cfg: DictConfig,
    ctx: int,
    weights: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict]:
    """One record per utterance: chat prompt, gold labels, and class weight.

    Context is built against `full_df` (positional lookups walk backwards
    through the conversation) while the rows to emit come from `df`.
    """
    from baseline.single_call import build_messages

    pos_of = {int(v): i for i, v in enumerate(full_df["corp_utt_idx"].tolist())}
    records = []
    for _, row in df.iterrows():
        speaker = row["speaker"]
        t1, t2 = row.get("t1_label_GT"), row.get("t2_label_GT")
        if speaker not in {"counsellor", "client"}:
            continue
        if not isinstance(t1, str) or not isinstance(t2, str):
            continue
        pos = pos_of[int(row["corp_utt_idx"])]
        weight = 1.0
        if weights:
            weight = float(weights.get(speaker, {}).get(t2, 1.0))
        records.append(
            {
                "prompt": build_messages(
                    full_df, pos, cfg.annotator.context_mode, ctx, emit_rationale=True
                ),
                "speaker": speaker,
                "t1_gold": t1,
                "t2_gold": t2,
                "weight": weight,
                "corp_utt_idx": int(row["corp_utt_idx"]),
            }
        )
    return records


def oversample(records: List[Dict], counts: Dict[str, Dict[str, int]]) -> List[Dict]:
    """Repeat rare-gold prompts so an epoch contains enough steps involving them.

    `SU` is 9 of 1040 counsellor rows; without this an epoch contains barely a
    handful of gradient steps that touch the code at all. This compounds with
    the reward weight, which is why `oversample_max_copies` is small.
    """
    out = []
    for rec in records:
        n = counts.get(rec["speaker"], {}).get(rec["t2_gold"], 1)
        out.extend([rec] * max(1, n))
    return out


# -----------------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------------
def _completion_text(completion) -> str:
    """TRL hands back message lists for conversational prompts, strings for
    standard ones. Accept either so the reward does not depend on that."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
        return str(last)
    return ""


class RewardStats:
    """Rolling window of reward components, read by the collapse guard and
    logged so it is visible which part of the reward is actually moving."""

    def __init__(self, window: int = 512):
        self.window = window
        self.rationale_words: List[float] = []
        self.components: Dict[str, List[float]] = {
            k: [] for k in ("reward", "format", "hierarchy", "t1_hit", "t2_hit")
        }

    def push(self, scored: Dict[str, object]) -> None:
        self.rationale_words.append(float(scored["rationale_words"]))
        for key in self.components:
            self.components[key].append(float(scored[key]))
        for series in (self.rationale_words, *self.components.values()):
            if len(series) > self.window:
                del series[: len(series) - self.window]

    def mean_rationale_words(self) -> float:
        return statistics.fmean(self.rationale_words) if self.rationale_words else 0.0

    def summary(self) -> Dict[str, float]:
        out = {
            k: (statistics.fmean(v) if v else 0.0) for k, v in self.components.items()
        }
        out["rationale_words"] = self.mean_rationale_words()
        return out


def make_reward_fn(stats: RewardStats, weighted: bool):
    """Build the TRL reward callable.

    Returns raw reward in [0, 1] scaled by the per-example class weight. The
    scaling survives into the gradient only because `scale_rewards` is off; see
    the module docstring.
    """
    from baseline.single_call import score_completion

    def reward_fn(
        completions,
        speaker,
        t1_gold,
        t2_gold,
        weight,
        log_metric=None,
        **kwargs,
    ):
        rewards = []
        for comp, spk, g1, g2, w in zip(completions, speaker, t1_gold, t2_gold, weight):
            scored = score_completion(
                _completion_text(comp), spk, g1, g2, expect_rationale=True
            )
            stats.push(scored)
            rewards.append(scored["reward"] * (float(w) if weighted else 1.0))
        if log_metric is not None:
            for name, value in stats.summary().items():
                log_metric(f"reward/{name}", value)
        return rewards

    reward_fn.__name__ = "misc_joint_reward"
    return reward_fn


def _check_weighting_is_live(scale_rewards, weighted: bool) -> None:
    """Refuse to run a rare-class-weighted job whose weighting cannot bite.

    A per-prompt constant multiplies every rollout in its group identically, so
    dividing by the within-group std cancels it exactly. The run would look
    correctly configured, cost the same GPU hours, and produce the unweighted
    result under a weighted name -- which would then be compared against the
    genuine unweighted ablation and show no difference.
    """
    if not weighted:
        return
    if scale_rewards not in (False, "none"):
        raise SystemExit(
            f"grpo.scale_rewards={scale_rewards!r} cancels the rare-class weight.\n"
            "Every rollout in a group shares one gold label and therefore one\n"
            "weight, so (w*r - mean(w*r)) / std(w*r) is independent of w.\n"
            "Set grpo.scale_rewards=none, or run with --no-rare-weighting."
        )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------
def _make_callbacks(cfg, stats, val_records, out_dir, tokenizer):
    from transformers import TrainerCallback

    class RationaleCollapseGuard(TrainerCallback):
        """Stop the run if the policy trades its rationale away.

        Format is worth 0.1 and the labels 0.8, so abandoning the rationale is
        a cheap trade the optimiser can find. If it does, the run is no longer
        measuring reasoning and every further GPU hour is wasted.
        """

        def __init__(self, floor: float, check_every: int = 25):
            self.floor = float(floor)
            self.check_every = int(check_every)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step < self.check_every:
                return control
            if state.global_step % self.check_every:
                return control
            mean_words = stats.mean_rationale_words()
            if mean_words < self.floor:
                print(
                    f"\nABORT at step {state.global_step}: mean rationale is "
                    f"{mean_words:.1f} words, below the floor of {self.floor}. "
                    "The policy is farming label reward without reasoning."
                )
                control.should_training_stop = True
            return control

    class ValidationSelector(TrainerCallback):
        """Select the checkpoint on held-out HLQC, never on MIV6.3A.

        Without this, the natural thing to do is take the last step, or worse
        pick the step that scores best on the evaluation corpus, which would
        make every reported MIV6.3A number a training metric.
        """

        def __init__(self, every: int, max_examples: int):
            self.every = int(every)
            self.max_examples = int(max_examples)
            self.best = -1.0
            self.history: List[Dict[str, float]] = []

        def _evaluate(self, model, step: int) -> Dict[str, float]:
            return evaluate_records(
                model,
                tokenizer,
                val_records[: self.max_examples],
                max_new_tokens=int(cfg.grpo.max_completion_length),
                desc=f"val@{step}",
            )

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not state.global_step or state.global_step % self.every:
                return control
            was_training = model.training
            model.eval()
            try:
                metrics = self._evaluate(model, state.global_step)
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
            print(
                f"\n[val step {state.global_step}] "
                f"t1_acc={metrics['t1_acc']:.3f} t2_acc={metrics['t2_acc']:.3f} "
                f"macroF1(t1)={metrics['t1_macro_f1']:.3f} "
                f"macroF1(t2)={metrics['t2_macro_f1']:.3f} "
                f"mean={score:.3f}{marker}"
            )
            return control

    guard = RationaleCollapseGuard(cfg.grpo.min_mean_rationale_words)
    selector = ValidationSelector(
        int(cfg.grpo.val_every_steps), int(cfg.grpo.get("val_max_examples", 200))
    )
    return guard, selector


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def evaluate_records(
    model,
    tokenizer,
    records: Sequence[Dict],
    max_new_tokens: int = 256,
    batch_size: int = 8,
    desc: str = "eval",
) -> Dict[str, float]:
    """Greedy-decode the records and score both tiers.

    Reported per tier: accuracy, and macro-F1 over the codes present in the
    gold labels. Selection uses the mean of the two macro-F1s rather than
    accuracy, because accuracy is what the ladder has already saturated -- the
    open gap against gpt-4o is entirely in the long tail.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm

    from baseline.single_call import parse_completion

    preds = {"t1": [], "t2": []}
    golds = {"t1": [], "t2": []}
    rationale_words: List[int] = []

    model_device = next(model.parameters()).device
    for start in tqdm(range(0, len(records), batch_size), desc=desc, leave=False):
        batch = records[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                r["prompt"], tokenize=False, add_generation_prompt=True
            )
            for r in batch
        ]
        # Left padding so every sequence's generation starts at the same index.
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model_device)
        tokenizer.padding_side = prev_side
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        for rec, ids in zip(batch, gen):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            parsed = parse_completion(text, rec["speaker"])
            preds["t1"].append(parsed["t1"])
            preds["t2"].append(parsed["t2"])
            golds["t1"].append(rec["t1_gold"])
            golds["t2"].append(rec["t2_gold"])
            rationale_words.append(len(parsed["rationale"].split()))

    metrics: Dict[str, float] = {}
    for tier in ("t1", "t2"):
        metrics[f"{tier}_acc"] = float(accuracy_score(golds[tier], preds[tier]))
        gold_classes = sorted(set(golds[tier]))
        metrics[f"{tier}_macro_f1"] = float(
            f1_score(
                golds[tier], preds[tier], labels=gold_classes, average="macro",
                zero_division=0,
            )
        )
    metrics["macro_f1_mean"] = (metrics["t1_macro_f1"] + metrics["t2_macro_f1"]) / 2
    metrics["rationale_words"] = (
        statistics.fmean(rationale_words) if rationale_words else 0.0
    )
    return metrics


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def load_policy(cfg: DictConfig, ctx: int, cold_start: bool):
    """Load the policy, either continuing the sc_ft_bare adapter or fresh LoRA.

    Returns `(model, tokenizer, peft_config)`. When continuing an adapter the
    model is already a PeftModel and `peft_config` is None, so TRL leaves it
    alone and uses the adapter-disabled base as the KL reference -- which costs
    no extra memory, unlike a separately loaded reference model.
    """
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from baseline.sc_arm import sft_adapter_dir
    from components.hf_load import resolve_hf_token

    token = resolve_hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.base_model, token=token, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        token=token,
        dtype=torch.bfloat16 if cfg.model.bf16 else torch.float32,
        attn_implementation=cfg.model.get("attn_implementation", "sdpa"),
    )

    if cold_start:
        print("Cold start: fresh LoRA on the base model")
        peft_config = LoraConfig(
            r=int(cfg.model.peft_r),
            lora_alpha=int(cfg.model.peft_alpha),
            lora_dropout=float(cfg.model.peft_dropout),
            target_modules=list(cfg.model.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        return model, tokenizer, peft_config

    adapter = sft_adapter_dir(cfg, "bare", ctx)
    if not adapter.exists():
        raise SystemExit(
            f"Missing sc_ft_bare adapter at {adapter}. Train it first:\n"
            f"  PYTHONPATH=src python -m baseline.sc_arm train --target bare --ctx {ctx}"
        )
    print(f"Initialising policy from {adapter}")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
    return model, tokenizer, None


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
def cmd_train(args) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from automisc_ft.data import load_manual
    from baseline.sc_arm import grpo_adapter_dir
    from baseline.single_call import oversample_counts, speaker_class_weights

    cfg = load_config(args.overrides)
    ctx, seed = args.ctx, args.seed
    weighted = bool(cfg.grpo.rare_class_weighting) and not args.no_rare_weighting
    _check_weighting_is_live(cfg.grpo.scale_rewards, weighted)

    # Cold start is reported as its own arm even though it keeps the weighting,
    # since it answers a different question (does GRPO need a competent
    # initialisation) than the weighting ablation does.
    if args.cold_start:
        variant = "coldstart"
    elif weighted:
        variant = "weighted"
    else:
        variant = "unweighted"
    out_dir = grpo_adapter_dir(cfg, ctx, seed, variant)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    train_df, val_df = split_by_conversation(
        full_df, int(cfg.grpo.val_folds), int(cfg.grpo.val_fold), seed=42
    )
    print(
        f"HLQC split by conversation: train={len(train_df)} rows "
        f"({train_df['conv_id'].nunique()} convs), "
        f"val={len(val_df)} rows ({val_df['conv_id'].nunique()} convs)"
    )

    # Weights come from the TRAINING slice only. Deriving them from the full
    # corpus would leak the validation label distribution into the objective.
    weights = None
    counts = {}
    if weighted:
        weights = speaker_class_weights(
            train_df,
            power=float(cfg.grpo.weight_power),
            floor=float(cfg.grpo.weight_floor),
            ceiling=float(cfg.grpo.weight_ceiling),
        )
        counts = {
            spk: oversample_counts(
                [], w, max_copies=int(cfg.grpo.oversample_max_copies)
            )
            for spk, w in weights.items()
        }
        for spk, w in weights.items():
            top = sorted(w.items(), key=lambda kv: -kv[1])[:4]
            print(f"  {spk} top weights: " + ", ".join(f"{c}={v:.2f}" for c, v in top))

    train_records = build_records(train_df, full_df, cfg, ctx, weights)
    val_records = build_records(val_df, full_df, cfg, ctx, None)
    if args.limit:
        train_records = train_records[: int(args.limit)]
        val_records = val_records[: max(8, int(args.limit) // 4)]
    if weighted:
        before = len(train_records)
        train_records = oversample(train_records, counts)
        print(f"Oversampled rare codes: {before} -> {len(train_records)} prompts")

    model, tokenizer, peft_config = load_policy(cfg, ctx, args.cold_start)
    stats = RewardStats()
    reward_fn = make_reward_fn(stats, weighted)
    guard, selector = _make_callbacks(cfg, stats, val_records, out_dir, tokenizer)

    grpo_args = GRPOConfig(
        output_dir=str(out_dir / "trainer"),
        seed=seed,
        num_generations=int(cfg.grpo.num_generations),
        num_train_epochs=float(cfg.grpo.num_train_epochs),
        learning_rate=float(cfg.grpo.learning_rate),
        temperature=float(cfg.grpo.temperature),
        top_p=float(cfg.grpo.top_p),
        max_completion_length=int(cfg.grpo.max_completion_length),
        beta=float(cfg.grpo.beta),
        epsilon=float(cfg.grpo.epsilon),
        num_iterations=int(cfg.grpo.num_iterations),
        scale_rewards=cfg.grpo.scale_rewards,
        loss_type=str(cfg.grpo.loss_type),
        per_device_train_batch_size=int(cfg.grpo.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.grpo.gradient_accumulation_steps),
        gradient_checkpointing=bool(cfg.grpo.gradient_checkpointing),
        max_grad_norm=float(cfg.grpo.max_grad_norm),
        warmup_ratio=float(cfg.grpo.warmup_ratio),
        lr_scheduler_type=str(cfg.grpo.lr_scheduler_type),
        bf16=bool(cfg.model.bf16),
        use_vllm=bool(cfg.vllm.enabled),
        vllm_mode=str(cfg.vllm.mode),
        vllm_gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        vllm_max_model_length=int(cfg.vllm.max_model_length),
        vllm_enable_sleep_mode=bool(cfg.vllm.get("enable_sleep_mode", False)),
        log_completions=bool(cfg.grpo.log_completions),
        num_completions_to_print=int(cfg.grpo.num_completions_to_print),
        logging_steps=int(cfg.grpo.logging_steps),
        save_steps=int(cfg.grpo.save_steps),
        save_strategy="no",  # the validation selector owns checkpointing
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=reward_fn,
        train_dataset=Dataset.from_list(train_records),
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[guard, selector],
    )

    print(
        f"GRPO ctx={ctx} seed={seed} weighted={weighted} cold_start={args.cold_start}\n"
        f"  G={cfg.grpo.num_generations} epochs={cfg.grpo.num_train_epochs} "
        f"lr={cfg.grpo.learning_rate} beta={cfg.grpo.beta} "
        f"scale_rewards={cfg.grpo.scale_rewards} loss={cfg.grpo.loss_type}\n"
        f"  prompts={len(train_records)} val={len(val_records)} -> {out_dir}"
    )
    started = time.time()
    trainer.train()
    elapsed = time.time() - started

    # If validation never improved on the initial -1.0 the selector saved
    # nothing, so fall back to the final policy rather than leaving no adapter.
    if not (out_dir / "adapter_config.json").exists():
        print("No validation improvement recorded; saving the final policy.")
        trainer.model.save_pretrained(str(out_dir))

    from baseline.sc_arm import GRPO_VARIANTS

    meta = {
        "format": "single_call",
        "arm": GRPO_VARIANTS[variant],
        "variant": variant,
        "seed": seed,
        "num_context_turns": ctx,
        "rare_class_weighting": weighted,
        "cold_start": bool(args.cold_start),
        "base_model": cfg.model.base_model,
        "init_from": "base" if args.cold_start else str(
            (REPO_ROOT / cfg.paths.sft_adapter_dir / f"ctx{ctx}" / "bare")
        ),
        "grpo": OmegaConf.to_container(cfg.grpo, resolve=True),
        "n_train_prompts": len(train_records),
        "n_val_prompts": len(val_records),
        "train_convs": int(train_df["conv_id"].nunique()),
        "val_convs": int(val_df["conv_id"].nunique()),
        "class_weights": weights,
        "wall_clock_hours": round(elapsed / 3600, 3),
        "best_val_macro_f1_mean": selector.best,
        "val_history": selector.history,
        "final_reward_components": stats.summary(),
    }
    (out_dir / "train_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    print(
        f"\nDone in {elapsed / 3600:.2f}h. Best val macro-F1 (mean) "
        f"{selector.best:.3f}. Adapter -> {out_dir}"
    )


# -----------------------------------------------------------------------------
# calibrate
# -----------------------------------------------------------------------------
def _prompt_tokens(tokenizer, messages) -> int:
    """Token count of a rendered chat prompt.

    Not `apply_chat_template(tokenize=True)`: transformers 5 returns a
    BatchEncoding there where 4.x returned a list of ids, so `len()` silently
    reports the number of dict keys. That reads as a plausible "2" rather than
    as an error, and the whole point of calibration is to size the run from
    real prompt lengths.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def cmd_calibrate(args) -> None:
    """Measure throughput on a few prompts, then extrapolate the phase budget.

    The GPU-hour estimates in docs/GRPO.md carry roughly a factor-of-two
    uncertainty because they assume a vLLM prefix-cache hit rate nobody has
    measured on this prompt shape. Running this first turns the budget into
    something derived from this cluster rather than from an assumption.
    """
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from automisc_ft.data import load_manual

    cfg = load_config(args.overrides)
    ctx = args.ctx
    n_prompts = int(args.limit)

    full_df = load_manual(REPO_ROOT / cfg.dataset.train_csv)
    train_df, _ = split_by_conversation(
        full_df, int(cfg.grpo.val_folds), int(cfg.grpo.val_fold), seed=42
    )
    records = build_records(train_df, full_df, cfg, ctx, None)[:n_prompts]

    model, tokenizer, peft_config = load_policy(cfg, ctx, args.cold_start)
    lengths = [_prompt_tokens(tokenizer, r["prompt"]) for r in records]
    by_speaker: Dict[str, List[int]] = {}
    for rec, n in zip(records, lengths):
        by_speaker.setdefault(rec["speaker"], []).append(n)
    print("\n=== prompt token lengths (ctx=%d) ===" % ctx)
    for spk, ns in by_speaker.items():
        print(
            f"  {spk:11s} n={len(ns):4d}  mean={statistics.fmean(ns):7.0f}  "
            f"max={max(ns):7d}"
        )

    stats = RewardStats()
    grpo_args = GRPOConfig(
        output_dir=str(REPO_ROOT / "outputs" / "grpo" / "calibrate"),
        seed=0,
        num_generations=int(cfg.grpo.num_generations),
        max_steps=int(args.steps),
        learning_rate=float(cfg.grpo.learning_rate),
        temperature=float(cfg.grpo.temperature),
        max_completion_length=int(cfg.grpo.max_completion_length),
        beta=float(cfg.grpo.beta),
        scale_rewards=cfg.grpo.scale_rewards,
        loss_type=str(cfg.grpo.loss_type),
        per_device_train_batch_size=int(cfg.grpo.per_device_train_batch_size),
        gradient_accumulation_steps=int(cfg.grpo.gradient_accumulation_steps),
        gradient_checkpointing=bool(cfg.grpo.gradient_checkpointing),
        bf16=bool(cfg.model.bf16),
        use_vllm=bool(cfg.vllm.enabled),
        vllm_mode=str(cfg.vllm.mode),
        vllm_gpu_memory_utilization=float(cfg.vllm.gpu_memory_utilization),
        vllm_max_model_length=int(cfg.vllm.max_model_length),
        vllm_enable_sleep_mode=bool(cfg.vllm.get("enable_sleep_mode", False)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        reward_funcs=make_reward_fn(stats, False),
        train_dataset=Dataset.from_list(records),
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print(f"\n=== timing {args.steps} GRPO steps ===")
    import torch

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    elapsed = time.time() - started
    # Colocate lives or dies on this number: two 15.2GB weight copies plus KV
    # plus activations have to fit one 40GB card, and the margin decides whether
    # vllm.gpu_memory_utilization can stay where it is.
    peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"peak reserved {peak_gb:.1f} GiB of {total_gb:.1f} GiB")

    per_step = elapsed / max(1, int(args.steps))
    prompts_per_step = (
        int(cfg.grpo.per_device_train_batch_size)
        * int(cfg.grpo.gradient_accumulation_steps)
        // int(cfg.grpo.num_generations)
    )
    per_prompt = per_step / max(1, prompts_per_step)
    n_full = 1925 * (1 - 1 / int(cfg.grpo.val_folds))
    epoch_hours = n_full * per_prompt / 3600
    run_hours = epoch_hours * float(cfg.grpo.num_train_epochs)

    report = {
        "steps_timed": int(args.steps),
        "wall_clock_s": round(elapsed, 1),
        "s_per_step": round(per_step, 2),
        "prompts_per_step": prompts_per_step,
        "s_per_prompt": round(per_prompt, 3),
        "projected_epoch_hours": round(epoch_hours, 2),
        "projected_run_hours": round(run_hours, 2),
        "projected_phase1_hours": round(run_hours * 3 + 4 + 1.5, 1),
        "projected_phase2_hours": round(run_hours * 4 + 1, 1),
        "projected_phase3_hours": round(run_hours + 3, 1),
        "reward_components": stats.summary(),
        "peak_reserved_gib": round(peak_gb, 1),
        "gpu_total_gib": round(total_gb, 1),
        "vllm_gpu_memory_utilization": float(cfg.vllm.gpu_memory_utilization),
        "vllm_sleep_mode": bool(cfg.vllm.get("enable_sleep_mode", False)),
        "prompt_tokens": {
            spk: {"mean": round(statistics.fmean(ns)), "max": max(ns)}
            for spk, ns in by_speaker.items()
        },
    }
    out = REPO_ROOT / "outputs" / "grpo" / f"calibration_ctx{ctx}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n=== calibration ===")
    for k, v in report.items():
        if not isinstance(v, dict):
            print(f"  {k:26s} {v}")
    print(f"\nWritten to {out}")
    print(
        "\nRe-cost the phases in docs/GRPO.md against projected_run_hours before "
        "submitting the 3-seed ladder."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train one GRPO policy")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument(
        "--no-rare-weighting",
        action="store_true",
        help="Phase 2 ablation: uniform class weights and no oversampling",
    )
    # `train` uses the whole training split unless capped for a smoke test.
    p_train.add_argument("--limit", type=int, default=None)

    p_cal = sub.add_parser("calibrate", help="measure throughput and re-cost the phases")
    p_cal.add_argument("--steps", type=int, default=5)
    p_cal.add_argument("--limit", type=int, default=50)

    for p in (p_train, p_cal):
        p.add_argument("--ctx", type=int, default=None)
        p.add_argument(
            "--cold-start",
            action="store_true",
            help="Phase 2 ablation: start from the base model, not sc_ft_bare",
        )
        p.add_argument("overrides", nargs="*")

    args = parser.parse_args()
    if args.ctx is None:
        args.ctx = int(load_config(args.overrides).annotator.num_context_turns)
    else:
        args.overrides = list(args.overrides) + [f"annotator.num_context_turns={args.ctx}"]

    {"train": cmd_train, "calibrate": cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
