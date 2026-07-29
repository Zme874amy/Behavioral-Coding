"""Azure OpenAI fine-tuning for the gpt-4o tier of the experiment grid.

Training data comes from the held-out HLQC_balanced_manual.csv: for every
utterance we emit one T1 example and one T2 example mirroring the tiered prompts
used at inference time, so the fine-tuned model drops into the same annotation
loop.

Two training targets, matching the two fine-tuning arms of the grid:

    bare  assistant replies {"label": ...}, on the label-only prompts
    rat   assistant replies {"explanation": ..., "label": ...}, on the
          rationale-first prompts, using the frozen gpt-4o rationales from
          `baseline.rationalize`

The context length (prior volleys per example) is a first-class parameter: one
fine-tuned model per (target, context) pair, trained and evaluated with matching
prompts. Artifacts live in data/fine_tuning/baseline/. The `bare` target keeps
the original un-suffixed filenames so existing jobs stay addressable.

Subcommands:
    build    write train/valid JSONL and print token/cost estimates
    submit   upload files and create the fine-tune job
    status   poll the job
    deploy   attempt data-plane deployment of the finished model

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.finetune build --ctx 3
    PYTHONPATH=src .venv/bin/python -m baseline.finetune build --target rat --ctx 5
    PYTHONPATH=src .venv/bin/python -m baseline.finetune submit --target rat --ctx 5
    PYTHONPATH=src .venv/bin/python -m baseline.finetune status --target rat --ctx 5
    PYTHONPATH=src .venv/bin/python -m baseline.finetune deploy --target rat --ctx 5

Note: creating a deployment is a control-plane (ARM) operation. An API key is a
data-plane credential, so `deploy` will fail with 404 unless the caller holds
Azure AD credentials with a role like Cognitive Services OpenAI Contributor; in
that case deploy from Azure AI Foundry instead.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from components.context import build_context_excerpt
from components.prompts.loader import render_prompt, render_user_prompt
from components.utils import get_azure_client

REPO_ROOT = Path(__file__).resolve().parents[2]
FT_DIR = REPO_ROOT / "data" / "fine_tuning" / "baseline"

HLQC_PATH = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"
BASE_MODEL = "gpt-4o-2024-08-06"
N_EPOCHS = 1  # keep training cost bounded; raise if underfit
SEED = 42
CONTEXT_MODE = "interval"
DEFAULT_CONTEXT_TURNS = 5
VALID_FRAC = 0.1


TARGETS = ("bare", "rat")


def _tag(target: str) -> str:
    """Filename infix. `bare` stays empty so pre-existing artifacts still resolve."""
    return "" if target == "bare" else f"_{target}"


def train_path(ctx: int, target: str = "bare") -> Path:
    return FT_DIR / f"hlqc_train{_tag(target)}_ctx{ctx}.jsonl"


def valid_path(ctx: int, target: str = "bare") -> Path:
    return FT_DIR / f"hlqc_valid{_tag(target)}_ctx{ctx}.jsonl"


def meta_path(ctx: int, target: str = "bare") -> Path:
    """Job metadata file; the pre-ctx-suffix job_metadata.json was the ctx5 job."""
    suffixed = FT_DIR / f"job_metadata{_tag(target)}_ctx{ctx}.json"
    if suffixed.exists():
        return suffixed
    legacy = FT_DIR / "job_metadata.json"
    if ctx == 5 and target == "bare" and legacy.exists():
        return legacy
    return suffixed


def deployment_name(ctx: int, target: str = "bare") -> str:
    return f"automisc-ft{_tag(target).replace('_', '-')}-ctx{ctx}"


def build_examples(ctx: int, target: str = "bare") -> list[dict]:
    """Build the chat-format fine-tuning examples for one target and context.

    `bare` trains label-only replies on the label-only prompts; `rat` trains
    explanation + label replies on the rationale-first prompts, so in each case
    the training target matches the prompt the model is trained under.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")

    rationales = {}
    if target == "rat":
        from baseline.rationalize import load_rationales, rationales_path

        rationales = load_rationales(ctx)
        if not rationales:
            raise SystemExit(
                f"No frozen rationales at {rationales_path(ctx)}. Generate them:\n"
                f"  PYTHONPATH=src python -m baseline.rationalize --ctx {ctx}"
            )

    suffix = "_bare" if target == "bare" else ""
    df = pd.read_csv(HLQC_PATH).reset_index(drop=True)
    examples = []
    n_missing = 0
    for i, row in df.iterrows():
        speaker = row["speaker"]
        context = build_context_excerpt(df, i, CONTEXT_MODE, ctx)
        user_prompt = render_user_prompt(
            transcript=context, speaker=speaker, utterance=row["utt_text"]
        )
        t1_label = row["t1_label_GT"]
        t2_label = row["t2_label_GT"]

        entry = rationales.get(str(row["corp_utt_idx"])) if target == "rat" else None
        if target == "rat" and not (
            entry and entry.get("t1_explanation") and entry.get("t2_explanation")
        ):
            n_missing += 1
            continue

        def reply(label: str, tier: str) -> str:
            if target == "bare":
                return json.dumps({"label": label})
            return json.dumps(
                {"explanation": entry[f"{tier}_explanation"], "label": label}
            )

        t1_system = render_prompt(speaker=speaker, structure=f"t1{suffix}")
        examples.append({
            "messages": [
                {"role": "system", "content": t1_system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": reply(t1_label, "t1")},
            ]
        })
        t2_system = render_prompt(
            speaker=speaker, structure=f"t2{suffix}", label=t1_label
        )
        examples.append({
            "messages": [
                {"role": "system", "content": t2_system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": reply(t2_label, "t2")},
            ]
        })
    if n_missing:
        print(
            f"WARNING: skipped {n_missing} of {len(df)} utterances with no frozen "
            f"rationale for ctx={ctx}. Run baseline.rationalize to completion for "
            "a full training set."
        )
    return examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def cmd_build(args) -> None:
    import random

    ctx, target = args.ctx, args.target
    examples = build_examples(ctx, target)
    rng = random.Random(SEED)
    rng.shuffle(examples)
    n_valid = int(len(examples) * VALID_FRAC)
    valid, train = examples[:n_valid], examples[n_valid:]
    write_jsonl(train, train_path(ctx, target))
    write_jsonl(valid, valid_path(ctx, target))

    n_chars = sum(len(m["content"]) for ex in examples for m in ex["messages"])
    est_tokens = n_chars / 4
    print(f"target: {target}  ctx: {ctx}")
    print(f"train: {len(train)} examples -> {train_path(ctx, target)}")
    print(f"valid: {len(valid)} examples -> {valid_path(ctx, target)}")
    print(f"estimated training tokens/epoch: ~{est_tokens/1e6:.1f}M "
          f"(~${est_tokens/1e6*25:.0f} per epoch at $25/M for {BASE_MODEL})")


def _wait_for_import(client, file_id: str, timeout_s: int = 600) -> None:
    """Azure imports uploaded files asynchronously; block until processed."""
    import time

    start = time.time()
    while time.time() - start < timeout_s:
        f = client.files.retrieve(file_id)
        if f.status == "processed":
            return
        if f.status in ("error", "deleted"):
            raise RuntimeError(f"file {file_id} import failed: status={f.status}")
        time.sleep(5)
    raise TimeoutError(f"file {file_id} not processed after {timeout_s}s")


def cmd_submit(args) -> None:
    ctx, target = args.ctx, args.target
    client = get_azure_client()
    if args.train_file and args.valid_file:
        train_id, valid_id = args.train_file, args.valid_file
        print(f"reusing uploaded files train={train_id} valid={valid_id}")
    else:
        with open(train_path(ctx, target), "rb") as f:
            train_id = client.files.create(file=f, purpose="fine-tune").id
        with open(valid_path(ctx, target), "rb") as f:
            valid_id = client.files.create(file=f, purpose="fine-tune").id
        print(f"uploaded train={train_id} valid={valid_id}")

    _wait_for_import(client, train_id)
    _wait_for_import(client, valid_id)
    print("file imports complete")

    # Azure caps the suffix length, so keep it short but target-distinguishing.
    suffix = f"automisc{_tag(target).replace('_', '-')}-ctx{ctx}"
    job = client.fine_tuning.jobs.create(
        model=BASE_MODEL,
        training_file=train_id,
        validation_file=valid_id,
        seed=SEED,
        hyperparameters={"n_epochs": N_EPOCHS},
        suffix=suffix,
    )
    meta = {"job_id": job.id, "status": job.status,
            "train_file": train_id, "valid_file": valid_id,
            "base_model": BASE_MODEL, "n_epochs": N_EPOCHS,
            "num_context_turns": ctx, "target": target}
    out_meta = FT_DIR / f"job_metadata{_tag(target)}_ctx{ctx}.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"fine-tune job created: {job.id} (status: {job.status})")
    print(f"metadata saved to {out_meta}")


def _job_id(args) -> str:
    if args.job:
        return args.job
    path = meta_path(args.ctx, args.target)
    if not path.exists():
        raise SystemExit(
            f"No job metadata at {path}. Submit the job first:\n"
            f"  PYTHONPATH=src python -m baseline.finetune submit "
            f"--target {args.target} --ctx {args.ctx}"
        )
    with open(path) as f:
        return json.load(f)["job_id"]


def cmd_status(args) -> None:
    client = get_azure_client()
    job = client.fine_tuning.jobs.retrieve(_job_id(args))
    print(f"status: {job.status}")
    print(f"fine_tuned_model: {job.fine_tuned_model}")
    if job.error:
        print(f"error: {job.error}")
    events = client.fine_tuning.jobs.list_events(job.id, limit=10)
    for ev in events.data:
        print(f"  [{ev.created_at}] {ev.message}")


def cmd_deploy(args) -> None:
    """Try the data-plane deployments API. If Azure rejects it, the fine-tuned
    model must be deployed from Azure AI Foundry / portal instead."""
    import httpx

    ctx, target = args.ctx, args.target
    client = get_azure_client()
    job = client.fine_tuning.jobs.retrieve(_job_id(args))
    if job.status != "succeeded" or not job.fine_tuned_model:
        print(f"job not ready: status={job.status}")
        return
    model_name = job.fine_tuned_model
    dep_name = args.deployment or deployment_name(ctx, target)
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    url = f"{endpoint}/openai/deployments/{dep_name}?api-version=2023-05-15"
    body = {"model": model_name, "scale_settings": {"scale_type": "standard"}}
    resp = httpx.put(url, json=body, headers={"api-key": api_key}, timeout=60)
    print(f"PUT {url} -> {resp.status_code}")
    print(resp.text)
    # Once deployed, each model is evaluated under BOTH inference styles; that
    # cross is what tests whether the training target overrides the prompt.
    arm = "ft_bare" if target == "bare" else "ft_rat"
    evals = "\n".join(
        f"  PYTHONPATH=src python -m baseline.main "
        f"condition=gpt4o_{arm}_inf_{style} model={dep_name} num_context_turns={ctx}"
        for style in ("bare", "cot")
    )
    if resp.status_code < 400:
        print(f"\nDeployed. Evaluate both cells with:\n{evals}")
    else:
        print(
            f"\nData-plane deployment failed ({resp.status_code}). Creating a "
            "deployment is a control-plane (ARM) operation and an API key is a "
            "data-plane credential, so this cannot succeed with the key alone.\n"
            f"Deploy '{model_name}' from Azure AI Foundry / the portal "
            f"(Deployments -> Create -> pick the fine-tuned model, name it "
            f"'{dep_name}'), or have someone with the Cognitive Services OpenAI "
            "Contributor role run it. Then:\n" + evals
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("build", "submit", "status", "deploy"):
        p = sub.add_parser(name)
        p.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT_TURNS,
                       help="number of prior context volleys (one FT model per setting)")
        p.add_argument("--target", choices=TARGETS, default="bare",
                       help="bare = label-only targets; rat = rationale + label targets")
        if name == "submit":
            p.add_argument("--train-file", default=None, help="reuse an already-uploaded file id")
            p.add_argument("--valid-file", default=None, help="reuse an already-uploaded file id")
        if name in ("status", "deploy"):
            p.add_argument("--job", default=None)
        if name == "deploy":
            p.add_argument("--deployment", default=None)
    args = parser.parse_args()

    # load .env for all subcommands
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    {"build": cmd_build, "submit": cmd_submit,
     "status": cmd_status, "deploy": cmd_deploy}[args.cmd](args)


if __name__ == "__main__":
    main()
