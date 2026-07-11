"""Azure OpenAI fine-tuning for the AutoMISC baseline study.

Training data comes from the held-out HLQC_balanced_manual.csv: for every
utterance we emit one T1 example and one T2 example that mirror the tiered
bare (label-only) prompts used at inference time, so the fine-tuned model can
be dropped into the same annotation loop.

Subcommands:
    build    write train/valid JSONL and print token/cost estimates
    submit   upload files and create the fine-tune job
    status   poll the job (pass --job <id>)
    deploy   attempt data-plane deployment of the finished model (pass --job <id>)

Usage:
    PYTHONPATH=src .venv/bin/python -m baseline.finetune build
    PYTHONPATH=src .venv/bin/python -m baseline.finetune submit
    PYTHONPATH=src .venv/bin/python -m baseline.finetune status --job ftjob-...
    PYTHONPATH=src .venv/bin/python -m baseline.finetune deploy --job ftjob-...
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
TRAIN_PATH = FT_DIR / "hlqc_train.jsonl"
VALID_PATH = FT_DIR / "hlqc_valid.jsonl"
META_PATH = FT_DIR / "job_metadata.json"

HLQC_PATH = REPO_ROOT / "data" / "manual" / "HLQC_balanced_manual.csv"
BASE_MODEL = "gpt-4o-2024-08-06"
N_EPOCHS = 1  # keep training cost bounded; raise if underfit
SEED = 42
CONTEXT_MODE = "interval"
NUM_CONTEXT_TURNS = 5
VALID_FRAC = 0.1


def build_examples() -> list[dict]:
    df = pd.read_csv(HLQC_PATH).reset_index(drop=True)
    examples = []
    for i, row in df.iterrows():
        speaker = row["speaker"]
        context = build_context_excerpt(df, i, CONTEXT_MODE, NUM_CONTEXT_TURNS)
        user_prompt = render_user_prompt(
            transcript=context, speaker=speaker, utterance=row["utt_text"]
        )
        t1_label = row["t1_label_GT"]
        t2_label = row["t2_label_GT"]

        t1_system = render_prompt(speaker=speaker, structure="t1_bare")
        examples.append({
            "messages": [
                {"role": "system", "content": t1_system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"label": t1_label})},
            ]
        })
        t2_system = render_prompt(speaker=speaker, structure="t2_bare", label=t1_label)
        examples.append({
            "messages": [
                {"role": "system", "content": t2_system},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"label": t2_label})},
            ]
        })
    return examples


def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def cmd_build(_args) -> None:
    import random

    examples = build_examples()
    rng = random.Random(SEED)
    rng.shuffle(examples)
    n_valid = int(len(examples) * VALID_FRAC)
    valid, train = examples[:n_valid], examples[n_valid:]
    write_jsonl(train, TRAIN_PATH)
    write_jsonl(valid, VALID_PATH)

    n_chars = sum(len(m["content"]) for ex in examples for m in ex["messages"])
    est_tokens = n_chars / 4
    print(f"train: {len(train)} examples -> {TRAIN_PATH}")
    print(f"valid: {len(valid)} examples -> {VALID_PATH}")
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
    client = get_azure_client()
    if args.train_file and args.valid_file:
        train_id, valid_id = args.train_file, args.valid_file
        print(f"reusing uploaded files train={train_id} valid={valid_id}")
    else:
        with open(TRAIN_PATH, "rb") as f:
            train_id = client.files.create(file=f, purpose="fine-tune").id
        with open(VALID_PATH, "rb") as f:
            valid_id = client.files.create(file=f, purpose="fine-tune").id
        print(f"uploaded train={train_id} valid={valid_id}")

    _wait_for_import(client, train_id)
    _wait_for_import(client, valid_id)
    print("file imports complete")

    job = client.fine_tuning.jobs.create(
        model=BASE_MODEL,
        training_file=train_id,
        validation_file=valid_id,
        seed=SEED,
        hyperparameters={"n_epochs": N_EPOCHS},
        suffix="automisc-baseline",
    )
    meta = {"job_id": job.id, "status": job.status,
            "train_file": train_id, "valid_file": valid_id,
            "base_model": BASE_MODEL, "n_epochs": N_EPOCHS}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"fine-tune job created: {job.id} (status: {job.status})")
    print(f"metadata saved to {META_PATH}")


def _job_id(args) -> str:
    if args.job:
        return args.job
    with open(META_PATH) as f:
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

    client = get_azure_client()
    job = client.fine_tuning.jobs.retrieve(_job_id(args))
    if job.status != "succeeded" or not job.fine_tuned_model:
        print(f"job not ready: status={job.status}")
        return
    model_name = job.fine_tuned_model
    deployment_name = args.deployment or "automisc-baseline-ft"
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

    url = f"{endpoint}/openai/deployments/{deployment_name}?api-version=2023-05-15"
    body = {"model": model_name, "scale_settings": {"scale_type": "standard"}}
    resp = httpx.put(url, json=body, headers={"api-key": api_key}, timeout=60)
    print(f"PUT {url} -> {resp.status_code}")
    print(resp.text)
    if resp.status_code >= 400:
        print("\nData-plane deployment failed. Deploy the model "
              f"'{model_name}' manually in the Azure portal (Deployments -> "
              f"Create -> select the fine-tuned model), then run:\n"
              f"  PYTHONPATH=src .venv/bin/python -m baseline.main "
              f"condition=finetuned model=<deployment-name>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--train-file", default=None, help="reuse an already-uploaded file id")
    p_submit.add_argument("--valid-file", default=None, help="reuse an already-uploaded file id")
    p_status = sub.add_parser("status")
    p_status.add_argument("--job", default=None)
    p_deploy = sub.add_parser("deploy")
    p_deploy.add_argument("--job", default=None)
    p_deploy.add_argument("--deployment", default=None)
    args = parser.parse_args()

    # load .env for all subcommands
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")

    {"build": cmd_build, "submit": cmd_submit,
     "status": cmd_status, "deploy": cmd_deploy}[args.cmd](args)


if __name__ == "__main__":
    main()
