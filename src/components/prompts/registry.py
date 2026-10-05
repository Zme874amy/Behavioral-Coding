"""Prompt registry: every prompt the project uses, with version, scheme, setting and sha256.

    PYTHONPATH=src python -m components.prompts.registry --write   # (re)generate REGISTRY.yaml
    PYTHONPATH=src python -m components.prompts.registry --check   # fail if any file drifted

A frozen entry must never change: --check fails if its hash differs from the
recorded one. Active entries may be edited, but only together with a --write in
the same commit, so the registry diff shows the change. Python-embedded prompts
are hashed at file level (any edit to that module changes the hash).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent                     # src/components/prompts
SRC = ROOT.parents[1]                            # src/
OUT = ROOT / "REGISTRY.yaml"

MISC_TEMPLATES = {  # name -> (setting, notes)
    "t1": ("two-call T1, CoT (rationale + label)", ""),
    "t2": ("two-call T2, CoT; spec of the predicted T1 group injected", ""),
    "t1_bare": ("two-call T1, label only (zero-shot / few-shot / SFT bare)", ""),
    "t2_bare": ("two-call T2, label only; spec injected", ""),
    "t12": ("single call, both tiers (single-call SFT / GRPO)", ""),
    "flat": ("flat T2 without the T1 step (legacy rich zero-shot)", "not used by the re-run"),
}
KNOWN_ISSUES_V1 = ("client t2.j2 and flat.j2 say \"counsellor's final utterance\"; spec YAML and counsellor "
                   "flat.j2 use ADWP/CON/DIR/RCWP for ADW/CO/DI/RCW; typos (Guild, expect IMC and IMC, "
                   "'about will happen', 'is a question is')")
CONVENTIONS = ("Deliberate project conventions kept in v2 (not in the MISC 2.5 manual): the CQ/OQ grammatical "
               "yes/no test with 'if ambiguous, CQ'; AutoMISC's T1 groups put GI under IMC (with permission), "
               "although MISC 2.5 does not require permission for GI.")

PY_PROMPTS = [  # (id, file, scheme, setting, status, notes)
    ("parser", "components/parser.py", "MISC 2.5", "utterance segmentation (volley -> utterances)", "active", ""),
    ("rationale_distill", "baseline/rationalize.py", "MISC 2.5", "teacher rationale generation for SFT CoT targets",
     "active", "Stage 2 regenerates rationales per training fold"),
    ("fewshot_rationale", "baseline/fewshot.py", "MISC 2.5", "frozen rationales for few-shot exemplars",
     "active", "duplicates rationale_distill; P2 merges them into one prompt"),
    ("agent_system", "agentic/agent.py", "MISC 2.5", "retrieval agent system prompt", "active",
     "Stage 4 only, if retrieval is kept"),
    ("synth_v1", "synth/prompts_v1.py", "MISC 2.5", "synthesis v1 generator", "frozen", "v1 data only"),
    ("synth_v2", "synth/prompts_v2.py", "MISC 2.5", "synthesis v2/v3 generator + segment + label", "frozen",
     "v2/v3 data; synthesis v4 gets a new module"),
    ("synth_judge", "synth/judge.py", "MISC 2.5", "synthesis quality judge", "active",
     "reads v2 scenario columns, so v3 windows are judged '(not specified)': fix before v4"),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    entries = []
    for version, status, tdir, sdir in (("v1", "frozen", ROOT / "templates", ROOT / "specs"),
                                         ("v2", "active", ROOT / "templates" / "v2", ROOT / "specs" / "v2")):
        for spk in ("counsellor", "client"):
            for name, (setting, notes) in MISC_TEMPLATES.items():
                p = tdir / spk / f"{name}.j2"
                entries.append({"id": f"misc.{version}.{spk}.{name}", "version": version, "scheme": "MISC 2.5",
                                "speaker": spk, "setting": setting, "status": status,
                                "path": str(p.relative_to(SRC)), "sha256": _sha(p),
                                **({"notes": notes} if notes else {})})
            p = sdir / f"{spk}_t2.yaml"
            entries.append({"id": f"misc.{version}.{spk}.spec_t2", "version": version, "scheme": "MISC 2.5",
                            "speaker": spk, "setting": "T2 code definitions per T1 group (injected)",
                            "status": status, "path": str(p.relative_to(SRC)), "sha256": _sha(p)})
        p = tdir / "user_prompt.j2"
        entries.append({"id": f"misc.{version}.user_prompt", "version": version, "scheme": "any",
                        "speaker": "both", "setting": "user turn: transcript excerpt + target utterance",
                        "status": status, "path": str(p.relative_to(SRC)), "sha256": _sha(p)})
    for path, eid, spk, setting in (
            ("templates/miti/_codes.j2", "miti.v1.codes", "counsellor", "shared MITI 4.2.1 code definitions + decision rules"),
            ("templates/miti/counsellor/t_bare.j2", "miti.v1.counsellor.bare", "counsellor", "single tier, label only"),
            ("templates/miti/counsellor/t_cot.j2", "miti.v1.counsellor.cot", "counsellor", "single tier, rationale + label")):
        entries.append({"id": eid, "version": "v1", "scheme": "MITI 4.2.1", "speaker": spk, "setting": setting,
                        "status": "active", "path": str((ROOT / path).relative_to(SRC)), "sha256": _sha(ROOT / path),
                        "notes": "definitions paraphrase MITI 4.2.1 sections E.4.a-g and F"})
    for path, eid, spk, setting in (
            ("templates/annomi/counsellor/main_bare.j2", "annomi.v1.counsellor.bare", "counsellor",
             "main behaviour:subtype, label only"),
            ("templates/annomi/client/talk_bare.j2", "annomi.v1.client.bare", "client", "talk type, label only")):
        entries.append({"id": eid, "version": "v1", "scheme": "AnnoMI", "speaker": spk, "setting": setting,
                        "status": "pending-source-check", "path": str((ROOT / path).relative_to(SRC)),
                        "sha256": _sha(ROOT / path),
                        "notes": "definitions written from the MI literature; the AnnoMI paper (Future Internet "
                                 "2023, section 4) was not reachable on 2026-10-05; check wording against it before use"})
    for eid, f, scheme, setting, status, notes in PY_PROMPTS:
        p = SRC / f
        entries.append({"id": eid, "version": "file", "scheme": scheme, "speaker": "-", "setting": setting,
                        "status": status, "path": f, "sha256": _sha(p), "hash_scope": "file",
                        **({"notes": notes} if notes else {})})
    final = {
        "zero-shot": "misc.v2.{speaker}.t1_bare + t2_bare (two-call), user_prompt v2",
        "few-shot": "same as zero-shot + exemplar block drawn per fold (seed recorded in .meta.json)",
        "sft_bare": "misc.v2.{speaker}.t1_bare + t2_bare; training target = label",
        "sft_cot": "misc.v2.{speaker}.t1 + t2; training target = rationale + label",
        "single_call_and_grpo": "misc.v2.{speaker}.t12",
        "miti_same_scheme": "miti.v1.counsellor.bare (or .cot)",
        "annomi_same_scheme": "annomi.v1.{speaker}.bare",
        "cross_scheme_transfer": "the MISC prompt, labelled 'MISC-prompt transfer'",
    }
    return {"generated_by": "python -m components.prompts.registry --write",
            "known_issues_v1": KNOWN_ISSUES_V1, "conventions": CONVENTIONS,
            "final_prompt_per_setting": final, "prompts": entries}


def check() -> int:
    recorded = yaml.safe_load(OUT.read_text())
    now = {e["id"]: e for e in build()["prompts"]}
    bad = 0
    for e in recorded["prompts"]:
        cur = now.get(e["id"])
        if cur is None:
            print(f"MISSING  {e['id']} ({e['path']})"); bad += 1
        elif cur["sha256"] != e["sha256"]:
            kind = "FROZEN-CHANGED" if e["status"] == "frozen" else "drift (re-run --write)"
            print(f"{kind:15s} {e['id']} ({e['path']})"); bad += 1
    for i in set(now) - {e["id"] for e in recorded["prompts"]}:
        print(f"UNREGISTERED {i}"); bad += 1
    print("registry ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.write:
        reg = build()
        OUT.write_text(yaml.safe_dump(reg, sort_keys=False, allow_unicode=True, width=120))
        print(f"wrote {OUT} ({len(reg['prompts'])} prompts)")
    else:
        sys.exit(check())


if __name__ == "__main__":
    main()
