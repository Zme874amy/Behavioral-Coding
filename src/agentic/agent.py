"""MISCAgent: a bounded, structured agent for MISC coding (not RAG).

Per utterance the model DIRECTS a bounded loop -- it may call tools when it wants
evidence, then proposes a label, which is checked deterministically and critiqued
against the codebook before being accepted:

    ASSESS/ACT  model emits one JSON action per turn: retrieve | define | answer
                (retrieve/define run a tool; the OBSERVATION is fed back)
    GATE        on `answer`, tools.check enforces hierarchy + vocabulary; a bad
                label is sent back for revision (does not count as a correct answer)
    CRITIQUE    exactly one forced self-check: the parent group's sibling
                definitions are injected and the model confirms or corrects its T2
                -- this is the mechanism aimed at the tail (e.g. SU vs AF)

Guardrails keep a 7B robust: strict single-JSON turns with fallback parsing, hard
caps (max_tool_calls / max_revise / max_steps), and a last-resort snap to a valid
hierarchical label so the arm always emits a well-formed code.

Reuses the loaded model via `TieredAnnotator._generate` (observations are injected
as user turns), the existing retriever as the `retrieve` tool, and the codebook /
validators in `agentic.tools`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from automisc_ft.data import t1_codes_for_speaker, t2_codes_for_speaker
from automisc_ft.infer import parse_label
from components.context import build_context_excerpt
from components.prompts.loader import render_user_prompt

from agentic import tools

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _system_prompt(speaker: str, max_tool_calls: int) -> str:
    t1s = t1_codes_for_speaker(speaker)
    return (
        f"You are an expert MISC behavioural coder. Classify the {speaker}'s FINAL "
        f"utterance into a T1 group and then a T2 code that is a CHILD of that group.\n"
        f"Valid T1 groups for the {speaker}: {t1s}.\n\n"
        "Work step by step and use tools when useful. Reply with EXACTLY ONE JSON "
        "object per turn and nothing else. Allowed actions:\n"
        '  {"action":"retrieve","tier":"t2","t1":"CRL"}   -- similar labelled examples '
        '(tier "t1" omits t1)\n'
        '  {"action":"define","code":"CRL"}                -- codebook definitions '
        "(a group or a code)\n"
        '  {"action":"answer","rationale":"<=1 sentence","t1":"<group>","t2":"<code>"}\n\n'
        f"Use at most {max_tool_calls} tool calls, then answer. Rare T2 codes "
        "(SU, EC, AF, GI) are easy to miss -- weigh them explicitly."
    )


@dataclass
class AgentResult:
    t1: str
    t2: str
    rationale: str
    n_steps: int
    n_tool_calls: int
    n_revises: int
    n_prompt_tokens: int
    n_gen_tokens: int
    trajectory: List[Dict] = field(default_factory=list)


class MISCAgent:
    def __init__(self, annotator, retriever, max_tool_calls: int = 3,
                 max_revise: int = 2, max_steps: int = 6):
        self.ann = annotator
        self.retriever = retriever
        self.max_tool_calls = int(max_tool_calls)
        self.max_revise = int(max_revise)
        self.max_steps = int(max_steps)

    # -- generation + parsing ---------------------------------------------------
    def _gen(self, messages: List[Dict], acc: Dict) -> str:
        text, n_p, n_g = self.ann._generate(messages)
        acc["p"] += int(n_p)
        acc["g"] += int(n_g)
        messages.append({"role": "assistant", "content": text})
        return text

    @staticmethod
    def _parse_action(text: str, speaker: str) -> Dict:
        """Best-effort parse of one JSON action; salvage a label if malformed."""
        m = _JSON_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and obj.get("action") in {"retrieve", "define", "answer"}:
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
        # Salvage: if the model wrote a label somewhere, treat it as an answer.
        t2 = parse_label(text, t2_codes_for_speaker(speaker))
        t1 = parse_label(text, t1_codes_for_speaker(speaker))
        if t2 != "UNKNOWN" or t1 != "UNKNOWN":
            return {"action": "answer", "rationale": "", "t1": t1, "t2": t2, "_salvaged": True}
        return {"action": "unparseable"}

    # -- the loop ---------------------------------------------------------------
    def code_row(self, df, pos: int, context_mode: str, ctx: int) -> AgentResult:
        speaker = df.iloc[pos]["speaker"]
        transcript = build_context_excerpt(df, pos, context_mode, ctx)
        utterance = df.iloc[pos]["utt_text"]

        messages = [
            {"role": "system", "content": _system_prompt(speaker, self.max_tool_calls)},
            {"role": "user", "content": render_user_prompt(transcript, speaker, utterance)
                + "\n\nBegin. Reply with ONE JSON action."},
        ]
        acc = {"p": 0, "g": 0}
        traj: List[Dict] = []
        tool_calls = revises = steps = 0
        final: Optional[Dict] = None

        while steps < self.max_steps and final is None:
            steps += 1
            text = self._gen(messages, acc)
            act = self._parse_action(text, speaker)
            traj.append({"step": steps, "action": act.get("action"), "raw": text[:400]})

            kind = act.get("action")
            if kind in ("retrieve", "define"):
                if tool_calls >= self.max_tool_calls:
                    messages.append({"role": "user", "content":
                        'Tool budget exhausted. Answer now as {"action":"answer",...}.'})
                    continue
                tool_calls += 1
                obs = self._run_tool(act, speaker, df, pos)
                traj[-1]["observation"] = obs[:400]
                messages.append({"role": "user", "content": "OBSERVATION:\n" + obs})
                continue

            if kind == "answer":
                t1, t2 = str(act.get("t1", "")), str(act.get("t2", ""))
                ok, reason = tools.check(speaker, t1, t2)
                if not ok and revises < self.max_revise:
                    revises += 1
                    messages.append({"role": "user", "content":
                        f"INVALID: {reason} Re-answer as {{\"action\":\"answer\",...}}."})
                    continue
                final = act if ok else {**act, **dict(zip(("t1", "t2"),
                                        tools.snap_to_hierarchy(speaker, t1, t2)))}
                break

            # unparseable
            messages.append({"role": "user", "content":
                'Reply with ONE JSON object: {"action":"answer","rationale":...,'
                '"t1":...,"t2":...}.'})

        if final is None:
            final = self._coerce_final(messages, speaker, acc, traj)

        final = self._critique(messages, speaker, final, acc, traj)

        return AgentResult(
            t1=final["t1"], t2=final["t2"], rationale=str(final.get("rationale", "")),
            n_steps=steps, n_tool_calls=tool_calls, n_revises=revises,
            n_prompt_tokens=acc["p"], n_gen_tokens=acc["g"], trajectory=traj,
        )

    # -- one mandatory self-critique grounded in sibling definitions ------------
    def _critique(self, messages, speaker, final, acc, traj) -> Dict:
        sib = tools.define(speaker, final["t1"])
        messages.append({"role": "user", "content":
            "Before finalizing, re-examine your T2 against these definitions and its "
            f"sibling codes:\n{sib}\nIf your T2 is wrong, correct it; otherwise keep it. "
            'Reply with the final {"action":"answer",...}.'})
        text = self._gen(messages, acc)
        act = self._parse_action(text, speaker)
        traj.append({"step": "critique", "action": act.get("action"), "raw": text[:400]})
        if act.get("action") == "answer":
            t1, t2 = str(act.get("t1", final["t1"])), str(act.get("t2", final["t2"]))
            ok, _ = tools.check(speaker, t1, t2)
            if not ok:
                t1, t2 = tools.snap_to_hierarchy(speaker, t1, t2)
            return {"action": "answer", "rationale": act.get("rationale", final.get("rationale", "")),
                    "t1": t1, "t2": t2}
        return final

    def _coerce_final(self, messages, speaker, acc, traj) -> Dict:
        """No valid answer within the step budget: salvage from the last turn."""
        last = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""
        t1 = parse_label(last, t1_codes_for_speaker(speaker))
        t2 = parse_label(last, t2_codes_for_speaker(speaker))
        t1, t2 = tools.snap_to_hierarchy(speaker, t1, t2)
        traj.append({"step": "coerce", "action": "answer", "t1": t1, "t2": t2})
        return {"action": "answer", "rationale": "", "t1": t1, "t2": t2}

    # -- tool dispatch ----------------------------------------------------------
    def _run_tool(self, act: Dict, speaker: str, df, pos: int) -> str:
        kind = act.get("action")
        if kind == "define":
            return tools.define(speaker, str(act.get("code", "")))
        if kind == "retrieve":
            tier = "t1" if act.get("tier") == "t1" else "t2"
            t1_label = act.get("t1")
            self.retriever.set_query(df.iloc[pos]["utt_text"], speaker)
            return tools.retrieve(self.retriever, speaker, tier, t1_label)
        return "(unknown tool)"
