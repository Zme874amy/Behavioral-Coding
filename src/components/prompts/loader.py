"""Render the annotation prompts, by prompt-set version.

Prompt sets are frozen once used. v1 is the original set every result up to
2026-10-05 used (templates/ and specs/ at the top level). v2 fixes its bugs
(client prompts naming the counsellor; ADWP/CON/DIR/RCWP spellings; typos) and
lives in templates/v2/ and specs/v2/. Scheme-specific families (MITI, AnnoMI)
live in templates/miti/ and templates/annomi/. REGISTRY.yaml lists every
template with its sha256 (`python -m components.prompts.registry --check`).

The version comes from the `version=` argument, else the PROMPT_VERSION
environment variable, else v1, so past runs reproduce unchanged. Every result's
.meta.json records the version and hashes in use (`prompt_fingerprint`).
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).parent
VERSIONS = {
    "v1": (ROOT / "templates", ROOT / "specs"),
    "v2": (ROOT / "templates" / "v2", ROOT / "specs" / "v2"),
}
# Kept for callers that read these directly; they point at v1.
PROMPT_DIR, SPECS_DIR = VERSIONS["v1"]


def current_version(version: str = None) -> str:
    v = version or os.environ.get("PROMPT_VERSION", "v1")
    if v not in VERSIONS:
        raise ValueError(f"unknown prompt version {v!r}; known: {sorted(VERSIONS)}")
    return v


@lru_cache(maxsize=None)
def _env(version: str) -> Environment:
    return Environment(loader=FileSystemLoader(VERSIONS[version][0]))


def load_spec(speaker: str, structure: str, version: str = None) -> dict:
    spec_file = VERSIONS[current_version(version)][1] / f"{speaker}_{structure}.yaml"
    if spec_file.exists():
        with open(spec_file, "r") as f:
            return yaml.safe_load(f)
    return {}


def render_prompt(speaker: str, structure: str, tier: str = None, version: str = None, **kwargs) -> str:
    version = current_version(version)
    file_name = f"{structure}.j2" if tier is None else f"{tier}.j2"
    template = _env(version).get_template(f"{speaker}/{file_name}")

    # If this is a t2 prompt and 'label' is passed, inject the relevant spec
    # (covers both "t2" and the label-only "t2_bare" variant)
    if structure.startswith("t2") and "label" in kwargs:
        spec_dict = load_spec(speaker, "t2", version)
        kwargs["spec"] = spec_dict.get(kwargs["label"], "")

    # The single-call template classifies both tiers at once, so it needs every
    # T2 group rather than one. Reading them from the same spec YAML the
    # two-call prompts use keeps the fine-grained definitions (notably the long
    # CQ/OQ rules) from drifting between the two formats.
    if structure.startswith("t12"):
        kwargs.setdefault("specs", load_spec(speaker, "t2", version))
        kwargs.setdefault("emit_rationale", True)

    return template.render(**kwargs)


def render_user_prompt(transcript: str, speaker: str, utterance: str, version: str = None) -> str:
    template = _env(current_version(version)).get_template("user_prompt.j2")
    return template.render(transcript=transcript, speaker=speaker, utterance=utterance)


SCHEME_DIR = ROOT / "templates"          # templates/miti/, templates/annomi/
SCHEME_PROMPTS = {
    "miti": {("counsellor", "bare"): "miti/counsellor/t_bare.j2",
             ("counsellor", "cot"): "miti/counsellor/t_cot.j2"},
    "annomi": {("counsellor", "bare"): "annomi/counsellor/main_bare.j2",
               ("client", "bare"): "annomi/client/talk_bare.j2"},
}


@lru_cache(maxsize=None)
def _scheme_env() -> Environment:
    return Environment(loader=FileSystemLoader(SCHEME_DIR))


def render_scheme_prompt(scheme: str, speaker: str, style: str = "bare") -> str:
    """System prompt for a non-MISC scheme (same-scheme evaluation uses the scheme's own prompt).

    The user prompt is the shared v2 `user_prompt.j2` (render_user_prompt(..., version="v2")).
    """
    try:
        path = SCHEME_PROMPTS[scheme][(speaker, style)]
    except KeyError:
        raise ValueError(f"no {scheme} prompt for {speaker}/{style}; have {sorted(SCHEME_PROMPTS.get(scheme, {}))}")
    return _scheme_env().get_template(path).render()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prompt_fingerprint(version: str = None) -> dict:
    """{version, files: {path relative to the prompts dir: sha256}} for one MISC set.

    Written into every result's .meta.json so a number can be traced to the
    exact prompt text that produced it.
    """
    version = current_version(version)
    tdir, sdir = VERSIONS[version]
    files = [p for spk in ("counsellor", "client") for p in (tdir / spk).glob("*.j2")]
    files += [tdir / "user_prompt.j2"] + list(sdir.glob("*.yaml"))
    return {"version": version,
            "files": {str(p.relative_to(ROOT)): _sha(p) for p in sorted(files) if p.exists()}}
