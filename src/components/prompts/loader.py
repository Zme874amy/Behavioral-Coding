from jinja2 import Environment, FileSystemLoader
import yaml
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "templates"
SPECS_DIR = Path(__file__).parent / "specs"

env = Environment(loader=FileSystemLoader(PROMPT_DIR))

def load_spec(speaker: str, structure: str) -> dict:
    spec_file = SPECS_DIR / f"{speaker}_{structure}.yaml"
    if spec_file.exists():
        with open(spec_file, "r") as f:
            return yaml.safe_load(f)
    return {}

def render_prompt(speaker: str, structure: str, tier: str = None, **kwargs) -> str:
    file_name = f"{structure}.j2" if tier is None else f"{tier}.j2"
    template = env.get_template(f"{speaker}/{file_name}")

    # If this is a t2 prompt and 'label' is passed, inject the relevant spec
    # (covers both "t2" and the label-only "t2_bare" variant)
    if structure.startswith("t2") and "label" in kwargs:
        spec_dict = load_spec(speaker, "t2")
        kwargs["spec"] = spec_dict.get(kwargs["label"], "")

    # The single-call template classifies both tiers at once, so it needs every
    # T2 group rather than one. Reading them from the same spec YAML the
    # two-call prompts use keeps the fine-grained definitions (notably the long
    # CQ/OQ rules) from drifting between the two formats.
    if structure.startswith("t12"):
        kwargs.setdefault("specs", load_spec(speaker, "t2"))
        kwargs.setdefault("emit_rationale", True)

    return template.render(**kwargs)

def render_user_prompt(transcript: str, speaker: str, utterance: str) -> str:
    template = env.get_template("user_prompt.j2")
    return template.render(transcript=transcript, speaker=speaker, utterance=utterance)