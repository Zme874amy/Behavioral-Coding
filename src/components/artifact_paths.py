"""Stable filenames for parsed / annotated artifacts (isolates parser × annotator combos)."""
from pathlib import Path


def model_slug(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def _parser_model(cfg) -> str:
    if hasattr(cfg.parser, "model") and cfg.parser.model is not None:
        return cfg.parser.model
    return cfg.parser.models[0].model


def _annotator_block(cfg):
    if hasattr(cfg.annotator, "model") and cfg.annotator.model is not None:
        return cfg.annotator
    return cfg.annotator.models[0]


def get_parsed_csv_path(cfg) -> Path:
    """Per-parser checkpoint under data/parsed/."""
    slug = model_slug(_parser_model(cfg))
    return Path("data/parsed") / f"{cfg.input_dataset.name}_{slug}_parsed.csv"


def annotated_csv_stem(cfg) -> str:
    """Filename stem (no .csv) for annotated outputs."""
    parser_m = _parser_model(cfg)
    ann = _annotator_block(cfg)
    ann_m = ann.model
    ctx = ann.context_mode
    nturns = ann.num_context_turns
    cls = ann.class_structure
    return (
        f"{cfg.input_dataset.name}_"
        f"{cfg.input_dataset.subset}_"
        f"{cls}_"
        f"{model_slug(parser_m)}_"
        f"{model_slug(ann_m)}_"
        f"{ctx}_"
        f"{nturns if ctx == 'interval' else ''}"
    )


def get_annotated_csv_path(cfg) -> Path:
    return Path("data/annotated") / f"{annotated_csv_stem(cfg)}_annotated.csv"


def corpus_hydra_export_name(cfg, state: str) -> str:
    """Basename for parsed/annotated exports under Hydra output dir."""
    return f"{annotated_csv_stem(cfg)}_{state}.csv"
