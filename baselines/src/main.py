"""Entrypoint for AutoMISC automated behavioural code classifier."""
import json
from pathlib import Path

import hydra
from hydra.utils import log
from omegaconf import OmegaConf
from datatypes.corpus import Corpus
from components.parser import Parser
from components.annotator import Annotator
import os
import openai
import logging
import lmstudio as lms

def validate_config(cfg) -> None:
    for model_name in [cfg.parser.model, cfg.annotator.model]:
        try:
            lms_models = {m.model_key for m in lms.list_downloaded_models("llm")}
        except Exception as e:
            raise RuntimeError(f"Failed to fetch LM Studio models: {e}")

        openai_models = set()
        if "OPENAI_API_KEY" in os.environ:
            try:
                openai.api_key = os.environ["OPENAI_API_KEY"]
                openai_models = {m.id for m in openai.models.list().data}
            except Exception as e:
                raise RuntimeError(f"Failed to fetch OpenAI models: {e}")
        else:
            log.warning("OPENAI_API_KEY not set; skipping OpenAI model validation")

        if model_name not in openai_models and model_name not in lms_models:
            log.info(f"Available LM Studio models: {lms_models}")
            log.info(f"Available OpenAI models: {openai_models}")
            raise ValueError(f"Model '{model_name}' not found in OpenAI or LM Studio models.")

        # --- Check that input file exists ---
        input_path = Path('data') / f'{cfg.input_dataset.name}.csv'
        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    log.info("Configuration successfully validated.")

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg) -> None:
    validate_config(cfg)    
    logging.getLogger("httpx").setLevel(logging.WARNING)

    log.info("Starting AutoMISC run with configuration:\n%s", OmegaConf.to_yaml(cfg))

    # 1) Load volley-level dataset
    corpus = Corpus(cfg)
    log.info(corpus)
    log.info("Dataset loaded. Now moving onto parsing")

    # 2) Parse each conversation into utterances
    parser = Parser(cfg)
    parsed = parser.parse_corpus(corpus)
    log.info("Parsing complete. Now moving onto annotation")
    parsed.save_to_csv()

    # 3) Annotate utterances
    annotator = Annotator(cfg)
    annotator.annotate_corpus(parsed)

    log.info("Annotating done. Returning from main")
    return


if __name__ == "__main__":
    main()