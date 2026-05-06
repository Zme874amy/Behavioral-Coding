"""Module for parsing Conversation objects into sequences of Utterance objects."""

from datatypes.corpus import Conversation, Corpus
from datatypes.corpus import Utterance
from datatypes.corpus import Volley
from .utils import call_chat_model, get_provider
from hydra.utils import log
import hydra
import pandas as pd
from pathlib import Path
from typing import List
from pydantic import BaseModel
from tqdm import tqdm

PARSER_SYSTEM_PROMPT = '''
You are a highly accurate Motivational Interviewing (MI) counselling session annotator.
Your task is to segment the given volley into utterances.

### Definitions:
1. **Volley**: An uninterrupted utterance or sequence of utterances spoken by one party, before the other party responds.
2. **Utterance**: A complete thought or thought unit expressed by a speaker. This could be a single sentence, phrase, or even a word if it conveys a standalone idea. Multiple utterances often run together without interruption in a volley.

### Output Format:
- Return the segmented utterances as a JSON list of strings (e.g. ["utt1", "utt2", ...]).
'''

few_shots = [
    {'role': 'user',      'content': "Why haven't you quit smoking - are you ever gonna quit?"},
    {'role': 'assistant', 'content': "[\"Why haven't you quit smoking - are you ever gonna quit?\"]"},
    {'role': 'user',      'content': "How long since your last drink? Do you feel ok?"},
    {'role': 'assistant', 'content': "[\"How long since your last drink?\", \"Do you feel ok?\"]"},
    {'role': 'user',      'content': "I can't quit. I just can't do it. I don't have what it takes. I just cannot stop."},
    {'role': 'assistant', 'content': "[\"I can't quit.\", \"I just can't do it.\", \"I don't have what it takes.\", \"I just cannot stop.\"]"},
    {'role': 'user',      'content': "I don't want to go to the bars every day. I don't want my kids to see that. I want my kids to have a better life than that."},
    {'role': 'assistant', 'content': "[\"I don't want to go to the bars every day.\", \"I don't want my kids to see that.\", \"I want my kids to have a better life than that.\"]"},
]

class MISCParser(BaseModel):
    utterances: List[str]

class Parser:
    """Parser for converting Conversation objects into sequences of Utterance objects."""

    def __init__(self, cfg):
        """
        Initialize parser with configuration.
        """
        self.cfg = cfg
        self.provider = get_provider(cfg.parser.model)

    def parse_conversation(self, conv: Conversation, existing_df: pd.DataFrame):
        '''parse a single conversation'''
        checkpoint_idx = 0
        conv_id = conv.conv_id
        if existing_df is not None and not existing_df.empty:
            if conv_id in existing_df['conv_id'].values:
                log.info(f"Conversation {conv_id} already exists in the existing DataFrame. Checking for volleys.")
                conv_group = existing_df[existing_df['conv_id']==conv_id]
                group_by_vol = conv_group.groupby('corp_vol_idx')
                grouped_dfs = [group for _, group in group_by_vol]
                for vol_idx, vol in enumerate(conv.volleys):
                    vol_utt_rows = grouped_dfs[vol_idx]
                    vol.parsed_utterances.extend([
                        Utterance(text=row['utt_text'], speaker=row['speaker'])
                        for _, row in vol_utt_rows.iterrows()
                    ])
                if len(conv.volleys) == len(group_by_vol):
                    log.info(f"Conversation {conv_id} already fully parsed. Skipping.")
                    return
                else:
                    log.info(f"Continuing parsing for conversation {conv_id}.")
                    checkpoint_idx = len(group_by_vol) 
                    log.info(f"Parsed up to volley index {checkpoint_idx} for conversation {conv_id}.")

        for vol_idx, vol in enumerate(tqdm(
                conv.volleys,
                desc=f"Parsing Conversation for {conv.conv_id}",
                unit="vol",
                total=len(conv.volleys)
            )):
            if vol_idx < checkpoint_idx:
                continue
            volley_text = vol.vol_text
            speaker_raw = vol.speaker.lower()
            speaker = (
                'counsellor' if speaker_raw in ['counsellor', 'therapist']
                else 'client' if speaker_raw in ['client', 'patient']
                else 'unknown'
            )

            messages = [
                {'role': 'system',  'content': PARSER_SYSTEM_PROMPT},
                *few_shots,
                {'role': 'user',    'content': volley_text},
            ]

            utterances = call_chat_model(
                messages=messages,
                model=getattr(self.cfg.parser, 'model', None),
                provider=self.provider,
                temperature=getattr(self.cfg.parser, 'temperature', 0.0),
                response_format=MISCParser
            )

            for utt in utterances['utterances']:
                vol.parsed_utterances.append(Utterance(text=utt, speaker=speaker))

        return

    def parse_corpus(self, corp: Corpus) -> Corpus:
        output_dir = Path('data/parsed')
        save_path = output_dir / f"{self.cfg.input_dataset.name}_parsed.csv"
        existing_df = pd.DataFrame()
        if save_path.exists():
            try: 
                existing_df = pd.read_csv(save_path)
                
                log.info(f"Parser: Loaded existing output file {save_path} with {len(existing_df)} rows.")
            except Exception as e:
                log.warning(f"Could not load existing output file: {e}", exc_info=True)

        try:
            for conv in tqdm(corp.conversations, desc="Parsing Corpus", unit="conv"):
                self.parse_conversation(conv, existing_df)
        except KeyboardInterrupt as e:
            log.warning(f"Manual interrupt while parsing corpus: {e}", exc_info=True)

        corp.state = 'parsed'
        return corp

