from pathlib import Path
from typing import Literal
import pandas as pd
from hydra.utils import log
import hydra

DEMO_DICT = {
    "MIV6.3A": "2024-11-14-MIV6.3A-2024-11-22-MIV6.3A_all_data_delta_with_post_keep_high_conf_True_merged.csv",
    "MIV6.3B": "2024-11-19-MIV6.1B_all_data_delta_with_post_keep_high_conf_True_merged.csv"
}

class Utterance:
    def __init__(self, text: str, speaker: str):
        self.text = text
        self.speaker = speaker
        self.labels: dict[str, str] = {} 

    def __repr__(self):
        return f"Utterance(speaker={self.speaker}, text_length={len(self.text)})"
    
class Volley:
    def __init__(self, text: str, speaker: str):
        self.vol_text = text
        self.speaker = speaker
        self.parsed_utterances: list[Utterance] = []

    def __repr__(self):
        return f"Volley(text={self.vol_text}, speaker={self.speaker}, utterances={(self.parsed_utterances)})"
    
class Conversation:
    def __init__(self, conv_id: str, volleys: list):
        self.conv_id = conv_id
        self.volleys: list[Volley] = volleys

    def __repr__(self):
        return f"Conversation(id={self.conv_id}, volleys={len(self.volleys)})"

class Corpus:
    def __init__(self, cfg):
        self.cfg = cfg
        self.state: Literal['loaded', 'parsed', 'annotated'] = 'loaded'
        self.conversations: list[Conversation] = self.load_dataframe_new()
    
    def filtered_ids(self, ids):
        # TODO: make sure lowconf is put before highconf in the output
        # TODO: make sure this works with HLQC, AnnoMI
        log.info("Filtering IDs based on dataset specifications")
        log.info(f"Dataset total unique IDs: {ids.nunique()}")
        if self.cfg.input_dataset.name in ['MIV6.3A', 'MIV6.3B']:
            demographics_fn = DEMO_DICT[self.cfg.input_dataset.name]
            subset_str = self.cfg.input_dataset.subset
            if subset_str == 'lowconf':
                status = 'low-confidence-or-discordant'
            elif subset_str == 'highconf':
                status = 'high-confidence-no-discordant'
            else:
                log.info("Returning all IDs without filtering")
                return ids.unique()[:self.cfg.n_conversations] if self.cfg.n_conversations else ids.unique()
            demo_info = pd.read_csv(Path("data") / demographics_fn)
            subset = demo_info[demo_info['Status'] == status]
            valid_ids = subset['Participant id'].unique()
            filtered = ids[ids.isin(valid_ids)]

            log.info(f"Valid IDs for {subset_str} subset: {filtered.nunique()}")

            return filtered.unique()[:self.cfg.n_conversations] if self.cfg.n_conversations else filtered.unique()
        return ids.unique()[:self.cfg.n_conversations] if self.cfg.n_conversations else ids.unique()

    def load_dataframe_new(self):
        '''
        create list of Conversation objects 
        '''
        conversations: list[Conversation] = []
        dataset_path = Path("data") / f'{self.cfg.input_dataset.name}.csv'
        df = pd.read_csv(dataset_path)
        valid_ids = self.filtered_ids(df['conv_id'])
        for id in valid_ids:
            conv_df = df[df['conv_id'] == id]
            volleys = []
            for _, row in conv_df.iterrows():
                volley_text = str(row['vol_text'])
                speaker_raw = row['speaker'].lower()
                speaker = (
                    'counsellor' if speaker_raw in ['counsellor', 'therapist']
                    else 'client' if speaker_raw in ['client', 'patient']
                    else 'unknown'
                )
                volleys.append(Volley(text=volley_text, speaker=speaker))
            conversation = Conversation(conv_id=id, volleys=volleys)
            log.info(f"Adding conversation {conversation.conv_id} with {len(conversation.volleys)} volleys")
            conversations.append(conversation)
        log.info(f"Loaded {len(conversations)} conversations from dataset {self.cfg.input_dataset.name}")

        return conversations
        
    def to_df(self):
        """export df"""
        df = []
        corp_vol_idx = 0
        corp_utt_idx = 0
        for corp_conv_idx, conv in enumerate(self.conversations):
            conv_utt_idx = 0
            for conv_vol_idx, volley in enumerate(conv.volleys):
                row = {
                        'corp_conv_idx': corp_conv_idx,
                        'conv_id': conv.conv_id,
                        'speaker': volley.speaker,
                        'corp_vol_idx': corp_vol_idx,
                        'conv_vol_idx': conv_vol_idx,
                        'vol_text': volley.vol_text,
                    }
                if self.state == 'loaded':
                    df.append(row)
                elif self.state == 'parsed':
                    for utterance in volley.parsed_utterances:
                        df.append({
                            **row,
                            'corp_utt_idx': corp_utt_idx,
                            'conv_utt_idx': conv_utt_idx,
                            'utt_text': utterance.text,
                        })
                        corp_utt_idx += 1
                        conv_utt_idx += 1
                elif self.state == 'annotated':
                    for utterance in volley.parsed_utterances:
                        df.append({
                            **row,
                            'corp_utt_idx': corp_utt_idx,
                            'conv_utt_idx': conv_utt_idx,
                            'utt_text': utterance.text,
                            **utterance.labels,  # Assuming labels are a dict of annotations
                        })
                        corp_utt_idx += 1
                        conv_utt_idx += 1
                corp_vol_idx += 1

        output_df = pd.DataFrame(df)
        return output_df
    
    def save_to_csv(self):
        output_dir = Path('data/test')
        output_dir.mkdir(parents=True, exist_ok=True)
        exp_output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        fn = (
            f"{self.cfg.input_dataset.name}_"
            f"{self.cfg.input_dataset.subset}_"
            f"{self.cfg.annotator.class_structure}_"
            f"{self.cfg.annotator.model.rsplit('/', 1)[-1]}_"
            f"{self.cfg.annotator.context_mode}_"
            f"{self.cfg.annotator.num_context_turns if self.cfg.annotator.context_mode == 'interval' else ''}" 
            f"_{self.state}.csv"
        )
        exp_save_path = exp_output_dir / fn
        df = self.to_df()
        df.to_csv(exp_save_path, index=False)
        log.info(f"Saved {self.cfg.input_dataset.name}-{self.state} dataframe to {exp_save_path}")

    def __repr__(self):
        avg_len = sum(len(conv.volleys) for conv in self.conversations) / len(self.conversations) if self.conversations else 0
        return (f"Corpus info:\n"
                f"Dataset: {self.cfg.input_dataset.name}.csv\n"
                f"Conversations: {len(self.conversations)}\n"
                f"Average conversation length: {avg_len:.2f} volleys")
