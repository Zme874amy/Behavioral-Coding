"""Print the annotation-agreement numbers for AnnoMI and Welivita.

Thin wrapper: the computation lives in `eda.registry` (annomi_agreement,
welivita_agreement), shared with notebooks/datasets_eda.ipynb. Releases are
downloaded to data/external/{annomi,welivita}/ on first use.

Usage: PYTHONPATH=src python scripts/audit_external_datasets.py
"""
import json

from eda.registry import annomi_agreement, welivita_agreement

if __name__ == "__main__":
    print("AnnoMI (7 ten-rater transcripts):")
    print(json.dumps(annomi_agreement(), indent=1, default=str))
    print("Welivita (ann1 vs ann2):")
    print(json.dumps(welivita_agreement(), indent=1))
