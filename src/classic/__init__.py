"""Traditional-ML + encoder pathway for MISC coding (separate from the LLM arms).

Mirrors the two source papers: n-gram / linguistic features and Qwen embeddings
fed to sklearn classifiers (classic.sklearn_run), and a fine-tuned roberta-base
encoder (classic.encoder). Kept apart from src/baseline and src/agentic; writes
to data/annotated/classic/.
"""
