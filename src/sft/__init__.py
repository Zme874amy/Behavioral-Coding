"""Supervised Fine-Tuning pipeline using human-labeled MI datasets.

This package replaces the legacy 'auto-annotate-then-train' flow. The SFT
pipeline NEVER trains on machine-generated labels. It uses only the
human-consensus labels in `data/manual/*.csv` and the AnnoMI manual labels
in `data/AnnoMI.csv` (Leave-One-Dataset-Out evaluation).
"""
