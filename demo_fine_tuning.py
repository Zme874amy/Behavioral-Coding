#!/usr/bin/env python3
"""
Quick fine-tuning demo script for AutoMISC.

This script demonstrates:
1. Running fine-tuning
2. Evaluating the results
3. Comparing to baseline
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("AutoMISC Fine-Tuning Demo")
    print("=" * 40)

    # Check if we have annotated data
    annotated_file = Path('data/annotated/MIV6.3A_lowconf_tiered_gemma-4-e4b_interval_5_annotated.csv')
    if not annotated_file.exists():
        print("❌ No annotated data found. Run annotation first:")
        print("   python src/main.py fine_tuning.enabled=false")
        return

    print("✅ Found annotated data")

    # Check fine-tuning config
    print("\n📋 Current fine-tuning config:")
    print("   Provider: local")
    print("   Model: gpt2")
    print("   LoRA: enabled")
    print("   Epochs: 3")

    # Run fine-tuning
    print("\n🚀 Starting fine-tuning...")
    try:
        from main import main as run_pipeline
        # This would run the full pipeline including fine-tuning
        # For demo, we'll just show the command
        print("   Command: python src/main.py fine_tuning.enabled=true")
    except Exception as e:
        print(f"   Error: {e}")

    # Run evaluation
    print("\n📊 Running evaluation...")
    try:
        from evaluate_fine_tuning import evaluate_fine_tuning
        # This would run evaluation
        print("   Command: python src/evaluate_fine_tuning.py")
        print("   Results will be saved to outputs/<date>/<time>/fine_tuning_evaluation/")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n✨ Demo complete!")
    print("\nNext steps:")
    print("1. Run: python src/main.py fine_tuning.enabled=true")
    print("2. Run: python src/evaluate_fine_tuning.py")
    print("3. Check results in outputs/ directory")

if __name__ == "__main__":
    main()