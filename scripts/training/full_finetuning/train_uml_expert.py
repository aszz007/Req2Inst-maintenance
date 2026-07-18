"""Train the FlowChart expert."""

import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_path_config
from src.training.full_finetuning_trainer import FullFineTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning.uml_expert')


def print_header():
    """Print header."""
    print("=" * 80)
    print(" " * 12 + "Full Fine-tuning FlowChart Expert Training (Conservative, High-Quality Strategy)")
    print("=" * 80)
    print()


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Train the FlowChart Expert with full fine-tuning')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='Train with 4-bit quantization (default: True)')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='Disable 4-bit quantization')
    args = parser.parse_args()

    print_header()

    path_cfg = get_path_config()

    print("=" * 80)
    print("Training strategy: conservative, high-quality configuration")
    print("=" * 80)
    print("Configuration:")
    print(f"  - LoRA Rank: 16 (high quality)")
    print(f"  - LoRA Alpha: 32")
    print(f"  - Max Seq Length: 2048 (covers 70% of FlowChart samples)")
    print(f"  - Batch Size: 2")
    print(f"  - Gradient Accumulation: 64 (effective batch=128)")
    print(f"  - 4-bit quantization: {args.use_4bit}")
    print("Note: Short FlowChart samples are fully covered; very long samples (~7,000 tokens) are heavily truncated")
    print("Note: Very long samples cannot be trained in full with 24 GB of GPU memory (hardware limitation)")
    print("Expected: 16-18 GB GPU memory, 5-10% quality loss")
    print("=" * 80)
    print()

    logger.info("Initializing full fine-tuning FlowChart expert trainer...")
    trainer = FullFineTuningTrainer(
        expert_type='uml',
        use_4bit=args.use_4bit,
        use_rtx4090_optimization=True,
    )

    logger.info("Setting up model...")
    if not trainer.setup_model():
        logger.error("Model setup failed")
        return 1

    logger.info("Preparing data...")
    if not trainer.prepare_data():
        logger.error("Data preparation failed")
        return 1

    logger.info("Starting training...")
    if not trainer.train():
        logger.error("Training failed")
        return 1

    print()
    print("=" * 80)
    print(" " * 25 + "Training completed successfully!")
    print("=" * 80)
    print(f"Full fine-tuning weights saved to: {trainer.output_dir}")
    print()
    print("Training summary:")
    print("  - Sample coverage: about 70% of FlowChart (very long samples truncated)")
    print("  - Training quality: 5-10% loss (very good)")
    print("  - Configuration: batch=2, effective batch=128 (optimized)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
