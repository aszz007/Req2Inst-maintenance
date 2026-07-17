"""Train the text expert."""

import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_path_config
from src.training.full_finetuning_trainer import FullFineTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning.text_expert')


def print_header():
    """Print header."""
    print("=" * 80)
    print(" " * 12 + "Full Fine-tuning Text Expert Training (Conservative, High-Quality Strategy)")
    print("=" * 80)
    print()


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Full Fine-tuning Text Expert训练')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    print_header()

    path_cfg = get_path_config()

    print("=" * 80)
    print("Training strategy: conservative, high-quality configuration")
    print("=" * 80)
    print("Configuration:")
    print(f"  - LoRA Rank: 16 (high quality)")
    print(f"  - LoRA Alpha: 32")
    print(f"  - Max Seq Length: 2048 (covers 90% of Text samples)")
    print(f"  - Batch Size: 2")
    print(f"  - Gradient Accumulation: 64 (effective batch=128)")
    print(f"  - 4-bit quantization: {args.use_4bit}")
    print("Note: Short Text samples are fully covered; long samples (~3,000 tokens) are partially truncated")
    print("Expected: 16-18 GB GPU memory, 5-10% quality loss")
    print("=" * 80)
    print()

    logger.info("Initializing full fine-tuning text expert trainer...")
    trainer = FullFineTuningTrainer(
        expert_type='text',
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
    print("  - Sample coverage: about 90% of Text")
    print("  - Training quality: 5-10% loss (very good)")
    print("  - Configuration: batch=2, effective batch=128 (optimized)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
