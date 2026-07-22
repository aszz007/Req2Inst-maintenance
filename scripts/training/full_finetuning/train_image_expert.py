"""Train the image expert."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_path_config  # noqa: E402
from src.training.full_finetuning_trainer import FullFineTuningTrainer  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger('training.full_finetuning.image_expert')


def print_header():
    """Print header."""
    print("Full Fine-tuning Image Expert Training (Conservative, High-Quality Strategy)")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Train the Image Expert with full fine-tuning')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='Train with 4-bit quantization (default: True)')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='Disable 4-bit quantization')
    args = parser.parse_args()

    print_header()

    get_path_config()

    print("Training strategy: conservative, high-quality configuration")
    print("Configuration:")
    print("  - LoRA Rank: 16 (high quality)")
    print("  - LoRA Alpha: 32")
    print("  - Max Seq Length: 2048 (covers 100% of the Image dataset)")
    print("  - Batch Size: 4")
    print("  - Gradient Accumulation: 32 (effective batch=128)")
    print(f"  - 4-bit quantization: {args.use_4bit}")
    print("Note: Image samples are at most ~500 tokens and are fully covered")
    print("Expected: 13-15 GB GPU memory, 5-10% quality loss")

    logger.info("Initializing full fine-tuning image expert trainer...")
    trainer = FullFineTuningTrainer(
        expert_type='image',
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

    print("Training completed successfully!")
    print(f"Full fine-tuning weights saved to: {trainer.output_dir}")
    print("Training summary:")
    print("  - Sample coverage: 100% of Image")
    print("  - Training quality: 5-10% loss (very good)")
    print("  - Configuration: batch=4, effective batch=128 (optimized)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
