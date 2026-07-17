"""Train the image expert."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.prompt_tuning_trainer import PromptTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.prompt_tuning.image_expert')


def detect_rtx4090() -> bool:
    """Detect rtx4090."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
    except:
        pass
    return False


def print_header():
    """Print header."""
    print("=" * 80)
    print(" " * 15 + "Prompt Tuning Image Expert Training")
    print("=" * 80)
    print()


def validate_environment() -> bool:
    """Validate environment."""
    print("Checking runtime environment...")
    print("-" * 80)

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        v = transformers.__version__.split('.')
        major, minor = int(v[0]), int(v[1])
        if not (major > 4 or (major == 4 and minor >= 51)):
            logger.warning(f"transformers >=4.51.0 is recommended; current version: {transformers.__version__}")
    except ImportError:
        logger.error("transformers is not installed")
        return False

    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        logger.error("PEFT is not installed. Run: pip install peft")
        return False

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
        else:
            logger.warning("CUDA is unavailable; training will run on CPU and be extremely slow")
    except ImportError:
        logger.error("PyTorch is not installed")
        return False

    print("-" * 80)
    print()
    return True


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='使用Prompt Tuning训练图像专家')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    parser.add_argument('--no_rtx4090_opt', action='store_true',
                        help='禁用RTX 4090优化（默认：自动检测）')
    args = parser.parse_args()

    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090 and not args.no_rtx4090_opt

    if is_rtx4090:
        if use_rtx4090_opt:
            logger.info("Detected RTX 4090; enabling optimized settings")
        else:
            logger.info("Detected RTX 4090, but optimization is disabled")

    print_header()

    if not validate_environment():
        logger.error("Environment validation failed; check the required dependencies")
        return 1

    logger.info("Creating prompt-tuning image expert trainer...")
    try:
        trainer = PromptTuningTrainer(
            expert_type='image',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Setting up model and prompt-tuning configuration...")
    if not trainer.setup_model():
        logger.error("Model setup failed")
        return 1

    logger.info("Preparing training data...")
    if not trainer.prepare_data():
        logger.error("Training data preparation failed")
        return 1

    status = trainer.get_training_status()
    print(f"Dataset statistics:")
    print(f"  - Training samples: {status['train_samples']}")
    print(f"  - Validation samples: {status['val_samples']}")
    print()

    logger.info("Starting training...")
    print("=" * 80)
    print("Training started - this may take a while, please wait...")
    print("=" * 80)
    print()

    success = trainer.train()

    if success:
        print()
        print("=" * 80)
        print(" " * 25 + "Training completed successfully!")
        print("=" * 80)
        print()
        print(f"Prompt Tuning weights saved to: {trainer.output_dir}")
        print(f"Checkpoint directory: {trainer.output_dir / 'training_checkpoints'}")
        print()
        print("Next steps:")
        print("  1. Use these weights for inference testing")
        print("  2. Continue training the other experts (Text, UML, General)")
        print()
        return 0
    else:
        print()
        print("=" * 80)
        print(" " * 28 + "Training failed")
        print("=" * 80)
        print()
        logger.error("An error occurred during training; check the logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
