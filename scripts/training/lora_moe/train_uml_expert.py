"""Train the FlowChart expert."""

import sys
import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lora_trainer import LoRATrainer
from config.settings import get_path_config, get_training_config, get_lora_config
from src.utils.logger import get_logger

logger = get_logger('training.train_uml_expert')


def print_header():
    """Print header."""
    print("=" * 80)
    print(" " * 30 + "FlowChart Expert Training")
    print("=" * 80)
    print()


def print_config(use_4bit: bool, use_rtx4090: bool):
    """Print config."""
    path_cfg = get_path_config()
    train_cfg = get_training_config()
    lora_cfg = get_lora_config('conservative')

    print("Training configuration:")
    print("-" * 80)
    print(f"Expert type: FlowChart Expert")
    print(f"Dataset: uml_dataset.csv (1,500 samples)")
    print(f"Base model: {path_cfg.QWEN3_8B_PATH}")
    print(f"Output directory: checkpoints/lora_moe/uml_expert/")
    print()

    print(f"LoRA configuration:")
    print(f"  - Rank: {lora_cfg.rank}")
    print(f"  - Alpha: {lora_cfg.alpha}")
    print(f"  - Dropout: {lora_cfg.dropout}")
    print(f"  - Target Modules: {lora_cfg.target_modules}")
    print()

    if use_rtx4090:
        print(f"Training parameters (RTX 4090 optimized):")
        print(f"  - Batch Size: 8 (optimized)")
        print(f"  - Gradient Accumulation: 2 (optimized)")
        print(f"  - Effective Batch Size: 16")
        print(f"  - Epochs: {train_cfg.num_epochs}")
        print(f"  - Learning Rate: {train_cfg.learning_rate}")
        print(f"  - Max Seq Length: {train_cfg.max_seq_length}")
        print(f"  - 4-bit quantization: {use_4bit}")
        print(f"  - BF16 mixed precision: True")
        print(f"  - TF32 acceleration: True")
        print(f"  - Fused optimizer: True")
        print(f"  - Data loader workers: 8")
    else:
        print(f"Training parameters:")
        print(f"  - Batch Size: {train_cfg.batch_size}")
        print(f"  - Gradient Accumulation: {train_cfg.gradient_accumulation_steps}")
        print(f"  - Effective Batch Size: {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
        print(f"  - Epochs: {train_cfg.num_epochs}")
        print(f"  - Learning Rate: {train_cfg.learning_rate}")
        print(f"  - Max Seq Length: {train_cfg.max_seq_length}")
        print(f"  - 4-bit quantization: {use_4bit}")

    print("-" * 80)
    print()


def validate_environment():
    """Validate environment."""
    print("Checking runtime environment...")
    print("-" * 80)

    try:
        import transformers
        tf_version = transformers.__version__

        print(f"Transformers version: {tf_version}")

        try:
            v_parts = tf_version.split('.')
            major, minor = int(v_parts[0]), int(v_parts[1])
            if not (major > 4 or (major == 4 and minor >= 51)):
                logger.warning(f"Current transformers version is {tf_version}; version >=4.51.0 is recommended")
                logger.warning("Verify that the script is running in the instruction_generator environment")
        except (ValueError, IndexError):
            logger.warning(f"Unable to parse transformers version: {tf_version}")

    except ImportError:
        logger.error("transformers is not installed")
        return False

    try:
        import peft
        print(f"PEFT version: {peft.__version__}")
    except ImportError:
        logger.error("PEFT is not installed. Run: pip install peft --break-system-packages")
        return False

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"CUDA available: {gpu_name}")
            print(f"GPU memory: {gpu_memory:.2f}GB")

            is_rtx4090 = 'RTX 4090' in gpu_name or 'RTX 4090D' in gpu_name
            if is_rtx4090:
                print(f"RTX 4090 detected; enabling optimized configuration")

        else:
            logger.warning("CUDA is unavailable; training will run on CPU and be extremely slow")
    except ImportError:
        logger.error("PyTorch is not installed")
        return False

    print("-" * 80)
    print()
    return True


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


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Train the FlowChart expert')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='Train with 4-bit quantization (default: True)')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='Disable 4-bit quantization')
    parser.add_argument('--no_rtx4090_opt', action='store_true',
                        help='Disable RTX 4090 optimizations (default: auto-detect)')
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

    logger.info(f"Creating FlowChart expert trainer...")
    try:
        trainer = LoRATrainer(
            expert_type='uml',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Setting up model and LoRA configuration...")
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
    print(f"  - Dataset: uml_dataset.csv (1,500 rows)")
    print()
    print("Note: 1,500 samples use the standard 80:10:10 split")
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

        path_cfg = get_path_config()
        output_path = path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe' / 'uml_expert'
        print(f"LoRA weights saved to: {output_path}")
        print(f"Checkpoint directory: {output_path / 'training_checkpoints'}")
        print()
        print("Next steps:")
        print("  1. Use these weights for inference testing")
        print("  2. Continue training the General Expert")
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
