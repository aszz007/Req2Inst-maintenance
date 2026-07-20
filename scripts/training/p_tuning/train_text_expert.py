"""Train the text expert."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.p_tuning_trainer import PTuningTrainer
from config.settings import get_path_config, get_ptuning_config
from src.utils.logger import get_logger

logger = get_logger('training.p_tuning.text_expert')


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
    print("P-Tuning v2 Text Expert Training (Prefix Tuning)")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Train the Text Expert with P-Tuning v2')
    parser.add_argument('--use_4bit', action='store_true', dest='use_4bit',
                        help='Train with 4-bit quantization')
    parser.add_argument('--no_4bit', action='store_false', dest='use_4bit',
                        help='Disable 4-bit quantization')
    parser.set_defaults(use_4bit=True)
    args = parser.parse_args()

    print_header()

    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090

    if is_rtx4090:
        logger.info("Detected RTX 4090; enabling optimized settings")

    ptuning_cfg = get_ptuning_config()
    print("Comparison experiment: P-Tuning v2 vs LoRA")
    print("Method: P-Tuning v2 (Prefix Tuning)")
    print("Configuration:")
    print(f"  - Virtual Tokens: {ptuning_cfg.num_virtual_tokens}")
    print(f"  - Encoder Hidden Size: {ptuning_cfg.encoder_hidden_size}")
    print(f"  - Prefix Projection: {ptuning_cfg.prefix_projection}")

    logger.info("Creating P-Tuning v2 text expert trainer...")
    try:
        trainer = PTuningTrainer(
            expert_type='text',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Setting up model and P-Tuning v2 configuration...")
    if not trainer.setup_model():
        logger.error("Model setup failed")
        return 1

    logger.info("Preparing training data...")
    if not trainer.prepare_data():
        logger.error("Training data preparation failed")
        return 1

    status = trainer.get_training_status()
    print("Dataset statistics:")
    print(f"  - Training samples: {status['train_samples']}")
    print(f"  - Validation samples: {status['val_samples']}")

    logger.info("Starting training...")

    success = trainer.train()

    if success:
        print("Training completed successfully!")

        path_cfg = get_path_config()
        output_path = path_cfg.PTUNING_CKPTS['text']
        print(f"P-Tuning v2 weights saved to: {output_path}")
        print(f"Checkpoint directory: {output_path / 'training_checkpoints'}")
        print("Next steps:")
        print("  1. Use these weights for inference testing")
        print("  2. Continue training the other experts (Image, FlowChart, General)")

        return 0
    else:
        logger.error("An error occurred during training; check the logs")
        return 1


if __name__ == "__main__":
    sys.exit(main())
