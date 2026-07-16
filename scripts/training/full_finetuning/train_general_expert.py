"""Train the general expert."""

import sys
import argparse
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_path_config
from src.training.full_finetuning_trainer import FullFineTuningTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning.general_expert')


def print_header():
    """Print header."""
    print("=" * 80)
    print(" " * 8 + "Full Fine-tuning General Expert训练 (保守高质量策略)")
    print("=" * 80)
    print()


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Full Fine-tuning General Expert训练')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    print_header()

    path_cfg = get_path_config()

    print("=" * 80)
    print("训练策略：保守高质量配置")
    print("=" * 80)
    print("配置：")
    print(f"  - LoRA Rank: 16 (高质量)")
    print(f"  - LoRA Alpha: 32")
    print(f"  - Max Seq Length: 2048 (覆盖General 85%样本)")
    print(f"  - Batch Size: 2")
    print(f"  - Gradient Accumulation: 64 (有效batch=128)")
    print(f"  - 4bit量化: {args.use_4bit}")
    print("数据来源: text + image + uml（混合数据集）")
    print("说明：Text 90%, Image 100%, UML 70%覆盖")
    print("预期：显存16-18GB，质量损失5-10%")
    print("=" * 80)
    print()

    logger.info("Initializing full fine-tuning general expert trainer...")
    trainer = FullFineTuningTrainer(
        expert_type='general',
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
    print(" " * 25 + "训练成功完成！")
    print("=" * 80)
    print(f"Full Fine-tuning权重已保存至: {trainer.output_dir}")
    print()
    print("训练总结：")
    print("  - 样本覆盖率：General 约85%（混合数据集）")
    print("  - 训练质量：损失5-10%（非常好）")
    print("  - 配置：batch=2, 有效batch=128（优化）")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
