"""Train the unified expert."""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.lora_trainer import LoRATrainer
from config.settings import (
    get_path_config,
    get_training_config,
    get_lora_config
)
from src.utils.logger import get_logger

logger = get_logger('training.lora_single.unified_expert')


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
    print(" " * 18 + "LoRA-Single统一模型训练 (Unified Expert Training)")
    print("=" * 80)
    print()


def validate_environment():
    """Validate environment."""
    print("验证运行环境...")
    print("-" * 80)

    try:
        import transformers
        version = transformers.__version__
        print(f"Transformers版本: {version}")

        try:
            v_parts = version.split('.')
            major, minor = int(v_parts[0]), int(v_parts[1])
            if not (major > 4 or (major == 4 and minor >= 51)):
                logger.warning(f"Current transformers version is {version}; version >=4.51.0 is recommended")
                logger.warning("Verify that the script is running in the instruction_generator environment")
        except (ValueError, IndexError):
            logger.warning(f"Unable to parse transformers version: {version}")
    except ImportError:
        logger.error("transformers is not installed")
        return False

    try:
        import peft
        print(f"PEFT版本: {peft.__version__}")
    except ImportError:
        logger.error("PEFT is not installed. Run: pip install peft --break-system-packages")
        return False

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f}GB")
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
    parser = argparse.ArgumentParser(description='训练LoRA-Single统一模型')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                        help='使用4bit量化训练（默认：True）')
    parser.add_argument('--no_4bit', dest='use_4bit', action='store_false',
                        help='不使用4bit量化')
    args = parser.parse_args()

    print_header()

    if not validate_environment():
        logger.error("Environment validation failed; check the required dependencies")
        return 1

    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090

    if is_rtx4090:
        logger.info("Detected RTX 4090; enabling optimized settings")

    print("=" * 80)
    print("对比实验：LoRA-Single vs LoRA-MoE")
    print("=" * 80)
    print("目标：验证MoE架构相比单一模型的优势")
    print("配置：")
    print("  - 使用exp4最优LoRA超参数（rank=64, alpha=128, dropout=0.05）")
    print("  - 使用相同的训练数据集（text + image + uml）")
    print("  - 唯一区别：无MoE路由机制")
    print("预期：LoRA-MoE通过专家专业化达到更好的性能")
    print("=" * 80)
    print()

    logger.info("Creating LoRA-Single unified model trainer...")
    try:
        path_cfg = get_path_config()
        trainer = LoRATrainer(
            expert_type='general',
            method_name='lora_single',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt,
            use_domain_templates=True
        )

        trainer.output_dir = path_cfg.LORA_SINGLE_CKPT
        trainer.checkpoint_dir = path_cfg.LORA_SINGLE_CKPT / 'training_checkpoints'

    except Exception as e:
        logger.error(f"Failed to create trainer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Preparing training data...")
    if not trainer.prepare_data():
        logger.error("Training data preparation failed")
        return 1

    status = trainer.get_training_status()
    print(f"数据统计:")
    print(f"  - 训练样本: {status['train_samples']}")
    print(f"  - 验证样本: {status['val_samples']}")
    print(f"  - 数据来源: text + image + uml（混合数据集）")
    print()

    logger.info("Setting up model and LoRA configuration...")
    if not trainer.setup_model():
        logger.error("Model setup failed")
        return 1

    logger.info("Starting training...")
    print("=" * 80)
    print("训练开始 - 这可能需要较长时间，请耐心等待...")
    print("=" * 80)
    print()

    success = trainer.train()

    if success:
        print()
        print("=" * 80)
        print(" " * 25 + "训练成功完成！")
        print("=" * 80)
        print()

        path_cfg = get_path_config()
        print(f"LoRA权重已保存至: {path_cfg.LORA_SINGLE_CKPT}")
        print()
        print("下一步:")
        print("  1. 使用该权重进行推理测试")
        print("  2. 与LoRA-MoE进行性能对比实验")
        print("  3. 分析单一模型 vs MoE架构的差异")
        print()

        return 0
    else:
        print()
        print("=" * 80)
        print(" " * 28 + "训练失败")
        print("=" * 80)
        print()
        logger.error("An error occurred during training; check the logs")
        return 1



if __name__ == "__main__":
    sys.exit(main())
