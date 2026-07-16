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
    print("=" * 80)
    print(" " * 15 + "P-Tuning v2 Text Expert训练 (Prefix Tuning)")
    print("=" * 80)
    print()


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='使用P-Tuning v2训练Text Expert')
    parser.add_argument('--use_4bit', action='store_true', dest='use_4bit',
                        help='使用4bit量化训练')
    parser.add_argument('--no_4bit', action='store_false', dest='use_4bit',
                        help='不使用4bit量化')
    parser.set_defaults(use_4bit=True)
    args = parser.parse_args()

    print_header()

    is_rtx4090 = detect_rtx4090()
    use_rtx4090_opt = is_rtx4090

    if is_rtx4090:
        logger.info("检测到RTX 4090，启用优化配置")

    ptuning_cfg = get_ptuning_config()
    print("=" * 80)
    print("对比实验：P-Tuning v2 vs LoRA")
    print("=" * 80)
    print("方法：P-Tuning v2（Prefix Tuning）")
    print("配置：")
    print(f"  - Virtual Tokens: {ptuning_cfg.num_virtual_tokens}")
    print(f"  - Encoder Hidden Size: {ptuning_cfg.encoder_hidden_size}")
    print(f"  - Prefix Projection: {ptuning_cfg.prefix_projection}")
    print("=" * 80)
    print()

    logger.info("创建P-Tuning v2文本专家训练器...")
    try:
        trainer = PTuningTrainer(
            expert_type='text',
            use_4bit=args.use_4bit,
            use_rtx4090_optimization=use_rtx4090_opt
        )
    except Exception as e:
        logger.error(f"创建训练器失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("设置模型和P-Tuning v2配置...")
    if not trainer.setup_model():
        logger.error("模型设置失败")
        return 1

    logger.info("准备训练数据...")
    if not trainer.prepare_data():
        logger.error("数据准备失败")
        return 1

    status = trainer.get_training_status()
    print(f"数据统计:")
    print(f"  - 训练样本: {status['train_samples']}")
    print(f"  - 验证样本: {status['val_samples']}")
    print()

    logger.info("开始训练...")
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
        output_path = path_cfg.PTUNING_CKPTS['text']
        print(f"P-Tuning v2权重已保存至: {output_path}")
        print(f"检查点目录: {output_path / 'training_checkpoints'}")
        print()
        print("下一步:")
        print("  1. 可以使用该权重进行推理测试")
        print("  2. 继续训练其他专家（Image, UML, General）")
        print()

        return 0
    else:
        print()
        print("=" * 80)
        print(" " * 28 + "训练失败")
        print("=" * 80)
        print()
        logger.error("训练过程中出现错误，请查看日志")
        return 1


if __name__ == "__main__":
    sys.exit(main())