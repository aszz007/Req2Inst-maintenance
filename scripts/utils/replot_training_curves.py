"""Regenerate training-curve plots from saved histories."""

import json
import math
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

logger = get_logger('utils.replot_training_curves')


def plot_training_curves(training_history, expert_type, method_name, output_path):
    """Plot training curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib未安装，无法生成可视化")
        return False

    loss_steps = []
    losses = []
    eval_steps = []
    eval_losses = []
    grad_norm_steps = []
    grad_norms = []
    lr_steps = []
    learning_rates = []

    for entry in training_history:
        step = entry.get('step', 0)

        if 'loss' in entry:
            loss_val = entry['loss']
            if loss_val is not None and not (isinstance(loss_val, float) and math.isnan(loss_val)):
                loss_steps.append(step)
                losses.append(loss_val)

        if 'eval_loss' in entry:
            eval_val = entry['eval_loss']
            if eval_val is not None and not (isinstance(eval_val, float) and math.isnan(eval_val)):
                eval_steps.append(step)
                eval_losses.append(eval_val)

        if 'grad_norm' in entry:
            grad_val = entry['grad_norm']
            if grad_val is not None and not (isinstance(grad_val, float) and math.isnan(grad_val)):
                grad_norm_steps.append(step)
                grad_norms.append(grad_val)

        if 'learning_rate' in entry:
            lr_val = entry['learning_rate']
            if lr_val is not None and not (isinstance(lr_val, float) and math.isnan(lr_val)):
                lr_steps.append(step)
                learning_rates.append(lr_val)

    total_entries = len(training_history)
    nan_eval_count = sum(1 for e in training_history if 'eval_loss' in e and
                        (e['eval_loss'] is None or (isinstance(e['eval_loss'], float) and math.isnan(e['eval_loss']))))

    if total_entries < 10:
        logger.warning(f"训练历史记录很少（{total_entries}条），可能因早停提前结束")

    if len(losses) < 3:
        logger.warning(f"训练损失数据点很少（{len(losses)}个）")
    if len(eval_losses) == 0:
        if nan_eval_count > 0:
            logger.warning(f"所有{nan_eval_count}个验证损失都是NaN（已过滤），无法绘制验证曲线")
            logger.warning("这表明训练过程不稳定，建议:")
            logger.warning("  1. 降低学习率（当前可能过大）")
            logger.warning("  2. 调整参数高效微调配置")
            logger.warning("  3. 检查数据集质量和预处理")
        else:
            logger.warning("没有验证损失数据")
    elif len(eval_losses) < 3:
        if nan_eval_count > 0:
            logger.warning(f"验证损失数据点很少（{len(eval_losses)}个有效，{nan_eval_count}个NaN已过滤）")
        else:
            logger.warning(f"验证损失数据点很少（{len(eval_losses)}个）")
    elif nan_eval_count > 0:
        logger.info(f"过滤了{nan_eval_count}个NaN验证损失，保留{len(eval_losses)}个有效值")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {expert_type.upper()} Expert ({method_name})',
                 fontsize=16, fontweight='bold')

    # 1. Training Loss
    if losses:
        axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training loss data',
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')

    # 2. Eval Loss
    if eval_losses:
        axes[0, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='o', markersize=4)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Eval Loss')
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No validation loss data',
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Eval Loss')
        axes[0, 1].set_title('Validation Loss')

    # 3. Gradient Norm
    if grad_norms:
        axes[1, 0].plot(grad_norm_steps, grad_norms, 'g-', linewidth=1, alpha=0.6)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No gradient norm data',
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].set_title('Gradient Norm')

    # 4. Learning Rate
    if learning_rates:
        axes[1, 1].plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        axes[1, 1].text(0.5, 0.5, 'No learning rate data',
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"训练曲线已保存至: {output_path}")
    logger.info(f"数据统计: Loss={len(losses)}点, EvalLoss={len(eval_losses)}点, GradNorm={len(grad_norms)}点, LR={len(learning_rates)}点")
    return True


def infer_method_from_path(history_path):
    """Infer the training method from a path."""
    path_str = str(history_path)

    if 'lora_moe' in path_str or 'lora-moe' in path_str:
        return 'lora_moe'
    elif 'lora_single' in path_str or 'lora-single' in path_str:
        return 'lora_single'
    elif 'p_tuning' in path_str or 'p-tuning' in path_str:
        return 'p_tuning'
    elif 'prompt_tuning' in path_str or 'prompt-tuning' in path_str:
        return 'prompt_tuning'
    elif 'full_finetuning' in path_str or 'full-finetuning' in path_str:
        return 'full_finetuning'
    else:
        return 'unknown'


def replot_single_history(history_file, output_timestamp_dir=None):
    """Replot one saved training history."""
    history_path = Path(history_file)

    if not history_path.exists():
        logger.error(f"文件不存在: {history_path}")
        return False

    try:
        with open(history_path, 'r') as f:
            content = f.read()
            content = content.replace(': NaN', ': null')
            history_data = json.loads(content)

        expert_type = history_data.get('expert_type', 'unknown')
        method_name = history_data.get('method_name', None)

        if not method_name or method_name == 'unknown':
            method_name = infer_method_from_path(history_path)
            logger.info(f"从路径推断方法名: {method_name}")

        training_history = history_data.get('history', [])

        if not training_history:
            logger.error(f"训练历史为空: {history_path}")
            return False

        logger.info(f"正在重新绘制曲线: {expert_type} expert, {method_name} method")
        logger.info(f"训练步数: {len(training_history)}")

        if output_timestamp_dir:
            method_dir = output_timestamp_dir / method_name
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
            timestamp_dir = base_dir / timestamp
            method_dir = timestamp_dir / method_name

        method_dir.mkdir(parents=True, exist_ok=True)

        output_path = method_dir / f'{expert_type}_expert.png'

        success = plot_training_curves(training_history, expert_type, method_name, output_path)

        if success:
            logger.info(f"成功: {output_path}")
            return True
        else:
            logger.error(f"绘制失败: {history_path}")
            return False

    except Exception as e:
        logger.error(f"处理失败 {history_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def replot_all_histories():
    """Replot all saved training histories."""
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'

    if not checkpoints_dir.exists():
        logger.error(f"checkpoints目录不存在: {checkpoints_dir}")
        return 0, 0

    history_files = list(checkpoints_dir.glob('**/training_history.json'))

    if not history_files:
        logger.warning("未找到任何training_history.json文件")
        return 0, 0

    logger.info(f"找到 {len(history_files)} 个训练历史文件")
    logger.info("=" * 80)

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
    output_timestamp_dir = base_dir / timestamp
    output_timestamp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"批量输出目录: {output_timestamp_dir}")
    logger.info("=" * 80)

    success_count = 0
    fail_count = 0

    for i, history_file in enumerate(history_files, 1):
        logger.info(f"[{i}/{len(history_files)}] 处理: {history_file.relative_to(PROJECT_ROOT)}")

        if replot_single_history(history_file, output_timestamp_dir):
            success_count += 1
        else:
            fail_count += 1

        logger.info("-" * 80)

    logger.info("=" * 80)
    logger.info(f"批量处理完成: 成功 {success_count}, 失败 {fail_count}")
    logger.info(f"所有图片已保存至: {output_timestamp_dir}")
    logger.info("=" * 80)

    return success_count, fail_count


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description='从已保存的training_history.json重新绘制训练曲线'
    )
    parser.add_argument(
        'history_file',
        nargs='?',
        help='training_history.json文件路径'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='批量处理checkpoints目录下的所有训练历史'
    )

    args = parser.parse_args()

    if args.all:
        logger.info("批量处理模式：处理所有训练历史文件")
        success, fail = replot_all_histories()
        sys.exit(0 if fail == 0 else 1)
    elif args.history_file:
        success = replot_single_history(args.history_file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
