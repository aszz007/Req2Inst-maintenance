"""Regenerate training-curve plots from saved histories."""

import json
import math
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger('utils.replot_training_curves')


SHORT_HISTORY_THRESHOLD = 10
SPARSE_CURVE_POINT_THRESHOLD = 3


def _is_valid_metric(value):
    return value is not None and not (
        isinstance(value, float) and math.isnan(value)
    )


def _collect_training_curve_data(training_history):
    curves = {
        'loss': ([], []),
        'eval_loss': ([], []),
        'grad_norm': ([], []),
        'learning_rate': ([], []),
    }

    for entry in training_history:
        step = entry.get('step', 0)

        if 'loss' in entry:
            loss_val = entry['loss']
            if _is_valid_metric(loss_val):
                curves['loss'][0].append(step)
                curves['loss'][1].append(loss_val)

        if 'eval_loss' in entry:
            eval_val = entry['eval_loss']
            if _is_valid_metric(eval_val):
                curves['eval_loss'][0].append(step)
                curves['eval_loss'][1].append(eval_val)

        if 'grad_norm' in entry:
            grad_val = entry['grad_norm']
            if _is_valid_metric(grad_val):
                curves['grad_norm'][0].append(step)
                curves['grad_norm'][1].append(grad_val)

        if 'learning_rate' in entry:
            lr_val = entry['learning_rate']
            if _is_valid_metric(lr_val):
                curves['learning_rate'][0].append(step)
                curves['learning_rate'][1].append(lr_val)

    nan_eval_count = sum(
        1
        for entry in training_history
        if 'eval_loss' in entry
        and not _is_valid_metric(entry['eval_loss'])
    )
    return curves, nan_eval_count


def _log_curve_data_quality(training_history, curves, nan_eval_count):
    total_entries = len(training_history)
    losses = curves['loss'][1]
    eval_losses = curves['eval_loss'][1]

    if total_entries < SHORT_HISTORY_THRESHOLD:
        logger.warning(
            f"Training history has only {total_entries} entries, possibly due to early stopping"
        )

    if len(losses) < SPARSE_CURVE_POINT_THRESHOLD:
        logger.warning(f"Training loss has only {len(losses)} data points")
    if not eval_losses:
        if nan_eval_count > 0:
            logger.warning(
                f"All {nan_eval_count} validation loss values are NaN and were filtered; the validation curve cannot be plotted"
            )
            logger.warning("This may indicate unstable training; consider:")
            logger.warning("  1. Reducing the learning rate, which may be too high")
            logger.warning(
                "  2. Adjusting the parameter-efficient fine-tuning configuration"
            )
            logger.warning("  3. Checking dataset quality and preprocessing")
        else:
            logger.warning("No validation loss data found")
    elif len(eval_losses) < SPARSE_CURVE_POINT_THRESHOLD:
        if nan_eval_count > 0:
            logger.warning(
                f"Validation loss has only {len(eval_losses)} valid data points; {nan_eval_count} NaN values were filtered"
            )
        else:
            logger.warning(
                f"Validation loss has only {len(eval_losses)} data points"
            )
    elif nan_eval_count > 0:
        logger.info(
            f"Filtered {nan_eval_count} NaN validation loss values; {len(eval_losses)} valid values remain"
        )


def _plot_training_loss(axis, loss_steps, losses):
    if losses:
        axis.plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
        axis.set_xlabel('Step')
        axis.set_ylabel('Loss')
        axis.set_title('Training Loss')
        axis.grid(True, alpha=0.3)
    else:
        axis.text(
            0.5,
            0.5,
            'No training loss data',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Loss')
        axis.set_title('Training Loss')


def _plot_eval_loss(axis, eval_steps, eval_losses):
    if eval_losses:
        axis.plot(
            eval_steps,
            eval_losses,
            'r-',
            linewidth=2,
            marker='o',
            markersize=4,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Eval Loss')
        axis.set_title('Validation Loss')
        axis.grid(True, alpha=0.3)
    else:
        axis.text(
            0.5,
            0.5,
            'No validation loss data',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Eval Loss')
        axis.set_title('Validation Loss')


def _plot_grad_norm(axis, grad_norm_steps, grad_norms):
    if grad_norms:
        axis.plot(
            grad_norm_steps,
            grad_norms,
            'g-',
            linewidth=1,
            alpha=0.6,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Gradient Norm')
        axis.set_title('Gradient Norm')
        axis.grid(True, alpha=0.3)
    else:
        axis.text(
            0.5,
            0.5,
            'No gradient norm data',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Gradient Norm')
        axis.set_title('Gradient Norm')


def _plot_learning_rate(axis, lr_steps, learning_rates):
    if learning_rates:
        axis.plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
        axis.set_xlabel('Step')
        axis.set_ylabel('Learning Rate')
        axis.set_title('Learning Rate Schedule')
        axis.grid(True, alpha=0.3)
        axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        axis.text(
            0.5,
            0.5,
            'No learning rate data',
            ha='center',
            va='center',
            transform=axis.transAxes,
        )
        axis.set_xlabel('Step')
        axis.set_ylabel('Learning Rate')
        axis.set_title('Learning Rate Schedule')


def plot_training_curves(training_history, expert_type, method_name, output_path):
    """Plot training curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed; visualizations cannot be generated")
        return False

    curves, nan_eval_count = _collect_training_curve_data(training_history)
    _log_curve_data_quality(training_history, curves, nan_eval_count)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f'Training Curves - {expert_type.upper()} Expert ({method_name})',
        fontsize=16,
        fontweight='bold',
    )

    _plot_training_loss(axes[0, 0], *curves['loss'])
    _plot_eval_loss(axes[0, 1], *curves['eval_loss'])
    _plot_grad_norm(axes[1, 0], *curves['grad_norm'])
    _plot_learning_rate(axes[1, 1], *curves['learning_rate'])

    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    losses = curves['loss'][1]
    eval_losses = curves['eval_loss'][1]
    grad_norms = curves['grad_norm'][1]
    learning_rates = curves['learning_rate'][1]
    logger.info(f"Training curves saved to: {output_path}")
    logger.info(
        f"Data summary: Loss={len(losses)} points, EvalLoss={len(eval_losses)} points, GradNorm={len(grad_norms)} points, LR={len(learning_rates)} points"
    )
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
        logger.error(f"File not found: {history_path}")
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
            logger.info(f"Inferred method name from path: {method_name}")

        training_history = history_data.get('history', [])

        if not training_history:
            logger.error(f"Training history is empty: {history_path}")
            return False

        logger.info(f"Replotting curves for {expert_type} expert with {method_name}")
        logger.info(f"Training steps: {len(training_history)}")

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
            logger.info(f"Plot saved: {output_path}")
            return True
        else:
            logger.error(f"Failed to plot: {history_path}")
            return False

    except Exception as e:
        logger.error(f"Failed to process {history_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def replot_all_histories():
    """Replot all saved training histories."""
    checkpoints_dir = PROJECT_ROOT / 'checkpoints'

    if not checkpoints_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoints_dir}")
        return 0, 0

    history_files = list(checkpoints_dir.glob('**/training_history.json'))

    if not history_files:
        logger.warning("No training_history.json files found")
        return 0, 0

    logger.info(f"Found {len(history_files)} training history files")

    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = PROJECT_ROOT / 'outputs' / 'training_curves'
    output_timestamp_dir = base_dir / timestamp
    output_timestamp_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Batch output directory: {output_timestamp_dir}")

    success_count = 0
    fail_count = 0

    for i, history_file in enumerate(history_files, 1):
        logger.info(f"[{i}/{len(history_files)}] Processing: {history_file.relative_to(PROJECT_ROOT)}")

        if replot_single_history(history_file, output_timestamp_dir):
            success_count += 1
        else:
            fail_count += 1


    logger.info(f"Batch processing completed: {success_count} succeeded, {fail_count} failed")
    logger.info(f"All plots saved to: {output_timestamp_dir}")

    return success_count, fail_count


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Replot training curves from a saved training_history.json'
    )
    parser.add_argument(
        'history_file',
        nargs='?',
        help='Path to training_history.json'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all training histories in the checkpoints directory'
    )

    args = parser.parse_args()

    if args.all:
        logger.info("Batch mode: processing all training history files")
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
