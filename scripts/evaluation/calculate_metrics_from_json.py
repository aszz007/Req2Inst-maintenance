"""Calculate evaluation metrics from saved JSON predictions."""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_metrics import EnhancedMetrics, EvaluationThresholds
from src.utils.logger import get_logger

logger = get_logger('evaluation.calculate_metrics_from_json')


def load_predictions_json(filepath: str) -> Dict:
    """Load predictions JSON."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading prediction data: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = data.get('samples', [])

    if not samples:
        raise ValueError("The JSON file does not contain sample data")

    inputs = [s['input'] for s in samples]
    predictions = [s['prediction'] for s in samples]
    references = [s['reference'] for s in samples]

    logger.info(f"Loaded {len(samples)} samples")
    logger.info(f"Expert: {data.get('expert_name', 'unknown')}")
    logger.info(f"Timestamp: {data.get('timestamp', 'unknown')}")

    return {
        'expert_name': data.get('expert_name', 'unknown'),
        'original_timestamp': data.get('timestamp', 'unknown'),
        'inputs': inputs,
        'predictions': predictions,
        'references': references
    }


def calculate_metrics(
        predictions: List[str],
        references: List[str],
        use_bertscore: bool = True,
        rouge_threshold: float = None,
        bertscore_threshold: float = None,
        use_and_logic: bool = None,
        format_threshold: float = None
) -> Dict:
    """Calculate metrics."""
    logger.info("Starting metric computation")

    metrics = EnhancedMetrics(use_bertscore=use_bertscore)

    valid_pairs = [
        (pred, ref) for pred, ref in zip(predictions, references)
        if pred.strip()
    ]

    if not valid_pairs:
        logger.error("No valid predictions found")
        return {}

    valid_predictions = [pair[0] for pair in valid_pairs]
    valid_references = [pair[1] for pair in valid_pairs]

    logger.info(f"Valid samples: {len(valid_predictions)}/{len(predictions)}")

    logger.info("\n[1/4] Computing generation quality metrics...")
    quality_metrics = metrics.calculate_generation_quality(
        predictions=valid_predictions,
        references=valid_references
    )

    logger.info("\n[2/4] Computing format metrics...")
    format_metrics = metrics.calculate_format_metrics(
        instructions=valid_predictions
    )

    logger.info("\n[3/4] Computing binary classification metrics...")
    binary_metrics = metrics.calculate_binary_classification_metrics(
        predictions=valid_predictions,
        references=valid_references,
        format_threshold=format_threshold,
        rouge_threshold=rouge_threshold,
        bertscore_threshold=bertscore_threshold,
        use_and_logic=use_and_logic
    )

    logger.info("\n[4/4] Computing summary statistics...")
    statistical_metrics = metrics.calculate_statistical_metrics(
        instructions=valid_predictions
    )

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(predictions),
        'valid_samples': len(valid_predictions),
        'generation_quality': quality_metrics,
        'format_metrics': format_metrics,
        'binary_classification': binary_metrics,
        'statistical_metrics': statistical_metrics,
        'threshold_config': EvaluationThresholds.get_config()
    }

    logger.info("Metric computation completed")

    return results


def print_metrics_summary(results: Dict, expert_name: str):
    """Print metrics summary."""
    print("\n" + "=" * 80)
    print(f"Evaluation Summary - {expert_name}")
    print("=" * 80)

    print("\n[Generation Quality Metrics]")
    quality = results['generation_quality']
    print(f"  BLEU:        {quality['bleu']:.4f}")
    print(f"  ROUGE-L:     {quality['rougeL']:.4f}")
    print(f"  METEOR:      {quality['meteor']:.4f}")
    if 'bertscore_f1' in quality:
        print(f"  BERTScore F1: {quality['bertscore_f1']:.4f}")

    print("\n[Format Metrics]")
    format_m = results['format_metrics']
    print(f"  Format score:    {format_m['avg_format_score']:.4f}")
    print(f"  Pass rate:      {format_m['valid_rate']:.2%}")

    print("\n[Binary Classification Metrics]")
    binary = results['binary_classification']
    print(f"  Precision:   {binary['precision']:.4f}")
    print(f"  Recall:      {binary['recall']:.4f}")
    print(f"  F1 Score:    {binary['f1_score']:.4f}")
    print(f"  TP: {binary['TP']:<6d}  FP: {binary['FP']:<6d}  FN: {binary['FN']:<6d}")

    print("\n[Threshold Configuration]")
    print(f"  ROUGE-L threshold:      {binary['rouge_threshold']:.2f}")
    print(f"  BERTScore threshold:    {binary['bertscore_threshold']:.2f}")
    print(f"  Combination logic:         {'AND (both must pass)' if binary['use_and_logic'] else 'OR (either may pass)'}")
    print(f"  Format score threshold:     {binary['format_threshold']:.2f}")

    print("=" * 80 + "\n")


def save_results(results: Dict, expert_name: str, save_dir: str):
    """Save results."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{expert_name}_metrics_{timestamp}.json'
    filepath = save_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to: {filepath}")


def main_single(args):
    """
    Original single-file metric computation logic.
    Accepts a pre-parsed args namespace so it can be called from the unified
    __main__ entry point without re-parsing sys.argv.
    """
    try:
        data = load_predictions_json(args.input)
    except Exception as e:
        logger.error(f"Failed to load prediction data: {e}")
        sys.exit(1)

    try:
        results = calculate_metrics(
            predictions=data['predictions'],
            references=data['references'],
            use_bertscore=args.use_bertscore,
            rouge_threshold=args.rouge_threshold,
            bertscore_threshold=args.bertscore_threshold,
            use_and_logic=args.use_and_logic,
            format_threshold=args.format_threshold
        )
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    results['expert_name'] = data['expert_name']
    results['original_timestamp'] = data['original_timestamp']
    results['input_file'] = args.input

    print_metrics_summary(results, data['expert_name'])

    save_results(results, data['expert_name'], args.save_dir)

    logger.info("Completed")


def main():
    """
    Original entry point — kept for backward compatibility.
    Parses its own args and delegates to main_single().
    """
    import argparse as _argparse
    parser = _argparse.ArgumentParser(
        description='Quickly recompute evaluation metrics from prediction JSON',
        formatter_class=_argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default thresholds
  python calculate_metrics_from_json.py --input predictions.json

  # Adjust the ROUGE threshold
  python calculate_metrics_from_json.py --input predictions.json --rouge-threshold 0.6

  # Use OR logic
  python calculate_metrics_from_json.py --input predictions.json --use-or

  # Disable BERTScore for faster computation
  python calculate_metrics_from_json.py --input predictions.json --no-bertscore
        """
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to the prediction data JSON file')
    parser.add_argument('--save-dir', '-o', type=str, default='outputs/evaluations/metrics',
                        help='Output directory')
    parser.add_argument('--rouge-threshold', type=float, default=None,
                        help=f'ROUGE-L threshold (default: {EvaluationThresholds.ROUGE_L_THRESHOLD})')
    parser.add_argument('--bertscore-threshold', type=float, default=None,
                        help=f'BERTScore F1 threshold (default: {EvaluationThresholds.BERTSCORE_F1_THRESHOLD})')
    parser.add_argument('--format-threshold', type=float, default=None,
                        help=f'Format score threshold (default: {EvaluationThresholds.FORMAT_SCORE_THRESHOLD})')
    logic_group = parser.add_mutually_exclusive_group()
    logic_group.add_argument('--use-and', dest='use_and_logic', action='store_true',
                             help='Combine ROUGE and BERTScore with AND logic (default)')
    logic_group.add_argument('--use-or', dest='use_and_logic', action='store_false',
                             help='Combine ROUGE and BERTScore with OR logic')
    parser.set_defaults(use_and_logic=None)
    parser.add_argument('--use-bertscore', action='store_true', default=True,
                        help='Enable BERTScore (enabled by default)')
    parser.add_argument('--no-bertscore', dest='use_bertscore', action='store_false',
                        help='Disable BERTScore (faster computation)')
    args = parser.parse_args()
    main_single(args)


# Batch mode extensions (Phase 2 addition)
# Do NOT modify anything above this line.

def scan_cache_files(cache_dir: Path) -> List[Path]:
    """
    Recursively find all *_predictions.json files under cache_dir.

    Args:
        cache_dir: Root directory to search

    Returns:
        List of matching Path objects
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        logger.warning(f'Cache directory not found: {cache_dir}')
        return []
    return sorted(cache_dir.rglob('*_predictions.json'))


def main_batch(args):
    """
    Handle --list, --exp, --all batch metric recomputation modes.

    Args:
        args: Parsed argparse namespace containing batch-mode flags
    """
    cache_dir = project_root / 'outputs' / 'inference_cache'

    if args.list_caches:
        files = scan_cache_files(cache_dir)
        if not files:
            print(f"No cache file found in directory: {cache_dir}")
        for f in files:
            try:
                print(f.relative_to(cache_dir))
            except ValueError:
                print(f)
        return

    files = scan_cache_files(cache_dir)

    if args.exp:
        EXP_CACHE_MAP = {
            'exp1': ['baselines', 'lora_moe'],
            'exp2': ['lora_moe', 'lora_single', 'p_tuning', 'prompt_tuning', 'full_finetuning'],
            'exp3': ['lora_moe', 'lora_single', 'exp3_cross_domain'],
            'exp4': ['lora_moe_exp4'],
            'exp5': ['lora_moe_exp5', 'lora_single_exp5', 'full_finetuning_exp5'],
            'exp6': ['few_shot', 'lora_moe'],
        }
        valid_dirs = EXP_CACHE_MAP.get(args.exp, [])
        if not valid_dirs:
            logger.error(f"Unknown experiment: {args.exp}. Valid values: {list(EXP_CACHE_MAP.keys())}")
            return
        files = [f for f in files if any(d in str(f) for d in valid_dirs)]
        logger.info(f"Filtered to {len(files)} files for experiment {args.exp}")

    if args.method:
        files = [f for f in files if args.method in str(f)]
        logger.info(f"Filtered to {len(files)} files for method '{args.method}'")

    if not files:
        logger.warning("No matching cache files found")
        return

    logger.info(f"Processing {len(files)} cache files...")
    save_dir = args.save_dir or 'outputs/evaluations/metrics'
    success_count = 0
    fail_count = 0

    for filepath in sorted(files):
        logger.info(f"\n--- Processing: {filepath} ---")
        try:
            data = load_predictions_json(str(filepath))
            results = calculate_metrics(
                data['predictions'],
                data['references'],
                use_bertscore=not args.no_bertscore
            )
            results['expert_name'] = data.get('expert_name', 'unknown')
            results['input_file'] = str(filepath)
            print_metrics_summary(results, data.get('expert_name', 'unknown'))
            save_results(results, data.get('expert_name', 'unknown'), save_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            fail_count += 1

    logger.info(f"\nBatch processing completed: {success_count} succeeded, {fail_count} failed")


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        description='Quickly recompute evaluation metrics from prediction JSON (supports batch mode)',
        formatter_class=_argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file mode (original behavior):
  python calculate_metrics_from_json.py --input predictions.json

  # List all available cache files:
  python calculate_metrics_from_json.py --list

  # Recompute metrics for all exp1 caches:
  python calculate_metrics_from_json.py --exp exp1

  # Recompute all caches, filter by method:
  python calculate_metrics_from_json.py --all --method lora_moe

  # Disable BERTScore for faster computation:
  python calculate_metrics_from_json.py --all --no-bertscore
        """
    )

    # Single-file flags (--input now optional when batch flags are used)
    parser.add_argument('--input', '-i', type=str, required=False,
                        help='Path to the prediction data JSON file (single-file mode)')
    parser.add_argument('--save-dir', '-o', type=str, default='outputs/evaluations/metrics',
                        help='Output directory')

    # Threshold flags (used by single-file mode)
    parser.add_argument('--rouge-threshold', type=float, default=None,
                        help=f'ROUGE-L threshold (default: {EvaluationThresholds.ROUGE_L_THRESHOLD})')
    parser.add_argument('--bertscore-threshold', type=float, default=None,
                        help=f'BERTScore F1 threshold (default: {EvaluationThresholds.BERTSCORE_F1_THRESHOLD})')
    parser.add_argument('--format-threshold', type=float, default=None,
                        help=f'Format score threshold (default: {EvaluationThresholds.FORMAT_SCORE_THRESHOLD})')

    logic_group = parser.add_mutually_exclusive_group()
    logic_group.add_argument('--use-and', dest='use_and_logic', action='store_true',
                             help='Combine ROUGE and BERTScore with AND logic (default)')
    logic_group.add_argument('--use-or', dest='use_and_logic', action='store_false',
                             help='Combine ROUGE and BERTScore with OR logic')
    parser.set_defaults(use_and_logic=None)

    parser.add_argument('--use-bertscore', action='store_true', default=True,
                        help='Enable BERTScore (enabled by default)')
    parser.add_argument('--no-bertscore', dest='use_bertscore', action='store_false',
                        help='Disable BERTScore (faster computation)')

    # Batch mode flags
    parser.add_argument('--list', dest='list_caches', action='store_true',
                        help='List all available inference cache files')
    parser.add_argument('--exp', type=str, default=None,
                        help='Compute metrics for all caches of a given experiment '
                             '(exp1, exp2, exp3, exp4, exp5, exp6)')
    parser.add_argument('--all', dest='compute_all', action='store_true',
                        help='Compute metrics for all available cache files')
    parser.add_argument('--method', type=str, default=None,
                        help='Filter cache files by method name (used with --exp or --all)')

    args = parser.parse_args()

    if args.list_caches or args.exp or args.compute_all:
        main_batch(args)
    elif args.input:
        main_single(args)
    else:
        parser.error('--input is required unless --list, --exp, or --all is specified')
