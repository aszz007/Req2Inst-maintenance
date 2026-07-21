#!/usr/bin/env python3
"""
Experiment 6: Few-Shot vs Fine-Tuning

Compare zero/few-shot prompting with base Qwen3-8B against fine-tuned Multi-Expert LoRA
on the text expert test set.

Configurations:
  - 0-shot  (3 runs: seed 42, 43, 44 for zero-shot consistency)
  - 1-shot  (1 run)
  - 3-shot  (1 run)
  - 5-shot  (3 runs: seed 42, 43, 44)
  - Multi-Expert LoRA fine-tuned text expert

Output: outputs/evaluations/experiments/exp6_fewshot_learning/
"""

import sys
import random
import traceback
import argparse
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config.settings import get_path_config
from src.training.data_loader import TextDatasetLoader, split_dataset_for_expert
from src.baselines.zero_shot import ZeroShotGenerator
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp6')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'few_shot'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp6_fewshot_learning'

# (n_shots, run_ids) — run_id=1 uses seed 42, run_id=2 uses seed 43, run_id=3 uses seed 44
SHOT_CONFIGS = [
    (0, [1, 2, 3]),
    (1, [1]),
    (3, [1]),
    (5, [1, 2, 3]),
]

SEED_MAP = {1: 42, 2: 43, 3: 44}


def _is_full_run_cache(cache_dir, filename):
    """Return True if a non-test-mode cache file exists for this combination."""
    import json as _json
    filepath = Path(cache_dir) / filename
    if not filepath.exists():
        return False
    try:
        raw = _json.loads(filepath.read_text(encoding='utf-8'))
        return not (
            raw.get('test_mode', False)
            or raw.get('metadata', {}).get('test_mode', False)
        )
    except Exception:
        return False


def _cache_filename(n_shots, run_id):
    return f'{n_shots}shot_text_run{run_id}_predictions.json'


def _sample_examples(train_data, n, seed):
    """Sample n diverse training examples using different source files."""
    random.seed(seed)

    # Try to pick from different source CSV files for diversity
    by_source = {}
    for item in train_data:
        src = item.get('source', 'unknown')
        by_source.setdefault(src, []).append(item)

    selected = []
    sources = list(by_source.keys())
    random.shuffle(sources)

    # Round-robin across sources
    while len(selected) < n:
        for src in sources:
            if len(selected) >= n:
                break
            pool = by_source[src]
            if pool:
                idx = random.randint(0, len(pool) - 1)
                selected.append(pool.pop(idx))

    # Fallback if not enough sources
    remaining = n - len(selected)
    if remaining > 0:
        all_items = [item for items in by_source.values() for item in items]
        selected += random.sample(all_items, min(remaining, len(all_items)))

    examples = [{'input': item['input'], 'output': item['output']} for item in selected[:n]]
    return examples


def run_few_shot(n_shots, run_id, train_data, test_data, generator, args):
    """Run few shot."""
    filename = _cache_filename(n_shots, run_id)
    cached = load_predictions_cache(CACHE_DIR, filename)
    if cached and not args.force_regenerate:
        logger.info(f'{n_shots}-shot run {run_id}: loading from cache')
        return cached

    seed = SEED_MAP[run_id]
    examples = _sample_examples(train_data, n_shots, seed) if n_shots > 0 else []
    logger.info(f'{n_shots}-shot run {run_id}: generating with seed={seed} and {len(examples)} examples...')

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    predictions = generator.batch_generate(
        inputs, input_type='text', n_shots=n_shots, examples=examples
    )

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(
        samples, f'{n_shots}_shot', 'text',
        {'n_shots': n_shots, 'run_id': run_id, 'seed': seed},
        CACHE_DIR, filename
    )
    return load_predictions_cache(CACHE_DIR, filename)


def run_lora_moe(test_data, args):
    """Run LoRA moe."""
    cache_subdir = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'lora_moe'
    filename = 'text_predictions.json'
    cached = load_predictions_cache(cache_subdir, filename)
    if cached and not args.force_regenerate:
        logger.info('Multi-Expert LoRA: loading from cache')
        return cached

    logger.info('Multi-Expert LoRA: running text-expert inference...')
    from src.experts import TextExpert
    expert = TextExpert(lora_path=None, use_4bit=True)
    if not expert.load_model():
        return None

    inputs = [d['input'] for d in test_data]
    references = [d['output'] for d in test_data]

    if args.test_mode:
        inputs, references = inputs[:10], references[:10]

    try:
        predictions = expert.batch_generate_instruction(inputs, batch_size=4)
    finally:
        expert.unload_model()

    samples = [
        {'index': i, 'input': inp, 'prediction': pred, 'reference': ref}
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
    ]
    save_predictions_cache(samples, 'lora_moe', 'text', {}, cache_subdir, filename)
    return load_predictions_cache(cache_subdir, filename)


def compute_runs_stats(run_metrics):
    """Compute mean and std ROUGE-L across multiple runs."""
    rouge_vals = [
        m.get('generation_quality', {}).get('rougeL', 0)
        for m in run_metrics
    ]
    if not rouge_vals:
        return 0.0, 0.0
    return float(np.mean(rouge_vals)), float(np.std(rouge_vals))


def plot_bar_with_errorbars(shot_summary, lora_rougeL, exp_dir, test_mode=False):
    """Plot bar with errorbars."""
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_shot_labels = []
    means = []
    stds = []

    for n_shots in sorted(shot_summary.keys()):
        mean, std = shot_summary[n_shots]
        label = f'{n_shots}-shot'
        n_shot_labels.append(label)
        means.append(mean)
        stds.append(std)

    n_shot_labels.append('Multi-Expert LoRA')
    means.append(lora_rougeL)
    stds.append(0)

    x = np.arange(len(n_shot_labels))
    colors = ['#ff7f0e'] * (len(n_shot_labels) - 1) + ['#1f77b4']

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(n_shot_labels)
    ax.set_ylabel('ROUGE-L')
    title = 'Exp6: Few-Shot vs Fine-Tuning (ROUGE-L)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color='#ff7f0e', alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color='#1f77b4', alpha=0.85),
        ],
        labels=['Few-Shot (base model)', 'Multi-Expert LoRA (fine-tuned)']
    )
    plt.tight_layout()
    path = plots_dir / 'fewshot_vs_finetuning.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved: {path}')


def run(args):
    """Run the workflow."""
    logger.info('Experiment 6: Few-shot versus fine-tuning')

    logger.info('Loading text dataset...')
    all_data = TextDatasetLoader().load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    logger.info(f'Training samples={len(train_data)}, test samples={len(test_data)}')

    results = {
        'experiment': 'exp6_fewshot_vs_finetuning',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'shot_configs': {},
        'lora_moe': {},
    }

    generator = ZeroShotGenerator(use_4bit=True)
    if not generator.load_model():
        logger.error('Failed to load base model; few-shot generation cannot proceed')
        return

    try:
        shot_summary = {}
        all_run_metrics = {}

        for n_shots, run_ids in SHOT_CONFIGS:
            logger.info(f'\n=== {n_shots}-shot ({len(run_ids)} runs) ===')
            run_metrics = []

            for run_id in run_ids:
                if getattr(args, 'only_missing', False) and _is_full_run_cache(
                        CACHE_DIR, _cache_filename(n_shots, run_id)):
                    logger.info(f'{n_shots}-shot run {run_id}: cache exists, skipping (--only-missing)')
                    continue
                try:
                    cached = run_few_shot(n_shots, run_id, train_data, test_data, generator, args)
                    if cached is None:
                        continue

                    preds = [s['prediction'] for s in cached['samples']]
                    refs = [s['reference'] for s in cached['samples']]
                    m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)
                    run_metrics.append(m)

                    q = m.get('generation_quality', {})
                    logger.info(
                        f'{n_shots}-shot run {run_id}: ROUGE-L={q.get("rougeL", 0):.4f}'
                    )
                except Exception as e:
                    logger.error(f'{n_shots}-shot run {run_id} failed: {e}')
                    logger.error(traceback.format_exc())

            mean_rougeL, std_rougeL = compute_runs_stats(run_metrics)
            shot_summary[n_shots] = (mean_rougeL, std_rougeL)
            all_run_metrics[n_shots] = run_metrics

            per_run_results = []
            for i, m in enumerate(run_metrics):
                q = m.get('generation_quality', {})
                per_run_results.append({
                    'run_id': run_ids[i] if i < len(run_ids) else i + 1,
                    'generation_quality': q,
                    'binary_classification': m.get('binary_classification', {}),
                })
            results['shot_configs'][str(n_shots)] = {
                'n_shots': n_shots,
                'runs': per_run_results,
                'mean_rougeL': mean_rougeL,
                'std_rougeL': std_rougeL,
            }
            logger.info(f'{n_shots}-shot: mean ROUGE-L={mean_rougeL:.4f} (standard deviation={std_rougeL:.4f})')

    finally:
        generator.unload_model()

    logger.info('\n=== Multi-Expert LoRA (fine-tuned) ===')
    lora_moe_cache_subdir = path_cfg.OUTPUTS_DIR / 'inference_cache' / 'lora_moe'
    if getattr(args, 'only_missing', False) and _is_full_run_cache(
            lora_moe_cache_subdir, 'text_predictions.json'):
        logger.info('Multi-Expert LoRA: cache exists, skipping (--only-missing)')
        lora_rougeL = 0.0
    else:
        try:
            cached = run_lora_moe(test_data, args)
            if cached:
                preds = [s['prediction'] for s in cached['samples']]
                refs = [s['reference'] for s in cached['samples']]
                m = compute_all_metrics(preds, refs, use_bertscore=not args.no_bertscore)
                q = m.get('generation_quality', {})
                b = m.get('binary_classification', {})
                results['lora_moe'] = {
                    'n_samples': len(preds),
                    'generation_quality': q,
                    'binary_classification': b,
                }
                lora_rougeL = q.get('rougeL', 0)
                logger.info(f'Multi-Expert LoRA: ROUGE-L={lora_rougeL:.4f} F1={b.get("f1_score", 0):.4f}')
            else:
                lora_rougeL = 0.0
        except Exception as e:
            logger.error(f'Multi-Expert LoRA evaluation failed: {e}')
            lora_rougeL = 0.0

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        plot_bar_with_errorbars(shot_summary, lora_rougeL, EXP_DIR, test_mode=args.test_mode)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')

    logger.info('Few-shot consistency summary')
    logger.info(f'{"Configuration":<16} {"Mean ROUGE-L":>14} {"Std. dev.":>8}')
    for n_shots, (mean, std) in sorted(shot_summary.items()):
        logger.info(f'{n_shots}-shot{" ":>10} {mean:>14.4f} {std:>8.4f}')
    logger.info(f'Multi-Expert LoRA{" ":>10} {lora_rougeL:>14.4f} {"0.0000":>8}')
    logger.info(f'\nResults saved to: {EXP_DIR}')


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Exp6: Few-shot vs fine-tuning')
    parser.add_argument('--force-regenerate', action='store_true')
    parser.add_argument('--from-cache', action='store_true')
    parser.add_argument('--no-bertscore', action='store_true')
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--only-missing', action='store_true',
                        help='Skip shot/run combos that already have a full-run cache. '
                             'Test-mode caches are treated as missing and re-run automatically.')
    args = parser.parse_args()
    if args.from_cache:
        args.force_regenerate = False
    run(args)


if __name__ == '__main__':
    main()
