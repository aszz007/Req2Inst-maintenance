#!/usr/bin/env python3
"""Run Experiment 9 routing-strategy evaluation and write its reports."""

import sys
import gc
import argparse
import random as random_module
from datetime import datetime
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config.settings import get_path_config
from src.training.data_loader import (
    TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
    GeneralDatasetLoader, split_dataset_for_expert,
)
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger

logger = get_logger('experiments.exp9')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
PLOT_DIR = EXP_DIR / 'plots'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']



def _get_expert(expert_type, lora_path=None):
    """Return expert."""
    from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
    cls = {
        'text': TextExpert, 'image': ImageExpert,
        'uml': UMLExpert, 'general': GeneralExpert
    }[expert_type]
    return cls(lora_path=lora_path, use_4bit=True)


def _load_test_data(expert_type):
    """Load test data."""
    if expert_type == 'text':
        data = TextDatasetLoader().load_csv_files()
    elif expert_type == 'image':
        data = ImageDatasetLoader().load_csv_file()
    elif expert_type == 'uml':
        data = UMLDatasetLoader().load_csv_file()
    else:
        data = GeneralDatasetLoader().load_all_data()
    _, _, test_data = split_dataset_for_expert(data, expert_type)
    return test_data


def _is_full_run_cache(cache_dir, filename):
    """Return whether full run cache."""
    cached = load_predictions_cache(cache_dir, filename)
    if cached is None:
        return False
    n = cached.get('total_samples', 0)
    return n > 15


def _metrics_from_cache(cached, use_bertscore=True):
    """Metrics from cache."""
    if cached is None:
        return {}
    samples = cached.get('samples', [])
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)


def _get_rougeL(metrics_dict):
    """Return ROUGE l."""
    return metrics_dict.get('generation_quality', {}).get('rougeL', 0.0)


def _get_format_score(metrics_dict):
    """Return format score."""
    return metrics_dict.get('format_metrics', {}).get('format_score', 0.0)


def _cleanup_gpu():
    """Release GPU resources."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass



def _get_cache_location(expert_type, eval_domain):
    """Return cache location."""
    if expert_type == eval_domain:
        return CACHE_DIR / 'lora_moe', f'{eval_domain}_predictions.json'

    if expert_type == 'text' and eval_domain == 'general':
        return CACHE_DIR / 'exp3_moe3_general_via_text', 'general_via_text_predictions.json'

    if expert_type in SPECIALIZED_TYPES and eval_domain in SPECIALIZED_TYPES:
        return (CACHE_DIR / 'exp3_cross_domain',
                f'{expert_type}_expert_on_{eval_domain}_predictions.json')

    return (CACHE_DIR / 'exp9_oracle',
            f'{expert_type}_expert_on_{eval_domain}_predictions.json')



def _run_single_inference(expert_type, eval_domain, test_data, args):
    """Run single inference."""
    cache_dir, filename = _get_cache_location(expert_type, eval_domain)

    cached = load_predictions_cache(cache_dir, filename)
    if cached and not args.force_regenerate:
        n = cached.get('total_samples', 0)
        if n > 15 or args.test_mode:
            logger.info(f"Cache hit: {expert_type}->{eval_domain} ({n} samples)")
            return cached

    logger.info(f"Running inference: {expert_type}->{eval_domain}")
    expert = _get_expert(expert_type)
    if not expert.load_model():
        logger.error(f"Failed to load {expert_type} expert")
        return None

    data_subset = test_data[:10] if args.test_mode else test_data
    inputs = [d['input'] for d in data_subset]
    refs = [d['output'] for d in data_subset]

    try:
        preds = expert.batch_generate_instruction(inputs, batch_size=4)
    except Exception as e:
        logger.error(f"Inference failed for {expert_type}->{eval_domain}: {e}")
        preds = [''] * len(inputs)
    finally:
        expert.unload_model()
        _cleanup_gpu()

    samples = [
        {'index': i, 'input': inp, 'prediction': p, 'reference': r}
        for i, (inp, p, r) in enumerate(zip(inputs, preds, refs))
    ]

    save_predictions_cache(
        samples, 'exp9_oracle', eval_domain,
        {'expert': expert_type, 'eval_domain': eval_domain,
         'purpose': 'oracle_label_construction'},
        cache_dir, filename
    )
    return load_predictions_cache(cache_dir, filename)


def run_phase1(args):
    """Run phase1."""
    logger.info("Phase 1: Oracle upper- and lower-bound analysis")

    logger.info("Loading test sets...")
    test_datasets = {}
    for et in ALL_TYPES:
        try:
            test_datasets[et] = _load_test_data(et)
            logger.info(f"  {et}: {len(test_datasets[et])} samples")
        except Exception as e:
            logger.error(f"  Failed to load {et} test set: {e}")

    logger.info("\n--- Step 1: Collecting 16 sets of inference results ---")

    # all_caches[expert_type][eval_domain] = cached_data
    all_caches = {}
    new_inference_count = 0
    reused_count = 0

    for expert_type in ALL_TYPES:
        all_caches[expert_type] = {}
        for eval_domain in ALL_TYPES:
            if eval_domain not in test_datasets:
                continue

            cache_dir, filename = _get_cache_location(expert_type, eval_domain)
            cached = load_predictions_cache(cache_dir, filename)

            if cached and not args.force_regenerate and (
                cached.get('total_samples', 0) > 15 or args.test_mode
            ):
                all_caches[expert_type][eval_domain] = cached
                reused_count += 1
                logger.info(f"  [Reused] {expert_type}->{eval_domain}: "
                          f"{cached.get('total_samples', 0)} samples")
            else:
                cached = _run_single_inference(
                    expert_type, eval_domain,
                    test_datasets[eval_domain], args
                )
                all_caches[expert_type][eval_domain] = cached
                new_inference_count += 1

    logger.info(f"\nInference summary: reused={reused_count}, new={new_inference_count}, "
              f"total={reused_count + new_inference_count}")

    logger.info("\n--- Step 2: Computing metrics ---")

    # score_matrix[expert_type][eval_domain] = rougeL
    score_matrix = {}
    metrics_matrix = {}
    use_bs = not args.no_bertscore

    for expert_type in ALL_TYPES:
        score_matrix[expert_type] = {}
        metrics_matrix[expert_type] = {}
        for eval_domain in ALL_TYPES:
            cached = all_caches.get(expert_type, {}).get(eval_domain)
            if cached is None:
                score_matrix[expert_type][eval_domain] = 0.0
                continue
            m = _metrics_from_cache(cached, use_bertscore=use_bs)
            score_matrix[expert_type][eval_domain] = _get_rougeL(m)
            metrics_matrix[expert_type][eval_domain] = m
            logger.info(f"  {expert_type}->{eval_domain}: "
                      f"ROUGE-L={score_matrix[expert_type][eval_domain]:.4f}")

    logger.info("\n--- Step 3: Scoring five routing strategies ---")

    strategies = {}

    hard_scores = {}
    for domain in ALL_TYPES:
        hard_scores[domain] = score_matrix.get(domain, {}).get(domain, 0.0)
    hard_avg = np.mean(list(hard_scores.values())) if hard_scores else 0.0
    strategies['Hard Routing'] = {
        'per_domain': hard_scores,
        'average': float(hard_avg),
    }
    logger.info(f"Hard Routing: mean ROUGE-L={hard_avg:.4f}")

    oracle_scores = {}
    oracle_selections = {}  # domain -> {expert: count}
    for domain in ALL_TYPES:
        if domain not in test_datasets:
            continue
        n_samples = len(test_datasets[domain])
        if args.test_mode:
            n_samples = min(10, n_samples)

        domain_per_sample_best = []
        domain_selections = defaultdict(int)

        for sample_idx in range(n_samples):
            best_score = -1.0
            best_expert = domain

            for expert_type in ALL_TYPES:
                cached = all_caches.get(expert_type, {}).get(domain)
                if cached is None:
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')

                if not pred or not pred.strip():
                    continue

                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0

                if rl > best_score:
                    best_score = rl
                    best_expert = expert_type

            domain_per_sample_best.append(best_score)
            domain_selections[best_expert] += 1

        oracle_scores[domain] = float(np.mean(domain_per_sample_best)) if domain_per_sample_best else 0.0
        oracle_selections[domain] = dict(domain_selections)

    oracle_avg = np.mean(list(oracle_scores.values())) if oracle_scores else 0.0
    strategies['Oracle Routing'] = {
        'per_domain': oracle_scores,
        'average': float(oracle_avg),
        'selections': oracle_selections,
    }
    logger.info(f"Oracle Routing: mean ROUGE-L={oracle_avg:.4f}")

    worst_scores = {}
    for domain in ALL_TYPES:
        if domain not in test_datasets:
            continue
        n_samples = len(test_datasets[domain])
        if args.test_mode:
            n_samples = min(10, n_samples)

        domain_per_sample_worst = []
        for sample_idx in range(n_samples):
            worst_score = float('inf')
            has_valid = False

            for expert_type in ALL_TYPES:
                cached = all_caches.get(expert_type, {}).get(domain)
                if cached is None:
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')
                if not pred or not pred.strip():
                    continue

                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0

                if rl < worst_score:
                    worst_score = rl
                    has_valid = True

            domain_per_sample_worst.append(worst_score if has_valid else 0.0)

        worst_scores[domain] = float(np.mean(domain_per_sample_worst)) if domain_per_sample_worst else 0.0

    worst_avg = np.mean(list(worst_scores.values())) if worst_scores else 0.0
    strategies['Worst Routing'] = {
        'per_domain': worst_scores,
        'average': float(worst_avg),
    }
    logger.info(f"Worst Routing: mean ROUGE-L={worst_avg:.4f}")

    random_seeds = [42, 43, 44]
    random_runs = []

    for seed in random_seeds:
        random_module.seed(seed)
        run_scores = {}

        for domain in ALL_TYPES:
            if domain not in test_datasets:
                continue
            n_samples = len(test_datasets[domain])
            if args.test_mode:
                n_samples = min(10, n_samples)

            domain_per_sample = []
            for sample_idx in range(n_samples):
                chosen_expert = random_module.choice(ALL_TYPES)
                cached = all_caches.get(chosen_expert, {}).get(domain)
                if cached is None:
                    domain_per_sample.append(0.0)
                    continue
                samples = cached.get('samples', [])
                if sample_idx >= len(samples):
                    domain_per_sample.append(0.0)
                    continue

                pred = samples[sample_idx].get('prediction', '')
                ref = samples[sample_idx].get('reference', '')
                if not pred or not pred.strip():
                    domain_per_sample.append(0.0)
                    continue

                try:
                    from rouge_score import rouge_scorer
                    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                    score = scorer.score(ref, pred)
                    rl = score['rougeL'].fmeasure
                except Exception:
                    rl = 0.0
                domain_per_sample.append(rl)

            run_scores[domain] = float(np.mean(domain_per_sample)) if domain_per_sample else 0.0

        run_avg = float(np.mean(list(run_scores.values()))) if run_scores else 0.0
        random_runs.append({'seed': seed, 'per_domain': run_scores, 'average': run_avg})

    random_avgs = [r['average'] for r in random_runs]
    random_mean = float(np.mean(random_avgs))
    random_std = float(np.std(random_avgs))

    random_per_domain = {}
    for domain in ALL_TYPES:
        domain_vals = [r['per_domain'].get(domain, 0.0) for r in random_runs]
        random_per_domain[domain] = float(np.mean(domain_vals))

    strategies['Random Routing'] = {
        'per_domain': random_per_domain,
        'average': random_mean,
        'std': random_std,
        'runs': random_runs,
    }
    logger.info(f"Random Routing: mean ROUGE-L={random_mean:.4f} +/- {random_std:.4f}")

    general_only_scores = {}
    for domain in ALL_TYPES:
        general_only_scores[domain] = score_matrix.get('general', {}).get(domain, 0.0)
    general_only_avg = np.mean(list(general_only_scores.values())) if general_only_scores else 0.0
    strategies['General-Only'] = {
        'per_domain': general_only_scores,
        'average': float(general_only_avg),
    }
    logger.info(f"General-Only: mean ROUGE-L={general_only_avg:.4f}")

    gap = oracle_avg - hard_avg
    logger.info("Phase 1 decision analysis")
    logger.info(f"Oracle ROUGE-L: {oracle_avg:.4f}")
    logger.info(f"Hard   ROUGE-L: {hard_avg:.4f}")
    logger.info(f"Gap (Oracle - Hard): {gap:.4f} ({gap*100:.2f}%)")

    if gap >= 0.02:
        logger.info(">> Gap >= 2%: Phase 2 (Soft Routing) is recommended")
        phase2_recommended = True
    else:
        logger.info(">> Gap < 2%: Hard Routing is close to the theoretical optimum; Phase 2 is optional")
        phase2_recommended = False

    general_gap = oracle_scores.get('general', 0) - hard_scores.get('general', 0)
    logger.info(f"\nGeneral-domain Oracle-Hard gap: {general_gap:.4f} ({general_gap*100:.2f}%)")

    results = {
        'phase': 'phase1',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'strategies': strategies,
        'score_matrix': score_matrix,
        'oracle_selections': oracle_selections,
        'gap_analysis': {
            'overall_gap': float(gap),
            'general_domain_gap': float(general_gap),
            'phase2_recommended': phase2_recommended,
            'per_domain_gaps': {
                domain: float(oracle_scores.get(domain, 0) - hard_scores.get(domain, 0))
                for domain in ALL_TYPES
            },
        },
        'test_set_sizes': {et: len(test_datasets.get(et, [])) for et in ALL_TYPES},
    }

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'phase1_results.json')
    logger.info(f"\nPhase 1 results saved to: {EXP_DIR / 'phase1_results.json'}")

    return results


# ========== Phase 2: Soft Routing ==========

def run_phase2(args, phase1_results=None):
    """Run phase2."""
    logger.info("Phase 2: Soft Routing validation on the general domain")

    from src.routing.soft_router import check_peft_version, SoftRouter, \
        build_type_aware_weights, group_general_samples_by_type

    if not check_peft_version():
        logger.error("The installed PEFT version does not support add_weighted_adapter; skipping Phase 2")
        return {'phase': 'phase2', 'status': 'skipped', 'reason': 'peft_version'}

    logger.info("Loading general test set...")
    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    logger.info(f"General test set: {len(general_test)} samples")

    if args.test_mode:
        general_test = general_test[:10]
        logger.info(f"Test mode: limited to {len(general_test)} samples")

    type_groups = group_general_samples_by_type(general_test)

    adapter_paths = {}
    for expert_name in ['text', 'image', 'uml', 'general']:
        adapter_path = path_cfg.get_expert_weight_path(expert_name)
        adapter_paths[f'{expert_name}_expert'] = str(adapter_path)
        logger.info(f"  {expert_name}_expert: {adapter_path}")

    logger.info("\nLoading base model...")
    from models.language_model import LanguageModel
    lm = LanguageModel(use_4bit=True)

    soft_router = SoftRouter(
        base_model=lm.model,
        tokenizer=lm.tokenizer,
        adapter_paths=adapter_paths,
    )

    if not soft_router.load_all_adapters():
        logger.error("Failed to load adapters; skipping Phase 2")
        return {'phase': 'phase2', 'status': 'failed', 'reason': 'adapter_load'}

    alpha_values = [0.3, 0.5, 0.7]
    all_alpha_results = {}

    for alpha in alpha_values:
        logger.info(f"\n--- Fusion ratio alpha={alpha} ---")

        predictions = [''] * len(general_test)

        for data_type, indices in type_groups.items():
            if not indices:
                continue

            weights = build_type_aware_weights(data_type, alpha=alpha)
            logger.info(f"  {data_type} type ({len(indices)} samples): weights={weights}")

            if not soft_router.merge_adapters(weights, merged_name=f"merged_{data_type}"):
                logger.error(f"  Adapter fusion failed for {data_type}")
                continue

            batch_inputs = [general_test[i]['input'] for i in indices]

            from models.prompt_templates.general_template import GeneralInstructionTemplate

            for batch_start in range(0, len(batch_inputs), 4):
                batch_end = min(batch_start + 4, len(batch_inputs))
                batch = batch_inputs[batch_start:batch_end]

                prompts = []
                for inp in batch:
                    prompt = GeneralInstructionTemplate.build_prompt(inp, force_type=data_type)
                    prompts.append(prompt)

                try:
                    batch_preds = lm.generate_batch(
                        prompts=prompts,
                        max_new_tokens=2048,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        batch_size=len(prompts),
                    )
                except Exception as e:
                    logger.error(f"  Generation failed: {e}")
                    batch_preds = [''] * len(prompts)

                for j, pred in enumerate(batch_preds):
                    global_idx = indices[batch_start + j]
                    predictions[global_idx] = pred

        refs = [d['output'] for d in general_test]
        metrics = compute_all_metrics(
            predictions, refs, use_bertscore=not args.no_bertscore
        )
        rougeL = _get_rougeL(metrics)

        all_alpha_results[alpha] = {
            'rougeL': rougeL,
            'metrics': metrics,
        }
        logger.info(f"  alpha={alpha}: ROUGE-L={rougeL:.4f}")

        cache_dir = CACHE_DIR / 'exp9_soft_routing'
        samples = [
            {'index': i, 'input': general_test[i]['input'],
             'prediction': predictions[i], 'reference': refs[i]}
            for i in range(len(general_test))
        ]
        save_predictions_cache(
            samples, 'exp9_soft_routing', 'general',
            {'alpha': alpha, 'strategy': 'type_aware_soft'},
            cache_dir, f'general_soft_alpha{alpha}_predictions.json'
        )

    soft_router.cleanup()
    del lm
    _cleanup_gpu()

    best_alpha = max(all_alpha_results, key=lambda a: all_alpha_results[a]['rougeL'])
    best_rougeL = all_alpha_results[best_alpha]['rougeL']

    hard_general_rougeL = 0.0
    if phase1_results:
        hard_general_rougeL = phase1_results.get('strategies', {}).get(
            'Hard Routing', {}).get('per_domain', {}).get('general', 0.0)

    improvement = best_rougeL - hard_general_rougeL

    logger.info(f"\nBest fusion ratio: alpha={best_alpha}")
    logger.info(f"Soft Routing ROUGE-L: {best_rougeL:.4f}")
    logger.info(f"Hard Routing ROUGE-L: {hard_general_rougeL:.4f}")
    logger.info(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

    results = {
        'phase': 'phase2',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'alpha_search': {
            str(a): {'rougeL': r['rougeL']} for a, r in all_alpha_results.items()
        },
        'best_alpha': best_alpha,
        'best_rougeL': best_rougeL,
        'hard_baseline_rougeL': hard_general_rougeL,
        'improvement': improvement,
    }

    save_experiment_results(results, EXP_DIR, 'phase2_results.json')
    logger.info(f"Phase 2 results saved to: {EXP_DIR / 'phase2_results.json'}")

    return results



def run_phase3(args, phase1_results=None, phase2_results=None):
    """Run phase3."""
    logger.info("Phase 3: Contribution analysis and visualization")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if phase1_results is None:
        p1_path = EXP_DIR / 'phase1_results.json'
        if p1_path.exists():
            import json
            with open(p1_path, 'r') as f:
                phase1_results = json.load(f)
        else:
            logger.error("Phase 1 results file not found; visualization cannot proceed")
            return

    if phase2_results is None:
        p2_path = EXP_DIR / 'phase2_results.json'
        if p2_path.exists():
            import json
            with open(p2_path, 'r') as f:
                phase2_results = json.load(f)

    strategies = phase1_results.get('strategies', {})
    oracle_selections = phase1_results.get('oracle_selections', {})

    _plot_contribution_band(strategies, phase2_results)

    _plot_oracle_heatmap(oracle_selections)

    _plot_per_domain_comparison(strategies)

    _plot_general_oracle_distribution(oracle_selections)

    _plot_gap_analysis(phase1_results.get('gap_analysis', {}))

    _plot_random_variance(strategies.get('Random Routing', {}))

    if phase2_results and phase2_results.get('phase') == 'phase2':
        _plot_soft_vs_hard(phase2_results)

    _plot_summary_table(strategies, phase2_results)

    _generate_report(phase1_results, phase2_results)

    logger.info(f"\nAll plots saved to: {PLOT_DIR}")


def _plot_contribution_band(strategies, phase2_results=None):
    """Plot contribution band."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategy_order = ['Worst Routing', 'Random Routing', 'Hard Routing',
                      'General-Only', 'Oracle Routing']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#2ecc71']

    x_vals = []
    labels = []
    errs = []

    for name in strategy_order:
        s = strategies.get(name, {})
        avg = s.get('average', 0)
        std = s.get('std', 0)
        x_vals.append(avg)
        labels.append(name)
        errs.append(std)

    if phase2_results and 'best_rougeL' in phase2_results:
        insert_idx = 3
        x_vals.insert(insert_idx, phase2_results['best_rougeL'])
        labels.insert(insert_idx, f"Soft Routing\n(alpha={phase2_results.get('best_alpha', '?')})")
        errs.insert(insert_idx, 0)
        colors.insert(insert_idx, '#9b59b6')

    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, x_vals, xerr=errs, color=colors, edgecolor='white',
                   height=0.6, capsize=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Average ROUGE-L', fontsize=12)
    ax.set_title('Routing Strategy Comparison: Contribution Band', fontsize=14)

    for bar, val, err in zip(bars, x_vals, errs):
        if err > 0:
            label = f'{val:.4f} +/- {err:.4f}'
        else:
            label = f'{val:.4f}'
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=9)

    ax.set_xlim(0, max(x_vals) * 1.15 if x_vals else 1.0)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_contribution_band.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/8] routing_contribution_band.png")


def _plot_oracle_heatmap(oracle_selections):
    """Plot oracle heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))

    matrix = np.zeros((4, 4))
    for d_idx, domain in enumerate(ALL_TYPES):
        selections = oracle_selections.get(domain, {})
        total = sum(selections.values()) if selections else 1
        for e_idx, expert in enumerate(ALL_TYPES):
            count = selections.get(expert, 0)
            matrix[d_idx][e_idx] = count / total * 100 if total > 0 else 0

    sns.heatmap(
        matrix, annot=True, fmt='.1f', cmap='YlOrRd',
        xticklabels=[f'{t}_expert' for t in ALL_TYPES],
        yticklabels=[f'{t}_test' for t in ALL_TYPES],
        ax=ax, vmin=0, vmax=100,
        cbar_kws={'label': 'Oracle Selection Rate (%)'}
    )
    ax.set_title('Oracle Expert Selection Heatmap', fontsize=14)
    ax.set_xlabel('Expert Used', fontsize=12)
    ax.set_ylabel('Test Domain', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'oracle_selection_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/8] oracle_selection_heatmap.png")


def _plot_per_domain_comparison(strategies):
    """Plot per domain comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_names = ['Worst Routing', 'Random Routing', 'Hard Routing',
                      'General-Only', 'Oracle Routing']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#95a5a6', '#2ecc71']

    x = np.arange(len(ALL_TYPES))
    width = 0.15
    offsets = np.arange(len(strategy_names)) - len(strategy_names) / 2 + 0.5

    for i, (name, color) in enumerate(zip(strategy_names, colors)):
        s = strategies.get(name, {})
        per_domain = s.get('per_domain', {})
        vals = [per_domain.get(d, 0) for d in ALL_TYPES]
        ax.bar(x + offsets[i] * width, vals, width, label=name, color=color, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{t.capitalize()}' for t in ALL_TYPES], fontsize=11)
    ax.set_ylabel('ROUGE-L', fontsize=12)
    ax.set_title('Per-Domain Routing Strategy Comparison', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'per_domain_strategy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/8] per_domain_strategy_comparison.png")


def _plot_general_oracle_distribution(oracle_selections):
    """Plot general oracle distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))

    general_sel = oracle_selections.get('general', {})
    if not general_sel:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
        ax.set_title('General Domain: Oracle Expert Selection', fontsize=14)
    else:
        labels = [f'{k}' for k in general_sel.keys()]
        sizes = list(general_sel.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(labels)]

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', colors=colors,
            startangle=90, pctdistance=0.85
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title('General Domain: Oracle Expert Selection Distribution', fontsize=14)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'general_domain_oracle_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/8] general_domain_oracle_distribution.png")


def _plot_gap_analysis(gap_analysis):
    """Plot gap analysis."""
    fig, ax = plt.subplots(figsize=(8, 5))

    per_domain_gaps = gap_analysis.get('per_domain_gaps', {})
    domains = list(per_domain_gaps.keys())
    gaps = [per_domain_gaps[d] * 100 for d in domains]

    colors = ['#e74c3c' if g < 0 else '#2ecc71' for g in gaps]
    bars = ax.bar(domains, gaps, color=colors, edgecolor='white', width=0.6)

    for bar, val in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('ROUGE-L Gap (%)', fontsize=12)
    ax.set_title('Oracle - Hard Routing Gap by Domain', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticklabels([d.capitalize() for d in domains], fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_gap_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/8] routing_gap_analysis.png")


def _plot_random_variance(random_data):
    """Plot random variance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    runs = random_data.get('runs', [])
    if not runs:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
    else:
        for run in runs:
            seed = run['seed']
            per_domain = run.get('per_domain', {})
            domains = list(per_domain.keys())
            vals = [per_domain[d] for d in domains]
            ax.plot(domains, vals, 'o-', label=f'seed={seed}', markersize=6)

        mean_per_domain = {}
        for d in ALL_TYPES:
            d_vals = [r['per_domain'].get(d, 0) for r in runs]
            mean_per_domain[d] = np.mean(d_vals)
        ax.plot(list(mean_per_domain.keys()), list(mean_per_domain.values()),
                's--', color='black', label='Mean', markersize=8, linewidth=2)

        ax.set_ylabel('ROUGE-L', fontsize=12)
        ax.set_title('Random Routing Variance (3 Seeds)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_xticklabels([d.capitalize() for d in ALL_TYPES], fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'random_routing_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/8] random_routing_variance.png")


def _plot_soft_vs_hard(phase2_results):
    """Plot soft vs hard."""
    fig, ax = plt.subplots(figsize=(8, 5))

    alpha_search = phase2_results.get('alpha_search', {})
    hard_baseline = phase2_results.get('hard_baseline_rougeL', 0)

    alphas = sorted(alpha_search.keys(), key=float)
    rougeL_vals = [alpha_search[a]['rougeL'] for a in alphas]

    ax.plot([float(a) for a in alphas], rougeL_vals, 'o-', color='#9b59b6',
            label='Soft Routing', markersize=8, linewidth=2)
    ax.axhline(y=hard_baseline, color='#3498db', linestyle='--',
               linewidth=2, label=f'Hard Routing ({hard_baseline:.4f})')

    best_alpha = phase2_results.get('best_alpha')
    best_rougeL = phase2_results.get('best_rougeL', 0)
    if best_alpha is not None:
        ax.annotate(
            f'Best: alpha={best_alpha}\nROUGE-L={best_rougeL:.4f}',
            xy=(float(best_alpha), best_rougeL),
            xytext=(float(best_alpha) + 0.08, best_rougeL + 0.005),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8)
        )

    ax.set_xlabel('Alpha (Specialized Expert Weight)', fontsize=12)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('Soft Routing vs Hard Routing (General Domain)', fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'soft_vs_hard_general.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [7/8] soft_vs_hard_general.png")


def _plot_summary_table(strategies, phase2_results=None):
    """Plot summary table."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    strategy_order = ['Worst Routing', 'Random Routing', 'General-Only',
                      'Hard Routing', 'Oracle Routing']
    headers = ['Strategy', 'Text', 'Image', 'FlowChart', 'General', 'Average']

    table_data = []
    for name in strategy_order:
        s = strategies.get(name, {})
        per_d = s.get('per_domain', {})
        row = [name]
        for d in ALL_TYPES:
            row.append(f"{per_d.get(d, 0):.4f}")
        avg = s.get('average', 0)
        std = s.get('std', 0)
        if std > 0:
            row.append(f"{avg:.4f} +/- {std:.4f}")
        else:
            row.append(f"{avg:.4f}")
        table_data.append(row)

    if phase2_results and 'best_rougeL' in phase2_results:
        best_alpha = phase2_results.get('best_alpha', '?')
        row = [f'Soft Routing (a={best_alpha})', '-', '-', '-',
               f"{phase2_results['best_rougeL']:.4f}", '-']
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')

    hard_row_idx = strategy_order.index('Hard Routing') + 1
    for j in range(len(headers)):
        table[hard_row_idx, j].set_facecolor('#d4e6f1')

    ax.set_title('Exp9: Routing Strategy Comparison Summary', fontsize=14,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [8/8] summary_table.png")


def _generate_report(phase1_results, phase2_results=None):
    """Generate report."""
    strategies = phase1_results.get('strategies', {})
    gap_analysis = phase1_results.get('gap_analysis', {})

    lines = [
        "# Experiment 9: Routing Strategy Comparison",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Phase 1: Oracle上下界分析",
        "\n### 策略对比结果",
        "",
        "| 策略 | Text | Image | FlowChart | General | 平均 |",
        "|------|------|-------|-----|---------|------|",
    ]

    for name in ['Worst Routing', 'Random Routing', 'General-Only',
                 'Hard Routing', 'Oracle Routing']:
        s = strategies.get(name, {})
        per_d = s.get('per_domain', {})
        avg = s.get('average', 0)
        std_str = f" +/- {s.get('std', 0):.4f}" if s.get('std', 0) > 0 else ""
        lines.append(
            f"| {name} | {per_d.get('text', 0):.4f} | {per_d.get('image', 0):.4f} | "
            f"{per_d.get('uml', 0):.4f} | {per_d.get('general', 0):.4f} | "
            f"{avg:.4f}{std_str} |"
        )

    lines.extend([
        "",
        "### 差距分析",
        f"- Oracle-Hard 总体差距: {gap_analysis.get('overall_gap', 0)*100:.2f}%",
        f"- General域差距: {gap_analysis.get('general_domain_gap', 0)*100:.2f}%",
        f"- Phase 2建议: {'建议执行' if gap_analysis.get('phase2_recommended') else '可选'}",
    ])

    per_gaps = gap_analysis.get('per_domain_gaps', {})
    for d in ALL_TYPES:
        lines.append(f"  - {d}: {per_gaps.get(d, 0)*100:.2f}%")

    if phase2_results and phase2_results.get('phase') == 'phase2':
        lines.extend([
            "",
            "## Phase 2: Soft Routing结果",
            f"- 最优alpha: {phase2_results.get('best_alpha')}",
            f"- Soft Routing ROUGE-L: {phase2_results.get('best_rougeL', 0):.4f}",
            f"- Hard Routing ROUGE-L: {phase2_results.get('hard_baseline_rougeL', 0):.4f}",
            f"- 提升: {phase2_results.get('improvement', 0)*100:.2f}%",
        ])

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Report saved to: {report_path}")



def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Exp9: Routing Strategy Comparison')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only the specified phase (1/2/3)')
    parser.add_argument('--all', action='store_true',
                        help='Run all phases (Phase 1 + conditional Phase 2 + Phase 3)')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force inference rerun and ignore the cache')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Skip BERTScore computation (faster)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode (10 samples per domain)')
    parser.add_argument('--skip-phase2-check', action='store_true',
                        help='Skip the Phase 2 gap check and force execution')
    args = parser.parse_args()

    logger.info("Experiment 9: Routing strategy comparison and router contribution analysis")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: phase={args.phase}, all={args.all}, "
              f"test_mode={args.test_mode}, no_bertscore={args.no_bertscore}")

    phase1_results = None
    phase2_results = None

    if args.phase == 1 or args.all:
        phase1_results = run_phase1(args)

    if args.phase == 2 or args.all:
        if phase1_results is None:
            import json
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)

        if args.skip_phase2_check:
            logger.info("Skipping the gap check and forcing Phase 2")
            phase2_results = run_phase2(args, phase1_results)
        elif phase1_results:
            gap = phase1_results.get('gap_analysis', {}).get('overall_gap', 0)
            general_gap = phase1_results.get('gap_analysis', {}).get('general_domain_gap', 0)
            if gap >= 0.02 or general_gap >= 0.02:
                logger.info(f"Gap={gap:.4f} >= 0.02; running Phase 2")
                phase2_results = run_phase2(args, phase1_results)
            else:
                logger.info(f"Gap={gap:.4f} < 0.02; skipping Phase 2 because Hard Routing is close to the theoretical optimum")
                logger.info("Use --skip-phase2-check to force Phase 2")
        else:
            logger.warning("Phase 1 results are unavailable; skipping Phase 2")

    if args.phase == 3 or args.all:
        run_phase3(args, phase1_results, phase2_results)

    if args.all or args.phase is None:
        final_results = {
            'experiment': 'exp9_routing_strategy',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        if phase1_results:
            final_results['phase1'] = phase1_results
        if phase2_results:
            final_results['phase2'] = phase2_results
        save_experiment_results(final_results, EXP_DIR, 'results.json')

    logger.info(f"Experiment 9 completed | time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {EXP_DIR}")


if __name__ == '__main__':
    main()
