#!/usr/bin/env python3
"""
Run All Experiments

Orchestrate all 11 experiments sequentially via subprocess, using the same
conda environment (instruction_generator).

Usage:
  python run_all_experiments.py
  python run_all_experiments.py --experiments 1,2,3
  python run_all_experiments.py --skip 4,5
  python run_all_experiments.py --skip-failed
  python run_all_experiments.py --from-cache
  python run_all_experiments.py --test-mode
"""

import sys
import argparse
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger('experiments.run_all')

EXP_DIR = Path(__file__).parent

EXP_SCRIPTS = {
    1:  'exp1_baseline_comparison.py',
    2:  'exp2_compare_finetuning_methods.py',
    3:  'exp3_moe_architecture_validation.py',
    4:  'exp4_lora_hyperparameter_optimization.py',
    5:  'exp5_data_efficiency_analysis.py',
    6:  'exp6_fewshot_vs_finetuning.py',
    7:  'exp7_uml_hyperparameter_optimization.py',
    8:  'exp8_inference_efficiency.py',
    9:  'exp9_routing_strategy.py',
    10: 'exp10_advanced_routing.py',
    11: 'exp11_ablation_optimization.py',
}

EXP_EXTRA_ARGS = {
    9:  ['--all'],
    10: ['--all'],
    11: ['--all'],
}

EXP_NAMES = {
    1:  '基线方法对比',
    2:  '微调方法对比',
    3:  'MoE架构验证',
    4:  'LoRA超参数优化',
    5:  '数据效率分析',
    6:  'Few-Shot vs 微调对比',
    7:  'UML超参数优化',
    8:  '推理效率分析',
    9:  '路由策略对比',
    10: '高级路由策略',
    11: '消融实验优化',
}

STATUS_PASS = 'PASS'
STATUS_FAIL = 'FAIL'
STATUS_SKIP = 'SKIPPED'


def run_experiment(exp_num, args, skip_failed, previously_failed):
    """
    Run a single experiment script as a subprocess.

    Returns:
        (status, elapsed_seconds)
    """
    if exp_num in previously_failed:
        logger.info(f'Experiment {exp_num}: skipped because a prerequisite experiment failed')
        return STATUS_SKIP, 0.0

    script = EXP_DIR / EXP_SCRIPTS[exp_num]
    if not script.exists():
        logger.error(f'Experiment {exp_num}: script not found: {script}')
        return STATUS_FAIL, 0.0

    cmd = [sys.executable, str(script)]

    cmd.extend(EXP_EXTRA_ARGS.get(exp_num, []))

    if args.from_cache:
        cmd.append('--from-cache')
    if args.test_mode:
        cmd.append('--test-mode')
    if args.no_bertscore:
        cmd.append('--no-bertscore')

    logger.info(f'Experiment {exp_num} ({EXP_NAMES[exp_num]}): starting...')
    logger.info(f'Command: {" ".join(cmd)}')

    start = time.time()
    try:
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - start

        if result.returncode == 0:
            logger.info(
                f'Experiment {exp_num}: passed ({timedelta(seconds=int(elapsed))})'
            )
            return STATUS_PASS, elapsed
        else:
            logger.error(
                f'Experiment {exp_num}: failed (return code={result.returncode}, '
                f'elapsed={timedelta(seconds=int(elapsed))})'
            )
            return STATUS_FAIL, elapsed

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f'Experiment {exp_num}: raised an exception: {e}')
        return STATUS_FAIL, elapsed


def parse_int_list(s):
    """Parse comma-separated integer list, e.g. '1,2,3' -> [1, 2, 3]."""
    if not s:
        return []
    parts = s.replace(' ', '').split(',')
    return [int(p) for p in parts if p]


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Run all comparison experiments (exp1–exp11)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python run_all_experiments.py

  # Run specific experiments
  python run_all_experiments.py --experiments 1,2,3

  # Skip specific experiments
  python run_all_experiments.py --skip 4,5

  # Continue past failures
  python run_all_experiments.py --skip-failed

  # Load from inference cache (skip model loading)
  python run_all_experiments.py --from-cache

  # Quick validation with 10 samples
  python run_all_experiments.py --test-mode
        """
    )
    parser.add_argument('--experiments', type=str, default=None,
                        help='Comma-separated experiment numbers to run (default: all 1-11)')
    parser.add_argument('--skip', type=str, default=None,
                        help='Comma-separated experiment numbers to skip')
    parser.add_argument('--skip-failed', action='store_true',
                        help='Continue to next experiment if one fails')
    parser.add_argument('--from-cache', action='store_true',
                        help='Pass --from-cache to each experiment script')
    parser.add_argument('--test-mode', action='store_true',
                        help='Pass --test-mode to each experiment script (10 samples)')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Pass --no-bertscore to each experiment script')
    args = parser.parse_args()

    # Determine which experiments to run
    all_exp_nums = sorted(EXP_SCRIPTS.keys())

    if args.experiments:
        selected = parse_int_list(args.experiments)
        invalid = [n for n in selected if n not in EXP_SCRIPTS]
        if invalid:
            logger.error(f'Invalid experiment numbers: {invalid}')
            sys.exit(1)
    else:
        selected = all_exp_nums

    skip_set = set(parse_int_list(args.skip or ''))
    to_run = [n for n in selected if n not in skip_set]

    if not to_run:
        logger.warning('No experiments selected; exiting')
        sys.exit(0)

    logger.info('Experiment runner')
    logger.info(f'Experiments to run: {to_run}')
    if skip_set:
        logger.info(f'Skipped by configuration: {sorted(skip_set)}')
    logger.info(f'Test mode: {args.test_mode}')
    logger.info(f'Load from cache: {args.from_cache}')
    logger.info(f'Continue after failure: {args.skip_failed}')
    logger.info(f'Start time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    summary = {}
    previously_failed = set()
    total_start = time.time()

    for exp_num in to_run:
        if args.skip_failed:
            status, elapsed = run_experiment(exp_num, args, args.skip_failed, previously_failed)
        else:
            # If not skip-failed, treat all selected experiments as fresh
            status, elapsed = run_experiment(exp_num, args, False, set())

        summary[exp_num] = {
            'name': EXP_NAMES[exp_num],
            'status': status,
            'elapsed': elapsed,
        }

        if status == STATUS_FAIL:
            previously_failed.add(exp_num)
            if not args.skip_failed:
                logger.error(
                    f'Experiment {exp_num} failed. Use --skip-failed to continue after failures.'
                )
                break

    total_elapsed = time.time() - total_start

    logger.info('Experiment execution summary')
    logger.info(f'{"Number":<6} {"Experiment":<40} {"Status":<10} {"Duration":>10}')
    for exp_num in to_run:
        if exp_num not in summary:
            logger.info(f'{exp_num:<6} {EXP_NAMES[exp_num]:<40} {"not run":<10}')
            continue
        s = summary[exp_num]
        elapsed_str = str(timedelta(seconds=int(s['elapsed'])))
        logger.info(f'{exp_num:<6} {s["name"]:<40} {s["status"]:<10} {elapsed_str:>10}')

    logger.info(f'Total duration: {timedelta(seconds=int(total_elapsed))}')

    pass_count = sum(1 for s in summary.values() if s['status'] == STATUS_PASS)
    fail_count = sum(1 for s in summary.values() if s['status'] == STATUS_FAIL)
    skip_count = sum(1 for s in summary.values() if s['status'] == STATUS_SKIP)

    logger.info(f'Results: {pass_count} passed, {fail_count} failed, {skip_count} skipped')

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
