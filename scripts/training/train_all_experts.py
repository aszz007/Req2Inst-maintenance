"""Run the configured expert-training tasks."""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import time


PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


TRAINING_TASKS = {
    'lora_moe': {
        'image': 'scripts/training/lora_moe/train_image_expert.py',
        'uml': 'scripts/training/lora_moe/train_uml_expert.py',
        'general': 'scripts/training/lora_moe/train_general_expert.py',
    },
    'prompt_tuning': {
        'text': 'scripts/training/prompt_tuning/train_text_expert.py',
        'image': 'scripts/training/prompt_tuning/train_image_expert.py',
        'uml': 'scripts/training/prompt_tuning/train_uml_expert.py',
        'general': 'scripts/training/prompt_tuning/train_general_expert.py',
    },
    'p_tuning': {
        'text': 'scripts/training/p_tuning/train_text_expert.py',
        'image': 'scripts/training/p_tuning/train_image_expert.py',
        'uml': 'scripts/training/p_tuning/train_uml_expert.py',
        'general': 'scripts/training/p_tuning/train_general_expert.py',
    },
    'full_finetuning': {
        'text': 'scripts/training/full_finetuning/train_text_expert.py',
        'image': 'scripts/training/full_finetuning/train_image_expert.py',
        'uml': 'scripts/training/full_finetuning/train_uml_expert.py',
        'general': 'scripts/training/full_finetuning/train_general_expert.py',
    }
}


ESTIMATED_TIME = {
    'lora_moe': {'text': 0.8, 'image': 0.2, 'uml': 0.5, 'general': 1.5},
    'prompt_tuning': {'text': 1.0, 'image': 0.3, 'uml': 0.75, 'general': 1.9},
    'p_tuning': {'text': 1.3, 'image': 0.4, 'uml': 0.9, 'general': 2.3},
    'full_finetuning': {'text': 1.8, 'image': 0.6, 'uml': 1.25, 'general': 3.0},
}


def print_header():
    """Print header."""
    print("\n" + "=" * 80)
    print(" " * 22 + "Train All Experts")
    print("=" * 80)
    print("\nThis script will train 16 models in sequence:")
    print("  - Round 1: LoRA-MoE (4 experts, about 3 hours)")
    print("  - Round 2: Prompt Tuning (4 experts, about 4 hours)")
    print("  - Round 3: P-Tuning v2 (4 experts, about 5 hours)")
    print("  - Round 4: Near-full fine-tuning (4 experts, about 7 hours)")
    print("\nEstimated total time: about 19 hours")
    print("=" * 80 + "\n")


def print_session_header(session_num, method_name, total_time):
    """Print session header."""
    print("\n" + "=" * 80)
    print(f"Round {session_num}: {method_name}")
    print(f"Estimated time: {total_time:.1f} hours")
    print("=" * 80 + "\n")


def format_time(seconds):
    """Format time."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"


def run_training_task(method, expert, script_path):
    """Run training task."""
    full_path = PROJECT_ROOT / script_path

    if not full_path.exists():
        print(f"Error: script not found: {full_path}")
        return False

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting training: {method}/{expert}")
    print(f"Script: {script_path}")
    print("-" * 80)

    start_time = time.time()

    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=True,
            capture_output=False
        )

        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {method}/{expert}")
        print(f"Elapsed: {format_time(elapsed)}")
        print(f"Status: succeeded")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Failed: {method}/{expert}")
        print(f"Elapsed: {format_time(elapsed)}")
        print(f"Status: failed")
        print(f"Exit code: {e.returncode}")

        return False

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print("\n" + "-" * 80)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Interrupted: {method}/{expert}")
        print(f"Elapsed: {format_time(elapsed)}")
        print(f"Status: interrupted by user")

        raise


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Train all experts in one command',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all methods and experts
  python scripts/training/train_all_experts.py
  
  # Train Prompt Tuning only
  python scripts/training/train_all_experts.py --method prompt_tuning
  
  # Train P-Tuning v2 and full fine-tuning (skip Prompt Tuning)
  python scripts/training/train_all_experts.py --method p_tuning full_finetuning --skip-failed
  
  # Train the text expert only (all methods)
  python scripts/training/train_all_experts.py --expert text
  
  # Continue automatically after failed tasks
  python scripts/training/train_all_experts.py --skip-failed
        """
    )

    parser.add_argument(
        '--method',
        nargs='+',
        choices=['lora_moe', 'prompt_tuning', 'p_tuning', 'full_finetuning', 'all'],
        default=['all'],
        help='Training methods; may specify multiple (default: all)'
    )

    parser.add_argument(
        '--expert',
        choices=['text', 'image', 'uml', 'general', 'all'],
        default='all',
        help='Train only the specified expert (default: all)'
    )

    parser.add_argument(
        '--skip-failed',
        action='store_true',
        default=False,
        help='Continue automatically after failed tasks (default: stop on failure)'
    )

    args = parser.parse_args()

    print_header()

    methods_to_train = (
        list(TRAINING_TASKS.keys()) if 'all' in args.method
        else args.method
    )

    experts_to_train = (
        ['text', 'image', 'uml', 'general'] if args.expert == 'all'
        else [args.expert]
    )

    overall_start = time.time()
    results = []
    failed_tasks = []

    try:
        session_num = 1
        for method in methods_to_train:
            method_display = {
                'lora_moe': 'LoRA-MoE（混合专家微调）',
                'prompt_tuning': 'Prompt Tuning（软提示）',
                'p_tuning': 'P-Tuning v2（前缀微调）',
                'full_finetuning': '准全参数微调'
            }[method]

            total_method_time = sum(
                ESTIMATED_TIME[method].get(expert, 0)
                for expert in experts_to_train
                if expert in TRAINING_TASKS[method]
            )

            print_session_header(session_num, method_display, total_method_time)
            session_num += 1

            for expert in experts_to_train:
                if expert not in TRAINING_TASKS[method]:
                    print(f"\nSkipping: {method}/{expert} (commented out)")
                    continue

                script_path = TRAINING_TASKS[method][expert]
                success = run_training_task(method, expert, script_path)

                results.append({
                    'method': method,
                    'expert': expert,
                    'success': success
                })

                if not success:
                    failed_tasks.append((method, expert))

                    if args.skip_failed:
                        print("\n" + "=" * 80)
                        print("Task failed; skipping it and continuing with the next task")
                        print("=" * 80)
                        print(f"Failed task: {method}/{expert}")
                        print("Continuing...")
                        print("=" * 80 + "\n")
                        continue
                    else:
                        print("\n" + "=" * 80)
                        print("Training failed!")
                        print("=" * 80)
                        print(f"Failed task: {method}/{expert}")
                        print("Stopping execution.")
                        print("Tip: use --skip-failed to skip failed tasks automatically")
                        print("=" * 80 + "\n")
                        return 1

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Training interrupted by user")
        print("=" * 80 + "\n")
        return 1

    overall_elapsed = time.time() - overall_start

    print("\n\n" + "=" * 80)
    if failed_tasks:
        print(" " * 25 + "Training finished (some tasks failed)")
    else:
        print(" " * 28 + "Training complete!")
    print("=" * 80)
    print(f"\nTotal elapsed: {format_time(overall_elapsed)}")

    success_count = sum(1 for r in results if r['success'])
    print(f"Successful tasks: {success_count}/{len(results)}")

    if failed_tasks:
        print(f"Failed tasks: {len(failed_tasks)}/{len(results)}")

    print("\nResults:")
    print("-" * 80)

    for result in results:
        status = "✓ 成功" if result['success'] else "✗ 失败"
        print(f"  {result['method']:20s} / {result['expert']:10s} : {status}")

    if failed_tasks:
        print("\n" + "=" * 80)
        print("Failed task details:")
        print("=" * 80)
        for method, expert in failed_tasks:
            print(f"  - {method}/{expert}")
        print("\nSuggestions:")
        print("  1. Check the training logs for the affected expert: logs/training/")
        print("  2. For OOM errors, consider reducing the sequence length or batch size further")
        print("  3. For configuration errors, check config/settings.py")
        print("  4. Retrain failed experts individually if needed")

    print("=" * 80 + "\n")

    return 0 if not failed_tasks or args.skip_failed else 1


if __name__ == "__main__":
    sys.exit(main())
