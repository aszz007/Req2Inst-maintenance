#!/usr/bin/env python
"""Run all configured LoRA-MoE expert-training tasks."""

import subprocess
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent


class Colors:
    """Define ANSI terminal color codes."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.END}\n")


def print_success(message: str):
    """Print success."""
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")


def print_error(message: str):
    """Print error."""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")


def print_info(message: str):
    """Print info."""
    print(f"{Colors.CYAN}[INFO] {message}{Colors.END}")


def print_warning(message: str):
    """Print warning."""
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")


class TrainingTask:
    """Describe one training command and its metadata."""

    def __init__(self, task_id: int, expert_type: str, description: str = ""):
        self.task_id = task_id
        self.expert_type = expert_type  # 'uml', 'general'
        self.description = description
        self.script_path = PROJECT_ROOT / 'training' / 'lora_moe' / f'train_{expert_type}_expert.py'

    def get_command(self, test_mode: bool = False) -> List[str]:
        """Return command."""
        cmd = [
            sys.executable,
            str(self.script_path)
        ]


        return cmd

    def get_env_vars(self, test_mode: bool = False) -> Dict[str, str]:
        """Return env vars."""
        env_vars = {}
        if test_mode:
            env_vars['TRAIN_EPOCHS'] = '1'
        return env_vars

    def __str__(self):
        return f"Task {self.task_id}: {self.expert_type.upper()} Expert"


def create_all_tasks() -> List[TrainingTask]:
    """Create all tasks."""
    tasks = []
    task_id = 1

    tasks.append(TrainingTask(
        task_id=task_id,
        expert_type='text',
        description="文本需求转众包指令专家(text_dataset)"
    ))
    task_id += 1

    tasks.append(TrainingTask(
        task_id=task_id,
        expert_type='image',
        description="图像描述转标注指令专家(image_dataset)"
    ))
    task_id += 1

    tasks.append(TrainingTask(
        task_id=task_id,
        expert_type='uml',
        description="UML描述转换专家(1500条数据)"
    ))
    task_id += 1

    tasks.append(TrainingTask(
        task_id=task_id,
        expert_type='general',
        description="通用兜底专家(text + image + uml数据)"
    ))
    task_id += 1

    return tasks


def run_task(task: TrainingTask, test_mode: bool = False) -> bool:
    """Run task."""
    print_header(f"训练任务 {task.task_id}/4: {task.description}")
    print_info(str(task))

    if not task.script_path.exists():
        print_error(f"训练脚本不存在: {task.script_path}")
        return False

    cmd = task.get_command(test_mode)
    env_vars = task.get_env_vars(test_mode)

    print_info(f"执行命令: {' '.join(cmd)}")
    if env_vars:
        print_info(f"环境变量: {env_vars}")

    start_time = time.time()

    try:
        import os
        env = os.environ.copy()
        env.update(env_vars)

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            check=False
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print_success(f"任务{task.task_id}完成! 耗时: {elapsed_time/60:.1f}分钟")
            return True
        else:
            print_error(f"任务{task.task_id}失败! 返回码: {result.returncode}")
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print_error(f"任务{task.task_id}异常: {e}")
        print_error(f"已耗时: {elapsed_time/60:.1f}分钟")
        return False


def save_report(results: List[Dict], output_dir: Path, test_mode: bool):
    """Save report."""
    report = {
        'mode': 'test' if test_mode else 'full',
        'timestamp': datetime.now().isoformat(),
        'total_tasks': len(results),
        'successful_tasks': sum(1 for r in results if r['success']),
        'failed_tasks': sum(1 for r in results if not r['success']),
        'tasks': results
    }

    report_file = output_dir / f"batch_training_report_{'test' if test_mode else 'full'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_info(f"训练报告已保存: {report_file}")

    return report


def print_summary(results: List[Dict]):
    """Print summary."""
    print_header("批量训练总结")

    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful

    print(f"总任务数: {total}")
    print(f"{Colors.GREEN}成功: {successful}{Colors.END}")
    print(f"{Colors.RED}失败: {failed}{Colors.END}")
    print()

    if failed > 0:
        print(f"{Colors.RED}失败任务:{Colors.END}")
        for r in results:
            if not r['success']:
                print(f"  - 任务{r['task_id']}: {r['description']}")
    print()


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='批量训练所有Expert')

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--test', action='store_true',
                            help='测试模式:每个Expert仅训练1个epoch,快速验证流程')
    mode_group.add_argument('--all', action='store_true',
                            help='完整训练模式:训练所有4个Expert')
    mode_group.add_argument('--expert', type=str, choices=['text', 'image', 'uml', 'general'],
                            help='仅训练指定类型的Expert')

    parser.add_argument('--resume-from', type=int, metavar='N',
                        help='从第N个任务继续训练(1-4)')

    args = parser.parse_args()

    all_tasks = create_all_tasks()

    if args.expert:
        tasks = [t for t in all_tasks if t.expert_type == args.expert]
        if not tasks:
            print_error(f"没有找到类型为'{args.expert}'的任务")
            return 1
    else:
        tasks = all_tasks

    if args.resume_from:
        if args.resume_from < 1 or args.resume_from > len(tasks):
            print_error(f"无效的任务ID: {args.resume_from}")
            return 1
        tasks = [t for t in tasks if t.task_id >= args.resume_from]
        print_info(f"从任务{args.resume_from}开始训练")

    print_header("批量训练计划")
    mode_text = "测试模式(1 epoch)" if args.test else "完整训练模式"
    print(f"模式: {mode_text}")
    print(f"总任务数: {len(tasks)}")
    print("\n任务列表:")
    for task in tasks:
        print(f"  {task}")
    print()

    if not args.test:
        confirm = input(f"{Colors.YELLOW}确认开始训练? (yes/no): {Colors.END}")
        if confirm.lower() != 'yes':
            print_info("取消训练")
            return 0

    start_time = time.time()
    results = []

    for task in tasks:
        success = run_task(task, test_mode=args.test)
        results.append({
            'task_id': task.task_id,
            'expert_type': task.expert_type,
            'description': task.description,
            'success': success
        })

        if task != tasks[-1]:
            print_info("等待5秒后继续下一个任务...")
            time.sleep(5)

    total_time = time.time() - start_time

    output_dir = PROJECT_ROOT / 'outputs' / 'reports'
    output_dir.mkdir(parents=True, exist_ok=True)
    save_report(results, output_dir, args.test)

    print_summary(results)
    print(f"总耗时: {total_time/3600:.2f}小时")

    failed_count = sum(1 for r in results if not r['success'])
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
