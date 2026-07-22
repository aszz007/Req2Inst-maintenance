#!/usr/bin/env python3
"""Run Experiment 8 inference-efficiency benchmarks."""

import sys
import gc
import time
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
from src.baselines.inference_utils import save_experiment_results
from src.utils.logger import get_logger

logger = get_logger('experiments.exp8')

path_cfg = get_path_config()
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp8_inference_efficiency'
PLOTS_DIR = EXP_DIR / 'plots'


COLOR_MAP = {
    'bm25': '#4ECDC4',
    'lsa': '#45B7A0',
    'template': '#36A882',
    'zeroshot': '#ff7f0e',
    'lora_moe': '#1f77b4',
    'lora_single': '#ff9933',
    'p_tuning': '#d62728',
    'prompt_tuning': '#e74c3c',
    'full_finetuning': '#9467bd',
}


def _get_method_color(method):
    """Return method color."""
    return COLOR_MAP.get(method, '#999999')



N_WARMUP = 3
N_LATENCY = 50
N_THROUGHPUT = 100
N_LATENCY_TEST = 5
N_THROUGHPUT_TEST = 10


THROUGHPUT_BATCH = {
    'bm25': 'bulk',
    'lsa': 'bulk',
    'template': 'bulk',
    'zeroshot': 4,
    'lora_moe': 8,
    'lora_single': 8,
    'p_tuning': 1,
    'prompt_tuning': 1,
    'full_finetuning': 4,
}

METHOD_LABELS = {
    'bm25': 'BM25',
    'lsa': 'LSA',
    'template': 'Template',
    'zeroshot': 'Zero-Shot',
    'lora_moe': 'Multi-Expert LoRA',
    'lora_single': 'LoRA (Unified)',
    'p_tuning': 'P-Tuning v2',
    'prompt_tuning': 'Prompt Tuning',
    'full_finetuning': 'Full FT',
}

CPU_METHODS = {'bm25', 'lsa', 'template'}
GPU_METHODS = {'zeroshot', 'lora_moe', 'lora_single',
               'p_tuning', 'prompt_tuning', 'full_finetuning'}

METHOD_ORDER = [
    'bm25', 'lsa', 'template',
    'zeroshot', 'lora_moe', 'lora_single',
    'p_tuning', 'prompt_tuning', 'full_finetuning',
]

ALL_METHODS = list(METHOD_ORDER)



def _gpu_available():
    """Return whether a CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_sync():
    """Synchronize queued GPU work."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _clear_gpu():
    """Clear cached GPU memory."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_peak_mb():
    """Return peak GPU memory use in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0


def _gpu_current_mb():
    """Return current GPU memory use in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0



def _disk_size_mb(method):
    """Return file or directory size in MB."""
    try:
        ckpt_map = {
            'lora_moe': path_cfg.LORA_MOE_CKPTS.get('text'),
            'lora_single': getattr(path_cfg, 'LORA_SINGLE_CKPT', None),
            'p_tuning': path_cfg.PTUNING_CKPTS.get('text') if hasattr(path_cfg, 'PTUNING_CKPTS') else None,
            'prompt_tuning': path_cfg.PROMPT_TUNING_CKPTS.get('text') if hasattr(path_cfg,
                                                                                 'PROMPT_TUNING_CKPTS') else None,
            'full_finetuning': path_cfg.FULL_FINETUNING_CKPTS.get('text') if hasattr(path_cfg,
                                                                                     'FULL_FINETUNING_CKPTS') else None,
        }
        ckpt = ckpt_map.get(method)
        if ckpt is None:
            return 0.0
        ckpt = Path(ckpt)
        if not ckpt.exists():
            return 0.0
        total = sum(f.stat().st_size for f in ckpt.rglob('*') if f.is_file())
        return total / (1024 ** 2)
    except Exception:
        return 0.0


#     throughput_info: dict {n_samples, wall_time_s, batch_size, samples_per_sec}

def _benchmark_cpu_method(method, train_data, test_inputs, n_warmup, n_latency, n_throughput):
    """Benchmark a CPU baseline."""
    from src.baselines.ir_methods import BM25Retriever, LSARetriever
    from src.baselines.template_filling import TemplateFiller

    t0 = time.perf_counter()
    if method == 'bm25':
        obj = BM25Retriever()
        obj.build_index(train_data)
        predict_one = lambda inp: obj.batch_retrieve([inp])[0]
        predict_batch = lambda inps: obj.batch_retrieve(inps)
    elif method == 'lsa':
        obj = LSARetriever(n_components=100)
        obj.build_index(train_data)
        predict_one = lambda inp: obj.batch_retrieve([inp])[0]
        predict_batch = lambda inps: obj.batch_retrieve(inps)
    else:  # template
        obj = TemplateFiller()
        predict_one = lambda inp: obj.batch_fill([inp])[0]
        predict_batch = lambda inps: obj.batch_fill(inps)
    load_time = time.perf_counter() - t0

    for inp in test_inputs[:n_warmup]:
        _ = predict_one(inp)

    latencies = []
    for inp in test_inputs[n_warmup:n_warmup + n_latency]:
        t0 = time.perf_counter()
        _ = predict_one(inp)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    batch_inputs = test_inputs[:n_throughput]
    t0 = time.perf_counter()
    _ = predict_batch(batch_inputs)
    wall = time.perf_counter() - t0

    throughput_info = {
        'n_samples': len(batch_inputs),
        'wall_time_s': round(wall, 4),
        'batch_size': len(batch_inputs),
        'samples_per_sec': round(len(batch_inputs) / max(wall, 1e-9), 2),
    }
    return load_time, latencies, throughput_info, []


def _infer_one(gen_obj, inp, method):
    """Run inference for one sample."""
    if method == 'zeroshot':
        return gen_obj.batch_generate([inp], input_type='text', n_shots=0)
    else:
        return gen_obj.batch_generate_instruction([inp], batch_size=1)


def _infer_batch(gen_obj, inputs, method, batch_size):
    """Run batched inference."""
    if method == 'zeroshot':
        return gen_obj.batch_generate(inputs, input_type='text', n_shots=0)
    else:
        return gen_obj.batch_generate_instruction(inputs, batch_size=batch_size)


def _benchmark_gpu_method(method, test_inputs, n_warmup, n_latency, n_throughput):
    """Benchmark a GPU method."""
    import torch

    use_4bit = False
    batch_size = THROUGHPUT_BATCH.get(method, 4)

    if method == 'zeroshot':
        from src.baselines.zero_shot import ZeroShotGenerator
        logger.info('  [DEBUG] Loading base model without an adapter, use_4bit=False (FP16 for all methods)')
        _clear_gpu()
        t0 = time.perf_counter()
        gen = ZeroShotGenerator(use_4bit=False)
        if not gen.load_model():
            logger.error(f'{method}: failed to load model')
            return None, None, None
        load_time = time.perf_counter() - t0
    else:
        from src.experts import TextExpert
        ckpt_map = {
            'lora_moe': lambda: str(path_cfg.LORA_MOE_CKPTS['text']),
            'lora_single': lambda: str(getattr(path_cfg, 'LORA_SINGLE_CKPT', '')),
            'p_tuning': lambda: str(getattr(path_cfg, 'PTUNING_CKPTS', {}).get('text', '')),
            'prompt_tuning': lambda: str(getattr(path_cfg, 'PROMPT_TUNING_CKPTS', {}).get('text', '')),
            'full_finetuning': lambda: str(getattr(path_cfg, 'FULL_FINETUNING_CKPTS', {}).get('text', '')),
        }
        ckpt_path = ckpt_map[method]()
        if not ckpt_path or not Path(ckpt_path).exists():
            logger.error(f'{method}: checkpoint path is missing or not configured: {ckpt_path}')
            return None, None, None
        logger.info(f'  [DEBUG] Checkpoint path: {ckpt_path}')
        logger.info(f'  [DEBUG] use_4bit={use_4bit} (FP16 for all methods), throughput_batch_size={batch_size}')
        _clear_gpu()
        t0 = time.perf_counter()
        gen = TextExpert(lora_path=ckpt_path, use_4bit=use_4bit)
        if not gen.load_model():
            logger.error(f'{method}: failed to load model')
            return None, None, None
        load_time = time.perf_counter() - t0

    logger.info(f'  [DEBUG] GPU memory after model load: {_gpu_current_mb():.0f} MB (peak: {_gpu_peak_mb():.0f} MB)')
    logger.info(f'  [DEBUG] Throughput batch size: {batch_size} (FP16 inference for all methods)')

    for inp in test_inputs[:n_warmup]:
        _infer_one(gen, inp, method)

    latencies = []
    output_lengths = []
    for inp in test_inputs[n_warmup:n_warmup + n_latency]:
        _gpu_sync()
        t0 = time.perf_counter()
        result = _infer_one(gen, inp, method)
        _gpu_sync()
        latencies.append((time.perf_counter() - t0) * 1000)
        if isinstance(result, list) and len(result) > 0:
            out_text = result[0] if isinstance(result[0], str) else str(result[0])
        elif isinstance(result, str):
            out_text = result
        else:
            out_text = str(result) if result else ''
        output_lengths.append(len(out_text))
    if output_lengths:
        avg_len = sum(output_lengths) / len(output_lengths)
        min_len = min(output_lengths)
        max_len = max(output_lengths)
        logger.info(f'  [DEBUG] Output length: mean={avg_len:.0f} characters, '
                    f'min={min_len}, max={max_len}')

    batch_inputs = test_inputs[:n_throughput]
    _gpu_sync()
    t0 = time.perf_counter()
    _infer_batch(gen, batch_inputs, method, batch_size)
    _gpu_sync()
    wall = time.perf_counter() - t0

    gen.unload_model()

    throughput_info = {
        'n_samples': len(batch_inputs),
        'wall_time_s': round(wall, 4),
        'batch_size': batch_size,
        'samples_per_sec': round(len(batch_inputs) / max(wall, 1e-9), 2),
    }
    return load_time, latencies, throughput_info, output_lengths



def plot_latency_comparison(results_by_method, test_mode=False):
    """Plot latency comparison."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    medians = [results_by_method[m].get('latency_median_ms', 0) for m in methods]
    p95s = [results_by_method[m].get('latency_p95_ms', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    ax.barh(y, medians, color=colors, edgecolor='gray', height=0.55,
            label='Median')
    ax.scatter(p95s, y, marker='|', color='red', s=120, zorder=5, label='P95')
    for i, (med, p95) in enumerate(zip(medians, p95s)):
        offset = max(max(medians), 0.1) * 0.02
        ax.text(max(med, p95) + offset, i,
                f'{med:.1f}ms', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Latency per Sample (ms)')
    title = 'Exp8: Inference Latency Comparison (batch_size=1)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "latency_comparison.png"}')


def plot_latency_distribution(latencies_dict, test_mode=False):
    """Plot latency distribution."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in latencies_dict and len(latencies_dict[m]) > 0]
    if not methods:
        logger.warning('No latency data found; skipping box plot')
        return

    data = [latencies_dict[m] for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    bp = ax.boxplot(data, vert=False, patch_artist=True, labels=labels,
                    widths=0.5, showfliers=True,
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median_line in bp['medians']:
        median_line.set(color='black', linewidth=1.5)

    ax.set_xlabel('Latency per Sample (ms)')
    title = 'Exp8: Latency Distribution (Box Plot)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "latency_distribution.png"}')


def plot_throughput_comparison(results_by_method, test_mode=False):
    """Plot throughput comparison."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    throughputs = [results_by_method[m].get('throughput_samples_per_sec', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, throughputs, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, throughputs):
        offset = max(max(throughputs), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Throughput (samples/sec)')
    title = 'Exp8: Throughput Comparison (Optimal Batch Size)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'throughput_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "throughput_comparison.png"}')


def plot_gpu_memory_comparison(results_by_method, test_mode=False):
    """Plot GPU memory comparison."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    mem_vals = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, mem_vals, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, mem_vals):
        offset = max(max(mem_vals), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f} MB', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Peak GPU Memory (MB)')
    title = 'Exp8: GPU Memory Comparison'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    legend_handles = []
    legend_labels = []
    if 'lora_moe' in methods:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['lora_moe']))
        legend_labels.append('Multi-Expert LoRA')
    soft_prompt = {'p_tuning', 'prompt_tuning'}
    if soft_prompt & set(methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['p_tuning']))
        legend_labels.append('Soft-Prompt Methods')
    other_methods = {'zeroshot', 'lora_single', 'full_finetuning'}
    if other_methods & set(methods):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP['lora_single']))
        legend_labels.append('Other Methods')
    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels, fontsize=8,
                  title='All GPU methods: FP16', title_fontsize=7)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'gpu_memory_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "gpu_memory_comparison.png"}')


def plot_load_time_comparison(results_by_method, test_mode=False):
    """Plot load time comparison."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    load_times = [results_by_method[m].get('load_time_s', 0) for m in methods]
    colors = [_get_method_color(m) for m in methods]
    labels = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.7)))
    y = np.arange(len(methods))
    bars = ax.barh(y, load_times, color=colors, edgecolor='gray', height=0.55)
    for bar, val in zip(bars, load_times):
        offset = max(max(load_times), 0.1) * 0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}s', va='center', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Load Time (seconds)')
    title = 'Exp8: Model Load Time Comparison'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'load_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "load_time_comparison.png"}')


def plot_combined_efficiency(results_by_method, test_mode=False):
    """Plot combined efficiency."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method and m in GPU_METHODS]
    if len(methods) < 2:
        return

    latencies = [results_by_method[m].get('latency_median_ms', 0) for m in methods]
    memories = [results_by_method[m].get('peak_gpu_memory_mb', 0) for m in methods]
    adapter_sizes = [results_by_method[m].get('adapter_size_mb', 1) for m in methods]
    max_adapter = max(adapter_sizes) if max(adapter_sizes) > 0 else 1
    bubble_sizes = [max(40, (s / max_adapter) * 400) for s in adapter_sizes]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(methods):
        color = _get_method_color(m)
        edge = 'black' if m == 'lora_moe' else 'gray'
        lw = 2 if m == 'lora_moe' else 0.5
        ax.scatter(latencies[i], memories[i], s=bubble_sizes[i],
                   c=color, edgecolors=edge, linewidths=lw, alpha=0.8, zorder=5)
        ax.annotate(METHOD_LABELS.get(m, m),
                    (latencies[i], memories[i]),
                    textcoords='offset points', xytext=(8, 8), fontsize=8)

    ax.set_xlabel('Median Latency (ms/sample)')
    ax.set_ylabel('Peak GPU Memory (MB)')
    title = 'Exp8: Latency vs Memory (bubble = adapter size)'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'latency_vs_memory.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "latency_vs_memory.png"}')


def plot_summary_table(results_by_method, test_mode=False):
    """Plot summary table."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    methods = [m for m in METHOD_ORDER if m in results_by_method]
    if not methods:
        return

    columns = ['Method', 'Device', 'Quant', 'Load(s)', 'Latency(ms)', 'P95(ms)',
               'Thru(/s)', 'Memory(MB)', 'Adapter(MB)']
    cell_data = []
    for m in methods:
        e = results_by_method[m]
        cell_data.append([
            e['label'],
            e['device'],
            e['quantisation'],
            f"{e['load_time_s']:.2f}",
            f"{e['latency_median_ms']:.1f}",
            f"{e['latency_p95_ms']:.1f}",
            f"{e['throughput_samples_per_sec']:.1f}",
            f"{e['peak_gpu_memory_mb']:.0f}",
            f"{e['adapter_size_mb']:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(14, max(3, len(methods) * 0.45 + 1.5)))
    ax.axis('off')

    table = ax.table(cellText=cell_data, colLabels=columns, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i, m in enumerate(methods):
        row_idx = i + 1
        if m == 'lora_moe':
            for j in range(len(columns)):
                cell = table[row_idx, j]
                cell.set_facecolor('#d6eaf8')
                cell.set_text_props(fontweight='bold')
        else:
            for j in range(len(columns)):
                cell = table[row_idx, j]
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')

    title = 'Exp8: Inference Efficiency Summary'
    if test_mode:
        title += ' [Test Mode]'
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f'Plot saved to: {PLOTS_DIR / "summary_table.png"}')



def generate_report(results, results_by_method, test_mode=False):
    """Generate report."""
    lines = []
    lines.append('# 实验8: 推理效率基准测试报告\n')
    lines.append(f'**生成时间**: {results.get("timestamp", "N/A")}\n')
    if test_mode:
        lines.append('> **注意**: 本报告在测试模式下生成, 样本量较少, 仅供验证流程使用。\n')

    hw = results.get('hardware', {})
    lines.append('## 1. 实验环境\n')
    lines.append(f'- GPU: {hw.get("gpu_name", "N/A")}')
    lines.append(f'- GPU显存: {hw.get("gpu_memory_total_mb", "N/A")} MB')
    lines.append(f'- CUDA: {hw.get("cuda_version", "N/A")}')
    lines.append(f'- PyTorch: {hw.get("torch_version", "N/A")}')
    lines.append(f'- CPU核心数: {hw.get("cpu_count", "N/A")}')
    lines.append(f'- 内存: {hw.get("ram_total_gb", "N/A")} GB')
    lines.append('')

    lines.append('## 2. 测试配置\n')
    lines.append(f'- 预热样本数: {results.get("n_warmup", "N/A")}')
    lines.append(f'- 延迟测量样本数: {results.get("n_latency", "N/A")}')
    lines.append(f'- 吞吐测量样本数: {results.get("n_throughput", "N/A")}')
    lines.append('')

    lines.append('## 3. 结果汇总\n')
    lines.append(
        '| 方法 | 设备 | 量化 | 加载(s) | 延迟(ms) | P95(ms) | Min(ms) | Max(ms) | 吞吐(/s) | 显存(MB) | Adapter(MB) |')
    lines.append(
        '|------|------|------|---------|----------|---------|---------|---------|----------|----------|-------------|')
    for m in METHOD_ORDER:
        if m not in results_by_method:
            continue
        e = results_by_method[m]
        highlight = '**' if m == 'lora_moe' else ''
        lines.append(
            f'| {highlight}{e["label"]}{highlight} | {e["device"]} | {e["quantisation"]} | '
            f'{e["load_time_s"]:.2f} | {e["latency_median_ms"]:.1f} | '
            f'{e["latency_p95_ms"]:.1f} | {e.get("latency_min_ms", 0):.1f} | '
            f'{e.get("latency_max_ms", 0):.1f} | {e["throughput_samples_per_sec"]:.1f} | '
            f'{e["peak_gpu_memory_mb"]:.0f} | {e["adapter_size_mb"]:.1f} |'
        )
    lines.append('')

    lines.append('## 4. 分析要点\n')

    gpu_entries = [(m, results_by_method[m]) for m in METHOD_ORDER
                   if m in results_by_method and m in GPU_METHODS]
    if gpu_entries:
        best_latency = min(gpu_entries, key=lambda x: x[1]['latency_median_ms'])
        best_throughput = max(gpu_entries, key=lambda x: x[1]['throughput_samples_per_sec'])
        lowest_mem = min(gpu_entries, key=lambda x: x[1]['peak_gpu_memory_mb'])

        lines.append(f'- **延迟最低 (GPU)**: {METHOD_LABELS[best_latency[0]]} '
                     f'({best_latency[1]["latency_median_ms"]:.1f} ms)')
        lines.append(f'- **吞吐最高 (GPU)**: {METHOD_LABELS[best_throughput[0]]} '
                     f'({best_throughput[1]["throughput_samples_per_sec"]:.1f} samples/sec)')
        lines.append(f'- **显存最低 (GPU)**: {METHOD_LABELS[lowest_mem[0]]} '
                     f'({lowest_mem[1]["peak_gpu_memory_mb"]:.0f} MB)')

        if 'lora_moe' in results_by_method:
            moe = results_by_method['lora_moe']
            lines.append('\n### Multi-Expert LoRA 效率分析\n')
            lines.append(f'- 加载时间: {moe["load_time_s"]:.2f}s')
            lines.append(f'- 中位延迟: {moe["latency_median_ms"]:.1f}ms '
                         f'(P95={moe["latency_p95_ms"]:.1f}ms, '
                         f'Std={moe["latency_std_ms"]:.1f}ms)')
            lines.append(f'- 延迟范围: {moe.get("latency_min_ms", 0):.1f}ms ~ '
                         f'{moe.get("latency_max_ms", 0):.1f}ms')
            lines.append(f'- 吞吐量: {moe["throughput_samples_per_sec"]:.1f} samples/sec '
                         f'(batch_size={moe["throughput_batch_size"]})')
            lines.append(f'- GPU显存: {moe["peak_gpu_memory_mb"]:.0f} MB')
            lines.append(f'- Adapter大小: {moe["adapter_size_mb"]:.1f} MB')
    lines.append('')

    lines.append('## 5. 可视化图表\n')
    plot_descriptions = [
        ('latency_comparison.png', '延迟对比柱状图 (中位数+P95标记)'),
        ('latency_distribution.png', '延迟分布箱线图'),
        ('throughput_comparison.png', '吞吐量对比柱状图'),
        ('gpu_memory_comparison.png', 'GPU显存对比柱状图'),
        ('load_time_comparison.png', '模型加载时间对比'),
        ('latency_vs_memory.png', '延迟-显存权衡散点气泡图'),
        ('summary_table.png', '论文级综合汇总表格'),
    ]
    for fname, desc in plot_descriptions:
        lines.append(f'- `plots/{fname}`: {desc}')
    lines.append('')

    report_path = EXP_DIR / 'report.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f'Experiment report saved to: {report_path}')



def run(args):
    """Run the workflow."""
    logger.info('Experiment 8: Inference efficiency benchmark')

    n_latency = N_LATENCY_TEST if args.test_mode else N_LATENCY
    n_throughput = N_THROUGHPUT_TEST if args.test_mode else N_THROUGHPUT
    n_warmup = min(N_WARMUP, 1) if args.test_mode else N_WARMUP

    logger.info('Loading text dataset...')
    loader = TextDatasetLoader()
    all_data = loader.load_csv_files()
    train_data, _, test_data = split_dataset_for_expert(all_data, 'text')
    test_inputs = [d['input'] for d in test_data]
    n_needed = n_warmup + max(n_latency, n_throughput)
    if len(test_inputs) < n_needed:
        logger.warning(f'Test set has only {len(test_inputs)} samples; {n_needed} are required, so samples will be reused cyclically')
        while len(test_inputs) < n_needed:
            test_inputs = test_inputs + test_inputs
    logger.info(f'Test samples: {len(test_data)} | latency measurements: {n_latency} | throughput measurements: {n_throughput}')

    methods_to_run = list(METHOD_ORDER)
    if args.methods:
        methods_to_run = [m.strip() for m in args.methods.split(',')]
    if args.skip:
        skip_set = set(m.strip() for m in args.skip.split(','))
        methods_to_run = [m for m in methods_to_run if m not in skip_set]

    results = {
        'experiment': 'exp8_inference_efficiency',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': args.test_mode,
        'n_warmup': n_warmup,
        'n_latency': n_latency,
        'n_throughput': n_throughput,
        'hardware': _get_hardware_info(),
        'methods': {},
    }
    results_by_method = {}
    latencies_dict = {}

    for method in methods_to_run:
        logger.info(f'Benchmarking: {METHOD_LABELS.get(method, method)}')

        try:
            if method in CPU_METHODS:
                load_time, latencies, tp_info, out_lens = _benchmark_cpu_method(
                    method, train_data, test_inputs, n_warmup, n_latency, n_throughput
                )
                peak_mem = 0.0
            elif method in GPU_METHODS:
                load_time, latencies, tp_info, out_lens = _benchmark_gpu_method(
                    method, test_inputs, n_warmup, n_latency, n_throughput
                )
                if load_time is None:
                    logger.warning(f'{method}: skipped because model loading failed')
                    continue
                peak_mem = _gpu_peak_mb()
                _clear_gpu()
            else:
                logger.warning(f'Unknown method: {method}')
                continue

            latencies_arr = np.array(latencies) if latencies else np.array([0])
            adapter_mb = _disk_size_mb(method)

            latencies_dict[method] = latencies if latencies else []

            out_len_stats = {}
            if out_lens:
                out_arr = np.array(out_lens)
                out_len_stats = {
                    'output_length_mean': round(float(np.mean(out_arr)), 1),
                    'output_length_min': int(np.min(out_arr)),
                    'output_length_max': int(np.max(out_arr)),
                }

            entry = {
                'method': method,
                'label': METHOD_LABELS.get(method, method),
                'device': 'CPU' if method in CPU_METHODS else 'GPU',
                'quantisation': 'N/A' if method in CPU_METHODS else 'FP16',
                'load_time_s': round(load_time, 3),
                'latency_mean_ms': round(float(np.mean(latencies_arr)), 2),
                'latency_median_ms': round(float(np.median(latencies_arr)), 2),
                'latency_p95_ms': round(float(np.percentile(latencies_arr, 95)), 2),
                'latency_min_ms': round(float(np.min(latencies_arr)), 2),
                'latency_max_ms': round(float(np.max(latencies_arr)), 2),
                'latency_std_ms': round(float(np.std(latencies_arr)), 2),
                'latency_n_samples': len(latencies),
                'throughput_samples_per_sec': tp_info['samples_per_sec'],
                'throughput_batch_size': tp_info['batch_size'],
                'throughput_wall_s': tp_info['wall_time_s'],
                'throughput_n_samples': tp_info['n_samples'],
                'peak_gpu_memory_mb': round(peak_mem, 1),
                'adapter_size_mb': round(adapter_mb, 2),
                **out_len_stats,
            }
            results['methods'][method] = entry
            results_by_method[method] = entry

            logger.info(
                f'  Load time:          {load_time:.2f}s\n'
                f'  Median latency:     {entry["latency_median_ms"]:.1f}ms  '
                f'(P95={entry["latency_p95_ms"]:.1f}ms, '
                f'Min={entry.get("latency_min_ms", 0):.1f}ms, '
                f'Max={entry.get("latency_max_ms", 0):.1f}ms)\n'
                f'  Throughput:         {tp_info["samples_per_sec"]:.1f} samples/sec '
                f'(batch={tp_info["batch_size"]})\n'
                f'  GPU memory:         {peak_mem:.0f} MB\n'
                f'  Adapter:    {adapter_mb:.1f} MB'
            )
        except Exception as e:
            logger.error(f'{method}: benchmark failed: {e}')
            logger.error(traceback.format_exc())
            _clear_gpu()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    save_experiment_results(results, EXP_DIR, 'results.json')

    try:
        if results_by_method:
            plot_latency_comparison(results_by_method, args.test_mode)
            plot_latency_distribution(latencies_dict, args.test_mode)
            plot_throughput_comparison(results_by_method, args.test_mode)
            plot_gpu_memory_comparison(results_by_method, args.test_mode)
            plot_load_time_comparison(results_by_method, args.test_mode)
            plot_combined_efficiency(results_by_method, args.test_mode)
            plot_summary_table(results_by_method, args.test_mode)
    except Exception as e:
        logger.warning(f'Plotting failed: {e}')
        logger.warning(traceback.format_exc())

    try:
        generate_report(results, results_by_method, args.test_mode)
    except Exception as e:
        logger.warning(f'Report generation failed: {e}')
        logger.warning(traceback.format_exc())

    logger.info('Inference efficiency summary')
    logger.info(
        f'{"Method":<18} {"Device":<6} {"Quant.":<6} '
        f'{"Load(s)":>8} {"Latency(ms)":>10} {"P95(ms)":>10} '
        f'{"Min(ms)":>10} {"Max(ms)":>10} '
        f'{"Throughput(/s)":>10} {"Memory(MB)":>10} {"Adapter(MB)":>12}'
    )
    for m in METHOD_ORDER:
        if m not in results_by_method:
            continue
        e = results_by_method[m]
        logger.info(
            f'{e["label"]:<18} {e["device"]:<6} {e["quantisation"]:<6} '
            f'{e["load_time_s"]:>8.2f} {e["latency_median_ms"]:>10.1f} '
            f'{e["latency_p95_ms"]:>10.1f} {e.get("latency_min_ms", 0):>10.1f} '
            f'{e.get("latency_max_ms", 0):>10.1f} {e["throughput_samples_per_sec"]:>10.1f} '
            f'{e["peak_gpu_memory_mb"]:>10.0f} {e["adapter_size_mb"]:>12.1f}'
        )
    # Diagnostic: output length vs latency correlation
    gpu_with_outlen = [(m, results_by_method[m]) for m in METHOD_ORDER
                       if m in results_by_method and m in GPU_METHODS
                       and 'output_length_mean' in results_by_method[m]]
    if gpu_with_outlen:
        logger.info('Diagnostic: Output Length vs Latency Correlation')
        logger.info(f'{"Method":<18} {"Latency(ms)":>12} {"AvgOutput(ch)":>14} {"Min":>8} {"Max":>8} {"ms/char":>10}')
        for m, e in gpu_with_outlen:
            avg_out = e.get('output_length_mean', 0)
            ms_per_char = e['latency_median_ms'] / max(avg_out, 1)
            logger.info(
                f'{e["label"]:<18} {e["latency_median_ms"]:>12.1f} '
                f'{avg_out:>14.0f} {e.get("output_length_min", 0):>8} '
                f'{e.get("output_length_max", 0):>8} {ms_per_char:>10.2f}'
            )

    # Save debug diagnostics JSON
    try:
        import json as _json
        diag = {
            'checkpoint_paths': {},
            'output_length_analysis': {},
            'batch_efficiency': {},
        }
        try:
            diag['checkpoint_paths'] = {
                'lora_moe_text': str(path_cfg.LORA_MOE_CKPTS.get('text', '')),
                'lora_single': str(getattr(path_cfg, 'LORA_SINGLE_CKPT', '')),
                'p_tuning_text': str(getattr(path_cfg, 'PTUNING_CKPTS', {}).get('text', '')),
                'prompt_tuning_text': str(getattr(path_cfg, 'PROMPT_TUNING_CKPTS', {}).get('text', '')),
                'full_ft_text': str(getattr(path_cfg, 'FULL_FINETUNING_CKPTS', {}).get('text', '')),
            }
        except Exception:
            pass
        for m in METHOD_ORDER:
            if m in results_by_method and 'output_length_mean' in results_by_method[m]:
                e = results_by_method[m]
                avg_out = e.get('output_length_mean', 0)
                diag['output_length_analysis'][m] = {
                    'avg_output_chars': avg_out,
                    'latency_median_ms': e['latency_median_ms'],
                    'ms_per_char': round(e['latency_median_ms'] / max(avg_out, 1), 2),
                }
        for m in METHOD_ORDER:
            if m in results_by_method and m in GPU_METHODS:
                e = results_by_method[m]
                ps_batch = e['throughput_wall_s'] / max(e['throughput_n_samples'], 1) * 1000
                diag['batch_efficiency'][m] = {
                    'latency_median_ms': e['latency_median_ms'],
                    'per_sample_in_batch_ms': round(ps_batch, 1),
                    'batch_speedup': round(e['latency_median_ms'] / max(ps_batch, 0.1), 1),
                    'batch_size': e['throughput_batch_size'],
                }
        diag_path = EXP_DIR / 'debug_diagnostics.json'
        with open(diag_path, 'w', encoding='utf-8') as df:
            _json.dump(diag, df, indent=2, ensure_ascii=False)
        logger.info(f'Debug diagnostics saved: {diag_path}')
    except Exception as diag_err:
        logger.warning(f'Failed to save diagnostics: {diag_err}')

    logger.info(f'\nResults saved to: {EXP_DIR}')


def _get_hardware_info():
    """Return hardware info."""
    info = {}
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total_mb'] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            )
            info['cuda_version'] = torch.version.cuda or 'N/A'
        info['torch_version'] = torch.__version__
    except Exception:
        pass
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['ram_total_gb'] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        import os
        info['cpu_count'] = os.cpu_count()
    return info


def main():
    """Run the command-line entry point."""
    global N_LATENCY, N_THROUGHPUT

    parser = argparse.ArgumentParser(description='Experiment 8: inference efficiency benchmark')
    parser.add_argument('--test-mode', action='store_true',
                        help='Use the minimum sample count for quick pipeline validation')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated list of methods to test (default: all), '
                             'for example "lora_moe,zeroshot,p_tuning"')
    parser.add_argument('--skip', type=str, default=None,
                        help='Comma-separated list of methods to skip, '
                             'for example "bm25,lsa,template" to skip CPU baselines')
    parser.add_argument('--n-latency', type=int, default=None,
                        help=f'Override the sample count for latency measurement (default: {N_LATENCY})')
    parser.add_argument('--n-throughput', type=int, default=None,
                        help=f'Override the sample count for throughput measurement (default: {N_THROUGHPUT})')
    args = parser.parse_args()
    if args.n_latency is not None:
        N_LATENCY = args.n_latency
    if args.n_throughput is not None:
        N_THROUGHPUT = args.n_throughput

    run(args)


if __name__ == '__main__':
    main()
