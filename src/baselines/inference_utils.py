"""Shared cache I/O, diagnostics, and metric adapters for evaluation experiments.

Experiment scripts use one cache schema that is consumed by the cached-
prediction CLI in `scripts/evaluation/calculate_metrics_from_json.py`.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.enhanced_metrics import EnhancedMetrics  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger('baselines.inference_utils')


_DIAGNOSTICS_SAMPLE_COUNT = 5


def _build_diagnostics(samples: list[dict]) -> dict:
    """Build compact format, length, and sample diagnostics for debugging."""
    total = len(samples)
    empty_count = 0
    format_counts = {'has_definition': 0, 'has_emphasis': 0, 'has_avoid': 0, 'all_three': 0}
    pred_lengths = []
    ref_lengths = []
    starts: dict[str, int] = {}

    for s in samples:
        pred = s.get('prediction', '') or ''
        ref = s.get('reference', '') or ''

        if not pred.strip():
            empty_count += 1
            continue

        pred_lengths.append(len(pred))
        ref_lengths.append(len(ref))

        has_def = 'Definition:' in pred
        has_emph = 'Emphasis' in pred
        has_avoid = 'Things to Avoid' in pred
        if has_def:
            format_counts['has_definition'] += 1
        if has_emph:
            format_counts['has_emphasis'] += 1
        if has_avoid:
            format_counts['has_avoid'] += 1
        if has_def and has_emph and has_avoid:
            format_counts['all_three'] += 1

        start = pred.strip()[:60]
        starts[start] = starts.get(start, 0) + 1

    top_starts = sorted(starts.items(), key=lambda x: -x[1])[:5]

    def _avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 0.0

    valid_count = total - empty_count
    representative = []
    step = max(1, total // _DIAGNOSTICS_SAMPLE_COUNT)
    for i in range(0, total, step):
        if len(representative) >= _DIAGNOSTICS_SAMPLE_COUNT:
            break
        s = samples[i]
        pred = (s.get('prediction') or '')[:200]
        ref = (s.get('reference') or '')[:200]
        representative.append({'index': s.get('index', i), 'pred_preview': pred, 'ref_preview': ref})

    return {
        'total': total,
        'empty_predictions': empty_count,
        'valid_predictions': valid_count,
        'format_compliance': {
            k: {'count': v, 'rate': round(v / valid_count, 4) if valid_count else 0.0}
            for k, v in format_counts.items()
        },
        'pred_length': {
            'min': min(pred_lengths) if pred_lengths else 0,
            'max': max(pred_lengths) if pred_lengths else 0,
            'avg': _avg(pred_lengths),
        },
        'ref_length': {
            'min': min(ref_lengths) if ref_lengths else 0,
            'max': max(ref_lengths) if ref_lengths else 0,
            'avg': _avg(ref_lengths),
        },
        'top_repeated_starts': [
            {'start': s, 'count': c} for s, c in top_starts
        ],
        'representative_samples': representative,
    }


def save_predictions_cache(
    samples: list[dict],
    method: str,
    expert_type: str,
    config: dict,
    cache_dir: Path,
    filename: str = None
) -> Path:
    """
    Save predictions to a cache JSON file.

    The format matches what calculate_metrics_from_json.py expects:
      {
        "method": ...,
        "expert_type": ...,
        "expert_name": ...,
        "config": {...},
        "timestamp": ...,
        "total_samples": N,
        "diagnostics": { ...compact debugging statistics... },
        "samples": [{"index": 0, "input": ..., "prediction": ..., "reference": ...}]
      }

    Full prediction and reference text is stored in each sample so that
    compute_all_metrics() operates on complete strings. A 'diagnostics'
    section is prepended with format compliance stats, length distribution,
    top repeated starts, and representative sample previews (truncated only
    there for display compactness).

    Args:
        samples: List of dicts with keys: index, input, prediction, reference
        method: Method name, e.g. 'lora_moe', 'bm25', 'zero_shot'
        expert_type: Expert type, e.g. 'text', 'image', 'uml', 'general'
        config: Arbitrary config dict to store alongside predictions
        cache_dir: Directory to write cache file into (created if needed)
        filename: Override filename. Defaults to '{expert_type}_predictions.json'

    Returns:
        Path to the saved cache file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f'{expert_type}_predictions.json'

    filepath = cache_dir / filename

    expert_name = f'{expert_type}_expert'

    diagnostics = _build_diagnostics(samples)

    # Store full text so that metric computation uses complete prediction and
    # reference strings, preventing score underestimation from truncation.
    stored_samples = []
    for s in samples:
        stored_samples.append({
            'index': s.get('index', 0),
            'input': s.get('input', '') or '',
            'prediction': s.get('prediction', '') or '',
            'reference': s.get('reference', '') or '',
        })

    payload = {
        'method': method,
        'expert_type': expert_type,
        'expert_name': expert_name,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(samples),
        'diagnostics': diagnostics,
        'samples': stored_samples,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info(f'Cache saved: {filepath} ({len(samples)} samples)')
    return filepath


def load_predictions_cache(cache_dir: Path, filename: str) -> dict | None:
    """Load predictions cache."""
    filepath = Path(cache_dir) / filename
    if not filepath.exists():
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f'Cache loaded: {filepath} ({data.get("total_samples", "?")} samples)')
    return data


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    use_bertscore: bool = True
) -> dict:
    """
    Run the canonical EnhancedMetrics workflow and preserve experiment keys.

    Empty predictions are filtered by EnhancedMetrics, and the returned wrapper
    keeps the result structure consumed by experiments 1-11.
    """
    metrics = EnhancedMetrics(use_bertscore=use_bertscore)

    try:
        report = metrics.generate_comprehensive_report(
            predictions=predictions,
            references=references
        )
        if not report:
            return {}

        result = {
            'total_samples': report['total_samples'],
            'valid_samples': report['valid_samples'],
            'skipped_samples': report['skipped_samples'],
            'generation_quality': report['generation_quality'],
            'format_metrics': report['format_metrics'],
            'binary_classification': report['binary_classification'],
            'statistical_metrics': report['statistical_metrics'],
        }

        quality = result['generation_quality']
        binary = result['binary_classification']
        logger.info(
            f'Metric computation complete | ROUGE-L={quality.get("rougeL", 0):.4f} '
            f'F1={binary.get("f1_score", 0):.4f}'
        )

        return result
    finally:
        try:
            metrics.cleanup()
        except Exception:
            pass
        del metrics
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def save_experiment_results(
    results: dict,
    exp_dir: Path,
    filename: str = 'results.json'
) -> Path:
    """
    Save experiment results to a JSON file.

    Args:
        results: Results dict to serialize
        exp_dir: Output directory (created if needed)
        filename: Output filename

    Returns:
        Path to the saved file
    """
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    filepath = exp_dir / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f'Experiment results saved: {filepath}')
    return filepath
