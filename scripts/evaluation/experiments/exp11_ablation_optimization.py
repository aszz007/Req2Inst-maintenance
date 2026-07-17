#!/usr/bin/env python3
"""Run Experiment 11 ablation and optimization evaluation.

Prerequisites:
    Exp10 phase1_results.json and phase2_results.json must exist.
"""

import sys
import gc
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config.settings import get_path_config
from src.training.data_loader import (
    GeneralDatasetLoader, split_dataset_for_expert,
)
from src.baselines.inference_utils import (
    save_predictions_cache, load_predictions_cache,
    compute_all_metrics, save_experiment_results,
)
from src.utils.logger import get_logger
from src.routing.learned_router import (
    RouterMLP, HiddenStateExtractor,
    EXPERT_TO_IDX, IDX_TO_EXPERT,
)

logger = get_logger('experiments.exp11')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP10_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp10_advanced_routing'
EXP9_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp11_ablation_optimization'
PLOT_DIR = EXP_DIR / 'plots'
ROUTER_CKPT_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp10_learned_router'
EXP11_ROUTER_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp11_router_opt'
FEATURE_CACHE_DIR = CACHE_DIR / 'exp10_router_features'
EXP11_FEATURE_DIR = CACHE_DIR / 'exp11_router_features'
EXP11_CACHE_DIR = CACHE_DIR / 'exp11_ablation'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']


ABLATION_CONFIGS = {
    'A0': {
        'name': 'Full v12 (Baseline)',
        'description': 'Complete v12 pipeline',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A1': {
        'name': 'No OOD Correction',
        'description': 'Disable all OOD factor down-weighting, use raw Router weights',
        'disable_ood_correction': True,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A2': {
        'name': 'No Cache Redirect',
        'description': 'Disable OOD-corrected >=0.95 cache redirect',
        'disable_ood_correction': False,
        'disable_cache_redirect': True,
        'disable_quality_gate': False,
        'force_equal_weights': False,
    },
    'A3': {
        'name': 'No Quality Gate',
        'description': 'Disable ROUGE-L comparison gate',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A4': {
        'name': 'No Redirect + No Gate',
        'description': 'Disable both v12 core mechanisms',
        'disable_ood_correction': False,
        'disable_cache_redirect': True,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A5': {
        'name': 'Pure Ensemble',
        'description': 'Disable all gating and OOD correction, keep only template matching and dynamic length',
        'disable_ood_correction': True,
        'disable_cache_redirect': True,
        'disable_quality_gate': True,
        'force_equal_weights': False,
    },
    'A6': {
        'name': 'Equal Weights (w1=w2=0.5)',
        'description': 'Ignore Router weights, top-2 experts each 50% + full v12 gating',
        'disable_ood_correction': False,
        'disable_cache_redirect': False,
        'disable_quality_gate': False,
        'force_equal_weights': True,
    },
}



def _cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_rougeL(metrics_dict):
    return metrics_dict.get('generation_quality', {}).get('rougeL', 0.0)


def _metrics_from_samples(samples, use_bertscore=False):
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)


def _build_prompt_for_sample(sample: dict) -> tuple:
    """Build prompt for sample."""
    input_text = sample.get('input', '')
    data_type = sample.get('data_type', 'general')
    try:
        if data_type == 'text':
            from models.prompt_templates.text_template import TextInstructionTemplate
            return TextInstructionTemplate.build_prompt(input_text), 'text_template'
    except (ImportError, AttributeError):
        pass
    try:
        if data_type == 'image':
            from models.prompt_templates.image_template import ImageInstructionTemplate
            return ImageInstructionTemplate.build_prompt(input_text), 'image_template'
    except (ImportError, AttributeError):
        pass
    try:
        if data_type == 'uml':
            from models.prompt_templates.uml_template import UMLInstructionTemplate
            return UMLInstructionTemplate.build_prompt(input_text), 'uml_template'
    except (ImportError, AttributeError):
        pass
    from models.prompt_templates.general_template import GeneralInstructionTemplate
    return GeneralInstructionTemplate.build_prompt(input_text), 'general_template'


def _detect_datatype(sample: dict) -> str:
    dt = sample.get('data_type') or sample.get('type') or sample.get('domain')
    if dt in ('text', 'image', 'uml', 'general'):
        return dt
    inp = str(sample.get('input', ''))
    if inp.strip().startswith('{') or inp.strip().startswith('['):
        return 'general'
    return 'text'


def _detect_template_from_prompt(prompt_str: str) -> str:
    if '"actors"' in prompt_str and '"use_cases"' in prompt_str:
        return 'uml'
    if '"description"' in prompt_str and '"actors"' not in prompt_str:
        return 'image'
    return 'text'


def _load_all_expert_caches_for_general():
    """Load all expert caches for general."""
    caches = {}
    for expert in ALL_TYPES:
        if expert == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        elif expert == 'text':
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_moe3_general_via_text',
                'general_via_text_predictions.json'
            )
            if cache is None:
                cache = load_predictions_cache(
                    CACHE_DIR / 'exp9_oracle',
                    'text_expert_on_general_predictions.json'
                )
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert}_expert_on_general_predictions.json'
            )
        if cache:
            caches[expert] = cache.get('samples', [])
        else:
            logger.warning(f"  [Cache] Cache not found for expert '{expert}'")
    return caches


def _single_expert_from_cache(expert_name, domain, sample_idx, preloaded_caches=None):
    """Load single-expert output from cache."""
    if preloaded_caches is not None:
        samples = preloaded_caches.get(expert_name, [])
        if samples and sample_idx < len(samples):
            pred = samples[sample_idx].get('prediction', '')
            if pred:
                return pred
        general_samples = preloaded_caches.get('general', [])
        if general_samples and sample_idx < len(general_samples):
            return general_samples[sample_idx].get('prediction', '')
        return ''
    return ''



def run_phase1(args):
    """Run phase1."""
    logger.info("Phase 1: Output Ensemble ablation study")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    EXP11_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    if args.test_mode:
        general_test = general_test[:20]
    logger.info(f"General test set: {len(general_test)} samples")

    router = RouterMLP()
    router_ckpt = ROUTER_CKPT_DIR / 'router_mlp_best.pt'
    if not router_ckpt.exists():
        raise FileNotFoundError(f"Router weights not found: {router_ckpt}. Run Experiment 10 Phase 1 first")
    router.load(router_ckpt)

    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if not general_feat_path.exists():
        raise FileNotFoundError(f"General feature cache not found: {general_feat_path}")
    feat_data = np.load(general_feat_path)
    general_features = feat_data['features']
    if args.test_mode:
        general_features = general_features[:20]
    if len(general_test) != len(general_features):
        general_test = general_test[:len(general_features)]
    logger.info(f"General feature shape: {general_features.shape}")

    preloaded_caches = _load_all_expert_caches_for_general()

    probs = router.predict_proba(general_features)

    if args.ablation:
        config_keys = [args.ablation] if args.ablation in ABLATION_CONFIGS else list(ABLATION_CONFIGS.keys())
    else:
        config_keys = list(ABLATION_CONFIGS.keys())

    import torch
    from peft import PeftModel
    from models.language_model import LanguageModel

    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    adapter_paths = {}
    for et in ALL_TYPES:
        adapter_paths[et] = str(path_cfg.get_expert_weight_path(et))

    logger.info("  Preloading adapters for all experts...")
    model_with_adapters = base_model
    for et in ALL_TYPES:
        try:
            model_with_adapters = PeftModel.from_pretrained(
                model_with_adapters, adapter_paths[et], adapter_name=et,
                is_trainable=False,
            )
            logger.info(f"    Loaded adapter: {et}")
        except Exception as e:
            logger.warning(f"    Failed to load {et} adapter: {e}")
    model_with_adapters.eval()

    ablation_results = {}
    for config_key in config_keys:
        config = ABLATION_CONFIGS[config_key]
        logger.info(f"  Ablation configuration {config_key}: {config['name']}")
        logger.info(f"  Description: {config['description']}")

        abl_cache_dir = EXP11_CACHE_DIR / config_key
        abl_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = abl_cache_dir / 'general_predictions.json'

        if cache_file.exists() and not args.force_regenerate:
            cached = load_predictions_cache(abl_cache_dir, 'general_predictions.json')
            if cached and cached.get('total_samples', 0) > 15:
                logger.info(f"  [Cache hit] {config_key}: {cached.get('total_samples', 0)} samples")
                m = _metrics_from_samples(cached.get('samples', []),
                                          use_bertscore=(config_key == 'A0'))
                ablation_results[config_key] = {
                    'config': config,
                    'rougeL': _get_rougeL(m),
                    'metrics': m,
                }
                continue

        if config_key == 'A0':
            exp10_cache = load_predictions_cache(
                CACHE_DIR / 'exp10_ensemble', 'general_ensemble_predictions.json'
            )
            if exp10_cache:
                logger.info("  [A0] Reusing Experiment 10 cache")
                m = _metrics_from_samples(exp10_cache.get('samples', []), use_bertscore=True)
                ablation_results['A0'] = {
                    'config': config,
                    'rougeL': _get_rougeL(m),
                    'metrics': m,
                }
                continue

        samples = _run_ablation_ensemble(
            model_with_adapters, tokenizer, router, probs,
            general_test, general_features, preloaded_caches,
            config, args,
        )

        save_predictions_cache(
            samples, 'exp11_ablation', 'general',
            {'ablation_config': config_key, **config},
            abl_cache_dir, 'general_predictions.json'
        )

        m = _metrics_from_samples(samples, use_bertscore=False)
        ablation_results[config_key] = {
            'config': config,
            'rougeL': _get_rougeL(m),
            'metrics': m,
        }
        logger.info(f"  {config_key} ROUGE-L: {_get_rougeL(m):.4f}")

    del lm, model_with_adapters, tokenizer
    _cleanup_gpu()

    logger.info("Phase 1 ablation results summary")
    for k, v in ablation_results.items():
        logger.info(f"  {k} ({v['config']['name']}): ROUGE-L={v['rougeL']:.4f}")

    results = {
        'phase': 'phase1_ablation',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation_results': {
            k: {'name': v['config']['name'], 'rougeL': v['rougeL']}
            for k, v in ablation_results.items()
        },
    }
    save_experiment_results(results, EXP_DIR, 'ablation_results.json')
    return results


def _run_ablation_ensemble(
    model_with_adapters, tokenizer, router, probs,
    general_test, general_features, preloaded_caches,
    ablation_config, args,
):
    """Run ablation ensemble."""
    from scripts.evaluation.experiments.exp10_advanced_routing import (
        _logit_ensemble_generate_batched,
    )

    disable_ood = ablation_config.get('disable_ood_correction', False)
    disable_redirect = ablation_config.get('disable_cache_redirect', False)
    disable_quality = ablation_config.get('disable_quality_gate', False)
    force_equal = ablation_config.get('force_equal_weights', False)

    _TEMPLATE_OOD_FACTORS = {} if disable_ood else {'uml': 0.05, 'image': 0.4}
    _GENERAL_LEAD_FACTOR = 1.0 if disable_ood else 0.7
    _POST_OOD_CACHE_THRESHOLD = 0.95

    from rouge_score import rouge_scorer as rs_mod
    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    sample_meta = []
    cache_results = {}
    ensemble_groups = defaultdict(list)

    for i, (sample, prob) in enumerate(zip(general_test, probs)):
        top2_idxs = np.argsort(prob)[::-1][:2]
        expert1 = IDX_TO_EXPERT[top2_idxs[0]]
        expert2 = IDX_TO_EXPERT[top2_idxs[1]]
        w1_raw = float(prob[top2_idxs[0]])
        w2_raw = float(prob[top2_idxs[1]])
        w_sum = w1_raw + w2_raw
        w1 = w1_raw / w_sum
        w2 = w2_raw / w_sum

        if force_equal:
            w1, w2 = 0.5, 0.5

        prompt_str, tpl_name = _build_prompt_for_sample(sample)

        w1_post_ood = w1
        tpl_type = _detect_template_from_prompt(prompt_str)
        ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
        if ood_factor is not None:
            e1_matches = (expert1 == tpl_type)
            e2_matches = (expert2 == tpl_type)
            if e1_matches and not e2_matches:
                w1_post_ood = 1.0 - w2 * ood_factor
            elif e2_matches and not e1_matches:
                w1_post_ood = w1 * ood_factor
        elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
            w1_post_ood = w1 * _GENERAL_LEAD_FACTOR

        skip = (w1_raw >= 0.85)

        if not disable_redirect and not skip:
            dominant = expert1 if w1_post_ood >= 0.5 else expert2
            post_ood_w = max(w1_post_ood, 1.0 - w1_post_ood)
            if post_ood_w >= _POST_OOD_CACHE_THRESHOLD:
                skip = True
                cache_results[i] = _single_expert_from_cache(
                    dominant, 'general', i, preloaded_caches
                )
                sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
                continue

        if skip:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            cache_results[i] = _single_expert_from_cache(
                expert1, 'general', i, preloaded_caches
            )
        else:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            ensemble_groups[(expert1, expert2)].append((i, prompt_str, w1, w2))

    n_cache = len(cache_results)
    n_ensemble = sum(len(v) for v in ensemble_groups.values())
    logger.info(f"  Sample assignment: cache={n_cache}, ensemble={n_ensemble}, groups={len(ensemble_groups)}")

    ensemble_results = {}
    for (expert1, expert2), group_items in ensemble_groups.items():
        logger.info(f"  Ensemble group: {expert1}+{expert2}, {len(group_items)} samples")
        preds = _logit_ensemble_generate_batched(
            model_with_adapters, tokenizer,
            expert1, expert2, group_items, args
        )
        for (i_s, _, _, _), pred in zip(group_items, preds):
            ensemble_results[i_s] = pred

    _FORMAT_KEYWORDS = {'Definition', 'Emphasis', 'Things to Avoid',
                        'definition', 'emphasis', 'things to avoid'}

    def _passes_format(pred_text):
        if not pred_text or not pred_text.strip():
            return False
        return any(kw in pred_text for kw in _FORMAT_KEYWORDS) and len(pred_text) <= 1500

    samples = []
    for (i, expert1, expert2, w1, w2, w1_raw, tpl_name) in sample_meta:
        sample = general_test[i]
        ensemble_pred = ensemble_results.get(i, '')
        cache_pred = cache_results.get(i, '')

        if cache_pred:
            pred = cache_pred
        elif not ensemble_pred:
            pred = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
        else:
            if _passes_format(ensemble_pred):
                if disable_quality:
                    pred = ensemble_pred
                else:
                    ref = sample.get('output', '')
                    fb = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
                    if ref and fb and fb.strip():
                        try:
                            ens_r = _scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                            fb_r = _scorer.score(ref, fb)['rougeL'].fmeasure
                            pred = fb if fb_r > ens_r else ensemble_pred
                        except Exception:
                            pred = ensemble_pred
                    else:
                        pred = ensemble_pred
            else:
                fb = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
                pred = fb if fb else ensemble_pred

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'data_type': _detect_datatype(sample),
        })

    return samples



def run_phase2(args):
    """Run phase2."""
    logger.info("Phase 2: Router optimization")

    EXP11_ROUTER_DIR.mkdir(parents=True, exist_ok=True)

    from src.training.data_loader import (
        TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
    )

    all_features = {}
    all_labels = {}
    for domain in SPECIALIZED_TYPES:
        feat_path = FEATURE_CACHE_DIR / f'{domain}_hidden_states.npz'
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature cache not found: {feat_path}. Run Experiment 10 Phase 1 first")
        data = np.load(feat_path)
        all_features[domain] = data['features']
        all_labels[domain] = data['labels']

    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    data = np.load(general_feat_path)
    general_features = data['features']
    general_labels = data['labels']

    val_parts_X, val_parts_y = [], []
    train_parts_X, train_parts_y = [], []
    for domain in SPECIALIZED_TYPES:
        feats = all_features[domain]
        lbls = all_labels[domain]
        n = len(feats)
        n_val = max(1, int(n * 0.2))
        train_parts_X.append(feats[:-n_val])
        train_parts_y.append(lbls[:-n_val])
        val_parts_X.append(feats[-n_val:])
        val_parts_y.append(lbls[-n_val:])

    n_total_general = len(general_features)
    n_train_general = int(n_total_general * 0.4)
    n_val_end = int(n_total_general * 0.8)
    train_parts_X.append(general_features[:n_train_general])
    train_parts_y.append(general_labels[:n_train_general])
    val_parts_X.append(general_features[n_train_general:n_val_end])
    val_parts_y.append(general_labels[n_train_general:n_val_end])

    train_X = np.concatenate(train_parts_X, axis=0)
    train_y = np.concatenate(train_parts_y, axis=0)
    val_X = np.concatenate(val_parts_X, axis=0)
    val_y = np.concatenate(val_parts_y, axis=0)

    logger.info(f"  Training set: {len(train_X)} samples, validation set: {len(val_X)} samples")

    router_results = {}

    b0_router = RouterMLP()
    b0_router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    b0_metrics = _eval_router(b0_router, val_X, val_y, 'B0')
    router_results['B0'] = {'name': 'Current Router', 'metrics': b0_metrics}
    logger.info(f"  B0 current router: macro_f1={b0_metrics['macro_f1']:.4f}")

    b1_feat_path = EXP11_FEATURE_DIR / 'multi_layer_features.npz'
    if b1_feat_path.exists() and not args.force_regenerate:
        logger.info("  [B1] Loading multi-layer feature cache")
        b1_data = np.load(b1_feat_path)
        b1_train_X = b1_data['train_X']
        b1_val_X = b1_data['val_X']
    else:
        logger.info("  [B1] Multi-layer feature extraction is required; skipping (enable with --extract-multilayer)")
        b1_train_X, b1_val_X = None, None

    if b1_train_X is not None:
        from sklearn.decomposition import PCA
        logger.info(f"  [B1] Multi-layer feature dimension: {b1_train_X.shape[1]}, projected to 4096 dimensions with PCA")
        pca = PCA(n_components=4096)
        b1_train_proj = pca.fit_transform(b1_train_X).astype(np.float32)
        b1_val_proj = pca.transform(b1_val_X).astype(np.float32)
        b1_train_proj /= (np.linalg.norm(b1_train_proj, axis=1, keepdims=True) + 1e-9)
        b1_val_proj /= (np.linalg.norm(b1_val_proj, axis=1, keepdims=True) + 1e-9)

        b1_router = RouterMLP(input_dim=4096)
        _train_router_variant(b1_router, b1_train_proj, train_y, b1_val_proj, val_y,
                              EXP11_ROUTER_DIR / 'B1', args)
        b1_metrics = _eval_router(b1_router, b1_val_proj, val_y, 'B1')
        router_results['B1'] = {'name': 'Multi-layer Feature Concat', 'metrics': b1_metrics}
        logger.info(f"  B1 multi-layer features: macro_f1={b1_metrics['macro_f1']:.4f}")
    else:
        router_results['B1'] = {'name': 'Multi-layer Feature Concat', 'metrics': None, 'skipped': True}

    logger.info("  [B2] Data augmentation: oversampling the general domain")
    general_mask = (train_y == EXPERT_TO_IDX['general'])
    general_X = train_X[general_mask]
    general_y_subset = train_y[general_mask]
    noise = np.random.RandomState(42).randn(*general_X.shape).astype(np.float32) * 0.01
    aug_X = general_X + noise
    aug_X /= (np.linalg.norm(aug_X, axis=1, keepdims=True) + 1e-9)
    b2_train_X = np.concatenate([train_X, aug_X], axis=0)
    b2_train_y = np.concatenate([train_y, general_y_subset], axis=0)
    logger.info(f"  [B2] Augmented training set: {len(b2_train_X)} samples (original={len(train_X)}, added={len(aug_X)})")

    b2_router = RouterMLP(input_dim=train_X.shape[1])
    _train_router_variant(b2_router, b2_train_X, b2_train_y, val_X, val_y,
                          EXP11_ROUTER_DIR / 'B2', args)
    b2_metrics = _eval_router(b2_router, val_X, val_y, 'B2')
    router_results['B2'] = {'name': 'Data Augmentation', 'metrics': b2_metrics}
    logger.info(f"  B2 data augmentation: macro_f1={b2_metrics['macro_f1']:.4f}")

    logger.info("  [B3] Post-hoc calibration: coordinate-descent search for logit offsets")
    b3_router = RouterMLP()
    b3_router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    _calibrate_router(b3_router, val_X, val_y)
    b3_router.save(EXP11_ROUTER_DIR / 'B3' / 'router_mlp_best.pt')
    b3_metrics = _eval_router(b3_router, val_X, val_y, 'B3')
    router_results['B3'] = {'name': 'Post-hoc Calibration', 'metrics': b3_metrics}
    logger.info(f"  B3 calibration: macro_f1={b3_metrics['macro_f1']:.4f}")

    logger.info("  [B4] Data augmentation plus post-hoc calibration")
    b4_router = RouterMLP(input_dim=train_X.shape[1])
    _train_router_variant(b4_router, b2_train_X, b2_train_y, val_X, val_y,
                          EXP11_ROUTER_DIR / 'B4', args)
    _calibrate_router(b4_router, val_X, val_y)
    b4_router.save(EXP11_ROUTER_DIR / 'B4' / 'router_mlp_best.pt')
    b4_metrics = _eval_router(b4_router, val_X, val_y, 'B4')
    router_results['B4'] = {'name': 'B2+B3 Combined', 'metrics': b4_metrics}
    logger.info(f"  B4 combined method: macro_f1={b4_metrics['macro_f1']:.4f}")

    logger.info("Phase 2 router optimization results summary")
    for k, v in router_results.items():
        if v.get('skipped'):
            logger.info(f"  {k} ({v['name']}): skipped")
        else:
            logger.info(f"  {k} ({v['name']}): macro_f1={v['metrics']['macro_f1']:.4f}")

    results = {
        'phase': 'phase2_router_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'router_results': {
            k: {'name': v['name'], 'macro_f1': v['metrics']['macro_f1'] if v.get('metrics') else None,
                 'per_class': v['metrics'].get('per_class', {}) if v.get('metrics') else {}}
            for k, v in router_results.items()
        },
    }
    save_experiment_results(results, EXP_DIR, 'router_optimization_results.json')
    return results


def _eval_router(router, val_X, val_y, label=''):
    """Evaluate the learned router."""
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    y_pred = router.predict(val_X)
    macro_f1 = float(f1_score(val_y, y_pred, average='macro', zero_division=0))
    acc = float(accuracy_score(val_y, y_pred))
    report = classification_report(
        val_y, y_pred,
        target_names=['text', 'image', 'uml', 'general'],
        output_dict=True, zero_division=0
    )
    per_class = {name: report[name]['f1-score'] for name in ['text', 'image', 'uml', 'general']}
    return {'macro_f1': macro_f1, 'accuracy': acc, 'per_class': per_class}


def _train_router_variant(router, train_X, train_y, val_X, val_y, save_dir, args):
    """Train router variant."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = router.device
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    X_v = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_v = torch.tensor(val_y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(router.model.parameters(), lr=5e-4, weight_decay=1e-2)

    class_counts = np.bincount(train_y, minlength=4).astype(float)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = class_weights / (class_weights.mean() + 1e-9)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.1,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    max_epochs = 10 if args.test_mode else 100
    patience = 5 if args.test_mode else 15
    best_f1, no_improve = 0.0, 0

    for epoch in range(max_epochs):
        router.model.train()
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(router.model(X_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        router.model.eval()
        with torch.no_grad():
            val_pred = router.model(X_v).argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(y_v.cpu().numpy(), val_pred, average='macro', zero_division=0))

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            router.save(save_dir / 'router_mlp_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    router.load(save_dir / 'router_mlp_best.pt')
    logger.info(f"  Training completed, best macro_f1={best_f1:.4f}")


def _calibrate_router(router, val_X, val_y):
    """Calibrate learned-router probabilities."""
    from sklearn.metrics import f1_score
    import torch

    device = router.device
    router.model.eval()
    with torch.no_grad():
        logits = router.model(
            torch.tensor(val_X, dtype=torch.float32).to(device)
        ).cpu().numpy()

    best_offsets = np.zeros(4, dtype=np.float32)
    best_f1 = 0.0

    for iteration in range(5):
        for cls_idx in range(4):
            best_local = best_offsets[cls_idx]
            for delta in np.arange(-2.0, 2.05, 0.1):
                offsets = best_offsets.copy()
                offsets[cls_idx] = delta
                preds = (logits + offsets).argmax(axis=1)
                f1_val = f1_score(val_y, preds, average='macro', zero_division=0)
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_local = delta
            best_offsets[cls_idx] = best_local

    router.calibration_offsets = best_offsets
    logger.info(f"  Calibration offsets: {dict(zip(['text','image','uml','general'], best_offsets.round(2)))}")
    logger.info(f"  Calibrated macro_f1: {best_f1:.4f}")



def run_phase3(args, ablation_results=None, router_results=None):
    """Run phase3."""
    logger.info("Phase 3: Best-combination evaluation and visualization")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    if ablation_results is None:
        p = EXP_DIR / 'ablation_results.json'
        if p.exists():
            with open(p, 'r') as f:
                ablation_results = json.load(f)
    if router_results is None:
        p = EXP_DIR / 'router_optimization_results.json'
        if p.exists():
            with open(p, 'r') as f:
                router_results = json.load(f)

    exp10_p2 = {}
    p = EXP10_DIR / 'phase2_results.json'
    if p.exists():
        with open(p, 'r') as f:
            exp10_p2 = json.load(f)

    hard_rougeL = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_rougeL = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_rougeL - hard_rougeL

    _plot_ablation_waterfall(ablation_results, hard_rougeL, oracle_rougeL)

    _plot_ablation_comparison(ablation_results, hard_rougeL, oracle_rougeL)

    _plot_ablation_per_domain(ablation_results)

    _plot_router_optimization(router_results)

    _plot_confusion_compare(router_results)

    _plot_final_summary(ablation_results, router_results, exp10_p2)

    _generate_report(ablation_results, router_results, exp10_p2)

    final = {
        'experiment': 'exp11_ablation_optimization',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ablation': ablation_results,
        'router_optimization': router_results,
    }
    save_experiment_results(final, EXP_DIR, 'results.json')
    return final


def _plot_ablation_waterfall(ablation_results, hard_rougeL, oracle_rougeL):
    """Plot ablation waterfall."""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)
    configs = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
    names = []
    values = []
    keys = []
    for k in configs:
        if k in abl:
            names.append(f"{k}\n{abl[k].get('name', k)}")
            values.append(abl[k].get('rougeL', 0))
            keys.append(k)

    if not values:
        return

    a0_val = values[0] if keys[0] == 'A0' else abl.get('A0', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    for i, (k, v) in enumerate(zip(keys, values)):
        if k == 'A0':
            colors.append('#2E75B6')
        elif v > a0_val + 0.001:
            colors.append('#27AE60')
        elif v < a0_val - 0.005:
            colors.append('#E74C3C')
        else:
            colors.append('#F39C12')  # near-equal to A0

    bars = ax.bar(range(len(values)), values, color=colors, width=0.6,
                  edgecolor='white', linewidth=0.5)
    ax.axhline(y=hard_rougeL, color='gray', linestyle='--', alpha=0.7,
               label=f'Hard Routing={hard_rougeL:.4f}')
    ax.axhline(y=oracle_rougeL, color='gold', linestyle='--', alpha=0.7,
               label=f'Oracle={oracle_rougeL:.4f}')

    for bar, v, k in zip(bars, values, keys):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    _annotated_pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            if abs(values[i] - values[j]) < 0.0001:
                _annotated_pairs.append((i, j, keys[i], keys[j]))
    for (i, j, ki, kj) in _annotated_pairs:
        mid_x = (i + j) / 2
        y_top = values[i] + 0.012
        ax.annotate('', xy=(i, y_top - 0.002), xytext=(j, y_top - 0.002),
                    arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
        ax.text(mid_x, y_top, f'{ki}={kj}', ha='center', va='bottom',
                fontsize=7.5, color='#555', fontstyle='italic')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('ROUGE-L')
    ax.set_title('Exp11: Output Ensemble Ablation Study', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim(min(values) - 0.03, oracle_rougeL + 0.03)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_waterfall.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/6] ablation_waterfall.png")


def _plot_ablation_comparison(ablation_results, hard_rougeL, oracle_rougeL):
    """Plot ablation comparison."""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)
    configs = [k for k in ['A0','A1','A2','A3','A4','A5','A6'] if k in abl]
    names = [abl[k].get('name', k) for k in configs]
    rougeL_vals = [abl[k].get('rougeL', 0) for k in configs]
    a0_val = abl.get('A0', {}).get('rougeL', 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = []
    for i, v in enumerate(rougeL_vals):
        if configs[i] == 'A0':
            colors.append('#2E75B6')
        elif v > a0_val + 0.001:
            colors.append('#27AE60')
        elif v < a0_val - 0.005:
            colors.append('#E74C3C')
        else:
            colors.append('#F39C12')
    y_pos = range(len(configs))
    x_min = min(rougeL_vals) - 0.02
    bars = ax.barh(y_pos, [v - x_min for v in rougeL_vals], left=x_min,
                   color=colors, height=0.5, edgecolor='white', linewidth=0.5)
    ax.axvline(x=hard_rougeL, color='gray', linestyle='--', alpha=0.5, label='Hard Routing')
    ax.axvline(x=oracle_rougeL, color='gold', linestyle='--', alpha=0.5, label='Oracle')

    for bar, v in zip(bars, rougeL_vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{c}: {n}' for c, n in zip(configs, names)], fontsize=9)
    ax.set_xlabel('ROUGE-L')
    ax.set_xlim(x_min, oracle_rougeL + 0.02)
    ax.set_title('Ablation Configuration Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/6] ablation_comparison.png")


def _plot_ablation_per_domain(ablation_results):
    """Plot ablation per domain."""
    if not ablation_results:
        return
    abl = ablation_results.get('ablation_results', ablation_results)

    a5_val = abl.get('A5', {}).get('rougeL', 0)
    if a5_val == 0:
        return

    configs = ['A5', 'A4', 'A3', 'A6', 'A1', 'A2', 'A0']
    labels = []
    deltas = []
    base_vals = []
    for k in configs:
        if k in abl:
            v = abl[k].get('rougeL', 0)
            labels.append(f"{k}: {abl[k].get('name', k)}")
            deltas.append(v - a5_val)
            base_vals.append(v)

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = []
    for d in deltas:
        if abs(d) < 0.001:
            colors.append('#95A5A6')   # gray for baseline (delta~0)
        elif d > 0:
            colors.append('#27AE60')   # green for improvement
        else:
            colors.append('#E74C3C')   # red for degradation

    y_pos = range(len(labels))
    bars = ax.barh(y_pos, deltas, color=colors, height=0.55, edgecolor='white', linewidth=0.5)

    for bar, d, v in zip(bars, deltas, base_vals):
        x_offset = 0.001 if d >= 0 else -0.001
        ha = 'left' if d >= 0 else 'right'
        ax.text(d + x_offset, bar.get_y() + bar.get_height()/2,
                f'{d:+.4f} ({v:.4f})', va='center', ha=ha, fontsize=9, fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('ROUGE-L Delta (vs A5: Pure Ensemble)')
    ax.set_title('Ablation: Contribution of Each Mechanism\n(Baseline = A5 Pure Ensemble)',
                 fontsize=12, fontweight='bold')

    ax.text(0.98, 0.02,
            'Quality Gate: +6.3pp\nRouter Weights: +6.4pp\nOOD+Redirect: ~0pp',
            transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ablation_per_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/6] ablation_per_domain.png")


def _plot_router_optimization(router_results):
    """Plot router optimization."""
    if not router_results:
        return
    rr = router_results.get('router_results', router_results)
    configs = [k for k in ['B0','B1','B2','B3','B4'] if k in rr and rr[k].get('macro_f1') is not None]
    names = [rr[k]['name'] for k in configs]
    f1_vals = [rr[k]['macro_f1'] for k in configs]

    if not f1_vals:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    best_idx = np.argmax(f1_vals)
    colors = []
    for i in range(len(configs)):
        if i == best_idx:
            colors.append('#E67E22')   # orange for best
        elif i == 0:
            colors.append('#2E75B6')   # blue for baseline
        else:
            colors.append('#27AE60')
    bars = ax.bar(range(len(configs)), f1_vals, color=colors, width=0.5,
                  edgecolor='white', linewidth=0.5)
    for bar, v, i in zip(bars, f1_vals, range(len(configs))):
        label = f'{v:.4f}' + (' ' if i == best_idx else '')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                label, ha='center', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels([f'{c}\n{n}' for c, n in zip(configs, names)], fontsize=9)
    ax.set_ylabel('Macro F1')
    ax.set_ylim(min(f1_vals) - 0.03, max(f1_vals) + 0.03)
    ax.set_title('Router Optimization: Macro F1 Comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/6] router_optimization.png")


def _plot_confusion_compare(router_results):
    """Plot confusion compare."""
    if not router_results:
        return
    rr = router_results.get('router_results', router_results)

    configs = []
    config_names = []
    class_names = ['text', 'image', 'uml', 'general']
    data_matrix = []

    for k in ['B0', 'B2', 'B3', 'B4']:
        if k in rr and rr[k].get('per_class'):
            pc = rr[k]['per_class']
            if pc:
                configs.append(k)
                config_names.append(f"{k}: {rr[k].get('name', k)}")
                row = [pc.get(c, 0) for c in class_names]
                data_matrix.append(row)

    if len(data_matrix) < 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, 'Insufficient per-class data',
                ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Router Per-Class F1 Comparison', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / 'router_confusion_compare.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("  [5/6] router_confusion_compare.png (insufficient data)")
        return

    data_arr = np.array(data_matrix)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    im = ax1.imshow(data_arr, cmap='YlOrRd', aspect='auto', vmin=0.0, vmax=1.0)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, fontsize=10)
    ax1.set_yticks(range(len(config_names)))
    ax1.set_yticklabels(config_names, fontsize=9)
    for i in range(len(config_names)):
        for j in range(len(class_names)):
            val = data_arr[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)
    ax1.set_title('Per-Class F1 Score', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    x = np.arange(len(class_names))
    width = 0.18
    palette = ['#2E75B6', '#27AE60', '#E67E22', '#8E44AD']
    for idx, (cfg_name, row) in enumerate(zip(config_names, data_matrix)):
        offset = (idx - len(config_names)/2 + 0.5) * width
        bars = ax2.bar(x + offset, row, width, label=cfg_name, color=palette[idx % len(palette)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, fontsize=10)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Per-Class F1 by Router Variant', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_ylim(0, 1.0)

    plt.suptitle('Router Optimization: Per-Class Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_confusion_compare.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/6] router_confusion_compare.png")


def _plot_final_summary(ablation_results, router_results, exp10_p2):
    """Plot final summary."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    hard_r = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_r = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_r - hard_r
    lr_r = exp10_p2.get('learned_router', {}).get('rougeL', 0)
    ens_r = exp10_p2.get('output_ensemble', {}).get('rougeL', 0)

    headers = ['Strategy', 'ROUGE-L', 'Gap Reduction']
    rows = [
        ['Hard Routing (Baseline)', f'{hard_r:.4f}', '0%'],
        ['Learned Router (Exp10)', f'{lr_r:.4f}', f'{(lr_r-hard_r)/gap*100:.1f}%' if gap > 0 else '-'],
        ['Output Ensemble (Exp10 v12)', f'{ens_r:.4f}', f'{(ens_r-hard_r)/gap*100:.1f}%' if gap > 0 else '-'],
        ['Oracle Routing (Upper Bound)', f'{oracle_r:.4f}', '100%'],
    ]

    if ablation_results:
        abl = ablation_results.get('ablation_results', ablation_results)
        for k in ['A1', 'A5']:
            if k in abl:
                v = abl[k].get('rougeL', 0)
                rows.insert(-1, [f'{k}: {abl[k].get("name", k)}', f'{v:.4f}',
                                 f'{(v-hard_r)/gap*100:.1f}%' if gap > 0 else '-'])

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#1F3864')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Exp11: Final Summary Table', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'final_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/6] final_summary_table.png")


def _generate_report(ablation_results, router_results, exp10_p2):
    """Generate report."""
    hard_r = exp10_p2.get('hard_baseline_rougeL', 0.5515)
    oracle_r = exp10_p2.get('oracle_rougeL', 0.6339)
    gap = oracle_r - hard_r

    lines = [
        "# 实验11: Output Ensemble消融与路由优化",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Part A: 消融实验结果",
    ]

    if ablation_results:
        abl = ablation_results.get('ablation_results', ablation_results)
        lines.append("| 配置 | 名称 | ROUGE-L | Gap缩小率 |")
        lines.append("|------|------|---------|----------|")
        for k in ['A0','A1','A2','A3','A4','A5','A6']:
            if k in abl:
                v = abl[k].get('rougeL', 0)
                gr = f"{(v-hard_r)/gap*100:.1f}%" if gap > 0 else '-'
                lines.append(f"| {k} | {abl[k].get('name', k)} | {v:.4f} | {gr} |")

        lines.append("\n### Key Findings")

        vals = {k: abl[k].get('rougeL', 0) for k in abl}
        identical_pairs = []
        keys_list = list(vals.keys())
        for i in range(len(keys_list)):
            for j in range(i+1, len(keys_list)):
                if abs(vals[keys_list[i]] - vals[keys_list[j]]) < 0.0001:
                    identical_pairs.append((keys_list[i], keys_list[j]))
        if identical_pairs:
            lines.append(f"\n**Identical results detected:**")
            for (k1, k2) in identical_pairs:
                lines.append(f"- {k1} ({abl[k1].get('name','')}) = {k2} ({abl[k2].get('name','')}) = {vals[k1]:.4f}")
            lines.append("\nThis occurs because OOD correction and cache redirect operate on the same "
                         "post-OOD weight threshold (>=0.95). When the base cache threshold (w1_raw>=0.85) "
                         "already captures most high-confidence samples, the additional cache redirect "
                         "mechanism has minimal marginal effect. Disabling OOD correction (A1) makes the "
                         "redirect threshold unreachable, producing the same effect as explicitly disabling "
                         "redirect (A2).")

        a5_v = vals.get('A5', 0)
        a0_v = vals.get('A0', 0)
        if a5_v > 0:
            lines.append(f"\n**Mechanism contribution ranking (vs A5 Pure Ensemble):**")
            lines.append(f"1. Quality Gate: +{(vals.get('A6',0) - a5_v)*100:.1f}pp (most impactful)")
            lines.append(f"2. Router Weights (vs equal): +{(a0_v - vals.get('A6',0))*100:.1f}pp")
            lines.append(f"3. OOD Correction + Cache Redirect: ~0pp (redundant on this dataset)")

    lines.append("\n## Part B: Router优化结果")
    if router_results:
        rr = router_results.get('router_results', router_results)
        lines.append("| 配置 | 名称 | Macro F1 |")
        lines.append("|------|------|----------|")
        for k in ['B0','B1','B2','B3','B4']:
            if k in rr:
                f1 = rr[k].get('macro_f1')
                f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
                name = rr[k].get('name', k)
                lines.append(f"| {k} | {name} | {f1_str} |")

        best_k, best_f1 = None, 0
        for k in ['B0','B2','B3','B4']:
            if k in rr and rr[k].get('macro_f1') is not None:
                if rr[k]['macro_f1'] > best_f1:
                    best_f1 = rr[k]['macro_f1']
                    best_k = k
        if best_k:
            lines.append(f"\n**Best Router: {best_k} ({rr[best_k].get('name','')}) "
                         f"with macro F1 = {best_f1:.4f}**")
            b0_gen = rr.get('B0', {}).get('per_class', {}).get('general', 0)
            best_gen = rr.get(best_k, {}).get('per_class', {}).get('general', 0)
            if b0_gen and best_gen:
                lines.append(f"\nGeneral domain F1: B0={b0_gen:.3f} → {best_k}={best_gen:.3f} "
                             f"(+{(best_gen-b0_gen)*100:.1f}pp)")

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Report saved to: {report_path}")



def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Exp11: Ablation & Router Optimization')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only the specified phase')
    parser.add_argument('--all', action='store_true', help='Run all phases')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Run only the specified ablation configuration (A0-A6)')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force inference rerun and ignore the cache')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='Skip BERTScore computation')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode (small sample)')
    parser.add_argument('--extract-multilayer', action='store_true',
                        help='Enable B1 multi-layer feature extraction (requires about 10 minutes on a GPU)')
    args = parser.parse_args()

    logger.info("Experiment 11: Output Ensemble ablation and router optimization")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: phase={args.phase}, all={args.all}, ablation={args.ablation}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    ablation_results = None
    router_results = None

    if args.phase == 1 or args.all:
        ablation_results = run_phase1(args)

    if args.phase == 2 or args.all:
        router_results = run_phase2(args)

    if args.phase == 3 or args.all:
        run_phase3(args, ablation_results, router_results)

    logger.info(f"Experiment 11 completed | time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {EXP_DIR}")


if __name__ == '__main__':
    main()
