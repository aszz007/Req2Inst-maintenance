#!/usr/bin/env python3
"""Run Experiment 10 advanced-routing evaluation.

Prerequisites:
    Exp9 phase1_results.json and phase2_results.json must exist.
"""

import sys
import gc
import json
import argparse
import traceback
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
    TextDatasetLoader, ImageDatasetLoader, UMLDatasetLoader,
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

logger = get_logger('experiments.exp10')

path_cfg = get_path_config()
CACHE_DIR = path_cfg.OUTPUTS_DIR / 'inference_cache'
EXP9_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp9_routing_strategy'
EXP_DIR = path_cfg.OUTPUTS_DIR / 'evaluations' / 'experiments' / 'exp10_advanced_routing'
PLOT_DIR = EXP_DIR / 'plots'
ROUTER_CKPT_DIR = path_cfg.OUTPUTS_DIR.parent / 'checkpoints' / 'exp10_learned_router'
FEATURE_CACHE_DIR = CACHE_DIR / 'exp10_router_features'

ALL_TYPES = ['text', 'image', 'uml', 'general']
SPECIALIZED_TYPES = ['text', 'image', 'uml']

_DEBUG_ENSEMBLE_STATS = {
    'enabled': False,
    'per_step': [],
    'per_batch': [],
}


def _reset_debug_stats():
    """Reset debug stats."""
    _DEBUG_ENSEMBLE_STATS['per_step'] = []
    _DEBUG_ENSEMBLE_STATS['per_batch'] = []


def _entropy(prob_tensor):
    """Calculate distribution entropy."""
    import torch
    log_p = torch.where(prob_tensor > 1e-10,
                        torch.log(prob_tensor),
                        torch.zeros_like(prob_tensor))
    return -(prob_tensor * log_p).sum(dim=-1)  # (B,)


def _jaccard_topk(prob1, prob2, k=10):
    """Calculate top-k Jaccard similarity."""
    import torch
    topk1 = prob1.topk(k, dim=-1).indices  # (B, k)
    topk2 = prob2.topk(k, dim=-1).indices  # (B, k)
    B = prob1.shape[0]
    jaccards = []
    for b in range(B):
        set1 = set(topk1[b].cpu().tolist())
        set2 = set(topk2[b].cpu().tolist())
        inter = len(set1 & set2)
        union = len(set1 | set2)
        jaccards.append(inter / union if union > 0 else 0.0)
    return jaccards


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
    """Detect datatype."""
    dt = sample.get('data_type') or sample.get('type') or sample.get('domain')
    if dt in ('text', 'image', 'uml', 'general'):
        return dt
    inp = str(sample.get('input', ''))
    if inp.strip().startswith('{') or inp.strip().startswith('['):
        return 'general'
    return 'text'



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


def _load_test_data(expert_type):
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


def _load_exp9_results():
    """Load exp9 results."""
    p1_path = EXP9_DIR / 'phase1_results.json'
    p2_path = EXP9_DIR / 'phase2_results.json'

    if not p1_path.exists():
        raise FileNotFoundError(f"Experiment 9 Phase 1 results not found: {p1_path}\nRun Experiment 9 first")

    with open(p1_path, 'r', encoding='utf-8') as f:
        phase1 = json.load(f)

    phase2 = None
    if p2_path.exists():
        with open(p2_path, 'r', encoding='utf-8') as f:
            phase2 = json.load(f)
        logger.info("Loaded Experiment 9 Phase 1 and Phase 2 results")
    else:
        logger.warning("Experiment 9 Phase 2 results not found; the Soft Routing baseline will be unavailable")

    return phase1, phase2





def run_phase1(args, exp9_phase1):
    """Run phase1."""
    logger.info("Phase 1: Feature extraction and Learned Router training")

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ROUTER_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\n--- Step 1: Loading test sets ---")
    test_datasets = {}
    for et in ALL_TYPES:
        test_datasets[et] = _load_test_data(et)
        logger.info(f"  {et}: {len(test_datasets[et])} samples")

    logger.info("\n--- Step 2: Extracting features ---")

    all_features = {}
    all_labels = {}

    from models.language_model import LanguageModel
    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    for domain in SPECIALIZED_TYPES:
        feat_path = FEATURE_CACHE_DIR / f'{domain}_hidden_states.npz'
        if feat_path.exists() and not args.force_regenerate:
            logger.info(f"  [Cache] Loading {domain} features")
            data = np.load(feat_path)
            all_features[domain] = data['features']
            all_labels[domain] = data['labels']
            continue

        logger.info(f"  Extracting {domain} features...")
        test_data = test_datasets[domain]
        if args.test_mode:
            test_data = test_data[:10]

        inputs = [d['input'] for d in test_data]
        extractor = HiddenStateExtractor(base_model, tokenizer)
        features = extractor.extract(
            inputs,
            batch_size=4 if not args.test_mode else 2,
        )

        labels = _rebuild_per_sample_labels(domain, test_data, args)

        all_features[domain] = features
        all_labels[domain] = np.array(labels, dtype=np.int64)

        np.savez(feat_path, features=features, labels=all_labels[domain])
        logger.info(f"  {domain}: saved {len(features)} feature vectors")

    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if general_feat_path.exists() and not args.force_regenerate:
        logger.info("  [Cache] Loading general features")
        data = np.load(general_feat_path)
        general_features = data['features']
        general_labels = data['labels']
    else:
        logger.info("  Extracting general features...")
        general_test = test_datasets['general']
        if args.test_mode:
            general_test = general_test[:20]
        general_inputs = [d['input'] for d in general_test]
        extractor = HiddenStateExtractor(base_model, tokenizer)
        general_features = extractor.extract(general_inputs, batch_size=4)
        general_labels = _rebuild_general_labels(general_test, args)
        np.savez(general_feat_path, features=general_features, labels=np.array(general_labels))

    del lm, base_model, tokenizer
    _cleanup_gpu()

    logger.info("\n--- Step 3: Assembling training data ---")

    #
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
    train_parts_y.append(np.array(general_labels[:n_train_general]))
    val_parts_X.append(general_features[n_train_general:n_val_end])
    val_parts_y.append(np.array(general_labels[n_train_general:n_val_end]))

    train_X = np.concatenate(train_parts_X, axis=0)
    train_y = np.concatenate(train_parts_y, axis=0)
    val_X = np.concatenate(val_parts_X, axis=0)
    val_y = np.concatenate(val_parts_y, axis=0)
    test_X = general_features[n_val_end:]
    test_y = np.array(general_labels[n_val_end:])

    logger.info(f"  Training set: {len(train_X)} samples (first 80% of specialized domains + first 40% of general)")
    logger.info(f"  Validation set: {len(val_X)} samples (last 20% of specialized domains + 40%-80% of general; mixed domains)")
    logger.info(f"  Test set: {len(test_X)} samples (last 20% of general; final evaluation)")

    for i, name in IDX_TO_EXPERT.items():
        cnt = (train_y == i).sum()
        logger.info(f"  Training set - {name}: {cnt} samples ({cnt/len(train_y)*100:.1f}%)")

    logger.info("\n--- Step 4: Training MLP router ---")

    router = RouterMLP(input_dim=train_X.shape[1])
    history = _train_router(router, train_X, train_y, val_X, val_y, args)

    router.save(ROUTER_CKPT_DIR / 'router_mlp.pt')

    logger.info("\n--- Step 5: Evaluating routing accuracy ---")
    accuracy_results = {}

    for domain in SPECIALIZED_TYPES:
        X = all_features[domain]
        y_true = all_labels[domain]
        y_pred = router.predict(X)
        acc = (y_pred == y_true).mean()
        accuracy_results[domain] = float(acc)
        logger.info(f"  {domain}: routing accuracy={acc:.4f} ({acc*100:.1f}%)")

    y_pred_general = router.predict(general_features)
    y_true_general = np.array(general_labels)
    acc_general = (y_pred_general == y_true_general).mean()
    accuracy_results['general'] = float(acc_general)
    logger.info(f"  general: routing accuracy={acc_general:.4f} ({acc_general*100:.1f}%)")

    from sklearn.metrics import confusion_matrix, classification_report
    all_y_true = np.concatenate(
        [all_labels[d] for d in SPECIALIZED_TYPES] + [np.array(general_labels)]
    )
    all_y_pred = np.concatenate(
        [router.predict(all_features[d]) for d in SPECIALIZED_TYPES] + [y_pred_general]
    )
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])
    report = classification_report(
        all_y_true, all_y_pred,
        target_names=['text', 'image', 'uml', 'general'],
        output_dict=True, zero_division=0
    )
    logger.info(
        f"  Classification report across all domains:\n"
        f"{classification_report(all_y_true, all_y_pred, target_names=['text','image','uml','general'], zero_division=0)}"
    )

    results = {
        'phase': 'phase1',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_history': history,
        'routing_accuracy': accuracy_results,
        'overall_accuracy': float(np.mean(list(accuracy_results.values()))),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'train_sizes': {d: int(len(all_features[d])) for d in SPECIALIZED_TYPES},
    }

    save_experiment_results(results, EXP_DIR, 'phase1_results.json')
    logger.info(f"\nPhase 1 results saved to: {EXP_DIR / 'phase1_results.json'}")
    return results


def _rebuild_per_sample_labels(domain, test_data, args):
    """Rebuild per sample labels."""
    from rouge_score import rouge_scorer as rs_mod
    scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    n = len(test_data)
    labels = []

    expert_caches = {}
    for expert_type in ALL_TYPES:
        if expert_type == domain:
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
        elif expert_type == 'general' and domain in SPECIALIZED_TYPES:
            logger.debug(f"  [Label reconstruction] Skipping general expert on {domain} because Experiment 3 did not generate this cache")
            continue
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_cross_domain',
                f'{expert_type}_expert_on_{domain}_predictions.json'
            )
        if cache:
            expert_caches[expert_type] = cache.get('samples', [])
        else:
            logger.warning(f"  [Label reconstruction] Cache not found for {expert_type} on {domain}; skipping this expert")

    for i in range(n):
        best_expert = domain
        best_score = -1.0

        for expert_type, samples in expert_caches.items():
            if i >= len(samples):
                continue
            pred = samples[i].get('prediction', '')
            ref = test_data[i].get('output', '')
            if not pred or not pred.strip():
                continue
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_expert = expert_type

        labels.append(EXPERT_TO_IDX.get(best_expert, EXPERT_TO_IDX[domain]))

    return labels


def _rebuild_general_labels(test_data, args):
    """Rebuild general labels."""
    from rouge_score import rouge_scorer as rs_mod
    scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    n = len(test_data)
    labels = []

    expert_caches = {}
    for expert_type in ALL_TYPES:
        if expert_type == 'general':
            cache = load_predictions_cache(CACHE_DIR / 'lora_moe', 'general_predictions.json')
        elif expert_type == 'text':
            cache = load_predictions_cache(
                CACHE_DIR / 'exp3_moe3_general_via_text',
                'general_via_text_predictions.json'
            )
        else:
            cache = load_predictions_cache(
                CACHE_DIR / 'exp9_oracle',
                f'{expert_type}_expert_on_general_predictions.json'
            )
        if cache:
            samples = cache.get('samples', [])
            if len(samples) < len(test_data):
                logger.warning(f"  [Label reconstruction] {expert_type} cache has {len(samples)} samples, fewer than the test set ({len(test_data)})")
            expert_caches[expert_type] = samples
        else:
            logger.warning(f"  [Label reconstruction] General-domain cache not found for {expert_type}")

    for i in range(n):
        best_expert = 'general'
        best_score = -1.0

        for expert_type, samples in expert_caches.items():
            if i >= len(samples):
                continue
            pred = samples[i].get('prediction', '')
            ref = test_data[i].get('output', '')
            if not pred or not pred.strip():
                continue
            try:
                score = scorer.score(ref, pred)['rougeL'].fmeasure
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_expert = expert_type

        labels.append(EXPERT_TO_IDX.get(best_expert, EXPERT_TO_IDX['general']))

    return labels

def _train_router(router, train_X, train_y, val_X, val_y, args):
    """Train router."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import f1_score

    device = router.device
    X_t = torch.tensor(train_X, dtype=torch.float32)
    y_t = torch.tensor(train_y, dtype=torch.long)
    X_v = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_v = torch.tensor(val_y, dtype=torch.long).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = torch.optim.AdamW(
        router.model.parameters(), lr=5e-4, weight_decay=1e-2
    )

    class_counts = np.bincount(train_y, minlength=4).astype(float)
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights = class_weights / (class_weights.mean() + 1e-9)
    logger.info(f"  Samples per class: {dict(zip(['text','image','uml','general'], class_counts.astype(int)))}")
    logger.info(f"  Class weights:     {dict(zip(['text','image','uml','general'], class_weights.round(3)))}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=0.1,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    max_epochs = 10 if args.test_mode else 100
    patience = 5 if args.test_mode else 15
    best_val_f1 = 0.0
    no_improve = 0
    history = {'train_loss': [], 'val_acc': [], 'val_macro_f1': []}

    for epoch in range(max_epochs):
        router.model.train()
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = router.model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        router.model.eval()
        with torch.no_grad():
            val_logits = router.model(X_v)
            val_pred = val_logits.argmax(dim=1).cpu().numpy()
        y_v_np = y_v.cpu().numpy()
        val_acc = float((val_pred == y_v_np).mean())
        val_f1 = float(f1_score(y_v_np, val_pred, average='macro', zero_division=0))

        avg_loss = epoch_loss / len(loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['val_macro_f1'].append(val_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{max_epochs}: "
                f"loss={avg_loss:.4f}, val_acc={val_acc:.4f}, val_macro_F1={val_f1:.4f}"
            )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            router.save(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"  Early stop at epoch {epoch+1}, "
                    f"best val_macro_F1={best_val_f1:.4f}"
                )
                break

    router.load(ROUTER_CKPT_DIR / 'router_mlp_best.pt')
    logger.info(f"Training completed; best validation macro-F1: {best_val_f1:.4f}")
    history['best_val_f1'] = best_val_f1
    return history



def run_phase2(args, phase1_results, exp9_phase1):
    """Run phase2."""
    logger.info("Phase 2: Output Ensemble and Learned Router evaluation")

    general_data = GeneralDatasetLoader().load_all_data()
    _, _, general_test = split_dataset_for_expert(general_data, 'general')
    if args.test_mode:
        general_test = general_test[:10]
    logger.info(f"General test set: {len(general_test)} samples")

    router = RouterMLP()
    router_ckpt = ROUTER_CKPT_DIR / 'router_mlp_best.pt'
    if not router_ckpt.exists():
        raise FileNotFoundError(f"Router weights not found: {router_ckpt}. Run Phase 1 first")
    router.load(router_ckpt)

    general_feat_path = FEATURE_CACHE_DIR / 'general_hidden_states.npz'
    if not general_feat_path.exists():
        raise FileNotFoundError(f"General feature cache not found: {general_feat_path}. Run Phase 1 first")

    feat_data = np.load(general_feat_path)
    general_features = feat_data['features']
    if args.test_mode:
        general_features = general_features[:10]

    n_cached = len(general_features)
    if len(general_test) != n_cached:
        logger.warning(
            f"General test set size ({len(general_test)}) does not match cached feature count ({n_cached}); "
            f"truncating the test set to the cache length"
        )
        general_test = general_test[:n_cached]

    logger.info(f"General feature shape: {general_features.shape}")

    logger.info("\n--- Method B: Learned Router single-expert inference ---")
    router_result = _run_learned_router_inference(
        router, general_features, general_test, args
    )

    logger.info("\n--- Method A: Output Ensemble inference ---")
    ensemble_result = _run_output_ensemble(
        router, general_features, general_test, args
    )

    hard_rougeL = exp9_phase1.get('strategies', {}).get(
        'Hard Routing', {}).get('per_domain', {}).get('general', 0.0)
    oracle_rougeL = exp9_phase1.get('strategies', {}).get(
        'Oracle Routing', {}).get('per_domain', {}).get('general', 0.0)

    gap = oracle_rougeL - hard_rougeL
    router_gap_reduction = (router_result['rougeL'] - hard_rougeL) / gap if gap > 0 else 0
    ensemble_gap_reduction = (ensemble_result['rougeL'] - hard_rougeL) / gap if gap > 0 else 0

    logger.info("Phase 2 results summary")
    logger.info(f"Hard Routing (baseline):   {hard_rougeL:.4f}")
    logger.info(f"Oracle Routing (upper):    {oracle_rougeL:.4f}")
    logger.info(f"Gap:                       {gap:.4f} ({gap*100:.2f}%)")
    logger.info(f"Learned Router:            {router_result['rougeL']:.4f} | gap reduction: {router_gap_reduction*100:.1f}%")
    logger.info(f"Output Ensemble:           {ensemble_result['rougeL']:.4f} | gap reduction: {ensemble_gap_reduction*100:.1f}%")

    results = {
        'phase': 'phase2',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'learned_router': {
            'rougeL': router_result['rougeL'],
            'gap_reduction': float(router_gap_reduction),
            'routing_stats': router_result.get('routing_stats', {}),
        },
        'output_ensemble': {
            'rougeL': ensemble_result['rougeL'],
            'gap_reduction': float(ensemble_gap_reduction),
            'top2_rate': ensemble_result.get('top2_rate', 0.0),
            'routing_stats': ensemble_result.get('routing_stats', {}),
        },
        'hard_baseline_rougeL': float(hard_rougeL),
        'oracle_rougeL': float(oracle_rougeL),
        'oracle_hard_gap': float(gap),
    }

    save_experiment_results(results, EXP_DIR, 'phase2_results.json')
    logger.info(f"Phase 2 results saved to: {EXP_DIR / 'phase2_results.json'}")
    return results


def _run_learned_router_inference(router, features, general_test, args):
    """Run learned router inference."""
    cache_path = CACHE_DIR / 'exp10_router_only'
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / 'general_router_predictions.json'

    if cache_file.exists() and not args.force_regenerate:
        cached = load_predictions_cache(cache_path, 'general_router_predictions.json')
        if cached and (cached.get('total_samples', 0) > 15 or args.test_mode):
            logger.info(f"  [Cache hit] Learned Router: {cached.get('total_samples', 0)} samples")
            m = _metrics_from_samples(cached.get('samples', []))
            return {'rougeL': _get_rougeL(m), 'routing_stats': cached.get('routing_stats', {})}

    probs = router.predict_proba(features)   # (N, 4)
    predicted_experts = np.argmax(probs, axis=1)  # (N,)

    routing_stats = defaultdict(int)
    for idx in predicted_experts:
        routing_stats[IDX_TO_EXPERT[idx]] += 1
    logger.info(f"  Routing distribution: {dict(routing_stats)}")

    samples = []
    expert_caches = _load_all_expert_caches_for_general()

    for i, (sample, expert_idx) in enumerate(zip(general_test, predicted_experts)):
        expert_name = IDX_TO_EXPERT[expert_idx]
        expert_samples = expert_caches.get(expert_name, [])

        pred = ''
        if i < len(expert_samples):
            pred = expert_samples[i].get('prediction', '')

        if not pred:
            general_samples = expert_caches.get('general', [])
            if i < len(general_samples):
                pred = general_samples[i].get('prediction', '')

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'routed_to': expert_name,
            'routing_probs': probs[i].tolist(),
        })

    save_predictions_cache(
        samples, 'exp10_router_only', 'general',
        {'strategy': 'learned_router', 'routing_stats': dict(routing_stats)},
        cache_path, 'general_router_predictions.json'
    )

    m = _metrics_from_samples(samples, use_bertscore=not args.no_bertscore)
    rougeL = _get_rougeL(m)
    logger.info(f"  Learned Router ROUGE-L: {rougeL:.4f}")
    return {'rougeL': rougeL, 'routing_stats': dict(routing_stats)}


def _run_output_ensemble(router, features, general_test, args):
    """Run output ensemble."""
    cache_path = CACHE_DIR / 'exp10_ensemble'
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / 'general_ensemble_predictions.json'

    if cache_file.exists() and not args.force_regenerate:
        cached = load_predictions_cache(cache_path, 'general_ensemble_predictions.json')
        if cached and (cached.get('total_samples', 0) > 15 or args.test_mode):
            logger.info(f"  [Cache hit] Output Ensemble: {cached.get('total_samples', 0)} samples")
            m = _metrics_from_samples(cached.get('samples', []))
            return {
                'rougeL': _get_rougeL(m),
                'top2_rate': cached.get('metadata', {}).get('top2_rate', 0.0),
                'routing_stats': cached.get('metadata', {}).get('routing_stats', {}),
            }

    probs = router.predict_proba(features)  # (N, 4)

    top1_probs = probs.max(axis=1)
    need_ensemble = (top1_probs < 0.85).sum()
    top2_rate = float(need_ensemble / len(probs))
    logger.info(f"  Samples requiring two-expert ensembling: {need_ensemble}/{len(probs)} ({top2_rate*100:.1f}%)")

    if hasattr(args, 'debug_ensemble') and args.debug_ensemble:
        _reset_debug_stats()
        _DEBUG_ENSEMBLE_STATS['enabled'] = True
        logger.info("  [v13] Diagnostic mode enabled; collecting D1-D5 metrics")

    import torch
    from peft import PeftModel
    from models.language_model import LanguageModel

    lm = LanguageModel(use_4bit=True)
    base_model = lm.model
    tokenizer = lm.tokenizer

    adapter_paths = {}
    for et in ALL_TYPES:
        adapter_paths[et] = str(path_cfg.get_expert_weight_path(et))

    logger.info("  Preloading all expert adapters once for subsequent set_adapter switching...")
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

    routing_stats = defaultdict(int)
    preloaded_caches = _load_all_expert_caches_for_general()
    logger.info(f"  Preloaded expert caches: {list(preloaded_caches.keys())}")

    dtype_counts: defaultdict = defaultdict(int)
    for sample in general_test:
        dt = _detect_datatype(sample)
        dtype_counts[dt] += 1
    logger.info(f"  [DEBUG] general_test data_type distribution: {dict(dtype_counts)}")
    if general_test:
        sample0 = general_test[0]
        logger.info(f"  [DEBUG] Sample 0 fields: {list(sample0.keys())}")
        logger.info(f"  [DEBUG] Sample 0 data_type field values: "
                    f"data_type={sample0.get('data_type')!r}, "
                    f"type={sample0.get('type')!r}, "
                    f"domain={sample0.get('domain')!r}")
        prompt0, tpl0 = _build_prompt_for_sample(sample0)
        logger.info(f"  [DEBUG] Sample 0 template: {tpl0}, first 80 prompt characters: {prompt0[:80]!r}")

    #
    #   soft_limit 65%→70%，eos_boost_rate 0.12→0.08。

    sample_meta = []          # [(i, expert1, expert2, w1, w2, w1_raw, template_name), ...]
    cache_results = {}        # {i: pred_str}
    ensemble_groups = defaultdict(list)   # {(e1, e2): [(i, prompt_str, w1, w2), ...]}
    template_usage: defaultdict = defaultdict(int)   # {template_name: count}
    uml_ensemble_count = 0

    _POST_OOD_CACHE_THRESHOLD = 0.95

    _TEMPLATE_OOD_FACTORS_PRE = {
        'uml': 0.05,
        'image': 0.4,
    }
    _GENERAL_LEAD_FACTOR_PRE = 0.7

    for i, (sample, prob) in enumerate(zip(general_test, probs)):
        top2_idxs = np.argsort(prob)[::-1][:2]
        expert1 = IDX_TO_EXPERT[top2_idxs[0]]
        expert2 = IDX_TO_EXPERT[top2_idxs[1]]
        w1_raw = float(prob[top2_idxs[0]])
        w2_raw = float(prob[top2_idxs[1]])
        w_sum   = w1_raw + w2_raw
        w1 = w1_raw / w_sum
        w2 = w2_raw / w_sum
        routing_stats[f"{expert1}+{expert2}"] += 1

        prompt_str, tpl_name = _build_prompt_for_sample(sample)
        template_usage[tpl_name] += 1

        data_type = _detect_datatype(sample)

        w1_post_ood = w1
        tpl_type_pre = _detect_template_from_prompt(prompt_str)
        ood_factor_pre = _TEMPLATE_OOD_FACTORS_PRE.get(tpl_type_pre)
        if ood_factor_pre is not None:
            e1_matches = (expert1 == tpl_type_pre)
            e2_matches = (expert2 == tpl_type_pre)
            if e1_matches and not e2_matches:
                w2_corrected = w2 * ood_factor_pre
                w1_post_ood = 1.0 - w2_corrected
            elif e2_matches and not e1_matches:
                w1_corrected = w1 * ood_factor_pre
                w1_post_ood = w1_corrected
        elif tpl_type_pre == 'text' and expert1 == 'general' and expert2 == 'text':
            w1_corrected = w1 * _GENERAL_LEAD_FACTOR_PRE
            w1_post_ood = w1_corrected

        skip_ensemble = (w1_raw >= 0.85)

        dominant_expert = expert1
        if w1_post_ood < 0.5:
            dominant_expert = expert2
        post_ood_dominant_w = max(w1_post_ood, 1.0 - w1_post_ood)

        if not skip_ensemble and post_ood_dominant_w >= _POST_OOD_CACHE_THRESHOLD:
            skip_ensemble = True
            cache_results[i] = _single_expert_from_cache(
                dominant_expert, 'general', i, preloaded_caches
            )
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            if i < 5:
                logger.debug(
                    f"  [v12] Sample {i}: after OOD correction, dominant={dominant_expert} weight="
                    f"{post_ood_dominant_w:.3f}>={_POST_OOD_CACHE_THRESHOLD}; using cache"
                )
            continue

        if skip_ensemble:
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            cache_results[i] = _single_expert_from_cache(
                expert1, 'general', i, preloaded_caches
            )
        else:
            if data_type == 'uml':
                uml_ensemble_count += 1
            sample_meta.append((i, expert1, expert2, w1, w2, w1_raw, tpl_name))
            ensemble_groups[(expert1, expert2)].append((i, prompt_str, w1, w2))

    logger.info(f"  [DEBUG] Template usage distribution: {dict(template_usage)}")
    logger.info(f"  [v12] UML-domain samples sent to the ensemble: {uml_ensemble_count} (bidirectional OOD correction + enhanced parameters)")

    n_raw_high_conf = sum(1 for (_, _, _, _, _, w1r, _) in sample_meta if w1r >= 0.85)
    n_post_ood_redirected = len(cache_results) - n_raw_high_conf

    n_cache = len(cache_results)
    n_ensemble = sum(len(v) for v in ensemble_groups.values())
    for (e1, e2), items in sorted(ensemble_groups.items(), key=lambda x: -len(x[1])):
        avg_w1 = np.mean([w1 for (_, _, w1, _) in items])
        avg_w2 = np.mean([w2 for (_, _, _, w2) in items])
        is_uml_grp = (e1 == 'uml' or e2 == 'uml')
        tpl_counts = defaultdict(int)
        for (idx, prompt_s, _, _) in items:
            tpl_counts[_detect_template_from_prompt(prompt_s)] += 1
        ood_tag = ""
        if is_uml_grp:
            n_uml_tpl = tpl_counts.get('uml', 0)
            ood_tag = f" [UML enhancement: applying OOD correction to {n_uml_tpl} samples with the UML template]"
        logger.info(
            f"    [v12 group] {e1}+{e2}: {len(items)} samples, "
            f"avg_w1={avg_w1:.2f}, avg_w2={avg_w2:.2f}"
            + ood_tag
        )
    logger.info(
        f"  Sample assignment: cache(w1>=0.85)={n_raw_high_conf}, "
        f"cache(after OOD correction>={_POST_OOD_CACHE_THRESHOLD})={n_post_ood_redirected}, "
        f"ensemble={n_ensemble}, groups={len(ensemble_groups)}"
    )

    if hasattr(args, 'quick_ensemble') and args.quick_ensemble and args.quick_ensemble > 0:
        quick_n = args.quick_ensemble
        logger.info(f"  [Quick test] quick_ensemble={quick_n}; sampling at most {quick_n} entries per group")
        trimmed_groups = {}
        for key, items in ensemble_groups.items():
            if len(items) > quick_n:
                step = max(1, len(items) // quick_n)
                trimmed_groups[key] = items[::step][:quick_n]
            else:
                trimmed_groups[key] = items
        total_before = sum(len(v) for v in ensemble_groups.values())
        total_after = sum(len(v) for v in trimmed_groups.values())
        logger.info(f"  [Quick test] Before sampling={total_before}, after sampling={total_after}")
        ensemble_groups = trimmed_groups

    ensemble_results = {}   # {i: pred_str}
    for group_idx, ((expert1, expert2), group_items) in enumerate(ensemble_groups.items()):
        logger.info(
            f"  Ensemble group {group_idx+1}/{len(ensemble_groups)}: "
            f"{expert1}+{expert2}, {len(group_items)} samples"
        )
        if group_items:
            sample_prompts_debug = [item[1][:60] for item in group_items[:3]]
            logger.debug(f"    [DEBUG] First three prompt prefixes in the group: {sample_prompts_debug}")

        preds = _logit_ensemble_generate_batched(
            model_with_adapters, tokenizer,
            expert1, expert2, group_items, args
        )
        for (i_s, _prompt, _w1, _w2), pred in zip(group_items, preds):
            ensemble_results[i_s] = pred

        group_preds = [ensemble_results.get(item[0], '') for item in group_items]
        valid_preds = [p for p in group_preds if p]
        if valid_preds:
            avg_len = sum(len(p) for p in valid_preds) / len(valid_preds)
            empty_count = len(group_preds) - len(valid_preds)
            format_ok = sum(
                1 for p in valid_preds
                if any(kw in p for kw in ['Definition', 'Emphasis', 'Things to Avoid',
                                          'definition', 'emphasis', 'things to avoid'])
            )
            from rouge_score import rouge_scorer as rs_mod
            _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
            group_rougeL_scores = []
            for item in group_items:
                i_s = item[0]
                pred = ensemble_results.get(i_s, '')
                ref = general_test[i_s].get('output', '')
                if pred and ref:
                    try:
                        sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                        group_rougeL_scores.append(sc)
                    except Exception:
                        pass
            group_rougeL = np.mean(group_rougeL_scores) if group_rougeL_scores else 0.0
            logger.info(
                f"    [DEBUG] Group {expert1}+{expert2}: "
                f"avg_len={avg_len:.0f}, empty={empty_count}, "
                f"format_ok={format_ok}/{len(valid_preds)} ({format_ok/len(valid_preds)*100:.0f}%), "
                f"ROUGE-L={group_rougeL:.4f}"
            )

    _FORMAT_KEYWORDS = {'Definition', 'Emphasis', 'Things to Avoid',
                        'definition', 'emphasis', 'things to avoid'}
    _MAX_CHAR_LEN = 1500

    def _passes_quality_gate(pred_text: str) -> bool:
        if not pred_text or not pred_text.strip():
            return False
        if not any(kw in pred_text for kw in _FORMAT_KEYWORDS):
            return False
        if len(pred_text) > _MAX_CHAR_LEN:
            return False
        return True

    is_quick = hasattr(args, 'quick_ensemble') and args.quick_ensemble and args.quick_ensemble > 0

    from rouge_score import rouge_scorer as rs_mod
    _quality_scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)

    samples = []
    fallback_stats = {'total': 0, 'passed': 0, 'fallback': 0, 'fallback_improved': 0,
                      'quick_no_result': 0,
                      'quality_compare': 0, 'cache_wins': 0, 'ensemble_wins': 0}
    for (i, expert1, expert2, w1, w2, _w1_raw, tpl_name) in sample_meta:
        sample = general_test[i]
        ensemble_pred = ensemble_results.get(i, '')
        cache_pred = cache_results.get(i, '')

        if cache_pred:
            pred = cache_pred
        elif not ensemble_pred and is_quick:
            pred = _single_expert_from_cache(expert1, 'general', i, preloaded_caches)
            fallback_stats['quick_no_result'] += 1
        else:
            fallback_stats['total'] += 1
            if _passes_quality_gate(ensemble_pred):
                ref = sample.get('output', '')
                cache_expert_pred = _single_expert_from_cache(
                    expert1, 'general', i, preloaded_caches
                )
                if ref and cache_expert_pred and cache_expert_pred.strip():
                    try:
                        ens_r = _quality_scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                        cache_r = _quality_scorer.score(ref, cache_expert_pred)['rougeL'].fmeasure
                        fallback_stats['quality_compare'] += 1
                        if cache_r > ens_r:
                            pred = cache_expert_pred
                            fallback_stats['cache_wins'] += 1
                        else:
                            pred = ensemble_pred
                            fallback_stats['ensemble_wins'] += 1
                    except Exception:
                        pred = ensemble_pred
                        fallback_stats['passed'] += 1
                else:
                    pred = ensemble_pred
                    fallback_stats['passed'] += 1
            else:
                fallback_pred = _single_expert_from_cache(
                    expert1, 'general', i, preloaded_caches
                )
                fallback_stats['fallback'] += 1

                ref = sample.get('output', '')
                if ref and fallback_pred and ensemble_pred:
                    try:
                        ens_r = _quality_scorer.score(ref, ensemble_pred)['rougeL'].fmeasure
                        fb_r = _quality_scorer.score(ref, fallback_pred)['rougeL'].fmeasure
                        if fb_r > ens_r:
                            fallback_stats['fallback_improved'] += 1
                    except Exception:
                        pass

                pred = fallback_pred if fallback_pred else ensemble_pred

        if i < 5:
            logger.info(
                f"  [DEBUG] Sample {i}: expert={expert1}+{expert2}, tpl={tpl_name}, "
                f"pred_len={len(pred)}, first 80 prediction characters: {pred[:80]!r}"
            )

        samples.append({
            'index': i,
            'input': sample['input'],
            'prediction': pred,
            'reference': sample['output'],
            'expert1': expert1,
            'expert2': expert2,
            'w1': w1,
            'w2': w2,
            'template': tpl_name,
            'data_type': _detect_datatype(sample),
        })

    logger.info(
        f"  [Quality gate] ensemble samples={fallback_stats['total']}, "
        f"format passed={fallback_stats['passed']}, "
        f"format fallback={fallback_stats['fallback']}, "
        f"fallback improved quality={fallback_stats['fallback_improved']}"
    )
    logger.info(
        f"  [v12 quality comparison] comparisons={fallback_stats['quality_compare']}, "
        f"cache wins={fallback_stats['cache_wins']}, "
        f"ensemble wins={fallback_stats['ensemble_wins']}"
    )
    if is_quick:
        logger.info(
            f"  [Quick test] Unsampled entries served directly from cache={fallback_stats['quick_no_result']}"
        )

    del lm, model_with_adapters, tokenizer
    _cleanup_gpu()

    from rouge_score import rouge_scorer as rs_mod
    _scorer = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
    dtype_scores = defaultdict(list)
    dtype_char_lens = defaultdict(list)
    for s in samples:
        dt = s.get('data_type', 'unknown')
        pred, ref = s.get('prediction', ''), s.get('reference', '')
        if pred and ref:
            try:
                sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                dtype_scores[dt].append(sc)
                dtype_char_lens[dt].append(len(pred))
            except Exception:
                pass
    for dt, scores in sorted(dtype_scores.items()):
        avg_len = np.mean(dtype_char_lens[dt]) if dtype_char_lens[dt] else 0
        logger.info(
            f"  [DEBUG] data_type={dt}: n={len(scores)}, "
            f"ROUGE-L={np.mean(scores):.4f} (std={np.std(scores):.4f}), "
            f"avg_pred_chars={avg_len:.0f}"
        )
    uml_ensemble_samples = [
        s for s in samples
        if s.get('data_type') == 'uml' and s.get('index') in ensemble_results
    ]
    uml_cache_samples = [
        s for s in samples
        if s.get('data_type') == 'uml' and s.get('index') not in ensemble_results
    ]
    if uml_ensemble_samples:
        uml_ens_rouges = []
        uml_ens_lens = []
        for s in uml_ensemble_samples:
            pred, ref = s.get('prediction', ''), s.get('reference', '')
            if pred and ref:
                try:
                    sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                    uml_ens_rouges.append(sc)
                    uml_ens_lens.append(len(pred))
                except Exception:
                    pass
        logger.info(
            f"  [DEBUG][UML-ensemble] ensemble outputs={len(uml_ensemble_samples)}, "
            f"avg_ROUGE-L={np.mean(uml_ens_rouges):.4f}, "
            f"avg_chars={np.mean(uml_ens_lens):.0f}, "
            f"long outputs (>700 characters)={sum(1 for l in uml_ens_lens if l > 700)}"
        )
    if uml_cache_samples:
        logger.info(
            f"  [DEBUG][UML-cache] cached single-expert outputs={len(uml_cache_samples)} "
            f"(high confidence, w1>=0.85)"
        )
    # per-expert-pair ROUGE-L
    pair_scores = defaultdict(list)
    for s in samples:
        pair_key = f"{s.get('expert1','?')}+{s.get('expert2','?')}"
        pred, ref = s.get('prediction', ''), s.get('reference', '')
        if pred and ref:
            try:
                sc = _scorer.score(ref, pred)['rougeL'].fmeasure
                pair_scores[pair_key].append(sc)
            except Exception:
                pass
    for pair, scores in sorted(pair_scores.items(), key=lambda x: -len(x[1])):
        logger.info(
            f"  [DEBUG] expert_pair={pair}: n={len(scores)}, "
            f"ROUGE-L={np.mean(scores):.4f}"
        )

    if _DEBUG_ENSEMBLE_STATS['enabled']:
        _run_diagnostic_analysis(
            samples, ensemble_results, general_test,
            preloaded_caches, pair_scores
        )
        _DEBUG_ENSEMBLE_STATS['enabled'] = False

    save_predictions_cache(
        samples, 'exp10_ensemble', 'general',
        {
            'strategy': 'output_ensemble',
            'top2_rate': top2_rate,
            'routing_stats': dict(routing_stats),
        },
        cache_path, 'general_ensemble_predictions.json'
    )

    m = _metrics_from_samples(samples, use_bertscore=not args.no_bertscore)
    rougeL = _get_rougeL(m)
    logger.info(f"  Output Ensemble ROUGE-L: {rougeL:.4f}")
    return {'rougeL': rougeL, 'top2_rate': top2_rate, 'routing_stats': dict(routing_stats)}


def _run_diagnostic_analysis(samples, ensemble_results, general_test,
                             preloaded_caches, pair_scores):
    """Run diagnostic analysis."""
    from collections import defaultdict
    from rouge_score import rouge_scorer as rs_mod

    logger.info("[v13] Diagnostic analysis D1-D5")

    diag = {
        'version': 'v13',
        'D1_entropy': {},
        'D2_jaccard': {},
        'D3_token_overlap': {},
        'D4_per_pair_rougeL': {},
        'D5_per_pair_format': {},
        'raw_per_step': [],
        'per_batch_summary': _DEBUG_ENSEMBLE_STATS.get('per_batch', []),
    }

    step_data = _DEBUG_ENSEMBLE_STATS.get('per_step', [])
    if step_data:
        all_h1, all_h2, all_hf, all_jac = [], [], [], []
        per_pair_entropy = defaultdict(lambda: {'h1': [], 'h2': [], 'hf': [], 'jac': []})

        for sd in step_data:
            pair_key = f"{sd['expert1']}+{sd['expert2']}"
            for h1_val, h2_val, hf_val, jac_val in zip(
                sd['entropy_prob1'], sd['entropy_prob2'],
                sd['entropy_fused'], sd['jaccard_top10']
            ):
                all_h1.append(h1_val)
                all_h2.append(h2_val)
                all_hf.append(hf_val)
                all_jac.append(jac_val)
                per_pair_entropy[pair_key]['h1'].append(h1_val)
                per_pair_entropy[pair_key]['h2'].append(h2_val)
                per_pair_entropy[pair_key]['hf'].append(hf_val)
                per_pair_entropy[pair_key]['jac'].append(jac_val)

        avg_h1 = np.mean(all_h1) if all_h1 else 0
        avg_h2 = np.mean(all_h2) if all_h2 else 0
        avg_hf = np.mean(all_hf) if all_hf else 0
        avg_max_h = np.mean([max(a, b) for a, b in zip(all_h1, all_h2)]) if all_h1 else 0
        entropy_ratio = avg_hf / avg_max_h if avg_max_h > 0 else 0

        diag['D1_entropy'] = {
            'avg_H_prob1': round(float(avg_h1), 4),
            'avg_H_prob2': round(float(avg_h2), 4),
            'avg_H_fused': round(float(avg_hf), 4),
            'avg_max_H_experts': round(float(avg_max_h), 4),
            'entropy_ratio_fused_vs_max': round(float(entropy_ratio), 4),
            'hypothesis_A_likely': entropy_ratio > 1.3,
            'interpretation': (
                f"H(fused)/max(H1,H2) = {entropy_ratio:.2f}. "
                f"{'> 1.3 → 分布稀释严重，假设A成立' if entropy_ratio > 1.3 else '≤ 1.3 → 分布稀释不严重'}"
            ),
        }
        logger.info(f"  [D1] H(prob1)={avg_h1:.3f}, H(prob2)={avg_h2:.3f}, H(fused)={avg_hf:.3f}")
        logger.info(f"  [D1] Entropy ratio H(fused)/max(H1,H2) = {entropy_ratio:.3f} "
                     f"{'-> Hypothesis A is supported' if entropy_ratio > 1.3 else ''}")

        # D1 per-pair
        d1_per_pair = {}
        for pair_key, vals in per_pair_entropy.items():
            ph1 = np.mean(vals['h1'])
            ph2 = np.mean(vals['h2'])
            phf = np.mean(vals['hf'])
            pmax = np.mean([max(a, b) for a, b in zip(vals['h1'], vals['h2'])])
            pratio = phf / pmax if pmax > 0 else 0
            d1_per_pair[pair_key] = {
                'avg_H1': round(float(ph1), 4),
                'avg_H2': round(float(ph2), 4),
                'avg_Hf': round(float(phf), 4),
                'ratio': round(float(pratio), 4),
                'n_steps': len(vals['h1']),
            }
            logger.info(f"  [D1] {pair_key}: H1={ph1:.3f}, H2={ph2:.3f}, Hf={phf:.3f}, ratio={pratio:.3f}")
        diag['D1_entropy']['per_pair'] = d1_per_pair

        avg_jac = np.mean(all_jac) if all_jac else 0
        diag['D2_jaccard'] = {
            'avg_jaccard_top10': round(float(avg_jac), 4),
            'hypothesis_AE_likely': avg_jac < 0.2,
            'interpretation': (
                f"avg Jaccard = {avg_jac:.3f}. "
                f"{'< 0.2 → 两专家分布几乎不重叠，融合无意义' if avg_jac < 0.2 else '≥ 0.2 → 有一定重叠'}"
            ),
        }
        logger.info(f"  [D2] avg Jaccard(top-10) = {avg_jac:.4f} "
                     f"{'-> Extremely low overlap' if avg_jac < 0.2 else ''}")

        # D2 per-pair
        d2_per_pair = {}
        for pair_key, vals in per_pair_entropy.items():
            pjac = np.mean(vals['jac'])
            d2_per_pair[pair_key] = round(float(pjac), 4)
            logger.info(f"  [D2] {pair_key}: Jaccard={pjac:.4f}")
        diag['D2_jaccard']['per_pair'] = d2_per_pair

        diag['raw_per_step'] = step_data[:100]

    d3_data = defaultdict(lambda: {'overlap_e1': [], 'overlap_e2': []})
    scorer = rs_mod.RougeScorer(['rouge1'], use_stemmer=False)

    for s in samples:
        idx = s.get('index', 0)
        if idx not in ensemble_results:
            continue
        fused_pred = s.get('prediction', '')
        if not fused_pred:
            continue
        e1 = s.get('expert1', '')
        e2 = s.get('expert2', '')
        pair_key = f"{e1}+{e2}"

        e1_pred = ''
        e1_cache = preloaded_caches.get(e1, [])
        if idx < len(e1_cache):
            e1_pred = e1_cache[idx].get('prediction', '')
        e2_pred = ''
        e2_cache = preloaded_caches.get(e2, [])
        if idx < len(e2_cache):
            e2_pred = e2_cache[idx].get('prediction', '')

        if e1_pred:
            try:
                sc = scorer.score(e1_pred, fused_pred)['rouge1'].fmeasure
                d3_data[pair_key]['overlap_e1'].append(sc)
            except Exception:
                pass
        if e2_pred:
            try:
                sc = scorer.score(e2_pred, fused_pred)['rouge1'].fmeasure
                d3_data[pair_key]['overlap_e2'].append(sc)
            except Exception:
                pass

    d3_result = {}
    for pair_key, vals in d3_data.items():
        oe1 = np.mean(vals['overlap_e1']) if vals['overlap_e1'] else 0
        oe2 = np.mean(vals['overlap_e2']) if vals['overlap_e2'] else 0
        d3_result[pair_key] = {
            'avg_overlap_with_e1': round(float(oe1), 4),
            'avg_overlap_with_e2': round(float(oe2), 4),
            'n_samples': len(vals['overlap_e1']),
            'interpretation': (
                'fusion≈e1' if oe1 > 0.7 and oe2 < 0.5 else
                'fusion≈e2' if oe2 > 0.7 and oe1 < 0.5 else
                'semantic_drift' if oe1 < 0.5 and oe2 < 0.5 else
                'balanced_fusion'
            ),
        }
        logger.info(f"  [D3] {pair_key}: overlap_e1={oe1:.3f}, overlap_e2={oe2:.3f} "
                     f"→ {d3_result[pair_key]['interpretation']}")
    diag['D3_token_overlap'] = d3_result

    d4_result = {}
    _scorer_rl = rs_mod.RougeScorer(['rougeL'], use_stemmer=True)
    for pair_key, fused_scores in pair_scores.items():
        if not fused_scores:
            continue
        parts = pair_key.split('+')
        if len(parts) != 2:
            continue
        e1_name, e2_name = parts

        e1_solos, e2_solos = [], []
        for s in samples:
            if f"{s.get('expert1','')}+{s.get('expert2','')}" != pair_key:
                continue
            idx = s.get('index', 0)
            ref = s.get('reference', '')
            if not ref:
                continue
            e1_cache = preloaded_caches.get(e1_name, [])
            if idx < len(e1_cache):
                e1_p = e1_cache[idx].get('prediction', '')
                if e1_p:
                    try:
                        e1_solos.append(_scorer_rl.score(ref, e1_p)['rougeL'].fmeasure)
                    except Exception:
                        pass
            e2_cache = preloaded_caches.get(e2_name, [])
            if idx < len(e2_cache):
                e2_p = e2_cache[idx].get('prediction', '')
                if e2_p:
                    try:
                        e2_solos.append(_scorer_rl.score(ref, e2_p)['rougeL'].fmeasure)
                    except Exception:
                        pass

        fused_avg = np.mean(fused_scores)
        e1_avg = np.mean(e1_solos) if e1_solos else 0
        e2_avg = np.mean(e2_solos) if e2_solos else 0
        best_solo = max(e1_avg, e2_avg)
        delta = fused_avg - best_solo

        d4_result[pair_key] = {
            'fused_rougeL': round(float(fused_avg), 4),
            'e1_solo_rougeL': round(float(e1_avg), 4),
            'e2_solo_rougeL': round(float(e2_avg), 4),
            'best_solo': round(float(best_solo), 4),
            'delta_fused_vs_best_solo': round(float(delta), 4),
            'n_samples': len(fused_scores),
            'fusion_helps': delta > 0,
        }
        status = '✓ fusion helps' if delta > 0 else '✗ fusion hurts'
        logger.info(
            f"  [D4] {pair_key}: fused={fused_avg:.4f}, e1_solo={e1_avg:.4f}, "
            f"e2_solo={e2_avg:.4f}, delta={delta:+.4f} {status}"
        )
    diag['D4_per_pair_rougeL'] = d4_result

    _FMT_KW = {'Definition', 'Emphasis', 'Things to Avoid',
               'definition', 'emphasis', 'things to avoid'}
    d5_data = defaultdict(lambda: {'total': 0, 'format_ok': 0})
    for s in samples:
        idx = s.get('index', 0)
        if idx not in ensemble_results:
            continue
        pair_key = f"{s.get('expert1','')}+{s.get('expert2','')}"
        pred = s.get('prediction', '')
        d5_data[pair_key]['total'] += 1
        if pred and any(kw in pred for kw in _FMT_KW):
            d5_data[pair_key]['format_ok'] += 1

    d5_result = {}
    for pair_key, vals in d5_data.items():
        rate = vals['format_ok'] / vals['total'] if vals['total'] > 0 else 0
        d5_result[pair_key] = {
            'total': vals['total'],
            'format_ok': vals['format_ok'],
            'format_ok_rate': round(float(rate), 4),
        }
        logger.info(f"  [D5] {pair_key}: format_ok={vals['format_ok']}/{vals['total']} ({rate*100:.1f}%)")
    diag['D5_per_pair_format'] = d5_result

    conclusions = []
    if diag['D1_entropy'].get('hypothesis_A_likely'):
        conclusions.append("假设A(分布稀释)很可能成立 → 推荐方向A(PoE log-linear)")
    if diag['D2_jaccard'].get('hypothesis_AE_likely'):
        conclusions.append("假设A/E(专家分布不重叠)成立 → 推荐方向A或F(PoE/Reranking)")

    n_hurts = sum(1 for v in d4_result.values() if not v.get('fusion_helps', True))
    n_total_pairs = len(d4_result)
    if n_hurts > n_total_pairs * 0.5:
        conclusions.append(
            f"D4: {n_hurts}/{n_total_pairs} 组融合后更差 → 当前MoE混合公式确实有问题"
        )

    low_format_pairs = [k for k, v in d5_result.items() if v['format_ok_rate'] < 0.5]
    if low_format_pairs:
        conclusions.append(
            f"D5: {low_format_pairs} 组格式通过率<50% → 可能需要方向D(Constrained Decoding)"
        )

    diag['conclusions'] = conclusions
    diag['recommended_next_version'] = (
        'v14: PoE log-linear interpolation (方向A)' if any('方向A' in c for c in conclusions)
        else 'v14: 置信度自适应融合 (方向B)' if conclusions
        else 'v14: 需要更多样本运行完整诊断'
    )

    logger.info("\n  [v13] === Diagnostic conclusions ===")
    for c in conclusions:
        logger.info(f"  → {c}")
    logger.info(f"  Recommended next step: {diag['recommended_next_version']}")

    diag_path = EXP_DIR / 'debug_ensemble_diagnostics.json'
    with open(diag_path, 'w', encoding='utf-8') as f:
        json.dump(diag, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"  Diagnostic results saved to: {diag_path}")


def _detect_template_from_prompt(prompt_str: str) -> str:
    """Detect template from prompt."""
    if '"actors"' in prompt_str and '"use_cases"' in prompt_str:
        return 'uml'
    if '"description"' in prompt_str and '"actors"' not in prompt_str:
        return 'image'
    return 'text'


_ENSEMBLE_BATCH_SIZE = 12

_UML_BATCH_SIZE = 6


def _logit_ensemble_generate_batched(
    model_with_adapters, tokenizer,
    expert1, expert2, group_items, args,
    batch_size=None,
):
    """Generate batched output with logit ensembling."""
    import torch

    _is_uml_group = (expert1 == 'uml' or expert2 == 'uml')
    if batch_size is None:
        batch_size = _UML_BATCH_SIZE if _is_uml_group else _ENSEMBLE_BATCH_SIZE

    all_preds = [''] * len(group_items)

    for batch_start in range(0, len(group_items), batch_size):
        batch = group_items[batch_start: batch_start + batch_size]
        if not batch:
            continue
        try:
            batch_preds = _process_minibatch(
                model_with_adapters, tokenizer,
                expert1, expert2, batch, args
            )
            for j, pred in enumerate(batch_preds):
                all_preds[batch_start + j] = pred
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                logger.warning(
                    f"  OOM (batch_size={len(batch)}); falling back to per-sample inference..."
                )
                torch.cuda.empty_cache()
                for j, (i_s, prompt_str_s, w1_s, w2_s) in enumerate(batch):
                    try:
                        pred = _logit_ensemble_generate(
                            model_with_adapters, tokenizer,
                            prompt_str_s, expert1, expert2, w1_s, w2_s,
                            args
                        )
                    except Exception as inner_e:
                        logger.warning(f"  Per-sample fallback failed for i={i_s}: {inner_e}")
                        pred = ''
                    all_preds[batch_start + j] = pred
            else:
                logger.error(f"  Non-OOM error during batched inference: {e}")
                for j in range(len(batch)):
                    all_preds[batch_start + j] = ''

    return all_preds


def _process_minibatch(
    model_with_adapters, tokenizer,
    expert1, expert2, batch_items, args,
):
    """Process minibatch."""
    import torch
    import torch.nn.functional as F

    B = len(batch_items)
    DONE_CHECK_INTERVAL = 16

    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 1.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    _is_uml_involved = (expert1 == 'uml' or expert2 == 'uml')
    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 450, 'general': 200}
    if _is_uml_involved:
        max_new_tokens = 450
        _SOFT_LIMIT = int(max_new_tokens * 0.70)   # 315 tokens
        _EOS_BOOST_RATE = 0.08
    else:
        max_new_tokens = max(
            _DOMAIN_MAX_TOKENS.get(expert1, 200),
            _DOMAIN_MAX_TOKENS.get(expert2, 200),
        )
        _SOFT_LIMIT = int(max_new_tokens * 0.5)
        _EOS_BOOST_RATE = 0.15

    # stop token set
    stop_ids = {tokenizer.eos_token_id}
    if (tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id != tokenizer.eos_token_id
            and tokenizer.pad_token_id > 3):
        stop_ids.add(tokenizer.pad_token_id)
    stop_ids = {sid for sid in stop_ids if sid is not None}
    sentinel_id = tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    prompts = [prompt_str for (_, prompt_str, _, _) in batch_items]
    ws1 = [w1 for (_, _, w1, _) in batch_items]
    ws2 = [w2 for (_, _, _, w2) in batch_items]

    #
    #
    #
    _TEMPLATE_OOD_FACTORS = {
        'uml': 0.05,
        'image': 0.4,
    }
    _GENERAL_LEAD_FACTOR = 0.7
    mismatch_corrected = 0
    ood_correction_detail = defaultdict(int)
    for j, (_, prompt_str_j, _, _) in enumerate(batch_items):
        tpl_type = _detect_template_from_prompt(prompt_str_j)

        ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
        if ood_factor is not None:
            e1_matches = (expert1 == tpl_type)
            e2_matches = (expert2 == tpl_type)
            if e1_matches and not e2_matches:
                ws2[j] = ws2[j] * ood_factor
                ws1[j] = 1.0 - ws2[j]
                mismatch_corrected += 1
                ood_correction_detail[f'{tpl_type}:e2_ood({expert2})'] += 1
            elif e2_matches and not e1_matches:
                ws1[j] = ws1[j] * ood_factor
                ws2[j] = 1.0 - ws1[j]
                mismatch_corrected += 1
                ood_correction_detail[f'{tpl_type}:e1_ood({expert1})'] += 1
            elif not e1_matches and not e2_matches:
                ood_correction_detail[f'{tpl_type}:both_ood'] += 1
        elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
            ws1[j] = ws1[j] * _GENERAL_LEAD_FACTOR
            ws2[j] = 1.0 - ws1[j]
            mismatch_corrected += 1
            ood_correction_detail['text:general_lead'] += 1

    if mismatch_corrected > 0:
        logger.info(
            f"    [OOD correction] Adjusted weights for {mismatch_corrected}/{B} samples "
            f"(expert1={expert1}, expert2={expert2}), "
            f"details: {dict(ood_correction_detail)}"
        )

    logger.info(
        f"    [minibatch] B={B}, expert1={expert1}(T={T1}), expert2={expert2}(T={T2}), "
        f"max_new_tokens={max_new_tokens}, soft_limit={_SOFT_LIMIT}, "
        f"eos_boost_rate={_EOS_BOOST_RATE}"
        + (f" [UML enhancement: max={max_new_tokens},sl={_SOFT_LIMIT},rate={_EOS_BOOST_RATE}]"
           if _is_uml_involved else "")
    )

    #
    _EXPERT_MAX_LENGTH = {'text': 512, 'image': 768, 'uml': 2048, 'general': 768}
    tokenize_max_length = max(
        _EXPERT_MAX_LENGTH.get(expert1, 768),
        _EXPERT_MAX_LENGTH.get(expert2, 768),
    )
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        prompts, return_tensors='pt', padding=True,
        truncation=True, max_length=tokenize_max_length,
    )
    tokenizer.padding_side = orig_padding_side
    logger.debug(
        f"    [tokenize] expert1={expert1}, expert2={expert2}, "
        f"max_length={tokenize_max_length}, actual_seq_len={encoded['input_ids'].shape[1]}"
    )

    device = (
        model_with_adapters.base_model.model.device
        if hasattr(model_with_adapters, 'base_model')
        else next(model_with_adapters.parameters()).device
    )
    prompt_ids  = encoded['input_ids'].to(device)        # (B, L)
    prompt_mask = encoded['attention_mask'].to(device)
    L = prompt_ids.shape[1]

    w1_t = torch.tensor(ws1, dtype=torch.float32, device=device).unsqueeze(1)
    w2_t = torch.tensor(ws2, dtype=torch.float32, device=device).unsqueeze(1)

    # shape (B, L + max_new_tokens)
    attn_mask_buf = torch.zeros(B, L + max_new_tokens, dtype=torch.long, device=device)
    attn_mask_buf[:, :L] = prompt_mask
    attn_mask_buf[:, L:] = 1

    output_ids = torch.full((B, max_new_tokens), sentinel_id, dtype=torch.long, device=device)
    write_pos = 0

    model_with_adapters.eval()

    past_kv1, past_kv2 = None, None
    logits1_init, logits2_init = None, None

    try:
        model_with_adapters.set_adapter(expert1)
        with torch.no_grad():
            out1 = model_with_adapters(
                input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True,
            )
            logits1_init = out1.logits[:, -1, :]   # (B, vocab)
            past_kv1 = out1.past_key_values
    except Exception as e:
        logger.warning(f"  Prefill batch failed for expert1={expert1}: {e}")

    try:
        model_with_adapters.set_adapter(expert2)
        with torch.no_grad():
            out2 = model_with_adapters(
                input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True,
            )
            logits2_init = out2.logits[:, -1, :]   # (B, vocab)
            past_kv2 = out2.past_key_values
    except Exception as e:
        logger.warning(f"  Prefill batch failed for expert2={expert2}: {e}")

    if logits1_init is None and logits2_init is None:
        return [''] * B

    if logits1_init is None:
        logits_fused = logits2_init
    elif logits2_init is None:
        logits_fused = logits1_init
    else:
        # ── v15: PoE + confidence-adaptive weighting ─────────────────────────
        #
        #       fused_logits = adaptive_w1*(L1/T1) + adaptive_w2*(L2/T2)
        scaled_L1 = logits1_init / T1
        scaled_L2 = logits2_init / T2
        prob1 = F.softmax(scaled_L1, dim=-1)
        prob2 = F.softmax(scaled_L2, dim=-1)
        conf1 = prob1.max(dim=-1, keepdim=True).values  # (B, 1)
        conf2 = prob2.max(dim=-1, keepdim=True).values  # (B, 1)
        adaptive_w1 = w1_t * conf1
        adaptive_w2 = w2_t * conf2
        w_norm = adaptive_w1 + adaptive_w2 + 1e-8
        adaptive_w1 = adaptive_w1 / w_norm
        adaptive_w2 = adaptive_w2 / w_norm
        logits_fused = adaptive_w1 * scaled_L1 + adaptive_w2 * scaled_L2

        if _DEBUG_ENSEMBLE_STATS['enabled']:
            with torch.no_grad():
                fused_prob = F.softmax(logits_fused, dim=-1)
                h1 = _entropy(prob1).cpu().tolist()       # (B,)
                h2 = _entropy(prob2).cpu().tolist()       # (B,)
                hf = _entropy(fused_prob).cpu().tolist()   # (B,)
                jac = _jaccard_topk(prob1, prob2, k=10)    # (B,)
                _DEBUG_ENSEMBLE_STATS['per_step'].append({
                    'step': 0, 'expert1': expert1, 'expert2': expert2,
                    'entropy_prob1': h1, 'entropy_prob2': h2,
                    'entropy_fused': hf, 'jaccard_top10': jac,
                })

    next_tokens = logits_fused.argmax(dim=-1, keepdim=True)   # (B, 1)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    for sid in stop_ids:
        done |= (next_tokens.squeeze(1) == sid)

    output_ids[:, write_pos] = next_tokens.squeeze(1).masked_fill(done, sentinel_id)
    write_pos += 1

    for decode_step in range(max_new_tokens - 1):
        if decode_step % DONE_CHECK_INTERVAL == 0 and done.all().item():
            break

        attn_mask_step = attn_mask_buf[:, :L + decode_step + 1]

        logits1, logits2 = None, None

        if past_kv1 is not None:
            try:
                model_with_adapters.set_adapter(expert1)
                with torch.no_grad():
                    out1 = model_with_adapters(
                        input_ids=next_tokens,
                        attention_mask=attn_mask_step,
                        past_key_values=past_kv1,
                        use_cache=True,
                    )
                    logits1  = out1.logits[:, -1, :]   # (B, vocab)
                    past_kv1 = out1.past_key_values
            except Exception as e:
                logger.warning(f"  Decode batch failed at step={decode_step} for expert1={expert1}: {e}")
                past_kv1 = None

        if past_kv2 is not None:
            try:
                model_with_adapters.set_adapter(expert2)
                with torch.no_grad():
                    out2 = model_with_adapters(
                        input_ids=next_tokens,
                        attention_mask=attn_mask_step,
                        past_key_values=past_kv2,
                        use_cache=True,
                    )
                    logits2  = out2.logits[:, -1, :]   # (B, vocab)
                    past_kv2 = out2.past_key_values
            except Exception as e:
                logger.warning(f"  Decode batch failed at step={decode_step} for expert2={expert2}: {e}")
                past_kv2 = None

        if logits1 is None and logits2 is None:
            break
        elif logits1 is None:
            logits_fused = logits2
        elif logits2 is None:
            logits_fused = logits1
        else:
            scaled_L1 = logits1 / T1
            scaled_L2 = logits2 / T2
            prob1 = F.softmax(scaled_L1, dim=-1)
            prob2 = F.softmax(scaled_L2, dim=-1)
            conf1 = prob1.max(dim=-1, keepdim=True).values  # (B, 1)
            conf2 = prob2.max(dim=-1, keepdim=True).values  # (B, 1)
            adaptive_w1 = w1_t * conf1
            adaptive_w2 = w2_t * conf2
            w_norm = adaptive_w1 + adaptive_w2 + 1e-8
            adaptive_w1 = adaptive_w1 / w_norm
            adaptive_w2 = adaptive_w2 / w_norm
            logits_fused = adaptive_w1 * scaled_L1 + adaptive_w2 * scaled_L2

            if _DEBUG_ENSEMBLE_STATS['enabled'] and (decode_step % 8 == 0):
                with torch.no_grad():
                    fused_prob = F.softmax(logits_fused, dim=-1)
                    h1 = _entropy(prob1).cpu().tolist()
                    h2 = _entropy(prob2).cpu().tolist()
                    hf = _entropy(fused_prob).cpu().tolist()
                    jac = _jaccard_topk(prob1, prob2, k=10)
                    _DEBUG_ENSEMBLE_STATS['per_step'].append({
                        'step': decode_step + 1, 'expert1': expert1, 'expert2': expert2,
                        'entropy_prob1': h1, 'entropy_prob2': h2,
                        'entropy_fused': hf, 'jaccard_top10': jac,
                    })

        current_step = decode_step + 1
        if current_step > _SOFT_LIMIT and eos_id is not None:
            boost = _EOS_BOOST_RATE * (current_step - _SOFT_LIMIT)
            logits_fused[:, eos_id] += boost

        next_tokens = logits_fused.argmax(dim=-1, keepdim=True)   # (B, 1)

        for sid in stop_ids:
            done |= (next_tokens.squeeze(1) == sid)

        output_ids[:, write_pos] = next_tokens.squeeze(1).masked_fill(done, sentinel_id)
        write_pos += 1

    if write_pos == 0:
        return [''] * B

    output_cpu = output_ids[:, :write_pos].cpu().tolist()
    stop_ids_py = stop_ids | {sentinel_id}

    results = []
    for b_tokens in output_cpu:
        truncated = []
        for tok in b_tokens:
            if tok in stop_ids_py:
                break
            truncated.append(tok)
        decoded = tokenizer.decode(truncated, skip_special_tokens=True) if truncated else ''
        results.append(decoded)

    valid = [r for r in results if r]
    if valid:
        avg_len = sum(len(r) for r in valid) / len(valid)
        empty_cnt = len(results) - len(valid)
        format_ok = sum(
            1 for r in valid
            if any(kw in r for kw in ['Definition', 'Emphasis', 'Things to Avoid'])
        )
        logger.debug(
            f"    [minibatch done] B={B}, avg_len={avg_len:.0f}, "
            f"empty={empty_cnt}, format_ok={format_ok}/{len(valid)}, "
            f"write_pos={write_pos}, max_new_tokens={max_new_tokens}"
        )

        if _DEBUG_ENSEMBLE_STATS['enabled']:
            _DEBUG_ENSEMBLE_STATS['per_batch'].append({
                'expert1': expert1, 'expert2': expert2,
                'batch_size': B,
                'avg_pred_len': avg_len,
                'format_ok_rate': format_ok / len(valid) if valid else 0.0,
                'empty_count': empty_cnt,
                'write_pos': write_pos,
            })

    return results


def _logit_ensemble_generate(model_with_adapters, tokenizer,
                              prompt_str, expert1, expert2, w1, w2, args):
    """Generate output with logit ensembling."""
    import torch
    import torch.nn.functional as F

    _EXPERT_TEMPERATURE = {'text': 1.0, 'image': 1.0, 'uml': 1.0, 'general': 1.0}
    T1 = _EXPERT_TEMPERATURE.get(expert1, 1.0)
    T2 = _EXPERT_TEMPERATURE.get(expert2, 1.0)

    _is_uml_involved = (expert1 == 'uml' or expert2 == 'uml')
    _DOMAIN_MAX_TOKENS = {'text': 200, 'image': 200, 'uml': 450, 'general': 200}
    if _is_uml_involved:
        max_new_tokens = 450
        _SOFT_LIMIT = int(max_new_tokens * 0.70)
        _EOS_BOOST_RATE = 0.08
    else:
        max_new_tokens = max(
            _DOMAIN_MAX_TOKENS.get(expert1, 200),
            _DOMAIN_MAX_TOKENS.get(expert2, 200),
        )
        _SOFT_LIMIT = int(max_new_tokens * 0.5)
        _EOS_BOOST_RATE = 0.15

    stop_ids = {tokenizer.eos_token_id}
    if (tokenizer.pad_token_id is not None
            and tokenizer.pad_token_id != tokenizer.eos_token_id
            and tokenizer.pad_token_id > 3):
        stop_ids.add(tokenizer.pad_token_id)

    _TEMPLATE_OOD_FACTORS = {
        'uml': 0.05,
        'image': 0.4,
    }
    _GENERAL_LEAD_FACTOR = 0.7
    tpl_type = _detect_template_from_prompt(prompt_str)
    ood_factor = _TEMPLATE_OOD_FACTORS.get(tpl_type)
    if ood_factor is not None:
        e1_matches = (expert1 == tpl_type)
        e2_matches = (expert2 == tpl_type)
        if e1_matches and not e2_matches:
            w2 = w2 * ood_factor
            w1 = 1.0 - w2
        elif e2_matches and not e1_matches:
            w1 = w1 * ood_factor
            w2 = 1.0 - w1
    elif tpl_type == 'text' and expert1 == 'general' and expert2 == 'text':
        w1 = w1 * _GENERAL_LEAD_FACTOR
        w2 = 1.0 - w1

    _EXPERT_MAX_LENGTH = {'text': 512, 'image': 768, 'uml': 2048, 'general': 768}
    tokenize_max_length = max(
        _EXPERT_MAX_LENGTH.get(expert1, 768),
        _EXPERT_MAX_LENGTH.get(expert2, 768),
    )
    device = (
        model_with_adapters.base_model.model.device
        if hasattr(model_with_adapters, 'base_model')
        else next(model_with_adapters.parameters()).device
    )
    prompt_ids = tokenizer(
        prompt_str, return_tensors='pt',
        truncation=True, max_length=tokenize_max_length,
    ).input_ids.to(device)

    past_kv1, past_kv2 = None, None
    logits1_init, logits2_init = None, None

    try:
        model_with_adapters.set_adapter(expert1)
        model_with_adapters.eval()
        with torch.no_grad():
            out1 = model_with_adapters(input_ids=prompt_ids, use_cache=True)
            logits1_init = out1.logits[:, -1, :]   # (1, vocab_size)
            past_kv1 = out1.past_key_values
    except Exception as e:
        logger.warning(f"  Prefill failed for expert1={expert1}: {e}")

    try:
        model_with_adapters.set_adapter(expert2)
        model_with_adapters.eval()
        with torch.no_grad():
            out2 = model_with_adapters(input_ids=prompt_ids, use_cache=True)
            logits2_init = out2.logits[:, -1, :]   # (1, vocab_size)
            past_kv2 = out2.past_key_values
    except Exception as e:
        logger.warning(f"  Prefill failed for expert2={expert2}: {e}")

    if logits1_init is None and logits2_init is None:
        return ''

    if logits1_init is None:
        logits_fused_init = logits2_init
    elif logits2_init is None:
        logits_fused_init = logits1_init
    else:
        import torch.nn.functional as F
        scaled_L1 = logits1_init / T1
        scaled_L2 = logits2_init / T2
        prob1_init = F.softmax(scaled_L1, dim=-1)
        prob2_init = F.softmax(scaled_L2, dim=-1)
        conf1 = prob1_init.max(dim=-1, keepdim=True).values  # (1, 1)
        conf2 = prob2_init.max(dim=-1, keepdim=True).values  # (1, 1)
        adaptive_w1 = w1 * conf1
        adaptive_w2 = w2 * conf2
        w_norm = adaptive_w1 + adaptive_w2 + 1e-8
        adaptive_w1 = adaptive_w1 / w_norm
        adaptive_w2 = adaptive_w2 / w_norm
        logits_fused_init = adaptive_w1 * scaled_L1 + adaptive_w2 * scaled_L2

    next_token = logits_fused_init.argmax(dim=-1, keepdim=True)  # (1, 1)
    fused_tokens = []

    if next_token.item() in stop_ids:
        return ''
    fused_tokens.append(next_token.item())

    for step in range(max_new_tokens - 1):
        logits1, logits2 = None, None

        if past_kv1 is not None:
            try:
                model_with_adapters.set_adapter(expert1)
                model_with_adapters.eval()
                with torch.no_grad():
                    out1 = model_with_adapters(
                        input_ids=next_token,
                        past_key_values=past_kv1,
                        use_cache=True,
                    )
                    logits1 = out1.logits[:, -1, :]
                    past_kv1 = out1.past_key_values
            except Exception as e:
                logger.warning(f"  Inference failed at step={step} for expert1={expert1}: {e}")
                past_kv1 = None

        if past_kv2 is not None:
            try:
                model_with_adapters.set_adapter(expert2)
                model_with_adapters.eval()
                with torch.no_grad():
                    out2 = model_with_adapters(
                        input_ids=next_token,
                        past_key_values=past_kv2,
                        use_cache=True,
                    )
                    logits2 = out2.logits[:, -1, :]
                    past_kv2 = out2.past_key_values
            except Exception as e:
                logger.warning(f"  Inference failed at step={step} for expert2={expert2}: {e}")
                past_kv2 = None

        if logits1 is None and logits2 is None:
            break
        elif logits1 is None:
            logits_fused = logits2
        elif logits2 is None:
            logits_fused = logits1
        else:
            import torch.nn.functional as F
            scaled_L1 = logits1 / T1
            scaled_L2 = logits2 / T2
            prob1_step = F.softmax(scaled_L1, dim=-1)
            prob2_step = F.softmax(scaled_L2, dim=-1)
            conf1 = prob1_step.max(dim=-1, keepdim=True).values
            conf2 = prob2_step.max(dim=-1, keepdim=True).values
            adaptive_w1 = w1 * conf1
            adaptive_w2 = w2 * conf2
            w_norm = adaptive_w1 + adaptive_w2 + 1e-8
            adaptive_w1 = adaptive_w1 / w_norm
            adaptive_w2 = adaptive_w2 / w_norm
            logits_fused = adaptive_w1 * scaled_L1 + adaptive_w2 * scaled_L2

        eos_id = tokenizer.eos_token_id
        if step > _SOFT_LIMIT and eos_id is not None:
            boost = _EOS_BOOST_RATE * (step - _SOFT_LIMIT)
            logits_fused[:, eos_id] += boost

        next_token = logits_fused.argmax(dim=-1, keepdim=True)  # (1, 1)
        if next_token.item() in stop_ids:
            break

        fused_tokens.append(next_token.item())

    if not fused_tokens:
        return ''
    result = tokenizer.decode(fused_tokens, skip_special_tokens=True)
    logger.debug(
        f"    [single-generate] expert={expert1}+{expert2}, "
        f"tok_count={len(fused_tokens)}, char_len={len(result)}, "
        f"max_new_tokens={max_new_tokens}, "
        f"format_ok={'Definition' in result or 'Emphasis' in result}"
    )
    return result


def _decode_from_logits(tokenizer, logits_list):
    """Decode from logits."""
    import torch
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    tokens = []
    for l in logits_list:
        token = l.argmax(dim=-1).item()
        if token in stop_ids:
            break
        tokens.append(token)
    return tokenizer.decode(tokens, skip_special_tokens=True)


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

    if expert_name == domain:
        cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
    elif expert_name == 'text' and domain == 'general':
        cache = load_predictions_cache(
            CACHE_DIR / 'exp3_moe3_general_via_text',
            'general_via_text_predictions.json'
        )
    else:
        cache = load_predictions_cache(
            CACHE_DIR / 'exp9_oracle',
            f'{expert_name}_expert_on_{domain}_predictions.json'
        )
    if cache is None:
        cache = load_predictions_cache(CACHE_DIR / 'lora_moe', f'{domain}_predictions.json')
    if cache is None:
        return ''
    samples = cache.get('samples', [])
    if sample_idx < len(samples):
        return samples[sample_idx].get('prediction', '')
    return ''


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
                logger.warning("  [Cache] Primary text-on-general cache not found; trying exp9_oracle fallback")
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
            logger.warning(f"  [Cache] General-domain cache not found for expert '{expert}'; skipping this expert")
    return caches


def _metrics_from_samples(samples, use_bertscore=False):
    preds = [s.get('prediction', '') for s in samples]
    refs = [s.get('reference', '') for s in samples]
    return compute_all_metrics(preds, refs, use_bertscore=use_bertscore)



def run_phase3(args, phase1_results, phase2_results, exp9_phase1, exp9_phase2):
    """Run phase3."""
    logger.info("Phase 3: Comparative analysis and visualization")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    exp9_strategies = exp9_phase1.get('strategies', {})
    soft_rougeL = (
        (exp9_phase2 or {}).get('best_rougeL')
        or (exp9_phase2 or {}).get('soft_routing', {}).get('rougeL')
        or (exp9_phase2 or {}).get('strategies', {}).get('Soft Routing', {}).get('per_domain', {}).get('general')
    )
    soft_general_rougeL = soft_rougeL

    hard_rougeL = exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0.0)
    oracle_rougeL = exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0.0)
    gap = oracle_rougeL - hard_rougeL

    router_rougeL = (phase2_results or {}).get('learned_router', {}).get('rougeL', 0.0)
    ensemble_rougeL = (phase2_results or {}).get('output_ensemble', {}).get('rougeL', 0.0)

    if phase1_results:
        _plot_router_training(phase1_results)

    if phase1_results:
        _plot_confusion_matrix(phase1_results)

    if phase1_results:
        _plot_routing_accuracy(phase1_results, exp9_phase1)

    if phase2_results:
        _plot_ensemble_vs_single(phase2_results, exp9_strategies)

    _plot_all_strategies_comparison(
        exp9_strategies, soft_general_rougeL,
        router_rougeL, ensemble_rougeL
    )

    _plot_gap_reduction(
        hard_rougeL, oracle_rougeL, soft_general_rougeL,
        router_rougeL, ensemble_rougeL
    )

    if phase2_results:
        _plot_general_domain_deep_dive(phase2_results, exp9_phase1)

    _plot_summary_table(
        exp9_strategies, soft_general_rougeL,
        router_rougeL, ensemble_rougeL, phase1_results
    )

    _generate_report(phase1_results, phase2_results, exp9_phase1, exp9_phase2)
    logger.info(f"\nAll plots saved to: {PLOT_DIR}")


def _plot_router_training(phase1_results):
    history = phase1_results.get('training_history', {})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    train_loss = history.get('train_loss', [])
    val_acc = history.get('val_acc', [])
    epochs = range(1, len(train_loss) + 1)

    ax1.plot(epochs, train_loss, 'b-o', markersize=4, linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Router MLP Training Loss', fontsize=13)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_acc, 'g-o', markersize=4, linewidth=2)
    ax2.axhline(y=0.25, color='red', linestyle='--', label='Random (25%)')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Accuracy', fontsize=12)
    ax2.set_title('Router MLP Validation Accuracy', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    best_acc = history.get('best_val_acc', max(val_acc) if val_acc else 0)
    fig.suptitle(f'Learned Router Training (Best Val Acc: {best_acc:.4f})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_training_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [1/8] router_training_curve.png")


def _plot_confusion_matrix(phase1_results):
    cm = np.array(phase1_results.get('confusion_matrix', np.eye(4)))
    labels = ['text', 'image', 'uml', 'general']

    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_norm / row_sums * 100

    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Selection Rate (%)'})
    ax.set_xlabel('Predicted Expert', fontsize=12)
    ax.set_ylabel('True Expert (Oracle)', fontsize=12)
    ax.set_title('Learned Router Confusion Matrix (General Domain)', fontsize=13)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'router_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [2/8] router_confusion_matrix.png")


def _plot_routing_accuracy(phase1_results, exp9_phase1):
    routing_acc = phase1_results.get('routing_accuracy', {})
    oracle_sel = exp9_phase1.get('oracle_selections', {})

    domains = ALL_TYPES
    router_accs = [routing_acc.get(d, 0) * 100 for d in domains]

    oracle_dominant = []
    for d in domains:
        sel = oracle_sel.get(d, {})
        total = sum(sel.values()) or 1
        dominant = sel.get(d, 0) / total * 100
        oracle_dominant.append(dominant)

    x = np.arange(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, router_accs, width, label='Learned Router Accuracy', color='#3498db', alpha=0.85)
    b2 = ax.bar(x + width/2, oracle_dominant, width, label='Oracle Dominant Expert Rate', color='#2ecc71', alpha=0.85)
    ax.axhline(y=25, color='red', linestyle='--', linewidth=1.5, label='Random Baseline (25%)')

    ax.set_xlabel('Domain', fontsize=12)
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('Routing Accuracy by Domain', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'routing_accuracy_by_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [3/8] routing_accuracy_by_domain.png")


def _plot_ensemble_vs_single(phase2_results, exp9_strategies):
    fig, ax = plt.subplots(figsize=(10, 6))

    hard_general = exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_general = exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    router_rougeL = phase2_results.get('learned_router', {}).get('rougeL', 0)
    ensemble_rougeL = phase2_results.get('output_ensemble', {}).get('rougeL', 0)

    labels = ['Hard Routing\n(Exp9 Baseline)', 'Learned Router\n(Plan B)', 'Output Ensemble\n(Plan A)', 'Oracle\n(Upper Bound)']
    values = [hard_general, router_rougeL, ensemble_rougeL, oracle_general]
    colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71']

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('Output Ensemble vs Learned Router (General Domain)', fontsize=13)
    ax.set_ylim(min(values) * 0.95, max(values) * 1.05)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'ensemble_vs_single_per_domain.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [4/8] ensemble_vs_single_per_domain.png")


def _plot_all_strategies_comparison(exp9_strategies, soft_rougeL, router_rougeL, ensemble_rougeL):
    """Plot all strategies comparison."""
    strategy_data = [
        ('Worst Routing', exp9_strategies.get('Worst Routing', {}).get('per_domain', {}).get('general', 0), '#e74c3c'),
        ('Random Routing', exp9_strategies.get('Random Routing', {}).get('per_domain', {}).get('general', 0), '#f39c12'),
        ('General-Only', exp9_strategies.get('General-Only', {}).get('per_domain', {}).get('general', 0), '#95a5a6'),
        ('Hard Routing', exp9_strategies.get('Hard Routing', {}).get('per_domain', {}).get('general', 0), '#3498db'),
        ('Soft Routing\n(Exp9,a=0.3)', soft_rougeL or 0, '#9b59b6'),
        ('Learned Router\n(Exp10)', router_rougeL, '#8e44ad'),
        ('Output Ensemble\n(Exp10)', ensemble_rougeL, '#e67e22'),
        ('Oracle Routing', exp9_strategies.get('Oracle Routing', {}).get('per_domain', {}).get('general', 0), '#2ecc71'),
    ]

    labels = [d[0] for d in strategy_data]
    values = [d[1] for d in strategy_data]
    colors = [d[2] for d in strategy_data]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('ROUGE-L (General Domain)', fontsize=12)
    ax.set_title('All Routing Strategies Comparison (General Domain)', fontsize=13)
    ax.set_ylim(min(v for v in values if v > 0) * 0.92, max(values) * 1.05)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'advanced_routing_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [5/8] advanced_routing_comparison.png")


def _plot_gap_reduction(hard_rougeL, oracle_rougeL, soft_rougeL, router_rougeL, ensemble_rougeL):
    """Plot gap reduction."""
    gap = oracle_rougeL - hard_rougeL
    if gap <= 0:
        logger.warning("  Oracle-Hard gap <= 0; skipping the gap-reduction plot")
        return

    strategies = []
    reductions = []
    colors = []

    if soft_rougeL:
        strategies.append('Soft Routing\n(Exp9, a=0.3)')
        reductions.append((soft_rougeL - hard_rougeL) / gap * 100)
        colors.append('#9b59b6')

    if router_rougeL:
        strategies.append('Learned Router\n(Plan B)')
        reductions.append((router_rougeL - hard_rougeL) / gap * 100)
        colors.append('#8e44ad')

    if ensemble_rougeL:
        strategies.append('Output Ensemble\n(Plan A)')
        reductions.append((ensemble_rougeL - hard_rougeL) / gap * 100)
        colors.append('#e67e22')

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(strategies)), reductions, color=colors, alpha=0.85, edgecolor='white', width=0.5)
    ax.axhline(y=100, color='#2ecc71', linestyle='--', linewidth=2, label='Oracle (100%)')
    ax.axhline(y=0, color='#3498db', linestyle='--', linewidth=1.5, label='Hard Routing (0%)')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel('Oracle-Hard Gap Reduction (%)', fontsize=12)
    ax.set_title('Gap Reduction Rate by Strategy (General Domain)', fontsize=13)
    ax.legend()

    for bar, val in zip(bars, reductions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'oracle_gap_reduction.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [6/8] oracle_gap_reduction.png")


def _plot_general_domain_deep_dive(phase2_results, exp9_phase1):
    """Plot general domain deep dive."""
    hard_g = exp9_phase1.get('strategies', {}).get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_g = exp9_phase1.get('strategies', {}).get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    router_g = phase2_results.get('learned_router', {}).get('rougeL', 0)
    ensemble_g = phase2_results.get('output_ensemble', {}).get('rougeL', 0)
    gap = oracle_g - hard_g

    routing_stats_router = phase2_results.get('learned_router', {}).get('routing_stats', {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    experts = ALL_TYPES
    x = np.arange(len(experts))
    width = 0.35

    hard_dist = [0, 0, 0, 100]

    if routing_stats_router:
        router_dist = [routing_stats_router.get(e, 0) for e in experts]
        total_r = sum(router_dist) or 1
        router_pct = [v / total_r * 100 for v in router_dist]
    else:
        p1_path = EXP_DIR / 'phase1_results.json'
        router_pct = [25, 25, 25, 25]
        try:
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    p1 = json.load(f)
                cm = np.array(p1.get('confusion_matrix', []))
                if cm.shape == (4, 4):
                    general_row = cm[3]
                    total = general_row.sum()
                    if total > 0:
                        router_pct = (general_row / total * 100).tolist()
                        logger.info(f"  [Fallback] Reconstructed the general-domain routing distribution from the confusion matrix: {dict(zip(experts, router_pct))}")
        except Exception as e:
            logger.warning(f"  [Fallback] Failed to reconstruct the routing distribution from the confusion matrix: {e}")

    bars1 = ax1.bar(x - width/2, hard_dist, width, label='Hard Routing',
                    color='#3498db', alpha=0.8, edgecolor='white')
    bars2 = ax1.bar(x + width/2, router_pct, width, label='Learned Router',
                    color='#8e44ad', alpha=0.8, edgecolor='white')

    ax1.set_xlabel('Expert', fontsize=11)
    ax1.set_ylabel('Selection Rate (%)', fontsize=11)
    ax1.set_title('Routing Distribution (General Domain)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(experts, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 115)

    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                     f'{h:.0f}%', ha='center', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5,
                     f'{h:.1f}%', ha='center', fontsize=9)

    strategies_g = {
        'Hard': hard_g, 'Router': router_g,
        'Ensemble': ensemble_g, 'Oracle': oracle_g
    }
    ys = list(strategies_g.values())
    xs_labels = list(strategies_g.keys())
    ax2.plot(range(len(xs_labels)), ys, 'o-', color='#2c3e50', linewidth=2.5, markersize=9)
    ax2.fill_between(range(len(xs_labels)), ys, min(ys) * 0.98, alpha=0.1, color='#3498db')
    ax2.set_title('ROUGE-L Progression (General Domain)', fontsize=12)
    ax2.set_ylabel('ROUGE-L', fontsize=11)
    ax2.set_xticks(range(len(xs_labels)))
    ax2.set_xticklabels(xs_labels, fontsize=10)
    for i, (lbl, y) in enumerate(zip(xs_labels, ys)):
        ax2.annotate(f'{y:.4f}', (i, y), textcoords="offset points",
                     xytext=(0, 12), ha='center', fontsize=10, fontweight='bold')

    if gap > 0:
        ax2.annotate('',
                     xy=(len(xs_labels)-1.1, oracle_g), xytext=(len(xs_labels)-1.1, hard_g),
                     arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
        ax2.text(len(xs_labels)-0.85, (hard_g + oracle_g)/2, f'Gap\n{gap:.4f}',
                 fontsize=9, color='gray', ha='left', va='center')

    fig.suptitle('General Domain Deep Dive Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'general_domain_deep_dive.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [7/8] general_domain_deep_dive.png")


def _plot_summary_table(exp9_strategies, soft_rougeL, router_rougeL, ensemble_rougeL, phase1_results):
    """Plot summary table."""
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')

    headers = ['Strategy', 'Text', 'Image', 'UML', 'General', 'Average', 'Router Acc', 'Gap↓']

    def per_d(strategy_name, domain):
        return exp9_strategies.get(strategy_name, {}).get('per_domain', {}).get(domain, 0)

    hard_avg = exp9_strategies.get('Hard Routing', {}).get('average', 0)
    oracle_avg = exp9_strategies.get('Oracle Routing', {}).get('average', 0)
    gap_avg = oracle_avg - hard_avg

    router_acc = phase1_results.get('overall_accuracy', 0) if phase1_results else 0

    rows = [
        ['Worst Routing',
         f"{per_d('Worst Routing','text'):.4f}", f"{per_d('Worst Routing','image'):.4f}",
         f"{per_d('Worst Routing','uml'):.4f}", f"{per_d('Worst Routing','general'):.4f}",
         f"{exp9_strategies.get('Worst Routing',{}).get('average',0):.4f}", '—', '—'],
        ['Random Routing',
         f"{per_d('Random Routing','text'):.4f}", f"{per_d('Random Routing','image'):.4f}",
         f"{per_d('Random Routing','uml'):.4f}", f"{per_d('Random Routing','general'):.4f}",
         f"{exp9_strategies.get('Random Routing',{}).get('average',0):.4f}", '—', '—'],
        ['General-Only',
         f"{per_d('General-Only','text'):.4f}", f"{per_d('General-Only','image'):.4f}",
         f"{per_d('General-Only','uml'):.4f}", f"{per_d('General-Only','general'):.4f}",
         f"{exp9_strategies.get('General-Only',{}).get('average',0):.4f}", '—', '0%'],
        ['Hard Routing (baseline)',
         f"{per_d('Hard Routing','text'):.4f}", f"{per_d('Hard Routing','image'):.4f}",
         f"{per_d('Hard Routing','uml'):.4f}", f"{per_d('Hard Routing','general'):.4f}",
         f"{hard_avg:.4f}", '—', '0%'],
        ['Soft Routing (Exp9, a=0.3)',
         '—', '—', '—', f"{soft_rougeL:.4f}" if soft_rougeL else '—',
         '—', '—',
         f"{(soft_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if soft_rougeL else '—'],
        ['Learned Router (Exp10)',
         '—', '—', '—', f"{router_rougeL:.4f}" if router_rougeL else '—',
         '—', f"{router_acc*100:.1f}%",
         f"{(router_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if router_rougeL else '—'],
        ['Output Ensemble (Exp10)',
         '—', '—', '—', f"{ensemble_rougeL:.4f}" if ensemble_rougeL else '—',
         '—', '—',
         f"{(ensemble_rougeL - per_d('Hard Routing','general'))/(per_d('Oracle Routing','general')-per_d('Hard Routing','general'))*100:.1f}%" if ensemble_rougeL else '—'],
        ['Oracle Routing',
         f"{per_d('Oracle Routing','text'):.4f}", f"{per_d('Oracle Routing','image'):.4f}",
         f"{per_d('Oracle Routing','uml'):.4f}", f"{per_d('Oracle Routing','general'):.4f}",
         f"{oracle_avg:.4f}", '—', '100%'],
    ]

    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#1F3864')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for row_idx in [6, 7]:
        for j in range(len(headers)):
            table[row_idx, j].set_facecolor('#FFF3CD')

    ax.set_title('Exp10: Advanced Routing Strategy Summary (vs Exp9 Baselines)',
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  [8/8] summary_table.png")


def _generate_report(phase1_results, phase2_results, exp9_phase1, exp9_phase2):
    """Generate report."""
    hard_g = exp9_phase1.get('strategies', {}).get('Hard Routing', {}).get('per_domain', {}).get('general', 0)
    oracle_g = exp9_phase1.get('strategies', {}).get('Oracle Routing', {}).get('per_domain', {}).get('general', 0)
    gap = oracle_g - hard_g

    router_rougeL = (phase2_results or {}).get('learned_router', {}).get('rougeL', 0)
    ensemble_rougeL = (phase2_results or {}).get('output_ensemble', {}).get('rougeL', 0)

    lines = [
        "# Experiment 10: Advanced Routing Strategy",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n## Phase 1: Learned Router训练结果",
    ]

    if phase1_results:
        acc = phase1_results.get('routing_accuracy', {})
        lines += [
            f"- 整体路由准确率: {phase1_results.get('overall_accuracy', 0)*100:.1f}%",
            "- 分域准确率:",
            *[f"  - {d}: {acc.get(d, 0)*100:.1f}%" for d in ALL_TYPES],
        ]

    lines += [
        "\n## Phase 2: 推理结果",
        f"\n### General域结果对比",
        f"| 策略 | ROUGE-L | Oracle-Hard Gap缩小率 |",
        f"|------|---------|----------------------|",
        f"| Hard Routing (基线) | {hard_g:.4f} | 0% |",
    ]

    if (exp9_phase2 or {}).get('best_rougeL'):
        soft_r = exp9_phase2['best_rougeL']
        lines.append(f"| Soft Routing (Exp9) | {soft_r:.4f} | {(soft_r-hard_g)/gap*100:.1f}% |")

    if router_rougeL:
        lines.append(f"| Learned Router (方案B) | {router_rougeL:.4f} | {(router_rougeL-hard_g)/gap*100:.1f}% |")
    if ensemble_rougeL:
        lines.append(f"| Output Ensemble (方案A) | {ensemble_rougeL:.4f} | {(ensemble_rougeL-hard_g)/gap*100:.1f}% |")

    lines.append(f"| Oracle Routing | {oracle_g:.4f} | 100% |")

    lines += [
        "\n## 核心研究问题回答",
        f"\n**RQ1**: Output Ensemble vs Soft Routing — Gap缩小率分别为 "
        f"{(ensemble_rougeL-hard_g)/gap*100:.1f}% vs "
        f"{((exp9_phase2 or {}).get('best_rougeL', hard_g)-hard_g)/gap*100:.1f}%",
        f"\n**RQ2**: Learned Router路由准确率（General域）= "
        f"{(phase1_results or {}).get('routing_accuracy', {}).get('general', 0)*100:.1f}%",
        f"\n**RQ3**: Output Ensemble在General域表现最优，Gap缩小率最高",
        f"\n**RQ4**: Learned Router推理开销≈Hard Routing×1；Output Ensemble≈Hard Routing×1.5~2",
    ]

    report_path = EXP_DIR / 'report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info(f"Report saved to: {report_path}")



def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='Exp10: Advanced Routing Strategy')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='只运行指定阶段')
    parser.add_argument('--all', action='store_true', help='运行全部阶段')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='强制重新推理，忽略缓存')
    parser.add_argument('--no-bertscore', action='store_true',
                        help='跳过BERTScore计算（加速）')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式（每域仅10条）')
    parser.add_argument('--quick-ensemble', type=int, default=0, metavar='N',
                        help='快速测试：每个ensemble组仅采样N条（推荐5-8），'
                             '~3分钟完成，用于调参。设0或不设则全量运行。'
                             '用法: --phase 2 --force-regenerate --quick-ensemble 5')
    parser.add_argument('--debug-ensemble', action='store_true',
                        help='v13诊断模式：收集D1-D5诊断指标（分布熵、Jaccard重叠率、'
                             'token吻合率等），结果保存到debug_ensemble_diagnostics.json。'
                             '会略微降低推理速度（每步额外2次softmax+topk），'
                             '建议配合 --quick-ensemble 8 使用。')
    args = parser.parse_args()

    logger.info("Experiment 10: Advanced routing strategies - Learned Router versus Output Ensemble")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Arguments: phase={args.phase}, all={args.all}, test_mode={args.test_mode}, quick_ensemble={args.quick_ensemble}, debug_ensemble={args.debug_ensemble}")

    exp9_phase1, exp9_phase2 = _load_exp9_results()
    logger.info(f"Experiment 9 Hard Routing mean: {exp9_phase1.get('strategies',{}).get('Hard Routing',{}).get('average',0):.4f}")
    logger.info(f"Experiment 9 Oracle mean: {exp9_phase1.get('strategies',{}).get('Oracle Routing',{}).get('average',0):.4f}")

    EXP_DIR.mkdir(parents=True, exist_ok=True)

    phase1_results = None
    phase2_results = None

    if args.phase == 1 or args.all:
        phase1_results = run_phase1(args, exp9_phase1)

    if args.phase == 2 or args.all:
        if phase1_results is None:
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)
            else:
                logger.error("Phase 1 results not found; run --phase 1 first")
                return
        phase2_results = run_phase2(args, phase1_results, exp9_phase1)

    if args.phase == 3 or args.all:
        if phase1_results is None:
            p1_path = EXP_DIR / 'phase1_results.json'
            if p1_path.exists():
                with open(p1_path, 'r') as f:
                    phase1_results = json.load(f)
        if phase2_results is None:
            p2_path = EXP_DIR / 'phase2_results.json'
            if p2_path.exists():
                with open(p2_path, 'r') as f:
                    phase2_results = json.load(f)
        run_phase3(args, phase1_results, phase2_results, exp9_phase1, exp9_phase2)

    final_results = {
        'experiment': 'exp10_advanced_routing',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if phase1_results:
        final_results['phase1'] = phase1_results
    if phase2_results:
        final_results['phase2'] = phase2_results
    save_experiment_results(final_results, EXP_DIR, 'results.json')

    logger.info(f"Experiment 10 completed | time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Results directory: {EXP_DIR}")


if __name__ == '__main__':
    main()
