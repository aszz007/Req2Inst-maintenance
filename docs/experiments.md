# Experiment index

The canonical orchestrator is
`scripts/evaluation/experiments/run_all_experiments.py`.

| No. | Script | Purpose |
| --- | --- | --- |
| 1 | `exp1_baseline_comparison.py` | Baseline method comparison |
| 2 | `exp2_compare_finetuning_methods.py` | Paper-aligned fine-tuning method comparison plus repository-only full fine-tuning |
| 3 | `exp3_moe_architecture_validation.py` | Multi-expert architecture validation |
| 4 | `exp4_lora_hyperparameter_optimization.py` | LoRA hyperparameter search |
| 5 | `exp5_data_efficiency_analysis.py` | Data-efficiency analysis |
| 6 | `exp6_fewshot_vs_finetuning.py` | Few-shot versus fine-tuning |
| 7 | `exp7_uml_hyperparameter_optimization.py` | FlowChart expert hyperparameter search |
| 8 | `exp8_inference_efficiency.py` | Inference-efficiency analysis |
| 9 | `exp9_routing_strategy.py` | Routing strategy comparison |
| 10 | `exp10_advanced_routing.py` | Repository-only advanced routing and output-ensemble exploration |
| 11 | `exp11_ablation_optimization.py` | Ablation experiments |

Experiments 9-11 receive `--all` automatically from the orchestrator.

Experiment 10 is not part of the manuscript evaluation. The tracked script
retains the last complete v15 confidence-adaptive PoE implementation for
historical reproducibility, but its results should not be interpreted as a
validated improvement or a paper claim. Earlier weighting and probability-
mixture variants remain recoverable from Git history.

## Usage

```bash
# All experiments
python scripts/evaluation/experiments/run_all_experiments.py

# Selected experiments
python scripts/evaluation/experiments/run_all_experiments.py \
  --experiments 1,2,3

# Small validation and continue after failures
python scripts/evaluation/experiments/run_all_experiments.py \
  --test-mode --skip-failed

# Reuse compatible prediction caches
python scripts/evaluation/experiments/run_all_experiments.py \
  --from-cache --test-mode --skip-failed
```

## Result interpretation

- A cache hit is not equivalent to a fresh model run.
- Compare metrics only when dataset splits, prompt templates, checkpoints,
  precision/quantization settings, and metric versions match.
- Format validity and semantic similarity are separate measurements.
- Binary classifications depend on threshold and AND/OR policy; inspect the
  saved report rather than quoting only one headline number.
- Generated plots and cached predictions belong under `outputs/` and should not
  be committed by default.
