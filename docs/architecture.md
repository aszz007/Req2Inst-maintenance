# Architecture

This document distinguishes the manuscript architecture from the repository's
current executable paths. The distinction prevents later utilities or
compatibility code from being reported as part of the published experiment
design.

## System boundary

Req2Inst accepts text requirements, image requirements, flowchart requirements,
or recognized JSON. It produces English crowdsourcing instructions and
experiment metadata. Base model files, LoRA checkpoints, datasets, and routine
outputs are external local artifacts.

## Manuscript architecture

The manuscript presents a four-stage workflow:

1. collect text, image, and FlowChart requirements;
2. preprocess and augment the requirements, then construct structured
   requirement-instruction pairs;
3. adapt Qwen3-8B with parameter-efficient fine-tuning, including the proposed
   four-expert LoRA method; and
4. compare the resulting methods through automatic, routing, efficiency, and
   human evaluation.

In the manuscript setup, BLIP-2 extracts entities and attributes from images,
while Qwen3-VL-8B extracts procedural and logical information from FlowCharts.
Both models are used during offline preprocessing rather than Qwen3-8B training
or instruction-generation inference. The Router MLP then supports top-1 expert
selection and top-2 output-space fusion over Text, Image, FlowChart, and General
LoRA experts.

## Terminology and compatibility boundary

The manuscript uses **FlowChart** as the public name of the third requirement
domain. The implementation predates that terminology and retains `uml` in paths,
class names, dataset keys, and CLI values. Documentation and display text use
FlowChart, while internal uml contracts remain unchanged to preserve
reproducibility.

## Repository inference path

1. `scripts/inference/generate_instructions.py` scans `inputs/` or a supplied
   input directory.
2. Text files are loaded directly. Image and flowchart files can be passed to
   `scripts/inference/recognize_inputs.py`.
3. For local convenience, `models/vision_model.py` can use
   Qwen3-VL-8B-Instruct to produce a structured image or FlowChart description.
   This repository path is not the manuscript's BLIP-2/Qwen3-VL offline
   preprocessing setup.
4. `src/instruction_generation/generator.py` provides the public generation
   interface.
5. `src/routing/moe_model.py` normalizes the input and invokes a router.
6. The current default entry point uses `src/routing/expert_router.py` for
   type-based selection. The manuscript's Router MLP and output-ensemble path is
   implemented in the advanced routing experiment; soft and learned routing
   implementations remain under
   `src/routing/soft_router.py` and `src/routing/learned_router.py`.
7. A Text, Image, FlowChart, or General expert builds its domain prompt and
   calls the shared Qwen3-8B language-model wrapper with a LoRA adapter.
8. The generated instruction and routing metadata are written under
   `outputs/generated_instructions/`.

The default production-like path selects one domain expert. More advanced
mixture, soft-routing, learned-router, and ensemble behavior is evaluated in
the experiment scripts and should not be assumed to run in every invocation.

## Major components

### Configuration

`config/settings.py` centralizes model locations, dataset locations,
checkpoint roots, training parameters, inference parameters, and device
detection. It is the current source of truth for the Qwen3 baseline.

### Model wrappers

- `models/language_model.py`: Qwen3-8B loading, optional 4-bit execution, LoRA
  adapter management, generation, batching, and output cleanup.
- `models/vision_model.py`: repository-side Qwen3-VL-8B-Instruct loading and
  image/FlowChart recognition.
- `models/prompt_templates/`: domain-specific three-part instruction prompts.

### Experts and routing

- `src/experts/`: base expert and four domain expert implementations.
- `src/routing/expert_router.py`: default rule/type routing.
- `src/routing/moe_model.py`: expert lifecycle and unified routing interface.
- `src/routing/soft_router.py`: weighted adapter routing used by experiments.
- `src/routing/learned_router.py`: hidden-state feature extraction and learned
  routing used by advanced experiments.

### Training

`src/training/` contains dataset loaders, base training behavior, and trainer
variants. Method- and expert-specific launchers are under `scripts/training/`.
General-expert data is assembled dynamically from text, image, and FlowChart
sources.

### Evaluation

`src/utils/enhanced_metrics.py` is the canonical engine for generation,
format, binary, and statistical evaluation. The experiment-facing wrapper in
`src/baselines/inference_utils.py` and the cached-prediction CLI in
`scripts/evaluation/calculate_metrics_from_json.py` delegate to this engine so
that both paths use the same format parser, thresholds, and per-sample metric
values.

The manuscript task-specific success criterion requires the three-part
Definition, Emphasis & Caution, and Things to Avoid structure together with
ROUGE-L >= 0.5 and BERTScore F1 >= 0.85. Semantic adequacy uses AND logic, and
overall success requires both format and semantic adequacy. CLI threshold
overrides remain available for diagnostic comparisons, but they are not the
manuscript-default evaluation contract.

`scripts/evaluation/experiments/` contains experiments 1-11. Experiment 10
contains the manuscript-reported MLP Learned Router and Output Ensemble
evaluation, along with later repository-only diagnostics and weighting
explorations. Only the paper-described top-1 selection and top-2 logit fusion
configuration should be treated as manuscript evidence; later variants are not
separate validated paper improvements. Cached predictions and generated plots
are local artifacts, not source.

### Quality format

Prompt templates and expert normalization target three lines: Definition,
Emphasis & Caution, and Things to Avoid. Format and generation-quality
measurements are calculated by `src/utils/enhanced_metrics.py` during
evaluation.

## Source-of-truth policy

For manuscript-facing descriptions, the current paper draft defines the
datasets, preprocessing models, method, routing configuration, experiment
environment, and reported claims.

For executable repository behavior, the active baseline is:

- Qwen3-8B for instruction generation;
- Qwen3-VL-8B-Instruct for optional local image and FlowChart recognition;
- four domain experts;
- `checkpoints/lora_moe/` as the standard Multi-Expert LoRA checkpoint root.

The active expert registry uses the `qwen3_8b` model identifier. Central
configuration, model wrappers, CLI compatibility checks, and regression tests
define the supported model-version boundary.

## Known structural debt

- Several scripts combine orchestration, model execution, metrics, and plotting.
- Some paths are hard-coded for the original Windows or Linux workstation.
- Historical path conventions and environment names coexist with current entry
  points.
- Some helper logic is duplicated across experiment scripts.
- Repository quality automation checks tracked-file syntax and documentation
  links. It does not replace direct training, inference, or metric validation.

These items describe future refactoring scope; they are not evidence that the
completed experiment run was invalid.
