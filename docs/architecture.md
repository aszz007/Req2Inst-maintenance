# Architecture

This document describes the current implemented architecture without rewriting
legacy code or claiming that every historical experiment uses the same path.

## System boundary

Req2Inst accepts text requirements, image requirements, flowchart requirements,
or recognized
JSON. It produces English crowdsourcing instructions and experiment metadata.
Base model files, LoRA checkpoints, datasets, and routine outputs are external
local artifacts.

## Terminology and compatibility boundary

The manuscript uses **FlowChart** as the public name of the third requirement
domain. The implementation predates that terminology and retains `uml` in paths,
class names, dataset keys, and CLI values. Documentation and display text use
FlowChart, while internal uml contracts remain unchanged to preserve
reproducibility.

## End-to-end inference path

1. `scripts/inference/generate_instructions.py` scans `inputs/` or a supplied
   input directory.
2. Text files are loaded directly. Image and flowchart files can be passed to
   `scripts/inference/recognize_inputs.py`.
3. `models/vision_model.py` uses Qwen3-VL-8B-Instruct to produce a structured
   image or flowchart description.
4. `src/instruction_generation/generator.py` provides the public generation
   interface.
5. `src/routing/moe_model.py` normalizes the input and invokes a router.
6. The manuscript describes adaptive expert selection with a Router MLP,
   including single-expert routing and top-2 output ensembling. The current
   default
   entry point still uses `src/routing/expert_router.py` for type-based selection;
   soft and learned routing implementations remain under
   `src/routing/soft_router.py` and `src/routing/learned_router.py`.
7. A Text, Image, FlowChart, or General expert builds its domain prompt and
   calls the
   shared Qwen3-8B language-model wrapper with a LoRA adapter.
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
- `models/vision_model.py`: Qwen3-VL-8B-Instruct loading and image/flowchart
  recognition.
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

`src/utils/enhanced_metrics.py` implements generation, format, and binary
evaluation reports. `scripts/evaluation/experiments/` contains experiments
1-11. Experiment 10 contains the manuscript-reported MLP Learned Router and
Output Ensemble evaluation, along with later repository-only diagnostics and
weighting explorations. Only the paper-described top-1 selection and top-2 logit
fusion configuration should be treated as manuscript evidence; later variants
are not separate validated paper improvements. Cached predictions and generated
plots are local artifacts, not source.

### Quality format

Prompt templates and expert normalization target three lines: Definition,
Emphasis & Caution, and Things to Avoid. A separate
`src/instruction_generation/quality_validator.py` can validate this structure;
not every historical script invokes that validator through the same path.

## Current source-of-truth policy

The active baseline is:

- Qwen3-8B for instruction generation;
- Qwen3-VL-8B-Instruct for image and flowchart recognition;
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
- Some helper logic is duplicated across experiment and preprocessing scripts.
- The lightweight test suite covers tracked-file syntax, documentation links,
  manuscript terminology, model compatibility, and experiment defaults. It does
  not replace model-backed training, inference, or metric validation.

These items describe future refactoring scope; they are not evidence that the
completed experiment run was invalid.
