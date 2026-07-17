# Architecture

This document describes the current implemented architecture without rewriting
legacy code or claiming that every historical experiment uses the same path.

## System boundary

Req2Inst accepts text requirements, natural images, UML diagrams, or recognized
JSON. It produces English crowdsourcing instructions and experiment metadata.
Base model files, LoRA checkpoints, datasets, and routine outputs are external
local artifacts.

## End-to-end inference path

1. `scripts/inference/generate_instructions.py` scans `inputs/` or a supplied
   input directory.
2. Text files are loaded directly. Image and UML files can be passed to
   `scripts/inference/recognize_inputs.py`.
3. `models/vision_model.py` uses Qwen3-VL-8B-Instruct to produce a structured
   image or UML description.
4. `src/instruction_generation/generator.py` provides the public generation
   interface.
5. `src/routing/moe_model.py` normalizes the input and invokes a router.
6. `src/routing/expert_router.py` performs the default type-based selection.
   Experimental soft and learned routing implementations live in
   `src/routing/soft_router.py` and `src/routing/learned_router.py`.
7. A Text, Image, UML, or General expert builds its domain prompt and calls the
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
- `models/vision_model.py`: Qwen3-VL-8B-Instruct loading and image/UML
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
General-expert data is assembled dynamically from text, image, and UML sources.

### Evaluation

`src/utils/enhanced_metrics.py` implements generation, format, and binary
evaluation reports. `scripts/evaluation/experiments/` contains experiments
1-11. Cached predictions and generated plots are local artifacts, not source.

### Quality format

Prompt templates and expert normalization target three lines: Definition,
Emphasis & Caution, and Things to Avoid. A separate
`src/instruction_generation/quality_validator.py` can validate this structure;
not every historical script invokes that validator through the same path.

## Current source-of-truth policy

The active baseline is:

- Qwen3-8B for instruction generation;
- Qwen3-VL-8B-Instruct for recognition;
- four domain experts;
- `checkpoints/lora_moe/` as the standard LoRA-MoE checkpoint root.

Qwen-7B model metadata still present in `src/routing/expert_router.py` is known
legacy state. It is not modified during documentation cleanup because changing
runtime metadata belongs in a dedicated, regression-tested maintenance task.

## Known structural debt

- Several scripts combine orchestration, model execution, metrics, and plotting.
- Some paths are hard-coded for the original Windows or Linux workstation.
- Historical environment and model names coexist with current names.
- Some helper logic is duplicated across experiment and preprocessing scripts.
- There is no established lightweight test suite yet.

These items describe future refactoring scope; they are not evidence that the
completed experiment run was invalid.
