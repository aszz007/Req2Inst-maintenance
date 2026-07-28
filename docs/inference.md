# Inference guide

This guide covers the repository inference path only. It does not reproduce
training, the manuscript experiments, or the manuscript offline visual
preprocessing pipeline.

## Supported local path

The maintained inference path uses:

- Qwen3-8B for instruction generation;
- the Text, Image, FlowChart, and General LoRA adapters under
  `checkpoints/lora_moe/`;
- Qwen3-VL-8B-Instruct for the optional repository-side image and FlowChart
  recognition utility; and
- Python 3.10 with the CUDA 12.8 PyTorch 2.7.1 package set documented in
  `requirements.txt`.

The paper-facing domain name is FlowChart. The CLI value and local directory
remain `uml` for backward compatibility.

## Required local assets

The source repository does not distribute models, adapters, inputs, or
generated outputs. Place locally obtained assets at the paths defined by
`config/settings.py`:

```text
base_models/qwen3-8B/Qwen/Qwen3-8B/
base_models/qwen3-VL-8B/qwen/Qwen3-VL-8B-Instruct/

checkpoints/lora_moe/text_expert/
checkpoints/lora_moe/image_expert/
checkpoints/lora_moe/uml_expert/
checkpoints/lora_moe/general_expert/
```

Each model directory must contain its configuration, tokenizer or processor
metadata, and model weights. Each expert directory must contain
`adapter_config.json` and an adapter weight file.

## Environment preflight

Activate the intended environment and run:

```bash
python -m pip check
python scripts/diagnostics/check_environment.py --profile inference
```

The inference profile checks Python, declared inference dependencies, CUDA
visibility, the two base-model directories, all four Multi-Expert LoRA
adapters, the main inference entry points, writable local runtime directories,
and free disk space. A successful preflight is necessary but does not prove
that model-backed inference will complete.

## Input layout

```text
inputs/
  text/   *.txt requirement files
  image/  *.jpg, *.jpeg, or *.png images
  uml/    *.jpg, *.jpeg, or *.png FlowChart images
```

Inputs and outputs are local artifacts and are ignored by Git.

## Single-modality recognition

Recognize one image:

```bash
python scripts/inference/recognize_inputs.py \
  --input inputs/image/example.jpg \
  --type image \
  --version qwen3
```

Recognize one FlowChart:

```bash
python scripts/inference/recognize_inputs.py \
  --input inputs/uml/example.jpg \
  --type uml \
  --version qwen3
```

Add `--streaming` only when console streaming is required for FlowChart
recognition. The tracked default is non-streaming.

## End-to-end instruction generation

Run all supported inputs under `inputs/`:

```bash
python scripts/inference/generate_instructions.py --output-format json
```

Use a different input root when required:

```bash
python scripts/inference/generate_instructions.py \
  --input-dir path/to/inputs \
  --output-format json
```

For text or previously recognized JSON only, use `--no-recognition`. Generated
instructions are written under `outputs/generated_instructions/`; recognition
JSON is written under `outputs/recognition_results/`.

## GPU and model lifecycle

The same logical Python package set is intended for the RTX 4060 and RTX 5090,
but Windows and Linux require separate installations of their platform wheels.
Do not copy an installed environment directory between operating systems.

- On an 8 GB RTX 4060, the tracked device policy selects 4-bit vision loading,
  a small generation batch, and sequential execution. Run one model-backed job
  at a time.
- A 20 GB-class or larger GPU is classified as high-end. The vision wrapper
  then uses its FP16 path. The standard language-expert constructor retains its
  existing 4-bit default unless a direct API caller explicitly disables it.

The end-to-end launcher calls the vision recognizer in a subprocess. Image and
FlowChart recognition complete in their own process before the
`InstructionGenerator` loads Qwen3-8B. This process boundary is the preferred
low-memory path because the operating system releases the vision model before
the language model is initialized.

Direct API callers are responsible for lifecycle cleanup. Language experts
provide `unload_model()`. The vision wrapper does not expose a base-model
unload method, so short-lived process isolation is preferred for repeated
mixed-modality use on limited-memory GPUs.

## Failure interpretation

- A failed environment preflight means a required dependency or local asset is
  missing or incompatible.
- A missing LoRA directory may cause an expert to use the base model. Check
  `get_lora_status()` when adapter-backed output is required.
- Recognition functions return structured failure fields for per-input errors;
  the end-to-end launcher records failed items in local output JSON.
- Do not treat an empty generated instruction, a parsed fallback result, or a
  base-model fallback as successful adapter-backed inference.
- Do not change model IDs, prompts, labels, routing, output structure, or
  generation defaults merely to make a smoke run pass.

## Maintainer verification

At repository commit `948cf4823f622e658574c131a9256ad3abf19ff0`, a local
Windows verification used Python 3.10.20, PyTorch 2.7.1+cu128,
Transformers 4.57.0, and an RTX 4060 Laptop GPU with 8 GB VRAM. The inference
diagnostic reported 34 passes, 2 warnings, and 0 failures. One text LoRA
generation, one image recognition, and one FlowChart recognition completed
successfully with local assets. Models were executed sequentially and were
released before the next model family was loaded.

This record demonstrates that the maintained local inference path ran on that
machine and commit. It is not evidence of manuscript experiment reproduction,
dataset redistribution rights, or portability to an unverified environment.
