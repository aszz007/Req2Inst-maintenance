# Reproducibility guide

## Reproducibility status

The project completed an end-to-end research run, but the exact original Conda
environment export was not retained. `requirements.txt` is therefore a
reconstructed compatibility specification. A future public release should add
an environment lock produced on the final verification machine.

## Recommended host

- Python 3.10 in a Conda environment named `instruction_generator`.
- NVIDIA GPU and a CUDA-compatible PyTorch 2.7.1 build.
- Sufficient local storage for two 8B base models, multiple adapters,
  checkpoints, datasets, and caches.
- 24 GB-class VRAM for the configurations designed around RTX 4090; lower-memory
  devices may require explicit configuration changes and revalidation.

CPU-only execution is technically detectable but is not a practical baseline
for full training or multimodal inference.

## Install

```bash
conda create -n instruction_generator python=3.10 -y
conda activate instruction_generator
python -m pip install -r requirements.txt
python -m pip install -e .
```

For GPU workstations, install the matching PyTorch/CUDA wheel before the full
requirements file if the default package index is unsuitable.

## Read-only environment preflight

Run the unified diagnostic before starting a model-backed workflow:

```bash
python scripts/diagnostics/check_environment.py
```

The default `all` profile checks the current Python environment, declared
runtime dependencies, CUDA/GPU visibility, base-model directories, datasets,
method checkpoints, key entry points, writable runtime targets, and available
disk space. Narrow checks and machine-readable output are also available:

```bash
python scripts/diagnostics/check_environment.py --profile inference
python scripts/diagnostics/check_environment.py --profile training
python scripts/diagnostics/check_environment.py --profile evaluation --json
```

Exit status 0 means that no required static preflight check failed; exit status
1 lists missing or incompatible requirements. The diagnostic is deliberately
read-only and does not download packages, create directories, load model
weights, or prove that a full training or inference run succeeds.

## Local model layout

The central configuration expects:

```text
base_models/
  qwen3-8B/Qwen/Qwen3-8B/
  qwen3-VL-8B/qwen/Qwen3-VL-8B-Instruct/
```

Model weights are not included in the repository. Record upstream model IDs,
revisions, licenses, checksums, and download dates before the public release.

## Local dataset layout

```text
data/dataset/
  text/
    CCHIT_dataset.csv
    CM1_dataset.csv
    GANNT_dataset.csv
    InfusionPump_dataset.csv
    Modis_dataset.csv
    WARC_dataset.csv
  image/image_dataset.csv
  uml/uml_dataset.csv
  general/
```

Expected logical fields are:

- Text: a low-level requirement field and an instruction field.
- Image: a description field and an instruction field.
- FlowChart (legacy uml dataset key): a description field and an instruction
  field.
- General: dynamically combines the three domains rather than requiring a
  dedicated source CSV.

The loaders tolerate several column-name variants; inspect
`src/training/data_loader.py` before converting a new dataset.

## Checkpoint layout

```text
checkpoints/
  lora_moe/{text,image,uml,general}_expert/
  lora_single/unified_expert/
  p_tuning/{text,image,uml,general}_expert/
  prompt_tuning/{text,image,uml,general}_expert/
  full_finetuning/{text,image,uml,general}_expert/
```

The repository does not ship checkpoints. Do not substitute a checkpoint from
another experiment without recording the change.

## Basic commands

Recognize a single image or flowchart input:

```bash
python scripts/inference/recognize_inputs.py \
  --input path/to/file.png --type image --version qwen3
```

Run end-to-end inference:

```bash
python scripts/inference/generate_instructions.py --output-format json
```

Train selected components:

```bash
python scripts/training/train_all_experts.py --method lora_moe --expert text
```

Train the manuscript-aligned LoRA (Unified) comparison model. By default, all
three input domains use the unified General prompt template:

```bash
python scripts/training/lora_single/train_unified_expert.py
```

The earlier domain-template variant remains available as an explicit
reproducibility option:

```bash
python scripts/training/lora_single/train_unified_expert.py \
  --use_domain_templates
```

Run selected or cached experiments:

```bash
python scripts/evaluation/experiments/run_all_experiments.py \
  --experiments 1,2,3 --test-mode --skip-failed

python scripts/evaluation/experiments/run_all_experiments.py \
  --from-cache --test-mode --skip-failed
```

## Reproducibility controls

- Dataset splitting defaults to seed 42.
- Training and generation still depend on CUDA kernels, package versions,
  sampling parameters, and hardware.
- Inference defaults are centralized in `InferenceConfig`.
- Cached results are valid only for the exact method, checkpoint, dataset split,
  prompt, and metric configuration that produced them.
- BERTScore and `evaluate` metrics may download additional model or metric
  resources on first use.

## What to record for a release run

- Git commit SHA and whether the worktree was clean.
- Operating system, Python, CUDA, GPU, and driver versions.
- Exact `pip freeze` or Conda export.
- Base-model IDs/revisions and file checksums.
- Dataset versions, licenses, checksums, and split seeds.
- Checkpoint hashes and training commands.
- Inference and metric commands.
- Whether results were generated from models or loaded from cache.

Do not call a run reproduced if any of these boundaries materially changed.
