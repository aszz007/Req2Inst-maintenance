# Req2Inst

[![Repository quality](https://github.com/aszz007/Req2Inst-maintenance/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/aszz007/Req2Inst-maintenance/actions/workflows/quality.yml)

> Multimodal crowdsourcing instruction generation with Qwen3,
> specialized LoRA experts,
> and adaptive expert routing.

[Project overview](docs/project-overview.md) | [Documentation](docs/README.md) |
[Inference guide](docs/inference.md) | [Reproducibility](docs/reproducibility.md) |
[Contribution guide](CONTRIBUTING.md)

## Project status

This repository contains the continuously maintained implementation of
Req2Inst. The manuscript currently cites the public snapshot at
[`aszz007/Req2Inst`](https://github.com/aszz007/Req2Inst); this repository is
intended to replace that snapshot after release verification.

The accompanying manuscript, *Req2Inst: Toward Task Instruction Generation for
Crowdsourcing from Multimodal Software Requirements*, is under review. Its
authors are Shuai Hong, Yilin He, and Rong Chen. Final venue, DOI, volume, issue,
and publication-date metadata are therefore not yet available.

The source code is distributed under the Apache License 2.0. Datasets, model
weights, checkpoints, local inputs, and generated artifacts are not included
and remain subject to their original licenses and terms.

## Research overview

Req2Inst converts text requirements, image requirements (including UI
screenshots and open-domain images), and flowchart requirements into
concise English instructions for crowdsourcing workers. The expected output is
a three-part instruction:

```text
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...
```

The manuscript describes a four-stage research workflow:

```text
multimodal data collection
        -> preprocessing, augmentation, and dataset construction
        -> Qwen3-8B parameter-efficient fine-tuning
        -> comparative and human evaluation
```

For the manuscript experiments, text requirements are processed directly;
BLIP-2 extracts visual entities from image inputs; and Qwen3-VL-8B extracts
procedural and logical structure from FlowChart inputs. These visual models are
used for offline preprocessing rather than as part of language-model training
or instruction-generation inference.

The proposed generation method uses four independent LoRA adapters on a shared
Qwen3-8B backbone: Text, Image, FlowChart, and General. A Router MLP supports
top-1 expert selection and top-2 output-space logit fusion.

See the [project overview](docs/project-overview.md) for the paper-aligned
dataset, method, and evaluation summary.

## Manuscript and repository boundary

- Instruction generation: **Qwen3-8B**.
- Manuscript image preprocessing: **BLIP-2**.
- Manuscript FlowChart preprocessing: **Qwen3-VL-8B**.
- Expert set: Text, Image, FlowChart, and General.
- Proposed adaptation method: Multi-Expert LoRA.
- Paper comparison methods: LoRA (Unified), LoRA (Task-Specific), Prompt
  Tuning, and P-Tuning v2.
- Additional repository implementation: full fine-tuning, retained for local
  comparison but not listed as a main method in the manuscript's Table 3.

The paper-facing domain name is **FlowChart**. Existing source paths, class
names, dataset files, and CLI values retain the legacy `uml` identifier for
backward compatibility. The paper evaluates a Router MLP for single-expert
selection and top-2 output ensembling. The default repository CLI retains
type-based routing, and the repository also contains experimental routing
implementations.

The repository additionally provides Qwen3-VL-8B-Instruct recognition utilities
for local image and FlowChart inference. This executable convenience path is
separate from the manuscript's offline visual-preprocessing setup and should not
be reported as a reproduction of that setup without matching data, models,
checkpoints, and configuration.

For paper-facing claims, the manuscript is the source of truth. For executable
behavior, `config/settings.py`, the model wrappers, and the selected entry point
are the source of truth.

## Repository layout

```text
config/                 Central paths and experiment configuration
models/                 Language/vision wrappers and prompt templates
src/                    Core preprocessing, experts, routing, training, metrics
scripts/
  preprocessing/        Dataset construction and recognition utilities
  training/             Training entry points for each method and expert
  inference/            End-to-end recognition and generation entry points
  evaluation/           Metrics and experiments 1-11
  diagnostics/          Read-only environment readiness checks
data/                   Local datasets; not tracked
base_models/            Local base model weights; not tracked
checkpoints/            Local training checkpoints; not tracked
outputs/                Generated results and caches; not tracked
inputs/                 Local inference inputs; not tracked by default
docs/                   Architecture, reproducibility, and release notes
```

## Environment

The manuscript experiments used an NVIDIA RTX 5090 with 32 GB VRAM, BF16 + TF32
for training, and FP16 for inference. Python 3.10 is recommended for rebuilding
the current repository environment. The original Conda lock was not preserved;
`requirements.txt` is a reconstructed compatibility specification.

1. Create and activate an isolated environment.

   ```bash
   conda create --override-channels -c conda-forge -n instruction_generator python=3.10 pip -y
   conda activate instruction_generator
   ```

2. Install the CUDA 12.8 PyTorch build before the remaining dependencies.

   ```bash
   python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
   ```

   This package set targets both the local RTX 4060 and the RTX 5090 server.
   Windows and Linux download different platform wheels from the same index;
   do not copy an installed environment directory between operating systems.

3. Install the project dependencies and package.

   ```bash
   python -m pip install -r requirements.txt
   python -m pip install -e .
   ```

4. Place the model weights at the paths defined in `config/settings.py`:

   ```text
   base_models/qwen3-8B/Qwen/Qwen3-8B/
   base_models/qwen3-VL-8B/qwen/Qwen3-VL-8B-Instruct/
   ```

5. Place datasets and expert checkpoints under `data/` and `checkpoints/` as
   described in [the reproducibility guide](docs/reproducibility.md).

6. Run the read-only environment preflight.

   ```bash
   python scripts/diagnostics/check_environment.py
   ```

   The command exits with status 1 when required dependencies or local assets
   are missing. It does not create directories, download packages, load model
   weights, or start a training/inference run.

## Inference

See the [inference guide](docs/inference.md) for the environment preflight,
required local assets, single-modality commands, GPU guidance, and model
lifecycle boundary.

Create local input subdirectories as needed:

```text
inputs/text/   .txt requirement files
inputs/image/  .jpg/.jpeg/.png UI screenshots or open-domain images
inputs/uml/    .jpg/.jpeg/.png flowchart images (legacy internal directory name)
```

Run the end-to-end generator:

```bash
python scripts/inference/generate_instructions.py --output-format json
```

Useful options include `--input-dir`, `--vision-version qwen3`,
`--expert-variant`, `--no-recognition`, and `--streaming`. The streaming flag
shows FlowChart recognition as it is generated. Without the flag, the vision
model uses `DeviceConfig.enable_streaming`, whose tracked default is `False`.
Generated files are written under `outputs/generated_instructions/` and are
intentionally excluded from Git.

This command exercises the repository's current local inference path. It is not
by itself a reproduction of the manuscript experiment pipeline.

## Training

Train an individual expert through its method-specific entry point:

```bash
python scripts/training/lora_moe/train_text_expert.py
```

The other methods and domains have corresponding entry points under
`scripts/training/`.

Training is GPU-intensive and expects local datasets, base models, and enough
storage for checkpoints. Do not start a full run before reviewing
`config/settings.py` and the selected method-specific script.

## Evaluation

Run an individual experiment directly. For example, Experiment 1 supports a
small cached validation when compatible caches are available:

```bash
python scripts/evaluation/experiments/exp1_baseline_comparison.py \
  --test-mode --from-cache
```

Each experiment has its own command-line options. See
[the experiment index](docs/experiments.md) before running an experiment.

## Data, weights, and generated artifacts

Datasets, base models, checkpoints, browser profiles, caches, and routine
outputs are not part of the source repository. This keeps the code history
small and avoids redistributing third-party material without permission.
Curated release results may be added separately after provenance and licensing
review. See [Data and artifact policy](docs/data-and-artifacts.md).

## Documentation

- [Documentation index](docs/README.md)
- [Project overview](docs/project-overview.md)
- [Architecture](docs/architecture.md)
- [Inference guide](docs/inference.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Experiment index](docs/experiments.md)
- [Data and artifact policy](docs/data-and-artifacts.md)
- [Release readiness checklist](docs/public-release-checklist.md)
- [Changelog](CHANGELOG.md)

## Contributing and support

The repository is public, but external contributions are not yet actively
solicited while the paper is under review and the maintained implementation is
being stabilized. Follow [CONTRIBUTING.md](CONTRIBUTING.md) so that
documentation, maintenance, and behavior-changing work remain separated. See
[SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md) for reporting guidance.

## Citation

See [`CITATION.cff`](CITATION.cff) for the current authors and preferred paper
citation. DOI, venue, volume, issue, and publication-date fields remain omitted
until final publication metadata is available.

## License

Req2Inst source code is distributed under the
[Apache License 2.0](LICENSE). Third-party dependencies, models, datasets,
checkpoints, inputs, and generated artifacts remain subject to their own
licenses and terms.
