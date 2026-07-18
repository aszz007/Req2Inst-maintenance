# Req2Inst

> Multimodal crowdsourcing instruction generation with Qwen3,
> specialized LoRA experts,
> and adaptive expert routing.

[Documentation](docs/README.md) · [Reproducibility](docs/reproducibility.md) ·
[Contribution guide](CONTRIBUTING.md)

## Project status

Req2Inst is a research prototype accompanying a paper that is currently under
review. The repository is private and is **not yet distributed under an
open-source license**. Code, data, model weights, and paper metadata will be
reviewed separately before the public release.

The implementation has completed an end-to-end research run. Repository
cleanup and documentation are ongoing; behavior-preserving maintenance is kept
separate from later code refactoring.

## What Req2Inst does

Req2Inst converts text requirements, image requirements (including UI
screenshots and open-domain images), and flowchart requirements into
concise English instructions for crowdsourcing workers. The expected output is
a three-part instruction:

```text
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...
```

The current pipeline is:

```text
text / image / FlowChart input
        |
        +-- image and flowchart recognition with Qwen3-VL-8B-Instruct
        |
structured text or JSON representation
        |
expert routing (paper: Router MLP; default CLI: type-based)
        |
Qwen3-8B + Text/Image/FlowChart/General LoRA expert
        |
three-part instruction + evaluation artifacts
```

## Current model baseline

- Instruction generation: **Qwen3-8B**.
- Image and flowchart recognition: **Qwen3-VL-8B-Instruct**.
- Expert set: Text, Image, FlowChart, and General.
- Main adaptation method: Multi-Expert LoRA (implemented under the legacy
  lora_moe path name).
- Paper comparison methods: LoRA (Unified), LoRA (Task-Specific), Prompt
  Tuning, and P-Tuning v2.
- Additional repository implementation: full fine-tuning, retained for local
  comparison but not listed as a main method in the manuscript's Table 3.

The paper-facing domain name is **FlowChart**. Existing source paths, class
names,
dataset files, and CLI values retain the legacy `uml` identifier for backward
compatibility. The paper evaluates a Router MLP for single-expert selection and
top-2 output ensembling; the repository also retains the default type-based
router
and several experimental routing implementations.

Some old comments and metadata still mention Qwen-7B. They are historical
artifacts from an earlier project stage and do not describe the current model
configuration. Until the later code-cleanup phase, `config/settings.py` and
the model wrappers are the source of truth.

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
data/                   Local datasets; not tracked
base_models/            Local base model weights; not tracked
checkpoints/            Local training checkpoints; not tracked
outputs/                Generated results and caches; not tracked
inputs/                 Local inference inputs; not tracked by default
docs/                   Architecture, reproducibility, and release notes
```

## Environment

The original runs used a Conda environment named `instruction_generator` and
an NVIDIA GPU. Python 3.10 is recommended for reconstructing the environment.
The dependency list in `requirements.txt` was reconstructed from imports,
configuration comments, and the historical dependency file; the exact
original environment lock was not preserved.

1. Create and activate an isolated environment.

   ```bash
   conda create -n instruction_generator python=3.10 -y
   conda activate instruction_generator
   ```

2. Install a PyTorch 2.7.1 build compatible with the local CUDA driver. The
   plain packages are also listed in `requirements.txt`, but GPU workstations
   may need the platform-specific PyTorch index.

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

## Inference

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
`--expert-variant`, and `--no-recognition`. Generated files are written under
`outputs/generated_instructions/` and are intentionally excluded from Git.

## Training

Train selected experts through the orchestrator:

```bash
python scripts/training/train_all_experts.py --method lora_moe --expert text
```

Training is GPU-intensive and expects local datasets, base models, and enough
storage for checkpoints. Do not start a full run before reviewing
`config/settings.py` and the selected method-specific script.

## Evaluation

Run a quick cached validation when compatible caches are available:

```bash
python scripts/evaluation/experiments/run_all_experiments.py \
  --test-mode --from-cache --skip-failed
```

Run selected experiments with `--experiments 1,2,3`. See
[the experiment index](docs/experiments.md) before launching the complete
suite.

## Data, weights, and generated artifacts

Datasets, base models, checkpoints, browser profiles, caches, and routine
outputs are not part of the source repository. This keeps the code history
small and avoids redistributing third-party material without permission.
Curated release results may be added separately after provenance and licensing
review. See [Data and artifact policy](docs/data-and-artifacts.md).

## Documentation

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Reproducibility guide](docs/reproducibility.md)
- [Experiment index](docs/experiments.md)
- [Data and artifact policy](docs/data-and-artifacts.md)
- [Public release checklist](docs/public-release-checklist.md)
- [Changelog](CHANGELOG.md)

## Contributing and support

The repository is not accepting public contributions while the paper is under
review. Maintainers can still follow [CONTRIBUTING.md](CONTRIBUTING.md) so that
documentation, maintenance, and behavior-changing work remain separated. See
[SUPPORT.md](SUPPORT.md) and [SECURITY.md](SECURITY.md) for reporting guidance.

## Citation

Citation metadata is intentionally provisional during peer review. See
[`CITATION.cff`](CITATION.cff) and the public-release checklist; replace the
placeholder author entry with the final paper metadata before publication.

## License

Copyright is currently reserved by the Req2Inst authors. The current
[`LICENSE`](LICENSE) is an interim notice, not an open-source license. A public
release must replace it with the license explicitly chosen by the authors.
