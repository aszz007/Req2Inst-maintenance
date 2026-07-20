# Contributing to Req2Inst

Req2Inst is currently a private research repository associated with a paper
under review. Public contributions are not yet accepted, but maintainers should
follow these rules so that the future release has a reviewable history.

## Scope boundaries

Keep the following categories in separate branches or pull requests:

1. Documentation and repository hygiene.
2. Behavior-preserving refactoring.
3. Training, inference, routing, or evaluation behavior changes.
4. Dataset, model-weight, or result releases.

Do not combine a dependency cleanup with an experiment-logic change. Do not
silently update thresholds, prompts, model paths, or cached results in a
documentation-only change.

## Before making changes

- Read `README.md`, `docs/architecture.md`, `docs/reproducibility.md`, and
  `docs/data-and-artifacts.md`.
- Check `git status` and preserve unrelated local modifications.
- Treat `config/settings.py` as the current configuration source of truth.
- Treat internal compatibility identifiers such as `uml` and `lora_moe` as
  stable contracts unless a dedicated migration task explicitly authorizes
  their replacement.
- Never commit base models, checkpoints, private datasets, browser profiles,
  credentials, routine outputs, or inference caches.

## Development setup

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The reconstructed dependency file is not an exact environment lock. Record
Python, CUDA, GPU, PyTorch, Transformers, PEFT, and bitsandbytes versions when a
change depends on runtime behavior.

For repository-only checks that do not load models or require the full runtime
stack, install the lightweight CI dependencies instead:

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q tests/test_repository_quality.py tests/test_manuscript_terminology.py tests/test_model_compatibility.py tests/test_experiment_alignment.py tests/test_console_output.py -k "not config_exposes_only_supported_vision_version"
python -m ruff check --select E9 .
```

The same lightweight checks run automatically for pull requests targeting
`main` and for pushes to `main` or `maintenance/**`.

## Validation expectations

Documentation-only changes should verify links, paths, and Git status. Python
changes should at minimum pass syntax parsing and relevant lightweight tests.
Model or experiment changes require the smallest representative inference or
cached experiment check, plus the full relevant run before release claims are
made.

If a full GPU run is not possible, say exactly what was and was not verified.

## Pull requests

A pull request should include:

- The problem and intended scope.
- Files and runtime contracts affected.
- Validation commands and results.
- Whether model weights, datasets, caches, or output schemas changed.
- Any known limitations or follow-up work.

Generated artifacts should not be committed merely as proof that a command ran.
Attach or summarize them separately unless maintainers explicitly approve a
small curated result for the public release.

## Data and licensing

Do not infer redistribution permission from a paper, dataset download page,
model hub page, or existing local copy. Record the exact source and applicable
license before adding any third-party asset to Git.
