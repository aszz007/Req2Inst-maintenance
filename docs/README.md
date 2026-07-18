# Req2Inst documentation

This directory is the maintained documentation entry point for Req2Inst. The
paper is under review, so citation, license, dataset, and release metadata are
provisional.

## Guides

- [Architecture](architecture.md)
- [Reproducibility](reproducibility.md)
- [Experiment index](experiments.md)
- [Data and artifact policy](data-and-artifacts.md)
- [Public release checklist](public-release-checklist.md)

## Documentation source of truth

The current execution baseline is Qwen3-8B for instruction generation and
Qwen3-VL-8B-Instruct for image and flowchart recognition. `config/settings.py`,
`models/language_model.py`, and `models/vision_model.py` are the code-level
sources of truth.

The manuscript calls the third requirement domain **FlowChart**. Internal paths,
classes, dataset keys, and CLI values continue to use the legacy `uml` identifier
until a dedicated compatibility migration is designed.

Legacy Qwen-7B names, old absolute paths, and old environment labels remain in
some files. They are documented as maintenance debt rather than silently
rewritten during repository cleanup. Runtime-affecting cleanup will be handled
later with regression verification.

## Status labels used in the documentation

- **Current**: represented by central configuration and active entry points.
- **Historical**: retained from earlier experiments but not the current default.
- **Local-only**: required for execution but intentionally excluded from Git.
- **Release-blocking**: must be resolved before making the repository public.
