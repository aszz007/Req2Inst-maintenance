# Req2Inst documentation

This directory is the documentation entry point for Req2Inst. The manuscript is
under review, so final venue, DOI, and publication-date metadata are not yet
available. The source code is licensed under Apache-2.0.

The maintained repository is intended to replace the paper-facing
[`aszz007/Req2Inst`](https://github.com/aszz007/Req2Inst) snapshot after release
verification.

## Guides

- [Project overview](project-overview.md)
- [Architecture](architecture.md)
- [Inference guide](inference.md)
- [Reproducibility](reproducibility.md)
- [Experiment index](experiments.md)
- [Data and artifact policy](data-and-artifacts.md)
- [Release readiness checklist](public-release-checklist.md)

## Sources of truth

The latest manuscript is the source of truth for paper-facing terminology,
method definitions, dataset scope, experiment settings, and reported claims.
`config/settings.py`, `models/language_model.py`, `models/vision_model.py`, and
the selected entry point are the sources of truth for current executable
behavior.

The manuscript uses BLIP-2 for offline image preprocessing and Qwen3-VL-8B for
offline FlowChart preprocessing. The repository additionally supports local
Qwen3-VL-8B-Instruct recognition for image and FlowChart inputs. Documentation
must keep these paths distinct rather than treating the repository extension as
paper evidence.

The manuscript calls the third requirement domain **FlowChart**. Internal paths,
classes, dataset keys, and CLI values continue to use the legacy `uml` identifier
until a dedicated compatibility migration is designed.

Legacy absolute paths and historical environment labels remain in some research
scripts. They are documented as maintenance debt rather than silently rewritten
during repository cleanup. Runtime-affecting cleanup requires dedicated
regression verification.

## Status labels

- **Current**: represented by central configuration and active entry points.
- **Historical**: retained from earlier experiments but not the current default.
- **Local-only**: required for execution but intentionally excluded from Git.
- **Promotion-blocking**: must be resolved before this repository replaces the
  paper-facing snapshot or before a versioned release is published.

## Code statistics

Use the cloc-based diagnostic when a code-size report is explicitly needed:

```bash
python scripts/diagnostics/code_stats.py
```

The wrapper uses an existing `cloc` executable (including an ignored
`cloc.exe` placed at the project root) and does not implement its own line
counter. It reports the main code roots (`config/`, `models/`, `src/`, and
`scripts/`) separately from an optional local `tests/` tree, then shows their
combined total. A public checkout without local maintenance tests reports a
zero test-code total. It does not count documentation, datasets, model files,
checkpoints, logs, or routine outputs. Pass `--json` for machine-readable output
or `--cloc-command PATH` to select a specific cloc executable.
