# Req2Inst documentation

This directory is the maintained documentation entry point for Req2Inst. This
repository is the continuously maintained implementation and will replace the
paper-facing [`aszz007/Req2Inst`](https://github.com/aszz007/Req2Inst) snapshot
after the maintenance and release boundaries are revalidated. The paper remains
under review, so final venue, DOI, and publication-date metadata are not yet
available. The source code is licensed under Apache-2.0.

## Guides

- [Architecture](architecture.md)
- [Reproducibility](reproducibility.md)
- [Experiment index](experiments.md)
- [Data and artifact policy](data-and-artifacts.md)
- [Release readiness checklist](public-release-checklist.md)

## Documentation source of truth

The current execution baseline is Qwen3-8B for instruction generation and
Qwen3-VL-8B-Instruct for image and flowchart recognition. `config/settings.py`,
`models/language_model.py`, and `models/vision_model.py` are the code-level
sources of truth.

The manuscript calls the third requirement domain **FlowChart**. Internal paths,
classes, dataset keys, and CLI values continue to use the legacy `uml` identifier
until a dedicated compatibility migration is designed.

Legacy absolute paths and historical environment labels remain in some research
scripts. They are documented as maintenance debt rather than silently rewritten
during repository cleanup. Runtime-affecting cleanup requires dedicated
regression verification.

## Status labels used in the documentation

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
`scripts/`) separately from `tests/`, then shows their combined total. It does
not count documentation, datasets, model files, checkpoints, logs, or routine
outputs. Pass `--json` for machine-readable output or `--cloc-command PATH` to
select a specific cloc executable.
