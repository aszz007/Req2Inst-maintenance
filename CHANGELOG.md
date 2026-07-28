# Changelog

All notable repository-level changes are documented here. Research experiments
and generated outputs are described in their own result files and are not
individually listed in this changelog.

The project follows the spirit of Keep a Changelog. A formal semantic-versioning
policy will be adopted before the first versioned release.

## [Unreleased]

### Added

- Public-facing English README and documentation.
- Paper-aligned project overview and dataset/method summary.
- Architecture, reproducibility, experiment, artifact, and release guides.
- Contribution, conduct, support, security, citation, and Apache-2.0 license
  files.
- Reconstructed runtime and development requirements.
- Inference runbook covering environment checks, local assets, GPU profiles,
  model lifecycle, commands, and a model-backed maintainer verification.
- GitHub issue and pull-request templates.

### Changed

- Generated `outputs/` content and local input assets are treated as local
  artifacts rather than source-controlled files.
- Current Qwen3-8B/Qwen3-VL-8B model policy is documented without changing
  historical code metadata.
- The public maintenance repository is identified as the continuously
  maintained implementation that will replace the paper-facing snapshot after
  stabilization and release-boundary review.
- Paper experiment settings are distinguished from repository-only inference
  and routing extensions.
- Source comments, docstrings, and 1,492 safe diagnostic messages are
  standardized in English without changing prompts, result labels, generated
  outputs, log levels, or control flow.
- Direct console-only progress and status text is translated to English in
  safe non-browser paths; prompts, CLI help, result labels, and output schemas
  remain unchanged.
- Validation errors and advanced-routing diagnostic explanations are
  standardized in English without changing validation criteria, thresholds,
  or recommendation outcomes.
- Legacy generated artifacts and superseded dependency notes are removed from
  the current source snapshot while remaining recoverable from Git history.
- The repository-hygiene and stability-first code-maintenance phase is closed;
  subsequent work is scoped to inference readiness, concrete defects, or
  separately approved research behavior.
- Required expert LoRA adapters now fail closed when their path is missing,
  invalid, or cannot be loaded.

### Removed

- The obsolete `run_all_training.sh` workstation launcher.
- 236 separator-only log calls that added console noise without diagnostic
  content.
- Unused Markdown summary-report generators from Experiments 8-11.
- Redundant phrase-specific Chinese output-cleanup patterns already covered by
  the generic Han-character boundary.
- Silent fallback from a failed expert LoRA adapter to the Qwen3 base model.

## [0.1.0] - Research prototype

- Completed the original end-to-end research implementation.
- Added multimodal preprocessing, four expert types, routing strategies,
  training variants, evaluation metrics, and experiments 1-11.
