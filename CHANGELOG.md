# Changelog

All notable repository-level changes are documented here. Research experiments
and generated outputs are described in their own result files and are not
individually listed in this changelog.

The project follows the spirit of Keep a Changelog. A formal semantic-versioning
policy will be adopted before the first public release.

## [Unreleased]

### Added

- Public-facing English README and documentation.
- Architecture, reproducibility, experiment, artifact, and release guides.
- Contribution, conduct, support, security, citation, and interim license files.
- Reconstructed runtime and development requirements.
- GitHub issue and pull-request templates.

### Changed

- Generated `outputs/` content and local input assets are treated as local
  artifacts rather than source-controlled files.
- Current Qwen3-8B/Qwen3-VL-8B model policy is documented without changing
  historical code metadata.
- Source comments, docstrings, and 1,492 safe diagnostic messages are
  standardized in English without changing prompts, result labels, generated
  outputs, log levels, or control flow.
- Direct console-only progress and status text is translated to English in
  safe non-browser paths; prompts, CLI help, validation fields, and result
  schemas remain unchanged.
- Legacy generated artifacts and superseded dependency notes are removed from
  the current source snapshot while remaining recoverable from Git history.

### Removed

- The obsolete `run_all_training.sh` workstation launcher.
- 236 separator-only log calls that added console noise without diagnostic
  content.

## [0.1.0] - Research prototype

- Completed the original end-to-end research implementation.
- Added multimodal preprocessing, four expert types, routing strategies,
  training variants, evaluation metrics, and experiments 1-11.
