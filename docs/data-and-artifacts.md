# Data and artifact policy

## Purpose

Req2Inst uses large models, third-party datasets, local input images, generated
predictions, and experiment caches. Keeping these distinct from source code is
necessary for repository size, reproducibility, confidentiality, and licensing.

## Source-controlled material

The repository should normally track:

- Python source and launch scripts.
- Prompt templates and central configuration.
- Documentation and small metadata schemas.
- Small, curated result summaries that have been explicitly approved for a
  release and contain no private or non-redistributable material.

## Local-only material

The following paths are ignored and must remain local unless a dedicated
release decision says otherwise:

- `base_models/`
- `checkpoints/`
- `lora_weights/`
- `data/`
- `logs/`
- `outputs/`
- `inputs/` assets
- browser profiles and automated-browser state
- model binaries such as `.safetensors`, `.pt`, `.pth`, and `.ckpt`

`inputs/README.md` and `outputs/README.md` are tracked only to document the
expected folder contracts.

## Output policy

Routine training curves, metric dumps, recognition results, validation lists,
prediction caches, and test outputs are reproducible working artifacts. They
must not accumulate in Git history.

For a public release, a curated result should instead include:

- The commit and command that generated it.
- Model/checkpoint and dataset identifiers.
- Metric configuration and cache/fresh-run status.
- A license and confidentiality review.
- Only the minimum files needed to support the documented claim.

Historical root-level PDFs, `evaluation_report.json`, and validation dumps are
local research snapshots rather than canonical paper results. They are removed
from the current tracked snapshot but remain recoverable from Git history and
the private pre-cleanup backup repository.

## Third-party assets

Possession, download access, a paper citation, or a public webpage does not by
itself establish redistribution permission. Before tracking an external asset,
record:

- Original source and version.
- Copyright holder or dataset/model owner.
- Exact license or written permission.
- Required attribution and redistribution conditions.
- Whether derived outputs may be redistributed.

If these facts are unclear, provide a download/preparation script or reference
instructions instead of copying the asset into the repository.

## Removing historical tracked artifacts

Repository cleanup should remove generated files from the Git index while
leaving the user's local files intact. History rewriting is a separate,
potentially disruptive release decision and must not be performed implicitly.
