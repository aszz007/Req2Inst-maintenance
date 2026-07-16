# Public release checklist

Complete this checklist before changing the repository from private to public.

## Paper and identity

- [ ] Confirm the final project name and repository URL.
- [ ] Replace provisional `CITATION.cff` authors with the complete author list.
- [ ] Add the final paper title, venue/status, DOI or preprint URL, and preferred
      citation only when disclosure is permitted.
- [ ] Confirm acknowledgements and funding statements.

## License and third-party notices

- [ ] Select an explicit open-source license with all copyright holders.
- [ ] Replace the interim `LICENSE` notice.
- [ ] Inventory code copied or adapted from third parties.
- [ ] Verify model, dataset, image, font, JAR, and other binary licenses.
- [ ] Add required attribution and notice files.

## Data and models

- [ ] Decide which datasets, preparation scripts, and split manifests can be
      released.
- [ ] Decide whether LoRA/checkpoint weights can be released under upstream
      model terms.
- [ ] Publish checksums and upstream revisions for released artifacts.
- [ ] Remove or anonymize personal, confidential, or review-sensitive data.
- [ ] Replace unverified sample assets with authorized examples or download
      instructions.

## Reproducibility

- [ ] Export the final Conda environment and/or fully pinned requirements lock.
- [ ] Record Python, CUDA, driver, GPU, and package versions.
- [ ] Verify model and dataset directory preparation from a clean clone.
- [ ] Add lightweight automated tests for prompts, routing, input parsing, and
      quality validation.
- [ ] Run representative inference and the release experiment suite.
- [ ] Record fresh-run versus cache-derived results.

## Repository hygiene

- [ ] Confirm `git status` is clean and the intended branch is synchronized.
- [ ] Search the full Git history for secrets, credentials, private paths, and
      prohibited artifacts.
- [ ] Review large files and decide whether history rewriting is necessary.
- [ ] Confirm `outputs/`, datasets, model weights, browser profiles, and local
      inputs are not tracked.
- [ ] Review README commands and all relative links from a clean checkout.
- [ ] Add a small CI workflow only after compatible lightweight tests exist.

## Community readiness

- [ ] Confirm CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, SUPPORT, issue templates,
      and pull-request template.
- [ ] Define supported versions and maintainer contact channels.
- [ ] Configure repository topics, description, social preview, and release
      notes.
- [ ] Decide whether public Issues, Discussions, and security advisories should
      be enabled.
