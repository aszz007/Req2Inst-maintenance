# Local generated outputs

This directory is created and populated by training, inference, recognition,
validation, plotting, and experiment scripts. Its generated contents are
intentionally excluded from Git.

Typical subdirectories include:

```text
generated_instructions/
recognition_results/
evaluations/
inference_cache/
training_curves/
validation/
reports/
```

Do not commit routine outputs or caches. A curated release result must follow
`docs/data-and-artifacts.md` and include provenance, configuration, and license
review.
