# Project overview

## Paper identity

- **Title:** Req2Inst: Toward Task Instruction Generation for Crowdsourcing
  from Multimodal Software Requirements
- **Authors:** Shuai Hong, Yilin He, and Rong Chen
- **Status:** Manuscript under review

Req2Inst studies how heterogeneous software requirements can be transformed
into concise, standardized task instructions for crowdsourcing workers. The
target instruction contains three parts:

```text
Definition: ...
Emphasis & Caution: ...
Things to Avoid: ...
```

Final publication metadata is not yet available. Use `CITATION.cff` for the
current preferred citation and add venue, DOI, and publication details only
after they are confirmed.

## Research workflow

The manuscript describes four sequential stages.

1. **Multimodal data collection.** Text requirements are collected from
   established software-requirement datasets. Image requirements cover UI
   designs and open-domain scenes. FlowChart requirements represent procedural
   and logical task descriptions.
2. **Preprocessing, augmentation, and dataset construction.** Text requirements
   are summarized and structurally analyzed, selected text samples are
   augmented, and visual inputs are converted into textual representations.
   The resulting requirements are paired with structured task instructions.
3. **Model adaptation.** Qwen3-8B is adapted through parameter-efficient
   fine-tuning. The proposed Multi-Expert LoRA method uses separate Text,
   Image, FlowChart, and General adapters on the same backbone.
4. **Evaluation.** The proposed method is compared with retrieval, template,
   zero-shot, prompt-based, and LoRA baselines using automatic metrics,
   routing analysis, efficiency analysis, and human evaluation.

## Manuscript dataset scope

The manuscript reports the following dataset composition. These source assets
are not redistributed by this repository.

| Domain | Manuscript scope | Source description |
| --- | ---: | --- |
| Text | 1,756 original requirements; 2,472 after augmentation | GANNT, WARC, CCHIT, InfusionPump, CM1, and MODIS requirement datasets |
| Image | 1,000 images | 500 Design2Code UI screenshots and 500 MS COCO images |
| FlowChart | 1,500 images | Roboflow flowchart data |
| General | 4,972 mixed-domain samples | Combined Text, Image, and FlowChart data |

The manuscript uses an 80%/10%/10% train, validation, and test split. Any local
dataset prepared for this repository should record its source, license,
checksum, preprocessing steps, and split manifest before results are compared
with the paper.

## Preprocessing boundary

For the manuscript experiments:

- text requirements are processed with TextRank-based sentence extraction,
  dependency parsing, and selected augmentation methods;
- BLIP-2 extracts visual entities and attributes from image inputs;
- Qwen3-VL-8B extracts procedural steps and logical relationships from
  FlowChart inputs.

The visual models are described as offline preprocessing tools. They are not
part of Qwen3-8B fine-tuning or instruction-generation inference in the
manuscript setup.

The current repository also contains Qwen3-VL-8B-Instruct utilities for local
image and FlowChart recognition. That executable convenience path is distinct
from the paper's preprocessing setup and should be reported separately.

## Generation and routing method

All manuscript generation experiments use Qwen3-8B as the language-model
backbone. The evaluated fine-tuning methods include Prompt Tuning, P-Tuning v2,
and LoRA variants. The proposed method trains four independent LoRA adapters:

- Text expert
- Image expert
- FlowChart expert
- General expert trained on mixed-domain data

A Router MLP outputs probabilities over the four experts. The manuscript
evaluates two uses of those probabilities:

- **Learned Router:** select the top-1 expert for each sample.
- **Output Ensemble:** select the top-2 experts and combine their output logits
  using normalized routing probabilities.

The repository's default command-line path currently uses type-based routing.
The learned-router and output-ensemble implementation is retained in the
advanced routing experiment code. Results from later repository-only routing
variants must not be presented as manuscript results.

## Evaluation contract

The manuscript reports BLEU, ROUGE-L, METEOR, BERTScore, and a task-specific
Binary Precision/Recall/F1 evaluation. A generated instruction is successful
only when both conditions hold:

1. it contains Definition, Emphasis & Caution, and Things to Avoid; and
2. ROUGE-L is at least 0.5 and BERTScore F1 is at least 0.85.

The experiments also include expert-routing analysis, efficiency analysis, and
human evaluation of clarity, completeness, and usability. The manuscript
environment uses an NVIDIA RTX 5090 with 32 GB VRAM, BF16 + TF32 for training,
and FP16 for inference.

## Repository scope

This repository provides source code, configuration, entry points, and
documentation. It does not include the third-party datasets, base-model files,
trained adapters, checkpoints, local inputs, caches, or routine outputs needed
for a complete model-backed run.

Paper-facing terminology uses **FlowChart**. Internal paths, class names,
dataset keys, and CLI values retain the legacy `uml` identifier for backward
compatibility. See the [architecture](architecture.md),
[reproducibility guide](reproducibility.md), and
[data and artifact policy](data-and-artifacts.md) before attempting a run.
