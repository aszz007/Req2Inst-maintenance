"""Regression coverage for dataset-length expert-type dispatch."""

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "utils" / "calculate_dataset_lengths.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "req2inst_dataset_length_utility",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DATASET_LENGTHS = _load_module()


class _Tokenizer:
    def __call__(self, text):
        return {"input_ids": list(range(len(text)))}


@pytest.mark.parametrize(
    ("expert_type", "template_name", "prefix"),
    [
        pytest.param("image", "ImageInstructionTemplate", "image:", id="image"),
        pytest.param("uml", "UMLInstructionTemplate", "uml:", id="uml"),
    ],
)
def test_image_domain_types_select_description_and_instruction_columns(
    monkeypatch,
    expert_type,
    template_name,
    prefix,
):
    template = getattr(DATASET_LENGTHS, template_name)
    captured_descriptions = []

    def build_prompt(description):
        captured_descriptions.append(description)
        return prefix

    monkeypatch.setattr(
        template,
        "build_prompt",
        staticmethod(build_prompt),
    )
    dataframe = DATASET_LENGTHS.pd.DataFrame(
        {"description": ["diagram"], "instruction": ["annotate"]}
    )

    lengths = DATASET_LENGTHS.calculate_lengths_from_df(
        dataframe,
        _Tokenizer(),
        expert_type,
    )

    assert captured_descriptions == ["diagram"]
    assert lengths == [len(f"{prefix}annotate")]
