"""Regression coverage for image and FlowChart sample display output."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CASES = (
    pytest.param(
        ROOT
        / "scripts"
        / "preprocessing"
        / "dataset_sampling"
        / "sample_image_dataset.py",
        (
            "Note: Image Expert input is a JSON text description, not an image",
        ),
        id="image",
    ),
    pytest.param(
        ROOT
        / "scripts"
        / "preprocessing"
        / "dataset_sampling"
        / "sample_uml_dataset.py",
        (
            "Note: FlowChart Expert input is a JSON text description, not a "
            "FlowChart diagram",
            "Note: Both FlowChart Expert and General Expert use this dataset",
        ),
        id="flowchart",
    ),
)


def _load_module(path: Path):
    module_name = f"sample_display_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _expected_output(notes, body):
    return "\n".join(
        [
            "=" * 70,
            "Random sample (1 rows)",
            *notes,
            "=" * 70,
            "",
            "[Sample 1]",
            "-" * 50,
            *body,
            "-" * 50,
            "",
        ]
    )


@pytest.mark.parametrize(("path", "notes"), SCRIPT_CASES)
def test_rich_sample_display_contract(path, notes, capsys):
    module = _load_module(path)
    samples = pd.DataFrame(
        [
            {
                "Description": '{"Description": "Primary description"}',
                "Low_Requirements": "Low requirement",
                "Instruction": "Three-part instruction",
                "High_Requirements": "Display-only high",
                "Extra": "extra value",
                "Ignored": float("nan"),
            }
        ]
    )

    module.display_samples(samples)

    assert capsys.readouterr().out == _expected_output(
        notes,
        [
            "[Description (training input, extracted from JSON)]",
            "Primary description",
            "",
            "[Low_Requirements]",
            "Low requirement",
            "",
            "[Instruction]",
            "Three-part instruction",
            "",
            "[High_Requirements (display only; not used for training)]",
            "Display-only high",
            "",
            "[Extra]: extra value",
        ],
    )


@pytest.mark.parametrize(("path", "notes"), SCRIPT_CASES)
def test_low_requirement_fallback_and_empty_field_filtering(path, notes, capsys):
    module = _load_module(path)
    samples = pd.DataFrame(
        [
            {
                "Low_Requirements": "Only low requirement",
                "Instruction": float("nan"),
                "High_Requirements": float("nan"),
                "Extra": "",
            }
        ]
    )

    module.display_samples(samples)

    assert capsys.readouterr().out == _expected_output(
        notes,
        [
            "[Low_Requirements]",
            "Only low requirement",
        ],
    )
