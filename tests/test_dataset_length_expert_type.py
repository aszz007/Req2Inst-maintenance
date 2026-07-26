"""Regression coverage for dataset-length expert-type dispatch."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_text_type_selects_low_requirements_and_instruction_columns(monkeypatch):
    captured_requirements = []

    def build_prompt(requirement):
        captured_requirements.append(requirement)
        return "text:"

    monkeypatch.setattr(
        DATASET_LENGTHS.TextInstructionTemplate,
        "build_prompt",
        staticmethod(build_prompt),
    )
    dataframe = DATASET_LENGTHS.pd.DataFrame(
        {"low_requirements": ["requirement"], "instruction": ["annotate"]}
    )

    lengths = DATASET_LENGTHS.calculate_lengths_from_df(
        dataframe,
        _Tokenizer(),
        "text",
    )

    assert captured_requirements == ["requirement"]
    assert lengths == [len("text:annotate")]


def test_main_preserves_dataset_order_and_console_output(
    monkeypatch,
    tmp_path,
    capsys,
):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    text_csv = text_dir / "sample.csv"
    image_csv = tmp_path / "image.csv"
    uml_csv = tmp_path / "uml.csv"
    for path in (text_csv, image_csv, uml_csv):
        path.write_text("fixture", encoding="utf-8")

    tokenizer = object()
    calls = []
    monkeypatch.setattr(
        DATASET_LENGTHS,
        "path_cfg",
        SimpleNamespace(
            TEXT_DATASET_DIR=text_dir,
            IMAGE_DATASET_CSV=image_csv,
            UML_DATASET_CSV=uml_csv,
        ),
    )
    monkeypatch.setattr(DATASET_LENGTHS, "get_tokenizer", lambda: tokenizer)

    def read_csv(path):
        path = Path(path)
        calls.append(("read", path.name))
        return f"frame:{path.name}", "fixture-encoding"

    def calculate(frame, actual_tokenizer, expert_type, source_name=""):
        assert actual_tokenizer is tokenizer
        calls.append(("calculate", frame, expert_type, source_name))
        return {"text": [10], "image": [20], "uml": [30]}[expert_type]

    def record_stats(name, lengths):
        calls.append(("stats", name, tuple(lengths)))

    monkeypatch.setattr(DATASET_LENGTHS, "read_csv_safely", read_csv)
    monkeypatch.setattr(DATASET_LENGTHS, "calculate_lengths_from_df", calculate)
    monkeypatch.setattr(DATASET_LENGTHS, "print_stats", record_stats)

    DATASET_LENGTHS.main()

    assert calls == [
        ("read", "sample.csv"),
        ("calculate", "frame:sample.csv", "text", "sample.csv"),
        ("stats", "Text Expert (all files combined)", (10,)),
        ("read", "image.csv"),
        ("calculate", "frame:image.csv", "image", ""),
        ("stats", "Image Expert", (20,)),
        ("read", "uml.csv"),
        ("calculate", "frame:uml.csv", "uml", ""),
        ("stats", "FlowChart Expert", (30,)),
    ]
    assert capsys.readouterr().out == (
        "\nComputing dataset lengths for all experts (including content previews)...\n\n"
        "[Text] Found 1 files\n"
        "[Text] Read sample.csv successfully | encoding: fixture-encoding\n"
        "[Image] Read successfully | encoding: fixture-encoding\n"
        "[FlowChart] Read successfully | encoding: fixture-encoding\n"
    )


def test_main_preserves_missing_dataset_messages(monkeypatch, tmp_path, capsys):
    text_dir = tmp_path / "missing-text"
    image_csv = tmp_path / "missing-image.csv"
    uml_csv = tmp_path / "missing-uml.csv"
    monkeypatch.setattr(
        DATASET_LENGTHS,
        "path_cfg",
        SimpleNamespace(
            TEXT_DATASET_DIR=text_dir,
            IMAGE_DATASET_CSV=image_csv,
            UML_DATASET_CSV=uml_csv,
        ),
    )
    monkeypatch.setattr(DATASET_LENGTHS, "get_tokenizer", object)

    def unexpected(*_args, **_kwargs):
        pytest.fail("Missing dataset paths must not be read or processed")

    monkeypatch.setattr(DATASET_LENGTHS, "read_csv_safely", unexpected)
    monkeypatch.setattr(DATASET_LENGTHS, "calculate_lengths_from_df", unexpected)
    monkeypatch.setattr(DATASET_LENGTHS, "print_stats", unexpected)

    DATASET_LENGTHS.main()

    assert capsys.readouterr().out == (
        "\nComputing dataset lengths for all experts (including content previews)...\n\n"
        f"[Text] Directory not found: {text_dir}\n"
        f"[Image] File not found: {image_csv}\n"
        f"[FlowChart] File not found: {uml_csv}\n"
    )


def test_main_short_circuits_when_tokenizer_loading_fails(monkeypatch, capsys):
    monkeypatch.setattr(DATASET_LENGTHS, "get_tokenizer", lambda: None)

    DATASET_LENGTHS.main()

    assert capsys.readouterr().out == ""


def test_main_preserves_unreadable_dataset_behavior(monkeypatch, tmp_path, capsys):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    text_csv = text_dir / "sample.csv"
    image_csv = tmp_path / "image.csv"
    uml_csv = tmp_path / "uml.csv"
    for path in (text_csv, image_csv, uml_csv):
        path.write_text("fixture", encoding="utf-8")

    reads = []
    stats = []
    monkeypatch.setattr(
        DATASET_LENGTHS,
        "path_cfg",
        SimpleNamespace(
            TEXT_DATASET_DIR=text_dir,
            IMAGE_DATASET_CSV=image_csv,
            UML_DATASET_CSV=uml_csv,
        ),
    )
    monkeypatch.setattr(DATASET_LENGTHS, "get_tokenizer", object)

    def unreadable(path):
        reads.append(Path(path).name)
        return None, None

    def unexpected(*_args, **_kwargs):
        pytest.fail("Unreadable datasets must not be tokenized")

    monkeypatch.setattr(DATASET_LENGTHS, "read_csv_safely", unreadable)
    monkeypatch.setattr(DATASET_LENGTHS, "calculate_lengths_from_df", unexpected)
    monkeypatch.setattr(
        DATASET_LENGTHS,
        "print_stats",
        lambda name, lengths: stats.append((name, tuple(lengths))),
    )

    DATASET_LENGTHS.main()

    assert reads == ["sample.csv", "image.csv", "uml.csv"]
    assert stats == [("Text Expert (all files combined)", ())]
    assert capsys.readouterr().out == (
        "\nComputing dataset lengths for all experts (including content previews)...\n\n"
        "[Text] Found 1 files\n"
        "[Text] Error: could not read sample.csv; all encoding attempts failed\n"
    )


def test_main_preserves_text_exception_boundary_and_continues(
    monkeypatch,
    tmp_path,
    capsys,
):
    text_dir = tmp_path / "text"
    text_dir.mkdir()
    text_csv = text_dir / "sample.csv"
    image_csv = tmp_path / "image.csv"
    uml_csv = tmp_path / "uml.csv"
    for path in (text_csv, image_csv, uml_csv):
        path.write_text("fixture", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        DATASET_LENGTHS,
        "path_cfg",
        SimpleNamespace(
            TEXT_DATASET_DIR=text_dir,
            IMAGE_DATASET_CSV=image_csv,
            UML_DATASET_CSV=uml_csv,
        ),
    )
    monkeypatch.setattr(DATASET_LENGTHS, "get_tokenizer", object)

    def read_csv(path):
        name = Path(path).name
        calls.append(("read", name))
        if name == "sample.csv":
            raise RuntimeError("fixture failure")
        return f"frame:{name}", "fixture-encoding"

    def calculate(frame, _tokenizer, expert_type, source_name=""):
        calls.append(("calculate", frame, expert_type, source_name))
        return {"image": [20], "uml": [30]}[expert_type]

    def record_stats(name, lengths):
        calls.append(("stats", name, tuple(lengths)))

    monkeypatch.setattr(DATASET_LENGTHS, "read_csv_safely", read_csv)
    monkeypatch.setattr(DATASET_LENGTHS, "calculate_lengths_from_df", calculate)
    monkeypatch.setattr(DATASET_LENGTHS, "print_stats", record_stats)

    DATASET_LENGTHS.main()

    assert calls == [
        ("read", "sample.csv"),
        ("stats", "Text Expert (all files combined)", ()),
        ("read", "image.csv"),
        ("calculate", "frame:image.csv", "image", ""),
        ("stats", "Image Expert", (20,)),
        ("read", "uml.csv"),
        ("calculate", "frame:uml.csv", "uml", ""),
        ("stats", "FlowChart Expert", (30,)),
    ]
    assert capsys.readouterr().out == (
        "\nComputing dataset lengths for all experts (including content previews)...\n\n"
        "[Text] Found 1 files\n"
        "[Text] Unexpected error while processing sample.csv: fixture failure\n"
        "[Image] Read successfully | encoding: fixture-encoding\n"
        "[FlowChart] Read successfully | encoding: fixture-encoding\n"
    )
