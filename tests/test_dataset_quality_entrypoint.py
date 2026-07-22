"""Regression coverage for the dataset-quality command-line entry point."""

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts"
    / "evaluation"
    / "data_quality"
    / "validate_dataset_quality.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_dataset_quality_entrypoint",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _UnusedReturn:
    """Fail if the entry point starts consuming a currently ignored return value."""

    def _fail(self, *_args, **_kwargs):
        raise AssertionError("The command-line entry point consumed an ignored return value")

    __bool__ = _fail
    __getattr__ = _fail
    __iter__ = _fail
    __len__ = _fail
    __str__ = _fail


def test_main_preserves_validation_and_report_side_effect_calls(
    monkeypatch,
    tmp_path,
):
    module = _load_module()
    calls = []
    loaded_dataset = object()

    class FakeValidator:
        def __init__(
            self,
            dataset_path,
            enable_period_check=False,
            encoding=None,
        ):
            calls.append(("init", dataset_path, enable_period_check, encoding))
            self.error_count = 0
            self.warning_count = 0

        def validate_dataset(self):
            calls.append(("validate_dataset",))
            return _UnusedReturn()

        def load_dataset(self):
            calls.append(("load_dataset",))
            return loaded_dataset

        def generate_report(self, save_path=None):
            calls.append(("generate_report", save_path))
            return _UnusedReturn()

        def save_problematic_instructions(self, output_dir, dataset):
            calls.append(("save_problematic_instructions", output_dir, dataset))

        def get_error_rows(self):
            raise AssertionError("No error rows should be requested when error_count is zero")

    report_path = tmp_path / "validation" / "report.csv"
    problematic_dir = report_path.parent / "problematic_instructions"
    monkeypatch.setattr(module, "UMLDatasetValidator", FakeValidator)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--dataset",
            "fixture.csv",
            "--enable-period-check",
            "--encoding",
            "gb18030",
            "--report-output",
            str(report_path),
        ],
    )

    module.main()

    assert calls == [
        ("init", "fixture.csv", True, "gb18030"),
        ("validate_dataset",),
        ("load_dataset",),
        ("generate_report", str(report_path)),
        (
            "save_problematic_instructions",
            str(problematic_dir),
            loaded_dataset,
        ),
    ]
