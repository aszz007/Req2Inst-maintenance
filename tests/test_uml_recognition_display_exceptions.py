"""Regression coverage for FlowChart recognition display exceptions."""

import ast
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECOGNIZER = ROOT / "scripts/preprocessing/raw_to_interim/uml/recognize_uml.py"


def _model(description):
    class VisionModel:
        def __init__(self, version):
            self.version = version

        def get_model_info(self):
            return {"model_name": "fixture-model", "device": "cpu"}

        def recognize_uml(self, _path, streaming=False):
            return {"success": True, "description": description}

    return VisionModel


def _load(vision_model, loads=json.loads):
    tree = ast.parse(RECOGNIZER.read_text(encoding="utf-8"))
    names = {
        "extract_metadata_from_path",
        "recognize_single_uml",
        "batch_recognize_uml",
    }
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "Dict": Dict,
        "List": List,
        "Path": Path,
        "VisionModel": vision_model,
        "datetime": datetime,
        "get_path_config": lambda: pytest.fail("unexpected path lookup"),
        "json": SimpleNamespace(loads=loads, dump=json.dump),
    }
    exec(compile(module, str(RECOGNIZER), "exec"), namespace)
    return namespace["recognize_single_uml"], namespace["batch_recognize_uml"]


def test_parse_failures_preserve_existing_display_fallbacks(tmp_path, capsys):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fixture")
    recognize_single, batch_recognize = _load(_model("not-json"))

    single_result = recognize_single(str(image_path))
    output_path = tmp_path / "results.json"
    batch_results = batch_recognize(str(tmp_path), output_file=str(output_path))

    assert single_result["description"] == "not-json"
    assert len(batch_results) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == batch_results
    output = capsys.readouterr().out
    assert "Description: not-json..." in output
    assert "Incomplete/missing: 1" in output


@pytest.mark.parametrize("target", ["single", "batch-summary", "statistics"])
def test_display_boundaries_do_not_swallow_keyboard_interrupt(tmp_path, target):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fixture")
    calls = 0

    def interrupt(_description):
        nonlocal calls
        calls += 1
        if target == "statistics" and calls == 1:
            return {"actors": [], "use_cases": [], "relationships": []}
        raise KeyboardInterrupt

    recognize_single, batch_recognize = _load(_model("{}"), interrupt)

    with pytest.raises(KeyboardInterrupt):
        if target == "single":
            recognize_single(str(image_path))
        else:
            batch_recognize(
                str(tmp_path),
                output_file=str(tmp_path / "results.json"),
            )

    assert calls == (2 if target == "statistics" else 1)
