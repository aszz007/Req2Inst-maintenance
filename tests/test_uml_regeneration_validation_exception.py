"""Regression coverage for the FlowChart regeneration validation fallback."""

import ast
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
REGENERATOR = (
    ROOT
    / "scripts"
    / "preprocessing"
    / "build_final_dataset"
    / "uml"
    / "regenerate_failed_uml.py"
)


def _load_validator():
    tree = ast.parse(
        REGENERATOR.read_text(encoding="utf-8"),
        filename=str(REGENERATOR),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UMLBatchRepairer"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_new_response"
    )
    isolated_module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "By": SimpleNamespace(CSS_SELECTOR="css selector"),
    }
    exec(compile(isolated_module, str(REGENERATOR), "exec"), namespace)
    return namespace["_validate_new_response"]


VALIDATE = _load_validator()


class _FailingDriver:
    def find_elements(self, *_args, **_kwargs):
        raise RuntimeError("browser fixture failure")


def test_validation_exception_is_accepted_without_changing_console_contract(capsys):
    repairer = SimpleNamespace(
        driver=_FailingDriver(),
        response_count_before_send=0,
    )

    assert VALIDATE(repairer) is True
    assert capsys.readouterr().out == "[Validation exception, accepted]"
