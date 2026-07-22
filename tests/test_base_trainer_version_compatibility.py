'''Focused regression coverage for BaseTrainer version compatibility.'''

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_TRAINER = ROOT / 'src' / 'training' / 'base_trainer.py'


def _load_strategy_selector(version_getter):
    tree = ast.parse(
        BASE_TRAINER.read_text(encoding='utf-8'),
        filename=str(BASE_TRAINER),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == '_should_use_eval_strategy'
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {'_get_transformers_version': version_getter}
    exec(compile(isolated_module, str(BASE_TRAINER), 'exec'), namespace)
    return namespace['_should_use_eval_strategy']


@pytest.mark.parametrize(
    ('version', 'expected'),
    [
        ((4, 45), False),
        ((4, 46), True),
        ((5, 0), True),
    ],
)
def test_eval_strategy_selection_preserves_version_boundary(version, expected):
    selector = _load_strategy_selector(lambda: version)

    assert selector() is expected


def test_eval_strategy_selection_keeps_ordinary_error_fallback():
    def raise_runtime_error():
        raise RuntimeError('version lookup failed')

    selector = _load_strategy_selector(raise_runtime_error)

    assert selector() is False


@pytest.mark.parametrize('control_exception', [KeyboardInterrupt, SystemExit])
def test_eval_strategy_selection_does_not_swallow_control_exceptions(
    control_exception,
):
    def raise_control_exception():
        raise control_exception()

    selector = _load_strategy_selector(raise_control_exception)

    with pytest.raises(control_exception):
        selector()
