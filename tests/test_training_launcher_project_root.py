'''Regression coverage for cross-platform training launcher project roots.'''

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_LAUNCHERS = [
    Path('scripts/training/full_finetuning/train_general_expert.py'),
    Path('scripts/training/full_finetuning/train_image_expert.py'),
    Path('scripts/training/full_finetuning/train_text_expert.py'),
    Path('scripts/training/full_finetuning/train_uml_expert.py'),
    Path('scripts/training/lora_moe/train_general_expert.py'),
    Path('scripts/training/lora_moe/train_image_expert.py'),
    Path('scripts/training/lora_moe/train_text_expert.py'),
    Path('scripts/training/lora_moe/train_uml_expert.py'),
    Path('scripts/training/lora_single/train_unified_expert.py'),
    Path('scripts/training/p_tuning/train_general_expert.py'),
    Path('scripts/training/p_tuning/train_image_expert.py'),
    Path('scripts/training/p_tuning/train_text_expert.py'),
    Path('scripts/training/p_tuning/train_uml_expert.py'),
    Path('scripts/training/prompt_tuning/train_general_expert.py'),
    Path('scripts/training/prompt_tuning/train_image_expert.py'),
    Path('scripts/training/prompt_tuning/train_text_expert.py'),
    Path('scripts/training/prompt_tuning/train_uml_expert.py'),
]


def _parse_launcher(relative_path: Path) -> tuple[Path, ast.Module]:
    launcher = ROOT / relative_path
    tree = ast.parse(
        launcher.read_text(encoding='utf-8'),
        filename=str(launcher),
    )
    return launcher, tree


def _project_root_assignment(tree: ast.Module) -> ast.Assign:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == 'PROJECT_ROOT'
            for target in node.targets
        )
    )


@pytest.mark.parametrize(
    'relative_path',
    TRAINING_LAUNCHERS,
    ids=[path.as_posix() for path in TRAINING_LAUNCHERS],
)
def test_training_launcher_project_root_resolves_repository_root(relative_path):
    launcher, tree = _parse_launcher(relative_path)
    assignment = _project_root_assignment(tree)
    module = ast.Module(body=[assignment], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        'Path': Path,
        '__file__': str(launcher),
    }

    exec(compile(module, str(launcher), 'exec'), namespace)

    assert namespace['PROJECT_ROOT'] == ROOT


@pytest.mark.parametrize(
    'relative_path',
    TRAINING_LAUNCHERS,
    ids=[path.as_posix() for path in TRAINING_LAUNCHERS],
)
def test_training_launcher_inserts_root_before_project_imports(relative_path):
    _launcher, tree = _parse_launcher(relative_path)
    assignment = _project_root_assignment(tree)
    insert_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == 'insert'
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == 'sys'
        and node.func.value.attr == 'path'
    )
    first_project_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (
            node.module == 'config'
            or node.module.startswith('config.')
            or node.module == 'src'
            or node.module.startswith('src.')
        )
    )

    assert assignment.lineno < insert_call.lineno < first_project_import.lineno
    assert len(insert_call.args) == 2  # noqa: PLR2004 - exact insert-call arity
    assert isinstance(insert_call.args[0], ast.Constant)
    assert insert_call.args[0].value == 0
