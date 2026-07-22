"""Regression contracts for full-finetuning entry-point imports."""

import ast
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FULL_FINETUNING_DIR = ROOT / "scripts" / "training" / "full_finetuning"
ENTRYPOINT_BODY_HASHES = {
    FULL_FINETUNING_DIR / "train_text_expert.py": (
        "10e676436745e9e58988a3d00f5c7943af1bc974eefe6926fc9b8220e164fa43"
    ),
    FULL_FINETUNING_DIR / "train_image_expert.py": (
        "3f424300f4e8501e95878d8d3b80f9bb65e22e3385293318f2ecaacdb107e724"
    ),
    FULL_FINETUNING_DIR / "train_uml_expert.py": (
        "796f99339c50c36b6f115435ff0d48a57785b9723da2fff469d9352a4980b4f7"
    ),
    FULL_FINETUNING_DIR / "train_general_expert.py": (
        "33a2416ec8b26b467649dc2b34f5bc0501d3edf69dea0c0b9deee83203d8de66"
    ),
}
EXPECTED_ENTRYPOINT_IMPORTS = [
    ("import", None, ("sys",)),
    ("import", None, ("argparse",)),
    ("from", "pathlib", ("Path",)),
    ("from", "config.settings", ("get_path_config",)),
    (
        "from",
        "src.training.full_finetuning_trainer",
        ("FullFineTuningTrainer",),
    ),
    ("from", "src.utils.logger", ("get_logger",)),
]
DOWNSTREAM_TORCH_MODULES = [
    ROOT / "config" / "settings.py",
    ROOT / "src" / "training" / "full_finetuning_trainer.py",
    ROOT / "src" / "training" / "base_trainer.py",
]


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _non_import_body_hash(path):
    tree = _parse(path)
    module = ast.Module(
        body=[
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
            and not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "PROJECT_ROOT"
                    for target in node.targets
                )
            )
        ],
        type_ignores=[],
    )
    return hashlib.sha256(
        ast.dump(module, include_attributes=False).encode("utf-8")
    ).hexdigest()


def _top_level_import_signature(path):
    signature = []
    for node in _parse(path).body:
        if isinstance(node, ast.Import):
            signature.append(
                ("import", None, tuple(alias.name for alias in node.names))
            )
        elif isinstance(node, ast.ImportFrom):
            signature.append(
                (
                    "from",
                    node.module,
                    tuple(alias.name for alias in node.names),
                )
            )
    return signature


def _loads_name(path, name):
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
        for node in ast.walk(_parse(path))
    )


@pytest.mark.parametrize("path,expected_hash", ENTRYPOINT_BODY_HASHES.items())
def test_entrypoint_non_import_body_matches_pre_cleanup_contract(
    path,
    expected_hash,
):
    assert _non_import_body_hash(path) == expected_hash


@pytest.mark.parametrize("path", ENTRYPOINT_BODY_HASHES)
def test_entrypoint_keeps_all_required_imports_except_torch(path):
    assert _top_level_import_signature(path) == EXPECTED_ENTRYPOINT_IMPORTS
    assert not _loads_name(path, "torch")


@pytest.mark.parametrize("path", DOWNSTREAM_TORCH_MODULES)
def test_downstream_runtime_modules_keep_their_torch_dependency(path):
    assert ("import", None, ("torch",)) in _top_level_import_signature(path)
    assert _loads_name(path, "torch")
