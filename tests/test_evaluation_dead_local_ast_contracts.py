"""Regression contracts for side-effect-free dead evaluation locals."""

import ast
import copy
import hashlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

SPECS = [
    pytest.param(
        "scripts/evaluation/experiments/exp4_lora_hyperparameter_optimization.py",
        "plot_heatmap_dropout_alpha",
        ("alphas", "dropouts"),
        "5e805f2e4dcd416178f34fd56f20dcdfb4b39075a13d8b9bed7db5a46165ece8",
        id="exp4-heatmap-axes",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp9_routing_strategy.py",
        "run_phase3",
        ("score_matrix",),
        "8876d7dc10d106f73376d2a5c98e745ae5bddab8157a77cc53e20de89b9f7ae4",
        id="exp9-score-matrix",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp10_advanced_routing.py",
        "_run_output_ensemble",
        ("n_cache",),
        "3f0b7fab466f4275194e4e8a34079b7a2bfc29cebbc688eb33eebf0ae214da3a",
        id="exp10-cache-count",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp10_advanced_routing.py",
        "run_phase1",
        ("test_y",),
        "63256ca2b13e2ac41a390cafa8f71f446586a661e54f72528797d85fac1aaf19",
        id="exp10-unused-test-labels",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp10_advanced_routing.py",
        "run_phase3",
        ("gap",),
        "4eda9c27a92b6dd246cf3e77c1d2148a4621b97746d12aa13b0d4bd63f58e265",
        id="exp10-phase3-gap",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp10_advanced_routing.py",
        "_plot_summary_table",
        ("gap_avg",),
        "af08b45c099c75ee56feef3fe02962dfa49732ffdecf61ba4a7c37317406518d",
        id="exp10-summary-gap",
    ),
    pytest.param(
        "scripts/evaluation/experiments/exp11_ablation_optimization.py",
        "run_phase3",
        ("gap",),
        "7ac7aa1d37fb83cc6fbd88ba49581e29650c66d59825b46649e8cf9af294d369",
        id="exp11-phase3-gap",
    ),
]


def _assignment_names(statement):
    if not isinstance(statement, ast.Assign):
        return set()
    return {
        target.id
        for target in statement.targets
        if isinstance(target, ast.Name)
    }


def _load_function(relative_path, function_name):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _normalized_hash(function, variable_names):
    function = copy.deepcopy(function)
    targets = set(variable_names)
    for owner in ast.walk(function):
        body = getattr(owner, "body", None)
        if not isinstance(body, list):
            continue
        owner.body = [
            statement
            for statement in body
            if not (_assignment_names(statement) & targets)
        ]
    ast.fix_missing_locations(function)
    dumped = ast.dump(function, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    ("relative_path", "function_name", "variable_names", "expected_hash"),
    SPECS,
)
def test_dead_local_removal_preserves_normalized_function_contract(
    relative_path,
    function_name,
    variable_names,
    expected_hash,
):
    function = _load_function(relative_path, function_name)
    loads = [
        (node.id, node.lineno)
        for node in ast.walk(function)
        if isinstance(node, ast.Name)
        and node.id in variable_names
        and isinstance(node.ctx, ast.Load)
    ]

    assert loads == []
    assert _normalized_hash(function, variable_names) == expected_hash
