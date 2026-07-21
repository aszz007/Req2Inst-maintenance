"""Regression coverage for unused evaluation plot return containers."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP10_SCRIPT = (
    ROOT
    / "scripts"
    / "evaluation"
    / "experiments"
    / "exp10_advanced_routing.py"
)
EXP11_SCRIPT = (
    ROOT
    / "scripts"
    / "evaluation"
    / "experiments"
    / "exp11_ablation_optimization.py"
)


def _load_function(path, function_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(path), "exec"), namespace)
    return namespace[function_name]


def test_exp10_routing_accuracy_preserves_both_bar_series(tmp_path):
    axis = MagicMock()
    annotated_bars = [
        SimpleNamespace(
            get_x=MagicMock(return_value=0.0),
            get_width=MagicMock(return_value=0.35),
            get_height=MagicMock(return_value=50.0),
        ),
        SimpleNamespace(
            get_x=MagicMock(return_value=1.0),
            get_width=MagicMock(return_value=0.35),
            get_height=MagicMock(return_value=75.0),
        ),
    ]
    axis.bar.side_effect = [annotated_bars, object()]
    plot = SimpleNamespace(
        subplots=MagicMock(return_value=(object(), axis)),
        tight_layout=MagicMock(),
        savefig=MagicMock(),
        close=MagicMock(),
    )
    logger = SimpleNamespace(info=MagicMock())
    plot_dir = tmp_path / "plots"
    plotter = _load_function(
        EXP10_SCRIPT,
        "_plot_routing_accuracy",
        {
            "ALL_TYPES": ("text", "image"),
            "PLOT_DIR": plot_dir,
            "np": np,
            "plt": plot,
            "logger": logger,
        },
    )

    plotter(
        {"routing_accuracy": {"text": 0.50, "image": 0.75}},
        {
            "oracle_selections": {
                "text": {"text": 3, "image": 1},
                "image": {"text": 1, "image": 3},
            }
        },
    )

    assert axis.bar.call_count == 2
    learned_call, oracle_call = axis.bar.call_args_list
    np.testing.assert_allclose(learned_call.args[0], [-0.175, 0.825])
    assert learned_call.args[1:] == ([50.0, 75.0], 0.35)
    assert learned_call.kwargs == {
        "label": "Learned Router Accuracy",
        "color": "#3498db",
        "alpha": 0.85,
    }
    np.testing.assert_allclose(oracle_call.args[0], [0.175, 1.175])
    assert oracle_call.args[1:] == ([75.0, 75.0], 0.35)
    assert oracle_call.kwargs == {
        "label": "Oracle Dominant Expert Rate",
        "color": "#2ecc71",
        "alpha": 0.85,
    }
    assert axis.text.call_count == 2
    output_path = plot_dir / "routing_accuracy_by_domain.png"
    plot.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
    plot.close.assert_called_once_with()
    logger.info.assert_called_once_with("  [3/8] routing_accuracy_by_domain.png")


def test_exp11_confusion_compare_preserves_each_variant_bar_series(tmp_path):
    first_axis = MagicMock()
    second_axis = MagicMock()
    first_axis.imshow.return_value = object()
    second_axis.bar.return_value = object()
    figure = MagicMock()
    plot = SimpleNamespace(
        subplots=MagicMock(return_value=(figure, [first_axis, second_axis])),
        suptitle=MagicMock(),
        tight_layout=MagicMock(),
        savefig=MagicMock(),
        close=MagicMock(),
    )
    logger = SimpleNamespace(info=MagicMock())
    plot_dir = tmp_path / "plots"
    plotter = _load_function(
        EXP11_SCRIPT,
        "_plot_confusion_compare",
        {
            "PLOT_DIR": plot_dir,
            "np": np,
            "plt": plot,
            "logger": logger,
        },
    )

    plotter(
        {
            "router_results": {
                "B0": {
                    "name": "Baseline",
                    "per_class": {
                        "text": 0.10,
                        "image": 0.20,
                        "uml": 0.30,
                        "general": 0.40,
                    },
                },
                "B2": {
                    "name": "Variant",
                    "per_class": {
                        "text": 0.50,
                        "image": 0.60,
                        "uml": 0.70,
                        "general": 0.80,
                    },
                },
            }
        }
    )

    assert second_axis.bar.call_count == 2
    baseline_call, variant_call = second_axis.bar.call_args_list
    np.testing.assert_allclose(baseline_call.args[0], [-0.09, 0.91, 1.91, 2.91])
    assert baseline_call.args[1:3] == ([0.10, 0.20, 0.30, 0.40], 0.18)
    assert baseline_call.kwargs == {
        "label": "B0: Baseline",
        "color": "#2E75B6",
    }
    np.testing.assert_allclose(variant_call.args[0], [0.09, 1.09, 2.09, 3.09])
    assert variant_call.args[1:3] == ([0.50, 0.60, 0.70, 0.80], 0.18)
    assert variant_call.kwargs == {
        "label": "B2: Variant",
        "color": "#27AE60",
    }
    figure.colorbar.assert_called_once()
    output_path = plot_dir / "router_confusion_compare.png"
    plot.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
    plot.close.assert_called_once_with()
    logger.info.assert_called_once_with("  [5/6] router_confusion_compare.png")
