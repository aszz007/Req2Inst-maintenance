"""Regression coverage for the Experiment 8 latency plot side effects."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
EXP8_SCRIPT = (
    ROOT
    / "scripts"
    / "evaluation"
    / "experiments"
    / "exp8_inference_efficiency.py"
)


def _load_plotter(namespace):
    tree = ast.parse(
        EXP8_SCRIPT.read_text(encoding="utf-8"),
        filename=str(EXP8_SCRIPT),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "plot_latency_comparison"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(EXP8_SCRIPT), "exec"), namespace)
    return namespace["plot_latency_comparison"]


def test_latency_plot_preserves_render_and_save_contract(tmp_path):
    axis = MagicMock()
    axis.barh.return_value = object()
    plot = SimpleNamespace(
        subplots=MagicMock(return_value=(object(), axis)),
        tight_layout=MagicMock(),
        savefig=MagicMock(),
        close=MagicMock(),
    )
    logger = SimpleNamespace(info=MagicMock())
    plots_dir = tmp_path / "plots"
    colors = {
        "lora_moe": "#1f77b4",
        "prompt_tuning": "#ff7f0e",
    }
    plotter = _load_plotter(
        {
            "PLOTS_DIR": plots_dir,
            "METHOD_ORDER": ("lora_moe", "prompt_tuning"),
            "METHOD_LABELS": {
                "lora_moe": "Multi-Expert LoRA",
                "prompt_tuning": "Prompt Tuning",
            },
            "_get_method_color": colors.__getitem__,
            "np": SimpleNamespace(arange=lambda size: tuple(range(size))),
            "plt": plot,
            "logger": logger,
        }
    )

    plotter(
        {
            "lora_moe": {
                "latency_median_ms": 12.5,
                "latency_p95_ms": 18.0,
            },
            "prompt_tuning": {
                "latency_median_ms": 20.0,
                "latency_p95_ms": 30.0,
            },
        },
        test_mode=True,
    )

    output_path = plots_dir / "latency_comparison.png"
    assert plots_dir.is_dir()
    plot.subplots.assert_called_once_with(figsize=(10, 5))
    axis.barh.assert_called_once_with(
        (0, 1),
        [12.5, 20.0],
        color=["#1f77b4", "#ff7f0e"],
        edgecolor="gray",
        height=0.55,
        label="Median",
    )
    axis.scatter.assert_called_once_with(
        [18.0, 30.0],
        (0, 1),
        marker="|",
        color="red",
        s=120,
        zorder=5,
        label="P95",
    )
    axis.set_title.assert_called_once_with(
        "Exp8: Inference Latency Comparison (batch_size=1) [Test Mode]"
    )
    plot.savefig.assert_called_once_with(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plot.close.assert_called_once_with()
    logger.info.assert_called_once_with(f"Plot saved to: {output_path}")
