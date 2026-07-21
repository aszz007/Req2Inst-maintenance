"""Regression coverage for the Experiment 6 plotting side effects."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[1]
EXP6_SCRIPT = (
    ROOT
    / "scripts"
    / "evaluation"
    / "experiments"
    / "exp6_fewshot_vs_finetuning.py"
)


def _load_plotter(namespace):
    tree = ast.parse(
        EXP6_SCRIPT.read_text(encoding="utf-8"),
        filename=str(EXP6_SCRIPT),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "plot_bar_with_errorbars"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    exec(compile(isolated_module, str(EXP6_SCRIPT), "exec"), namespace)
    return namespace["plot_bar_with_errorbars"]


def test_plot_bar_with_errorbars_preserves_render_and_save_contract(tmp_path):
    axis = MagicMock()
    axis.bar.return_value = object()
    plot = SimpleNamespace(
        subplots=MagicMock(return_value=(object(), axis)),
        Rectangle=MagicMock(side_effect=lambda *args, **kwargs: (args, kwargs)),
        tight_layout=MagicMock(),
        savefig=MagicMock(),
        close=MagicMock(),
    )
    logger = SimpleNamespace(info=MagicMock())
    plotter = _load_plotter(
        {
            "np": SimpleNamespace(arange=lambda size: tuple(range(size))),
            "plt": plot,
            "logger": logger,
        }
    )

    plotter(
        {1: (0.20, 0.01), 5: (0.40, 0.02)},
        0.60,
        tmp_path,
        test_mode=True,
    )

    plots_dir = tmp_path / "plots"
    output_path = plots_dir / "fewshot_vs_finetuning.png"
    assert plots_dir.is_dir()
    plot.subplots.assert_called_once_with(figsize=(9, 5))
    axis.bar.assert_called_once_with(
        (0, 1, 2),
        [0.20, 0.40, 0.60],
        yerr=[0.01, 0.02, 0],
        capsize=5,
        color=["#ff7f0e", "#ff7f0e", "#1f77b4"],
        alpha=0.85,
    )
    axis.set_title.assert_called_once_with(
        "Exp6: Few-Shot vs Fine-Tuning (ROUGE-L) [Test Mode]"
    )
    plot.savefig.assert_called_once_with(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )
    plot.close.assert_called_once_with()
    logger.info.assert_called_once_with(f"Plot saved: {output_path}")
