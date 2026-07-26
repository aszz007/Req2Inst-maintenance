"""Regression contracts for the saved-history training-curve plotter."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import matplotlib
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "utils" / "replot_training_curves.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "req2inst_replot_training_curves",
        SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None

    source_package = ModuleType("src")
    source_package.__path__ = []
    utils_package = ModuleType("src.utils")
    utils_package.__path__ = []
    logger_module = ModuleType("src.utils.logger")
    logger_module.get_logger = lambda _name: SimpleNamespace(
        error=MagicMock(),
        warning=MagicMock(),
        info=MagicMock(),
    )
    stubbed_modules = {
        "src": source_package,
        "src.utils": utils_package,
        "src.utils.logger": logger_module,
    }
    previous_modules = {
        name: sys.modules.get(name) for name in stubbed_modules
    }
    sys.modules.update(stubbed_modules)
    try:
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


REPLOT = _load_module()


def _install_plot_spy(monkeypatch):
    axis_map = {
        (0, 0): MagicMock(name="training_loss_axis"),
        (0, 1): MagicMock(name="eval_loss_axis"),
        (1, 0): MagicMock(name="grad_norm_axis"),
        (1, 1): MagicMock(name="learning_rate_axis"),
    }
    axes = MagicMock(name="axes")
    axes.__getitem__.side_effect = axis_map.__getitem__
    figure = MagicMock(name="figure")
    backend = MagicMock(name="use_backend")
    subplots = MagicMock(name="subplots", return_value=(figure, axes))
    tight_layout = MagicMock(name="tight_layout")
    savefig = MagicMock(name="savefig")
    close = MagicMock(name="close")
    logger = SimpleNamespace(
        error=MagicMock(name="error"),
        warning=MagicMock(name="warning"),
        info=MagicMock(name="info"),
    )

    monkeypatch.setattr(matplotlib, "use", backend)
    monkeypatch.setattr(plt, "subplots", subplots)
    monkeypatch.setattr(plt, "tight_layout", tight_layout)
    monkeypatch.setattr(plt, "savefig", savefig)
    monkeypatch.setattr(plt, "close", close)
    monkeypatch.setattr(REPLOT, "logger", logger)

    return SimpleNamespace(
        axes=axes,
        figure=figure,
        backend=backend,
        subplots=subplots,
        tight_layout=tight_layout,
        savefig=savefig,
        close=close,
        logger=logger,
    )


def test_plotter_preserves_filtered_series_rendering_and_logs(monkeypatch, tmp_path):
    spy = _install_plot_spy(monkeypatch)
    output_path = tmp_path / "text_expert.png"
    history = [
        {
            "step": 1,
            "loss": 1.25,
            "eval_loss": 2.5,
            "grad_norm": 0.75,
            "learning_rate": 0.0001,
        },
        {
            "step": 2,
            "loss": None,
            "eval_loss": float("nan"),
            "grad_norm": float("nan"),
            "learning_rate": None,
        },
    ]

    assert REPLOT.plot_training_curves(
        history,
        "text",
        "lora_moe",
        output_path,
    ) is True

    spy.backend.assert_called_once_with("Agg")
    spy.subplots.assert_called_once_with(2, 2, figsize=(15, 10))
    spy.figure.suptitle.assert_called_once_with(
        "Training Curves - TEXT Expert (lora_moe)",
        fontsize=16,
        fontweight="bold",
    )

    training_axis = spy.axes[0, 0]
    training_axis.plot.assert_called_once_with(
        [1],
        [1.25],
        "b-",
        linewidth=1.5,
        alpha=0.7,
    )
    training_axis.set_xlabel.assert_called_once_with("Step")
    training_axis.set_ylabel.assert_called_once_with("Loss")
    training_axis.set_title.assert_called_once_with("Training Loss")
    training_axis.grid.assert_called_once_with(True, alpha=0.3)

    eval_axis = spy.axes[0, 1]
    eval_axis.plot.assert_called_once_with(
        [1],
        [2.5],
        "r-",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    eval_axis.set_xlabel.assert_called_once_with("Step")
    eval_axis.set_ylabel.assert_called_once_with("Eval Loss")
    eval_axis.set_title.assert_called_once_with("Validation Loss")
    eval_axis.grid.assert_called_once_with(True, alpha=0.3)

    grad_axis = spy.axes[1, 0]
    grad_axis.plot.assert_called_once_with(
        [1],
        [0.75],
        "g-",
        linewidth=1,
        alpha=0.6,
    )
    grad_axis.set_xlabel.assert_called_once_with("Step")
    grad_axis.set_ylabel.assert_called_once_with("Gradient Norm")
    grad_axis.set_title.assert_called_once_with("Gradient Norm")
    grad_axis.grid.assert_called_once_with(True, alpha=0.3)

    learning_rate_axis = spy.axes[1, 1]
    learning_rate_axis.plot.assert_called_once_with(
        [1],
        [0.0001],
        "m-",
        linewidth=1.5,
    )
    learning_rate_axis.set_xlabel.assert_called_once_with("Step")
    learning_rate_axis.set_ylabel.assert_called_once_with("Learning Rate")
    learning_rate_axis.set_title.assert_called_once_with("Learning Rate Schedule")
    learning_rate_axis.grid.assert_called_once_with(True, alpha=0.3)
    learning_rate_axis.ticklabel_format.assert_called_once_with(
        style="sci",
        axis="y",
        scilimits=(0, 0),
    )

    spy.tight_layout.assert_called_once_with()
    spy.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")
    spy.close.assert_called_once_with()
    assert spy.logger.warning.call_args_list == [
        call("Training history has only 2 entries, possibly due to early stopping"),
        call("Training loss has only 1 data points"),
        call(
            "Validation loss has only 1 valid data points; "
            "1 NaN values were filtered"
        ),
    ]
    assert spy.logger.info.call_args_list == [
        call(f"Training curves saved to: {output_path}"),
        call(
            "Data summary: Loss=1 points, EvalLoss=1 points, "
            "GradNorm=1 points, LR=1 points"
        ),
    ]


def test_plotter_preserves_empty_series_panels_and_warnings(monkeypatch, tmp_path):
    spy = _install_plot_spy(monkeypatch)
    output_path = tmp_path / "empty.png"
    history = [{"step": step} for step in range(10)]

    assert REPLOT.plot_training_curves(
        history,
        "image",
        "prompt_tuning",
        output_path,
    ) is True

    assert spy.logger.warning.call_args_list == [
        call("Training loss has only 0 data points"),
        call("No validation loss data found"),
    ]
    expected_empty_panels = (
        (spy.axes[0, 0], "No training loss data", "Loss", "Training Loss"),
        (spy.axes[0, 1], "No validation loss data", "Eval Loss", "Validation Loss"),
        (spy.axes[1, 0], "No gradient norm data", "Gradient Norm", "Gradient Norm"),
        (spy.axes[1, 1], "No learning rate data", "Learning Rate", "Learning Rate Schedule"),
    )
    for axis, message, ylabel, title in expected_empty_panels:
        axis.plot.assert_not_called()
        axis.text.assert_called_once_with(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xlabel.assert_called_once_with("Step")
        axis.set_ylabel.assert_called_once_with(ylabel)
        axis.set_title.assert_called_once_with(title)


def test_plotter_preserves_all_nan_validation_guidance(monkeypatch, tmp_path):
    spy = _install_plot_spy(monkeypatch)
    history = [
        {
            "step": step,
            "loss": float(step),
            "eval_loss": float("nan"),
        }
        for step in range(10)
    ]

    assert REPLOT.plot_training_curves(
        history,
        "uml",
        "p_tuning",
        tmp_path / "nan.png",
    ) is True

    assert spy.logger.warning.call_args_list == [
        call(
            "All 10 validation loss values are NaN and were filtered; "
            "the validation curve cannot be plotted"
        ),
        call("This may indicate unstable training; consider:"),
        call("  1. Reducing the learning rate, which may be too high"),
        call("  2. Adjusting the parameter-efficient fine-tuning configuration"),
        call("  3. Checking dataset quality and preprocessing"),
    ]


def test_plotter_preserves_partial_nan_information(monkeypatch, tmp_path):
    spy = _install_plot_spy(monkeypatch)
    history = [
        {"step": 0, "loss": 1.0, "eval_loss": 3.0},
        {"step": 1, "loss": 0.9, "eval_loss": 2.0},
        {"step": 2, "loss": 0.8, "eval_loss": 1.0},
        {"step": 3, "eval_loss": float("nan")},
        *[{"step": step} for step in range(4, 10)],
    ]

    assert REPLOT.plot_training_curves(
        history,
        "general",
        "lora_single",
        tmp_path / "partial_nan.png",
    ) is True

    spy.logger.warning.assert_not_called()
    assert spy.logger.info.call_args_list[0] == call(
        "Filtered 1 NaN validation loss values; 3 valid values remain"
    )


def test_plotter_preserves_missing_matplotlib_failure(monkeypatch, tmp_path):
    logger = SimpleNamespace(error=MagicMock())
    monkeypatch.setattr(REPLOT, "logger", logger)
    real_import = builtins.__import__

    def reject_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ImportError("fixture import failure")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)

    assert REPLOT.plot_training_curves(
        [],
        "text",
        "lora_moe",
        tmp_path / "unused.png",
    ) is False
    logger.error.assert_called_once_with(
        "matplotlib is not installed; visualizations cannot be generated"
    )
