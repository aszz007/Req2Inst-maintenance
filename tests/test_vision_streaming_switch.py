"""Regression coverage for the FlowChart streaming-output switch."""

import ast
import queue
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
VISION_MODEL = ROOT / "models" / "vision_model.py"
RECOGNIZE_INPUTS = ROOT / "scripts" / "inference" / "recognize_inputs.py"
GENERATE_INSTRUCTIONS = (
    ROOT / "scripts" / "inference" / "generate_instructions.py"
)
SETTINGS = ROOT / "config" / "settings.py"


def _load_function(path, function_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function_name]


def _load_vision_method(method_name, namespace):
    tree = ast.parse(
        VISION_MODEL.read_text(encoding="utf-8"),
        filename=str(VISION_MODEL),
    )
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionModel"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(VISION_MODEL), "exec"), namespace)
    return namespace[method_name]


class _PollingStreamer:
    def __init__(self, events):
        self._events = iter(events)

    def __next__(self):
        event = next(self._events)
        if isinstance(event, BaseException):
            raise event
        return event


class _WorkerThread:
    def __init__(self, alive):
        self._alive = iter(alive)
        self.joined = False

    def is_alive(self):
        return next(self._alive, False)

    def join(self):
        self.joined = True


def test_stream_consumer_waits_for_a_live_worker_and_preserves_text(capsys):
    logger = SimpleNamespace(debug=MagicMock())
    consume = _load_vision_method("_consume_streamer", {"logger": logger})
    streamer = _PollingStreamer([queue.Empty(), "Flow", "Chart"])
    thread = _WorkerThread([True])

    result = consume(
        SimpleNamespace(),
        streamer,
        thread,
        {"error": None},
    )

    assert result == "FlowChart"
    assert capsys.readouterr().out == "FlowChart"
    assert thread.joined is True
    assert logger.debug.call_count == 2  # noqa: PLR2004 - one call per streamed chunk


def test_stream_consumer_surfaces_a_stopped_worker_error():
    consume = _load_vision_method(
        "_consume_streamer",
        {"logger": SimpleNamespace(debug=MagicMock())},
    )
    streamer = _PollingStreamer([queue.Empty()])
    thread = _WorkerThread([False])

    with pytest.raises(RuntimeError, match="worker failed"):
        consume(
            SimpleNamespace(),
            streamer,
            thread,
            {"error": "worker failed"},
        )


def test_streaming_worker_uses_shared_generation_and_safe_fallback_contract():
    tree = ast.parse(VISION_MODEL.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VisionModel"
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "_generate_streaming"
    )
    call_names = {
        ast.unparse(node.func)
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
    }

    assert "self._model_generate_vision" in call_names
    assert "self._consume_streamer" in call_names
    assert "self._generate_standard" in call_names
    assert "self.model.generate" not in call_names
    assert "time.sleep" not in call_names
    assert not any(
        isinstance(node, ast.Name) and node.id == "last_output_time"
        for node in ast.walk(method)
    )

    thread_call = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "Thread"
    )
    daemon = next(
        keyword.value
        for keyword in thread_call.keywords
        if keyword.arg == "daemon"
    )
    assert isinstance(daemon, ast.Constant) and daemon.value is True


def test_streaming_failure_waits_before_standard_fallback():
    state = {"alive": False, "joined": False}

    class _Streamer:
        def __init__(self, tokenizer, **kwargs):
            self.tokenizer = tokenizer
            self.kwargs = kwargs

    class _Thread:
        def __init__(self, target, daemon):
            self.target = target
            state["daemon"] = daemon

        def start(self):
            state["alive"] = True

        def is_alive(self):
            return state["alive"]

        def join(self):
            state["joined"] = True
            state["alive"] = False

    def standard_fallback(inputs, task_type):
        assert state["joined"] is True
        return "fallback output"

    model = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=object()),
        uml_gen_config={},
        image_gen_config={},
        _build_gen_kwargs=MagicMock(return_value={}),
        _model_generate_vision=MagicMock(),
        _consume_streamer=MagicMock(side_effect=RuntimeError("stream failed")),
        _generate_standard=standard_fallback,
    )
    generate = _load_vision_method(
        "_generate_streaming",
        {
            "TextIteratorStreamer": _Streamer,
            "Thread": _Thread,
            "logger": SimpleNamespace(
                warning=MagicMock(),
                error=MagicMock(),
                info=MagicMock(),
            ),
        },
    )

    result = generate(model, object())

    assert result == "fallback output"
    assert state == {"alive": False, "joined": True, "daemon": True}
    model._model_generate_vision.assert_not_called()


@pytest.mark.parametrize("streaming", [None, True])
def test_unified_recognition_preserves_and_overrides_streaming(
    tmp_path,
    streaming,
):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fixture")
    calls = []

    class _VisionModel:
        def __init__(self, version):
            self.version = version

        def recognize_uml(self, path, streaming):
            calls.append((path, streaming))
            return {"success": True}

    recognize = _load_function(
        RECOGNIZE_INPUTS,
        "recognize_single_file",
        {
            "Dict": Dict,
            "Optional": Optional,
            "Path": Path,
            "VisionModel": _VisionModel,
            "datetime": datetime,
            "logger": SimpleNamespace(info=MagicMock(), error=MagicMock()),
        },
    )

    kwargs = {} if streaming is None else {"streaming": streaming}
    result = recognize(str(image_path), "uml", "qwen3", **kwargs)

    assert result["success"] is True
    assert calls == [(str(image_path), streaming)]


def test_recognition_cli_leaves_streaming_at_the_configured_default():
    tree = ast.parse(RECOGNIZE_INPUTS.read_text(encoding="utf-8"))
    parse_args = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    streaming_argument = next(
        node
        for node in ast.walk(parse_args)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--streaming"
    )
    default = next(
        keyword.value
        for keyword in streaming_argument.keywords
        if keyword.arg == "default"
    )

    assert isinstance(default, ast.Constant)
    assert default.value is None


@pytest.mark.parametrize(
    ("rec_type", "streaming", "expects_flag"),
    [
        ("uml", True, True),
        ("uml", False, False),
        ("image", True, False),
    ],
)
def test_instruction_cli_forwards_streaming_only_for_flowcharts(
    tmp_path,
    monkeypatch,
    rec_type,
    streaming,
    expects_flag,
):
    output_path = tmp_path / "recognition.json"
    run = MagicMock(
        return_value=SimpleNamespace(stdout=f"OUTPUT_FILE:{output_path}\n")
    )
    monkeypatch.setattr(subprocess, "run", run)
    recognize = _load_function(
        GENERATE_INSTRUCTIONS,
        "recognize_images",
        {
            "Path": Path,
            "List": List,
            "Optional": Optional,
            "logger": SimpleNamespace(info=MagicMock(), error=MagicMock()),
            "project_root": tmp_path,
            "re": re,
            "subprocess": subprocess,
            "sys": SimpleNamespace(executable=sys.executable),
        },
    )

    result = recognize(
        [tmp_path / "diagram.png"],
        rec_type,
        "qwen3",
        streaming=streaming,
    )

    command = run.call_args.args[0]
    assert ("--streaming" in command) is expects_flag
    assert result == output_path


def test_streaming_defaults_remain_disabled():
    tree = ast.parse(SETTINGS.read_text(encoding="utf-8"))
    device_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeviceConfig"
    )
    field = next(
        node
        for node in device_config.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "enable_streaming"
    )

    assert isinstance(field.value, ast.Constant)
    assert field.value.value is False
