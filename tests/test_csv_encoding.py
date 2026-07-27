'''Regression coverage for deterministic CSV encoding selection.'''

import ast
from pathlib import Path

import pandas as pd
import pytest

from src import csv_encoding


ROOT = Path(__file__).resolve().parents[1]
DATA_LOADER = ROOT / 'src' / 'training' / 'data_loader.py'


def _write_csv(path: Path, encoding: str, values: list[str]) -> None:
    content = 'text\n' + '\n'.join(values) + '\n'
    path.write_bytes(content.encode(encoding))


def test_detects_utf8_sig_and_preserves_text(tmp_path):
    csv_path = tmp_path / 'utf8.csv'
    _write_csv(csv_path, 'utf-8-sig', ['café'])

    encoding = csv_encoding.detect_csv_encoding(csv_path)
    frame = pd.read_csv(csv_path, encoding=encoding)

    assert encoding == 'utf-8-sig'
    assert frame['text'].tolist() == ['café']


def test_prefers_gb18030_for_project_gbk_punctuation(tmp_path):
    csv_path = tmp_path / 'gbk.csv'
    expected = ['patient’s record', 'SRS 34–36', 'within ±1 ml']
    _write_csv(csv_path, 'gbk', expected)

    encoding = csv_encoding.detect_csv_encoding(csv_path)
    frame = pd.read_csv(csv_path, encoding=encoding)

    assert encoding == 'gb18030'
    assert frame['text'].tolist() == expected


def test_falls_back_to_cp1252_when_gb18030_is_invalid(tmp_path):
    csv_path = tmp_path / 'cp1252.csv'
    _write_csv(csv_path, 'cp1252', ['café'])

    encoding = csv_encoding.detect_csv_encoding(csv_path)
    frame = pd.read_csv(csv_path, encoding=encoding)

    assert encoding == 'cp1252'
    assert frame['text'].tolist() == ['café']


def test_explicit_encoding_override_is_authoritative(tmp_path):
    csv_path = tmp_path / 'latin1.csv'
    _write_csv(csv_path, 'latin1', ['café'])

    encoding = csv_encoding.detect_csv_encoding(
        csv_path,
        preferred_encoding='latin1',
    )

    assert encoding == 'iso8859-1'


@pytest.mark.parametrize('control_exception', [KeyboardInterrupt, SystemExit])
def test_control_exceptions_are_not_swallowed(
    monkeypatch,
    tmp_path,
    control_exception,
):
    csv_path = tmp_path / 'control.csv'
    _write_csv(csv_path, 'utf-8', ['value'])

    def raise_control_exception(*_args, **_kwargs):
        raise control_exception()

    monkeypatch.setattr(csv_encoding, '_decode_file', raise_control_exception)

    with pytest.raises(control_exception):
        csv_encoding.detect_csv_encoding(csv_path)


def test_decode_failure_reports_attempted_encodings(tmp_path):
    csv_path = tmp_path / 'invalid.csv'
    csv_path.write_bytes(b'\xff')

    with pytest.raises(UnicodeError, match='utf-8'):
        csv_encoding.detect_csv_encoding(
            csv_path,
            encodings=('utf-8',),
        )


def test_missing_file_is_not_hidden(tmp_path):
    with pytest.raises(FileNotFoundError):
        csv_encoding.detect_csv_encoding(tmp_path / 'missing.csv')


def test_training_loader_detector_delegates_to_shared_resolver():
    tree = ast.parse(
        DATA_LOADER.read_text(encoding='utf-8'),
        filename=str(DATA_LOADER),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'detect_csv_encoding'
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    calls = []

    def resolver(filepath):
        calls.append(filepath)
        return 'gb18030'

    namespace = {
        'Path': Path,
        'resolve_csv_encoding': resolver,
    }
    exec(compile(isolated_module, str(DATA_LOADER), 'exec'), namespace)

    path = Path('fixture.csv')
    assert namespace['detect_csv_encoding'](path) == 'gb18030'
    assert calls == [path]
