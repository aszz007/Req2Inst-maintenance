'''Regression coverage for per-row dataset loader exception boundaries.'''

from pathlib import Path

import pytest

from src.training import data_loader


def _write_csv(path: Path, header: str, rows: list[tuple[str, str]]) -> None:
    content = [header]
    content.extend(f'{input_value},{output_value}' for input_value, output_value in rows)
    path.write_text('\n'.join(content) + '\n', encoding='utf-8')


def _text_loader(dataset_dir: Path):
    loader = data_loader.TextDatasetLoader.__new__(
        data_loader.TextDatasetLoader,
    )
    loader.dataset_dir = dataset_dir
    return loader


def _image_loader(dataset_csv: Path):
    loader = data_loader.ImageDatasetLoader.__new__(
        data_loader.ImageDatasetLoader,
    )
    loader.dataset_csv = dataset_csv
    return loader


def test_text_loader_skips_ordinary_bad_row_and_preserves_valid_rows(
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / 'text.csv'
    _write_csv(
        csv_path,
        'low_requirements,instruction',
        [('first', 'output-1'), ('bad', 'output-2'), ('last', 'output-3')],
    )

    def build_prompt(value):
        if value == 'bad':
            raise RuntimeError('bad text row')
        return f'prompt:{value}'

    monkeypatch.setattr(
        data_loader.TextInstructionTemplate,
        'build_prompt',
        build_prompt,
    )

    assert _text_loader(tmp_path).load_csv_files() == [
        {
            'input': 'first',
            'input_with_prompt': 'prompt:first',
            'output': 'output-1',
            'source': 'text',
        },
        {
            'input': 'last',
            'input_with_prompt': 'prompt:last',
            'output': 'output-3',
            'source': 'text',
        },
    ]


def test_image_loader_skips_ordinary_bad_row_and_preserves_valid_rows(
    monkeypatch,
    tmp_path,
):
    csv_path = tmp_path / 'image.csv'
    _write_csv(
        csv_path,
        'description,instruction',
        [('first', 'output-1'), ('bad', 'output-2'), ('last', 'output-3')],
    )

    def build_prompt(value):
        if value == 'bad':
            raise RuntimeError('bad image row')
        return f'prompt:{value}'

    monkeypatch.setattr(
        data_loader.ImageInstructionTemplate,
        'build_prompt',
        build_prompt,
    )

    assert _image_loader(csv_path).load_csv_file() == [
        {
            'input': 'first',
            'input_with_prompt': 'prompt:first',
            'output': 'output-1',
            'source': 'image_dataset',
        },
        {
            'input': 'last',
            'input_with_prompt': 'prompt:last',
            'output': 'output-3',
            'source': 'image_dataset',
        },
    ]


@pytest.mark.parametrize('control_exception', [KeyboardInterrupt, SystemExit])
def test_text_loader_propagates_control_exceptions(
    monkeypatch,
    tmp_path,
    control_exception,
):
    csv_path = tmp_path / 'text.csv'
    _write_csv(csv_path, 'low_requirements,instruction', [('trigger', 'output')])

    def raise_control_exception(_value):
        raise control_exception()

    monkeypatch.setattr(
        data_loader.TextInstructionTemplate,
        'build_prompt',
        raise_control_exception,
    )

    with pytest.raises(control_exception):
        _text_loader(tmp_path).load_csv_files()


@pytest.mark.parametrize('control_exception', [KeyboardInterrupt, SystemExit])
def test_image_loader_propagates_control_exceptions(
    monkeypatch,
    tmp_path,
    control_exception,
):
    csv_path = tmp_path / 'image.csv'
    _write_csv(csv_path, 'description,instruction', [('trigger', 'output')])

    def raise_control_exception(_value):
        raise control_exception()

    monkeypatch.setattr(
        data_loader.ImageInstructionTemplate,
        'build_prompt',
        raise_control_exception,
    )

    with pytest.raises(control_exception):
        _image_loader(csv_path).load_csv_file()
