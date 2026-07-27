"""Provide safe file, path, checkpoint, and JSON utilities."""

import csv
import json
import shutil
import warnings
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_path_join(*paths: str | Path) -> Path:
    """Join path components safely."""
    if not paths:
        return Path('.')

    result = Path(paths[0])
    for p in paths[1:]:
        result = result / p

    return result


def get_relative_path(path: str | Path, base: str | Path) -> Path:
    """Return relative path."""
    path = Path(path).resolve()
    base = Path(base).resolve()

    try:
        return path.relative_to(base)
    except ValueError:
        return path


def validate_path_exists(
        path: str | Path,
        path_type: str = 'auto',
        raise_error: bool = True
) -> bool:
    """Validate path exists."""
    path = Path(path)

    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"Path does not exist: {path}")
        return False

    if path_type == 'file' and not path.is_file():
        if raise_error:
            raise ValueError(f"Expected a file, but the path is a directory: {path}")
        return False

    if path_type == 'dir' and not path.is_dir():
        if raise_error:
            raise ValueError(f"Expected a directory, but the path is a file: {path}")
        return False

    return True



def load_json(filepath: str | Path, encoding: str = 'utf-8') -> dict:
    """Load JSON."""
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON format ({filepath}): {str(e)}",
            e.doc, e.pos
        )


def save_json(
        data: dict,
        filepath: str | Path,
        indent: int = 2,
        encoding: str = 'utf-8',
        ensure_ascii: bool = False
) -> None:
    """Save JSON."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    with open(filepath, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def update_json(
        filepath: str | Path,
        updates: dict,
        create_if_missing: bool = True
) -> dict:
    """Update JSON."""
    filepath = Path(filepath)

    if filepath.exists():
        data = load_json(filepath)
    elif create_if_missing:
        data = {}
    else:
        raise FileNotFoundError(f"JSON file does not exist: {filepath}")

    data.update(updates)

    save_json(data, filepath)

    return data



def load_csv(
        filepath: str | Path,
        encoding: str = 'utf-8',
        delimiter: str = ',',
        skip_header: bool = False
) -> list[dict]:
    """Load CSV."""
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        if skip_header:
            next(reader, None)

        return list(reader)


def load_csv_chunks(
        filepath: str | Path,
        chunksize: int = 1000,
        encoding: str = 'utf-8',
        delimiter: str = ','
) -> Iterator[list[dict]]:
    """Load CSV chunks."""
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        chunk = []
        for i, row in enumerate(reader):
            chunk.append(row)

            if (i + 1) % chunksize == 0:
                yield chunk
                chunk = []

        if chunk:
            yield chunk


def save_csv(
        data: list[dict],
        filepath: str | Path,
        fieldnames: list[str] | None = None,
        encoding: str = 'utf-8',
        delimiter: str = ','
) -> None:
    """Save CSV."""
    if not data:
        warnings.warn("No data to save")
        return

    filepath = Path(filepath)
    ensure_dir(filepath.parent)

    if fieldnames is None:
        fieldnames = list(data[0].keys())

    with open(filepath, 'w', encoding=encoding, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(data)



def load_lora_weights(expert_name: str) -> Path | None:
    """Load LoRA weights."""
    try:
        from config import get_path_config
        path_cfg = get_path_config()
        weight_path = path_cfg.get_expert_weight_path(expert_name)

        if weight_path.exists():
            return weight_path
        else:
            warnings.warn(f"LoRA weights for the {expert_name} expert were not found: {weight_path}")
            return None
    except ImportError:
        warnings.warn("The configuration module is not loaded; unable to resolve the weight path")
        return None


def save_lora_weights(
        model: Any,
        expert_name: str,
        checkpoint_name: str | None = None,
        save_method: str = 'peft'
) -> Path:
    """Save LoRA weights."""
    try:
        from config import get_path_config
        path_cfg = get_path_config()

        weight_path = path_cfg.get_expert_weight_path(expert_name)
        ensure_dir(weight_path)

        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"checkpoint_{timestamp}"

        save_path = weight_path / checkpoint_name
        ensure_dir(save_path)

        if save_method == 'peft':
            model.save_pretrained(save_path)
        else:
            import torch
            torch.save(model.state_dict(), save_path / "model.pt")

        return save_path

    except Exception as e:
        raise RuntimeError(f"Failed to save LoRA weights: {str(e)}")


def list_checkpoints(expert_name: str) -> list[Path]:
    """List available checkpoints."""
    try:
        from config import get_path_config
        path_cfg = get_path_config()

        checkpoint_dir = path_cfg.get_expert_checkpoint_path(expert_name)

        if not checkpoint_dir.exists():
            return []

        checkpoints = [d for d in checkpoint_dir.iterdir() if d.is_dir()]

        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        return checkpoints

    except Exception as e:
        warnings.warn(f"Failed to list checkpoints: {str(e)}")
        return []



def scan_files(
        directory: str | Path,
        pattern: str = "*",
        recursive: bool = False
) -> list[Path]:
    """Scan files that match the requested criteria."""
    directory = Path(directory)
    validate_path_exists(directory, path_type='dir')

    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def batch_process_files(
        file_list: list[Path],
        process_fn: Callable[[Path], Any],
        error_handling: str = 'skip'
) -> list[Any]:
    """Process files in batches."""
    results = []
    errors = []

    for file_path in file_list:
        try:
            result = process_fn(file_path)
            results.append(result)
        except Exception as e:
            if error_handling == 'raise':
                raise
            elif error_handling == 'skip':
                warnings.warn(f"Failed to process file ({file_path}): {str(e)}")
                continue
            elif error_handling == 'collect':
                errors.append({'file': file_path, 'error': str(e)})
                continue

    if error_handling == 'collect' and errors:
        warnings.warn(f"Batch processing complete; {len(errors)} files failed")
        results.append({'errors': errors})

    return results



def get_file_size(filepath: str | Path, human_readable: bool = True) -> int | str:
    """Return file size."""
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    size_bytes = filepath.stat().st_size

    if not human_readable:
        return size_bytes

    unit_scale = 1024.0

    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < unit_scale:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= unit_scale

    return f"{size_bytes:.2f} PB"


def copy_file_safe(
        src: str | Path,
        dst: str | Path,
        overwrite: bool = False
) -> Path:
    """Copy a file with safety checks."""
    src = Path(src)
    dst = Path(dst)

    validate_path_exists(src, path_type='file')
    ensure_dir(dst.parent)

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination file already exists: {dst}")

    shutil.copy2(src, dst)
    return dst


def create_backup(
        filepath: str | Path,
        backup_dir: str | Path | None = None,
        timestamp: bool = True
) -> Path:
    """Create backup."""
    filepath = Path(filepath)
    validate_path_exists(filepath, path_type='file')

    if backup_dir is None:
        backup_dir = filepath.parent / 'backups'
    else:
        backup_dir = Path(backup_dir)

    ensure_dir(backup_dir)

    if timestamp:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{filepath.stem}_{timestamp_str}{filepath.suffix}"
    else:
        backup_name = f"{filepath.stem}_backup{filepath.suffix}"

    backup_path = backup_dir / backup_name

    shutil.copy2(filepath, backup_path)

    return backup_path
