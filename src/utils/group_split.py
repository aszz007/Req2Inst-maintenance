"""Split datasets while keeping related inputs in the same partition."""

import random
from collections import defaultdict


def group_split_by_input(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dedup_identical: bool = True,
    input_key: str = "input",
    output_key: str = "output",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records while grouping identical inputs."""
    if dedup_identical:
        seen = set()
        cleaned: list[dict] = []
        for item in data:
            key = (item[input_key], item[output_key])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        n_removed = len(data) - len(cleaned)
        if n_removed:
            print(f"[group_split] Removed {n_removed} duplicate rows "
                  f"({len(data)} → {len(cleaned)})")
        data = cleaned

    groups: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        groups[item[input_key]].append(item)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n = len(group_keys)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)
    train_keys = group_keys[:n_train]
    val_keys = group_keys[n_train : n_train + n_val]
    test_keys = group_keys[n_train + n_val :]

    train_data = [item for k in train_keys for item in groups[k]]
    val_data = [item for k in val_keys for item in groups[k]]
    test_data = [item for k in test_keys for item in groups[k]]

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test_data)

    return train_data, val_data, test_data
