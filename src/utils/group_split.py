"""Split datasets while keeping related inputs in the same partition."""

import random
from collections import defaultdict
from typing import Dict, List, Tuple


def group_split_by_input(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dedup_identical: bool = True,
    input_key: str = "input",
    output_key: str = "output",
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split records while grouping identical inputs."""
    if dedup_identical:
        seen = set()
        cleaned: List[Dict] = []
        for item in data:
            key = (item[input_key], item[output_key])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        n_removed = len(data) - len(cleaned)
        if n_removed:
            print(f"[group_split] 已删除 {n_removed} 条完全重复行 "
                  f"({len(data)} → {len(cleaned)})")
        data = cleaned

    groups: Dict[str, List[Dict]] = defaultdict(list)
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
