"""Create a reproducible sample of the image dataset."""

import argparse
import ast
import json
import os
import sys
import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "image" / "image_dataset.csv"

DESCRIPTION_FIELD = "Description"
TRAIN_INPUT_FIELD = "Low_Requirements"
OUTPUT_FIELD = "Instruction"

DATASET_TOTAL = 1000


def load_image_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load image dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Image dataset file not found: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"Loading image dataset: {dataset_path.name}")
    print(f"Total: {len(df)} rows, columns: {list(df.columns)}\n")
    return df


def extract_description(raw_value: str) -> str:
    """Extract description."""
    if not isinstance(raw_value, str) or not raw_value.strip():
        return ""

    raw = raw_value.strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(raw)
            if isinstance(parsed, dict) and "Description" in parsed:
                return str(parsed["Description"]).strip()
        except Exception:
            pass

    return raw


def sample_dataset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample a dataset."""
    if n > len(df):
        print(f"[Warning] Requested sample size {n} rows, but the dataset contains only {len(df)} rows; returning the full dataset.")
        return df.copy()

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def display_samples(samples: pd.DataFrame) -> None:
    """Display representative samples."""
    print("=" * 70)
    print(f"Random sample ({len(samples)} rows)")
    print("Note: Image Expert input is a JSON text description, not an image")
    print("=" * 70)

    for idx, row in samples.iterrows():
        print(f"\n[Sample {idx + 1}]")
        print("-" * 50)

        if DESCRIPTION_FIELD in row:
            raw = str(row[DESCRIPTION_FIELD]).strip()
            description = extract_description(raw)
            if description:
                print(f"[Description (training input, extracted from JSON)]\n{description}")
            else:
                print(f"[{DESCRIPTION_FIELD} (raw)]\n{raw}")
        elif TRAIN_INPUT_FIELD in row:
            val = str(row[TRAIN_INPUT_FIELD]).strip()
            if val and val != "nan":
                print(f"[{TRAIN_INPUT_FIELD}]\n{val}")

        if TRAIN_INPUT_FIELD in row and DESCRIPTION_FIELD in row:
            val = str(row.get(TRAIN_INPUT_FIELD, "")).strip()
            if val and val != "nan":
                print(f"\n[{TRAIN_INPUT_FIELD}]\n{val}")

        if OUTPUT_FIELD in row:
            instr = str(row[OUTPUT_FIELD]).strip()
            if instr and instr != "nan":
                print(f"\n[{OUTPUT_FIELD}]\n{instr}")

        high = str(row.get("High_Requirements", "")).strip()
        if high and high != "nan":
            print(f"\n[High_Requirements (display only; not used for training)]\n{high}")

        skip_cols = {DESCRIPTION_FIELD, TRAIN_INPUT_FIELD, OUTPUT_FIELD, "High_Requirements"}
        for col in row.index:
            if col not in skip_cols:
                val = str(row[col]).strip()
                if val and val != "nan":
                    print(f"\n[{col}]: {val}")

        print("-" * 50)


def save_samples(samples: pd.DataFrame, output_path: str) -> None:
    """Save samples."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(out, index=False, encoding="utf-8")
    print(f"\nSample saved to: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="从图像数据集（data/dataset/image/image_dataset.csv）随机采样并展示样本"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="采样数量（默认: 3）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，不指定时每次结果不同（默认: None）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="将采样结果保存到指定路径（可选，例如 outputs/samples/image_samples.csv）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"数据集CSV路径（默认: {IMAGE_DATASET_PATH}）"
    )
    return parser.parse_args()


def run_sampling(n: int = 3, seed: int = None, output: str = None,
                 dataset: str = None) -> pd.DataFrame:
    """Run sampling."""
    csv_path = Path(dataset) if dataset else IMAGE_DATASET_PATH

    actual_seed = seed if seed is not None else random.randint(0, 99999)
    print(f"Random seed: {actual_seed}")

    df = load_image_dataset(csv_path)

    samples = sample_dataset(df, n=n, seed=actual_seed)

    display_samples(samples)

    print(f"\nDataset size: {len(df)} rows (framework reference: {DATASET_TOTAL} rows)")
    print(f"Train/validation/test reference: 800 / 100 / 100")

    if output:
        save_samples(samples, output)

    return samples


if __name__ == "__main__":
    args = parse_args()
    run_sampling(
        n=args.n,
        seed=args.seed,
        output=args.output,
        dataset=args.dataset,
    )
