"""Create a reproducible sample of the text dataset."""

import argparse
import sys
import random
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

TEXT_DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / "text"

TRAIN_FIELD = "Low_Requirements"
OUTPUT_FIELD = "Instruction"

DATASET_TOTAL = 2472


def load_text_dataset(dataset_dir: Path) -> pd.DataFrame:
    """Load text dataset."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Text dataset directory not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the dataset directory: {dataset_dir}")

    print(f"Found {len(csv_files)} CSV file(s):")
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding="utf-8")
        print(f"  - {csv_file.name}: {len(df)} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined total: {len(combined)} rows\n")
    return combined


def sample_dataset(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample a dataset."""
    if n > len(df):
        print(f"[Warning] Requested sample size {n} rows; the dataset contains only {len(df)} rows; returning the full dataset.")
        return df.copy()

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def display_samples(samples: pd.DataFrame) -> None:
    """Display representative samples."""
    print("=" * 70)
    print(f"Random sample ({len(samples)} rows)")
    print("=" * 70)

    for idx, row in samples.iterrows():
        print(f"\n[Sample {idx + 1}]")
        print("-" * 50)
        if TRAIN_FIELD in row:
            req = str(row[TRAIN_FIELD]).strip()
            print(f"[{TRAIN_FIELD}]\n{req}")
        elif "High_Requirements" in row:
            high = str(row.get("High_Requirements", "")).strip()
            if high:
                print(f"[High_Requirements (display only; not used for training)]\n{high}")
        if OUTPUT_FIELD in row:
            instr = str(row[OUTPUT_FIELD]).strip()
            print(f"\n[{OUTPUT_FIELD}]\n{instr}")
        extra_cols = [
            c for c in row.index
            if c not in {TRAIN_FIELD, "High_Requirements", OUTPUT_FIELD}
        ]
        if extra_cols:
            for col in extra_cols:
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
        description="Randomly sample and display examples from the text dataset (data/dataset/text/)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of samples (default: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed; if omitted, results differ between runs (default: None)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save sampled results to the specified path (optional, e.g. outputs/samples/text_samples.csv)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help=f"Dataset directory path (default: {TEXT_DATASET_DIR})"
    )
    return parser.parse_args()


def run_sampling(n: int = 3, seed: int = None, output: str = None,
                 dataset_dir: str = None) -> pd.DataFrame:
    """Run sampling."""
    dir_path = Path(dataset_dir) if dataset_dir else TEXT_DATASET_DIR

    actual_seed = seed if seed is not None else random.randint(0, 99999)
    print(f"Random seed: {actual_seed}")

    print(f"Loading text dataset: {dir_path}")
    df = load_text_dataset(dir_path)

    samples = sample_dataset(df, n=n, seed=actual_seed)

    display_samples(samples)

    print(f"\nDataset columns: {list(df.columns)}")
    print(f"Dataset size: {len(df)} rows (framework reference: {DATASET_TOTAL} rows)")

    if output:
        save_samples(samples, output)

    return samples


if __name__ == "__main__":
    args = parse_args()
    run_sampling(
        n=args.n,
        seed=args.seed,
        output=args.output,
        dataset_dir=args.dataset_dir,
    )
