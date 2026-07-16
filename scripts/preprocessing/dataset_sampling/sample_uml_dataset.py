"""Create a reproducible sample of the UML dataset."""

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

UML_DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "uml" / "uml_dataset.csv"

DESCRIPTION_FIELD = "Description"
TRAIN_INPUT_FIELD = "Low_Requirements"
OUTPUT_FIELD = "Instruction"

DATASET_TOTAL = 1500


def load_uml_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load UML dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"UML数据集文件不存在: {dataset_path}")

    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"加载UML数据集: {dataset_path.name}")
    print(f"总计: {len(df)} 条，列: {list(df.columns)}\n")
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
        print(f"[警告] 请求采样 {n} 条，但数据集只有 {len(df)} 条，将返回全部数据。")
        return df.copy()

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def display_samples(samples: pd.DataFrame) -> None:
    """Display representative samples."""
    print("=" * 70)
    print(f"随机采样结果（共 {len(samples)} 条）")
    print("注意: UML Expert 输入是 JSON 文本描述，不是UML图")
    print("注意: UML Expert 和 General Expert 都使用此数据集")
    print("=" * 70)

    for idx, row in samples.iterrows():
        print(f"\n【样本 {idx + 1}】")
        print("-" * 50)

        if DESCRIPTION_FIELD in row:
            raw = str(row[DESCRIPTION_FIELD]).strip()
            description = extract_description(raw)
            if description:
                print(f"[Description（训练输入，提取自JSON）]\n{description}")
            else:
                print(f"[{DESCRIPTION_FIELD}（原始）]\n{raw}")
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
            print(f"\n[High_Requirements（仅展示，不用于训练）]\n{high}")

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
    print(f"\n采样结果已保存至: {out.resolve()}")


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="从UML数据集（data/dataset/uml/uml_dataset.csv）随机采样并展示样本"
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
        help="将采样结果保存到指定路径（可选，例如 outputs/samples/uml_samples.csv）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=f"数据集CSV路径（默认: {UML_DATASET_PATH}）"
    )
    return parser.parse_args()


def run_sampling(n: int = 3, seed: int = None, output: str = None,
                 dataset: str = None) -> pd.DataFrame:
    """Run sampling."""
    csv_path = Path(dataset) if dataset else UML_DATASET_PATH

    actual_seed = seed if seed is not None else random.randint(0, 99999)
    print(f"随机种子: {actual_seed}")

    df = load_uml_dataset(csv_path)

    samples = sample_dataset(df, n=n, seed=actual_seed)

    display_samples(samples)

    print(f"\n数据集规模: {len(df)} 条（框架参考: {DATASET_TOTAL} 条）")
    print(f"训练集/验证集/测试集参考: 1200 / 150 / 150")
    print(f"使用方: UML Expert + General Expert（共用同一数据集）")

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
