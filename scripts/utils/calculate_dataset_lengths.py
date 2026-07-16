"""Calculate token-length distributions for project datasets."""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

print(f"项目根目录: {project_root}")

try:
    from config.settings import get_path_config
    from models.prompt_templates.text_template import TextInstructionTemplate
    from models.prompt_templates.image_template import ImageInstructionTemplate
    from models.prompt_templates.uml_template import UMLInstructionTemplate

    path_cfg = get_path_config()

except ImportError as e:
    print("\n[致命错误] 无法导入项目配置或模块！")
    print(f"错误信息: {e}")
    sys.exit(1)



def get_tokenizer():
    """Return tokenizer."""
    model_path = path_cfg.get_text_model_path()
    print(f"\n[Tokenizer] 正在加载模型路径: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"[Tokenizer] 加载成功，词表大小: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        print(f"[Tokenizer] 加载失败: {e}")
        return None


def read_csv_safely(file_path):
    """Read a CSV file with encoding fallbacks."""
    encodings = ['utf-8-sig', 'utf-8', 'cp1252', 'gbk', 'gb18030', 'latin-1']

    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)

            return df, enc
        except Exception:
            continue

    return None, None


def calculate_lengths_from_df(df, tokenizer, expert_type, source_name=""):
    """Calculate lengths from df."""
    lengths = []
    desc = f"处理 {source_name}" if source_name else f"处理 {expert_type}"

    input_col = None
    output_col = None

    if expert_type == 'text':
        input_col = next((c for c in df.columns if c.lower() in ['low_requirements', 'input', 'lowrequirements']), None)
        output_col = next((c for c in df.columns if c.lower() in ['instruction', 'output', 'instructions']), None)
    elif expert_type == 'image' or expert_type == 'uml':
        input_col = next((c for c in df.columns if c.lower() in ['description', 'input']), None)
        output_col = next((c for c in df.columns if c.lower() in ['instruction', 'output']), None)

    if not input_col or not output_col:
        return []

    first_valid_row = df.dropna(subset=[input_col]).iloc[0] if not df.empty else None
    if first_valid_row is not None:
        preview_text = str(first_valid_row[input_col])[:60].replace('\n', ' ')
        print(f"  -> [预览 {source_name}] ({input_col}): {preview_text}...")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc, leave=False):
        try:
            input_raw = str(row[input_col])
            output_raw = str(row[output_col])

            if pd.isna(input_raw) or pd.isna(output_raw) or input_raw.lower() == 'nan':
                continue

            prompt = ""
            if expert_type == 'text':
                prompt = TextInstructionTemplate.build_prompt(input_raw)
            elif expert_type == 'image':
                prompt = ImageInstructionTemplate.build_prompt(input_raw)
            elif expert_type == 'uml':
                prompt = UMLInstructionTemplate.build_prompt(input_raw)

            full_text = prompt + output_raw
            tokens = tokenizer(full_text)['input_ids']
            lengths.append(len(tokens))

        except Exception:
            continue

    return lengths


def print_stats(name, lengths):
    """Print stats."""
    if not lengths:
        print(f"[{name}] 未找到有效数据或文件为空")
        return

    lengths = np.array(lengths)
    print(f"\n===== {name} 长度统计 (Tokens) =====")
    print(f"样本数量 : {len(lengths)}")
    print(f"最小长度 : {np.min(lengths)}")
    print(f"最大长度 : {np.max(lengths)}")
    print(f"平均长度 : {np.mean(lengths):.2f}")
    print(f"95%分位  : {np.percentile(lengths, 95):.2f}")
    print(f"99%分位  : {np.percentile(lengths, 99):.2f}")
    print("======================================\n")



def main():
    """Run the command-line entry point."""
    tokenizer = get_tokenizer()
    if not tokenizer:
        return

    print("\n开始统计各专家数据集长度 (含内容预览检测)...\n")

    # 1. Text Expert
    text_dir = Path(path_cfg.TEXT_DATASET_DIR)

    if text_dir.exists():
        all_text_lengths = []
        csv_files = list(text_dir.glob("*.csv"))

        print(f"[Text] 发现 {len(csv_files)} 个文件")
        for csv_file in csv_files:
            try:
                df, used_encoding = read_csv_safely(csv_file)

                if df is not None:
                    print(f"[Text] 读取 {csv_file.name} 成功 | 编码: {used_encoding}")

                    file_lengths = calculate_lengths_from_df(
                        df, tokenizer, 'text', source_name=csv_file.name
                    )
                    all_text_lengths.extend(file_lengths)
                else:
                    print(f"[Text] 错误: 无法读取 {csv_file.name}，所有编码尝试均失败")
            except Exception as e:
                print(f"[Text] 处理 {csv_file.name} 时发生未知错误: {e}")

        print_stats("Text Expert (所有文件合并)", all_text_lengths)
    else:
        print(f"[Text] 目录不存在: {text_dir}")

    # 2. Image Expert
    image_path = Path(path_cfg.IMAGE_DATASET_CSV)
    if image_path.exists():
        df, enc = read_csv_safely(image_path)
        if df is not None:
            print(f"[Image] 读取成功 | 编码: {enc}")
            lengths = calculate_lengths_from_df(df, tokenizer, 'image')
            print_stats("Image Expert", lengths)
    else:
        print(f"[Image] 文件不存在: {image_path}")

    # 3. UML Expert
    uml_path = Path(path_cfg.UML_DATASET_CSV)
    if uml_path.exists():
        df, enc = read_csv_safely(uml_path)
        if df is not None:
            print(f"[UML] 读取成功 | 编码: {enc}")
            lengths = calculate_lengths_from_df(df, tokenizer, 'uml')
            print_stats("UML Expert", lengths)
    else:
        print(f"[UML] 文件不存在: {uml_path}")


if __name__ == "__main__":
    main()
