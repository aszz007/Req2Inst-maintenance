"""Calculate token-length distributions for project datasets."""

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

print(f"Project root: {project_root}")

try:
    from config.settings import get_path_config
    from models.prompt_templates.text_template import TextInstructionTemplate
    from models.prompt_templates.image_template import ImageInstructionTemplate
    from models.prompt_templates.uml_template import UMLInstructionTemplate

    path_cfg = get_path_config()

except ImportError as e:
    print("\n[FATAL ERROR] Failed to import project configuration or modules!")
    print(f"Error details: {e}")
    sys.exit(1)



def get_tokenizer():
    """Return tokenizer."""
    model_path = path_cfg.get_text_model_path()
    print(f"\n[Tokenizer] Loading model from: {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print(f"[Tokenizer] Loaded successfully; vocabulary size: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        print(f"[Tokenizer] Failed to load: {e}")
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
    desc = f"Processing {source_name}" if source_name else f"Processing {expert_type}"

    input_col = None
    output_col = None

    if expert_type == 'text':
        input_col = next((c for c in df.columns if c.lower() in ['low_requirements', 'input', 'lowrequirements']), None)
        output_col = next((c for c in df.columns if c.lower() in ['instruction', 'output', 'instructions']), None)
    elif expert_type in ('image', 'uml'):
        input_col = next((c for c in df.columns if c.lower() in ['description', 'input']), None)
        output_col = next((c for c in df.columns if c.lower() in ['instruction', 'output']), None)

    if not input_col or not output_col:
        return []

    first_valid_row = df.dropna(subset=[input_col]).iloc[0] if not df.empty else None
    if first_valid_row is not None:
        preview_text = str(first_valid_row[input_col])[:60].replace('\n', ' ')
        print(f"  -> [Preview {source_name}] ({input_col}): {preview_text}...")

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
        print(f"[{name}] No valid data found or the file is empty")
        return

    lengths = np.array(lengths)
    print(f"\n===== {name} length statistics (tokens) =====")
    print(f"Sample count: {len(lengths)}")
    print(f"Minimum length: {np.min(lengths)}")
    print(f"Maximum length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.2f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.2f}")
    print("======================================\n")



def main():
    """Run the command-line entry point."""
    tokenizer = get_tokenizer()
    if not tokenizer:
        return

    print("\nComputing dataset lengths for all experts (including content previews)...\n")

    # 1. Text Expert
    text_dir = Path(path_cfg.TEXT_DATASET_DIR)

    if text_dir.exists():
        all_text_lengths = []
        csv_files = list(text_dir.glob("*.csv"))

        print(f"[Text] Found {len(csv_files)} files")
        for csv_file in csv_files:
            try:
                df, used_encoding = read_csv_safely(csv_file)

                if df is not None:
                    print(f"[Text] Read {csv_file.name} successfully | encoding: {used_encoding}")

                    file_lengths = calculate_lengths_from_df(
                        df, tokenizer, 'text', source_name=csv_file.name
                    )
                    all_text_lengths.extend(file_lengths)
                else:
                    print(f"[Text] Error: could not read {csv_file.name}; all encoding attempts failed")
            except Exception as e:
                print(f"[Text] Unexpected error while processing {csv_file.name}: {e}")

        print_stats("Text Expert (all files combined)", all_text_lengths)
    else:
        print(f"[Text] Directory not found: {text_dir}")

    # 2. Image Expert
    image_path = Path(path_cfg.IMAGE_DATASET_CSV)
    if image_path.exists():
        df, enc = read_csv_safely(image_path)
        if df is not None:
            print(f"[Image] Read successfully | encoding: {enc}")
            lengths = calculate_lengths_from_df(df, tokenizer, 'image')
            print_stats("Image Expert", lengths)
    else:
        print(f"[Image] File not found: {image_path}")

    # 3. FlowChart Expert
    uml_path = Path(path_cfg.UML_DATASET_CSV)
    if uml_path.exists():
        df, enc = read_csv_safely(uml_path)
        if df is not None:
            print(f"[FlowChart] Read successfully | encoding: {enc}")
            lengths = calculate_lengths_from_df(df, tokenizer, 'uml')
            print_stats("FlowChart Expert", lengths)
    else:
        print(f"[FlowChart] File not found: {uml_path}")


if __name__ == "__main__":
    main()
