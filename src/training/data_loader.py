"""Load, normalize, split, tokenize, and batch the project datasets."""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
import json
import pandas as pd
import random
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from config.settings import get_path_config, get_training_config
from src.utils.logger import get_logger

from models.prompt_templates.text_template import TextInstructionTemplate
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from models.prompt_templates.general_template import GeneralInstructionTemplate

logger = get_logger('training.data_loader')


def normalize_column_name(col_name: str) -> str:
    """Normalize column name."""
    col_name = col_name.replace('\ufeff', '').replace('\ufffe', '')
    col_name = col_name.strip()
    col_name = col_name.lower()
    col_name = ''.join(c for c in col_name if c.isprintable() or c.isspace())
    col_name = col_name.strip()

    return col_name


def detect_csv_encoding(filepath: Path) -> str:
    """Detect CSV encoding."""
    encodings = [
        'utf-8-sig',
        'utf-8',
        'cp1252',
        'gbk',
        'gb2312',
        'gb18030',
        'utf-16',
        'utf-16-le',       # UTF-16 Little Endian
        'utf-16-be',       # UTF-16 Big Endian
        'latin-1'
    ]

    for encoding in encodings:
        try:
            pd.read_csv(filepath, encoding=encoding, nrows=100)
            return encoding
        except:
            continue

    return 'latin-1'


class InstructionDataset(Dataset):
    """Expose tokenized instruction-training examples."""

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        """Initialize the instance."""
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        input_text = item.get('input_with_prompt', item['input'])
        output_text = item['output']

        MIN_VALID_LABELS = 10

        full_text = f"{input_text}{output_text}"

        encodings = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )

        prompt_encodings = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()

        labels = input_ids.clone()
        prompt_len = min(prompt_encodings['input_ids'].shape[1], labels.shape[0])
        labels[:prompt_len] = -100

        valid_labels = (labels != -100).sum().item()

        if valid_labels < MIN_VALID_LABELS:
            output_ids = self.tokenizer(
                output_text,
                truncation=True,
                max_length=self.max_length - MIN_VALID_LABELS,
                padding=False,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids'].squeeze()
            output_len = min(output_ids.shape[0], self.max_length - MIN_VALID_LABELS)
            output_ids = output_ids[:output_len]

            max_prompt_len = self.max_length - output_len
            prompt_ids = self.tokenizer(
                input_text,
                truncation=True,
                max_length=max_prompt_len,
                padding=False,
                return_tensors='pt'
            )['input_ids'].squeeze()

            input_ids = torch.cat([prompt_ids, output_ids])
            attention_mask = torch.ones(len(input_ids), dtype=torch.long)
            labels = input_ids.clone()
            labels[:len(prompt_ids)] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class InstructionDataCollator:
    """Pad and collate instruction-training batches."""

    def __init__(self, tokenizer, pad_to_multiple_of: int = None):
        """Initialize the instance."""
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        """Pad and collate a batch of tokenized features."""
        max_length = max(len(f['input_ids']) for f in features)

        if self.pad_to_multiple_of:
            max_length = (
                (max_length + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_token_id = self.tokenizer.pad_token_id

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature['input_ids']
            attention_mask = feature['attention_mask']
            labels = feature['labels']

            pad_len = max_length - len(input_ids)

            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=attention_mask.dtype)
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=labels.dtype)
            ])

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            'input_ids': torch.stack(batch_input_ids),
            'attention_mask': torch.stack(batch_attention_mask),
            'labels': torch.stack(batch_labels),
        }


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe columns."""
    column_mapping = {}
    for col in df.columns:
        normalized = normalize_column_name(col)
        column_mapping[col] = normalized

    df = df.rename(columns=column_mapping)

    return df


def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
    """Find column."""
    df_columns_lower = {col.lower(): col for col in df.columns}

    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]

    return None


def normalize_json_string(json_str: str) -> str:
    """Normalize JSON string."""
    try:
        obj = json.loads(json_str)
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except (json.JSONDecodeError, TypeError):
        return json_str


def filter_uml_json_positions(uml_data: dict) -> dict:
    """Filter token positions that belong to FlowChart JSON content."""
    if not isinstance(uml_data, dict):
        return uml_data

    import copy
    filtered_data = copy.deepcopy(uml_data)

    if 'actors' in filtered_data and isinstance(filtered_data['actors'], list):
        filtered_actors = []
        for actor in filtered_data['actors']:
            if isinstance(actor, dict):
                filtered_actor = {k: v for k, v in actor.items() if k != 'position'}
                filtered_actors.append(filtered_actor)
            else:
                filtered_actors.append(actor)
        filtered_data['actors'] = filtered_actors

    return filtered_data


class TextDatasetLoader:
    """Load text dataset data."""

    def __init__(self):
        """Initialize the instance."""
        path_cfg = get_path_config()
        self.dataset_dir = path_cfg.TEXT_DATASET_DIR
        logger.info(f"Initializing TextDatasetLoader; path: {self.dataset_dir}")

    def load_csv_files(self) -> List[Dict]:
        """Load CSV files."""
        all_data = []
        dataset_path = Path(self.dataset_dir)

        if not dataset_path.exists():
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            logger.error("Confirm that text_dataset.csv has been placed in this directory")
            logger.info("Total text records: 0")
            return all_data

        if not dataset_path.is_dir():
            logger.error(f"Path is not a directory: {dataset_path}")
            logger.info("Total text records: 0")
            return all_data

        csv_files = []
        for pattern in ("*.csv", "*.CSV", "*.Csv"):
            csv_files.extend(dataset_path.rglob(pattern))
        csv_files = list({f.resolve(): f for f in csv_files}.values())

        if len(csv_files) == 0:
            all_files = list(dataset_path.rglob("*"))
            logger.warning(f"No CSV files found in {dataset_path}")
            if all_files:
                logger.warning(f"Files currently present in the directory ({len(all_files)}):")
                for f in all_files[:20]:
                    logger.warning(f"  {f.relative_to(dataset_path)}")
            else:
                logger.warning(f"Directory is empty; place text_dataset.csv in: {dataset_path}")

        logger.info(f"Found {len(csv_files)} CSV files")

        for csv_file in csv_files:
            try:
                encoding = detect_csv_encoding(csv_file)
                df = pd.read_csv(csv_file, encoding=encoding)

                df = clean_dataframe_columns(df)

                logger.info(f"Reading {csv_file.name} with encoding '{encoding}'")

            except Exception:
                logger.error(f"Failed to load: {csv_file.name}")
                continue

            low_req_col = find_column(df, ['low_requirements', 'lowrequirements', 'low requirements'])
            instruction_col = find_column(df, ['instruction', 'instructions'])

            if not low_req_col or not instruction_col:
                logger.warning(f"Skipping file because required columns are missing: {csv_file.name}")
                logger.warning(f"  Actual columns: {list(df.columns)}")
                continue

            for _, row in df.iterrows():
                try:
                    low_req = str(row[low_req_col]).strip()
                    instruction = str(row[instruction_col]).strip()

                    if low_req and low_req != 'nan' and instruction and instruction != 'nan':
                        prompt = TextInstructionTemplate.build_prompt(low_req)

                        all_data.append({
                            'input': low_req,
                            'input_with_prompt': prompt,
                            'output': instruction,
                            'source': csv_file.stem
                        })
                except:
                    continue

            logger.info(f"Loaded {csv_file.name}: {len(df)} records")

        logger.info(f"Total text records: {len(all_data)}")
        return all_data


class ImageDatasetLoader:
    """Load image dataset data."""

    def __init__(self):
        """Initialize the instance."""
        path_cfg = get_path_config()
        self.dataset_csv = path_cfg.IMAGE_DATASET_CSV
        logger.info(f"Initializing ImageDatasetLoader; path: {self.dataset_csv}")

    def load_csv_file(self) -> List[Dict]:
        """Load CSV file."""
        all_data = []

        if not self.dataset_csv.exists():
            logger.warning(f"Image dataset file does not exist: {self.dataset_csv}")
            return all_data

        try:
            encoding = detect_csv_encoding(self.dataset_csv)
            df = pd.read_csv(self.dataset_csv, encoding=encoding)

            df = clean_dataframe_columns(df)

            logger.info(f"Reading image dataset with encoding '{encoding}'")

        except Exception:
            logger.error("Failed to load image dataset")
            return all_data

        desc_col = find_column(df, ['description', 'desc', 'descriptions'])
        instruction_col = find_column(df, ['instruction', 'instructions'])

        if not desc_col or not instruction_col:
            logger.error("CSV is missing a required column: Description or Instruction")
            logger.error(f"Actual columns: {list(df.columns)}")
            return all_data

        for idx, row in df.iterrows():
            try:
                desc_str = str(row[desc_col])

                try:
                    desc_json = json.loads(desc_str)
                    if 'description' in desc_json:
                        filtered_json = {
                            'description': desc_json.get('description', ''),
                            'details': desc_json.get('details', {})
                        }
                        description = normalize_json_string(json.dumps(filtered_json, ensure_ascii=False))
                    else:
                        logger.warning(f"Row {idx}: JSON does not contain a description field; skipping")
                        continue
                except (json.JSONDecodeError, TypeError, ValueError):
                    description = desc_str.strip()

                instruction = str(row[instruction_col]).strip()

                if description and description != 'nan' and instruction and instruction != 'nan':
                    prompt = ImageInstructionTemplate.build_prompt(description)

                    all_data.append({
                        'input': description,
                        'input_with_prompt': prompt,
                        'output': instruction,
                        'source': 'image_dataset'
                    })

            except:
                continue

        logger.info(f"Image dataset loaded: {len(all_data)} records")

        return all_data


class UMLDatasetLoader:
    """Load umldataset data."""

    def __init__(self):
        """Initialize the instance."""
        self.path_cfg = get_path_config()
        self.dataset_csv = self.path_cfg.UML_DATASET_CSV

        logger.info(f"Initializing FlowChart dataset loader - dataset: {self.dataset_csv}")

    def load_csv_file(self) -> List[Dict]:
        """Load CSV file."""
        csv_path = self.dataset_csv

        logger.info(f"Loading FlowChart dataset: {csv_path}")

        if not csv_path.exists():
            logger.error(f"FlowChart dataset file does not exist: {csv_path}")
            return []

        try:
            encoding = detect_csv_encoding(csv_path)
            logger.info(f"Detected encoding: {encoding}")

            df = pd.read_csv(csv_path, encoding=encoding)

            df.columns = [normalize_column_name(col) for col in df.columns]
            logger.info(f"Normalized columns: {list(df.columns)}")

            column_map = {
                'description': ['description', 'desc', 'uml_description', 'Description'],
                'instruction': ['instruction', 'Instruction', 'output', 'Output']
            }

            desc_col = None
            inst_col = None

            for standard_name, possible_names in column_map.items():
                possible_names_lower = [normalize_column_name(n) for n in possible_names]
                for col in df.columns:
                    if col in possible_names_lower:
                        if standard_name == 'description':
                            desc_col = col
                        elif standard_name == 'instruction':
                            inst_col = col
                        break

            if desc_col is None or inst_col is None:
                logger.error(f"Required columns were not found. Actual columns: {list(df.columns)}")
                logger.error("Required columns: description and instruction")
                return []

            logger.info(f"Using columns: description='{desc_col}', instruction='{inst_col}'")

            data_list = []
            for idx, row in df.iterrows():
                try:
                    description = row[desc_col]
                    instruction = row[inst_col]

                    if pd.isna(description) or pd.isna(instruction):
                        continue

                    if isinstance(description, str) and description.strip().startswith('{'):
                        try:
                            desc_json = json.loads(description)
                            if 'actors' in desc_json or 'use_cases' in desc_json:
                                filtered_json = filter_uml_json_positions(desc_json)
                                description = normalize_json_string(json.dumps(filtered_json, ensure_ascii=False))
                            elif 'description' in desc_json:
                                filtered_json = filter_uml_json_positions(desc_json)
                                description = normalize_json_string(json.dumps(filtered_json, ensure_ascii=False))
                            else:
                                logger.warning(f"Row {idx}: FlowChart JSON has an unexpected structure; skipping")
                                continue
                        except json.JSONDecodeError:
                            pass

                    prompt = UMLInstructionTemplate.build_prompt(str(description))

                    data_list.append({
                        'input': str(description),
                        'input_with_prompt': prompt,
                        'output': str(instruction),
                        'source': 'uml_dataset'
                    })

                except Exception as e:
                    logger.warning(f"Error while processing row {idx}: {e}")
                    continue

            logger.info(f"Loaded {len(data_list)} FlowChart records")
            return data_list

        except Exception as e:
            logger.error(f"Failed to load FlowChart dataset: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []


class GeneralDatasetLoader:
    """Load general dataset data."""

    def __init__(self, use_domain_templates: bool = False):
        """Initialize the instance."""
        self.use_domain_templates = use_domain_templates
        logger.info("Initializing GeneralDatasetLoader - loading text, image, and FlowChart data")
        logger.info(f"Template mode: {'domain-specific templates (lora_single)' if use_domain_templates else 'general template (general_expert)'}")

    def load_all_data(self) -> List[Dict]:
        """Load all data."""
        all_data = []

        logger.info("Loading text data...")
        text_loader = TextDatasetLoader()
        text_raw = text_loader.load_csv_files()

        for item in text_raw:
            if self.use_domain_templates:
                prompt = TextInstructionTemplate.build_prompt(item['input'])
            else:
                prompt = GeneralInstructionTemplate.build_prompt(
                    item['input'],
                    force_type='text'
                )
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'text'
            })

        logger.info(f"Text records: {len(text_raw)}")

        logger.info("Loading image data...")
        image_loader = ImageDatasetLoader()
        image_raw = image_loader.load_csv_file()

        for item in image_raw:
            if self.use_domain_templates:
                prompt = ImageInstructionTemplate.build_prompt(item['input'])
            else:
                prompt = GeneralInstructionTemplate.build_prompt(
                    item['input'],
                    force_type='image'
                )
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'image'
            })

        logger.info(f"Image records: {len(image_raw)}")

        logger.info("Loading FlowChart data...")
        uml_loader = UMLDatasetLoader()
        uml_raw = uml_loader.load_csv_file()

        for item in uml_raw:
            if self.use_domain_templates:
                prompt = UMLInstructionTemplate.build_prompt(str(item['input']))
            else:
                try:
                    uml_json = json.loads(item['input']) if isinstance(item['input'], str) else item['input']
                    uml_json = filter_uml_json_positions(uml_json)
                except (json.JSONDecodeError, TypeError):
                    uml_json = {"description": item['input'], "details": {"diagram_type": "use case diagram"}}
                prompt = GeneralInstructionTemplate.build_prompt(uml_json, force_type='uml')
            all_data.append({
                'input': item['input'],
                'input_with_prompt': prompt,
                'output': item['output'],
                'source': item['source'],
                'data_type': 'uml'
            })

        logger.info(f"FlowChart records: {len(uml_raw)}")

        logger.info(f"Total general-dataset records: {len(all_data)}")
        logger.info(f"  - Text: {len(text_raw)}")
        logger.info(f"  - Image: {len(image_raw)}")
        logger.info(f"  - FlowChart: {len(uml_raw)}")

        return all_data


def split_dataset(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset."""
    random.seed(seed)

    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]

    logger.info(f"Dataset split - train: {len(train_data)}, validation: {len(val_data)}, test: {len(test_data)}")

    return train_data, val_data, test_data


def split_dataset_for_expert(
    data: List[Dict],
    expert_type: str,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split dataset for expert."""
    data_size = len(data)

    if data_size < 500:
        train_ratio, val_ratio, test_ratio = 0.80, 0.15, 0.05
        logger.info(f"Small dataset ({data_size} records); using an 80:15:5 split")
    else:
        train_ratio, val_ratio, test_ratio = 0.80, 0.10, 0.10
        logger.info(f"Large dataset ({data_size} records); using an 80:10:10 split")

    return split_dataset(data, train_ratio, val_ratio, test_ratio, seed)


def create_dataloader(
    dataset: InstructionDataset,
    batch_size: int = None,
    shuffle: bool = True,
    num_workers: int = 8
) -> DataLoader:
    """Create dataloader."""
    if batch_size is None:
        train_cfg = get_training_config()
        batch_size = train_cfg.batch_size

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
