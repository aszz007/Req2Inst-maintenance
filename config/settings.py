"""Centralize model, path, training, inference, and device configuration."""

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Store text-model configuration."""

    version: str = "qwen3_8b"

    def get_model_name(self) -> str:
        """Return model name."""
        return "Qwen3-8B"

    def get_model_size(self) -> str:
        """Return model size."""
        return "8B"


@dataclass
class VisionModelConfig:
    """Store vision-model configuration."""

    version: str = "qwen3"

    SUPPORTED_VERSIONS: List[str] = None

    def __post_init__(self):
        """Finalize dataclass initialization."""
        if self.SUPPORTED_VERSIONS is None:
            self.SUPPORTED_VERSIONS = ["qwen3"]

        if self.version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported vision model version: {self.version}. "
                f"Supported versions: {self.SUPPORTED_VERSIONS}"
            )

    def get_model_name(self) -> str:
        """Return model name."""
        return "Qwen3-VL-8B-Instruct"

    def get_model_size(self) -> str:
        """Return model size."""
        return "8B"

class PathConfig:
    """Store project path configuration."""

    def __init__(self):
        """Initialize the instance."""
        self.PROJECT_ROOT = Path(__file__).parent.parent.resolve()

        self.BASE_MODELS_DIR = self.PROJECT_ROOT / "base_models"

        self.QWEN3_8B_PATH = (
            self.BASE_MODELS_DIR / "qwen3-8B" / "Qwen" / "Qwen3-8B"
        )

        self.TEXT_MODEL_PATHS = {
            'qwen3_8b': self.QWEN3_8B_PATH,
        }

        self.QWEN_VL_3_PATH = (
                self.BASE_MODELS_DIR / "qwen3-VL-8B" / "qwen" / "Qwen3-VL-8B-Instruct"
        )

        self.VISION_MODEL_PATHS = {
            'qwen3': self.QWEN_VL_3_PATH,
        }

        self.DATA_DIR = self.PROJECT_ROOT / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.INTERIM_DATA_DIR = self.DATA_DIR / "interim"

        self.RAW_IMAGE_DIR = self.RAW_DATA_DIR / "image"
        self.RAW_TEXT_DIR = self.RAW_DATA_DIR / "text"
        self.RAW_UML_DIR = self.RAW_DATA_DIR / "uml"

        self.COCO_1K_DIR = self.RAW_IMAGE_DIR / "coco_1k"
        self.ROBOFLOW_UML_DIR = self.RAW_UML_DIR / "roboflow_uml"
        self.MDPI_UML_DIR = self.RAW_UML_DIR / "mdpi_uml"
        self.PLANT_UML_DIR = self.RAW_UML_DIR / "plant_uml"

        self.INTERIM_IMAGE_DIR = self.INTERIM_DATA_DIR / "image"
        self.INTERIM_TEXT_DIR = self.INTERIM_DATA_DIR / "text"
        self.INTERIM_UML_DIR = self.INTERIM_DATA_DIR / "uml"

        self.DATASET_DIR = self.DATA_DIR / "dataset"
        self.TEXT_DATASET_DIR = self.DATASET_DIR / "text"
        self.IMAGE_DATASET_DIR = self.DATASET_DIR / "image"
        self.UML_DATASET_DIR = self.DATASET_DIR / "uml"
        self.GENERAL_DATASET_DIR = self.DATASET_DIR / "general"

        self.IMAGE_DATASET_CSV = self.IMAGE_DATASET_DIR / "image_dataset.csv"

        self.UML_DATASET_CSV = self.UML_DATASET_DIR / "uml_dataset.csv"


        self.TEXT_DATASET_FILES = {
            'CCHIT': self.TEXT_DATASET_DIR / "CCHIT_dataset.csv",
            'CM1': self.TEXT_DATASET_DIR / "CM1_dataset.csv",
            'GANNT': self.TEXT_DATASET_DIR / "GANNT_dataset.csv",
            'InfusionPump': self.TEXT_DATASET_DIR / "InfusionPump_dataset.csv",
            'Modis': self.TEXT_DATASET_DIR / "Modis_dataset.csv",
            'WARC': self.TEXT_DATASET_DIR / "WARC_dataset.csv"
        }

        self.INPUTS_DIR = self.PROJECT_ROOT / "inputs"
        self.INPUT_TEXT_DIR = self.INPUTS_DIR / "text"
        self.INPUT_IMAGE_DIR = self.INPUTS_DIR / "image"
        self.INPUT_UML_DIR = self.INPUTS_DIR / "uml"

        self.LORA_WEIGHTS_DIR = self.PROJECT_ROOT / "lora_weights"
        self.EXPERTS_DIR = self.LORA_WEIGHTS_DIR / "experts"

        self.TEXT_EXPERT_WEIGHTS = self.EXPERTS_DIR / "text_expert"

        self.IMAGE_EXPERT_WEIGHTS = self.EXPERTS_DIR / "image_expert"

        self.UML_EXPERT_WEIGHTS = self.EXPERTS_DIR / "uml_expert"

        self.GENERAL_EXPERT_WEIGHTS = self.EXPERTS_DIR / "general_expert"

        self.CHECKPOINTS_DIR = self.PROJECT_ROOT / "checkpoints"

        self.LORA_MOE_CKPTS = {
            'text': self.CHECKPOINTS_DIR / "lora_moe" / "text_expert",
            'image': self.CHECKPOINTS_DIR / "lora_moe" / "image_expert",
            'uml': self.CHECKPOINTS_DIR / "lora_moe" / "uml_expert",
            'general': self.CHECKPOINTS_DIR / "lora_moe" / "general_expert",
        }

        self.LORA_SINGLE_CKPT = self.CHECKPOINTS_DIR / "lora_single" / "unified_expert"

        self.PTUNING_CKPTS = {
            'text': self.CHECKPOINTS_DIR / "p_tuning" / "text_expert",
            'image': self.CHECKPOINTS_DIR / "p_tuning" / "image_expert",
            'uml': self.CHECKPOINTS_DIR / "p_tuning" / "uml_expert",
            'general': self.CHECKPOINTS_DIR / "p_tuning" / "general_expert",
        }

        self.PROMPT_TUNING_CKPTS = {
            'text': self.CHECKPOINTS_DIR / "prompt_tuning" / "text_expert",
            'image': self.CHECKPOINTS_DIR / "prompt_tuning" / "image_expert",
            'uml': self.CHECKPOINTS_DIR / "prompt_tuning" / "uml_expert",
            'general': self.CHECKPOINTS_DIR / "prompt_tuning" / "general_expert",
        }

        self.FULL_FINETUNING_CKPTS = {
            'text': self.CHECKPOINTS_DIR / "full_finetuning" / "text_expert",
            'image': self.CHECKPOINTS_DIR / "full_finetuning" / "image_expert",
            'uml': self.CHECKPOINTS_DIR / "full_finetuning" / "uml_expert",
            'general': self.CHECKPOINTS_DIR / "full_finetuning" / "general_expert",
        }

        self.EXPERT_LORA_PATHS = {
            # Text Expert
            'text': self.LORA_MOE_CKPTS['text'],
            'text_expert': self.LORA_MOE_CKPTS['text'],

            # Image Expert
            'image': self.LORA_MOE_CKPTS['image'],
            'image_expert': self.LORA_MOE_CKPTS['image'],

            # UML Expert
            'uml': self.LORA_MOE_CKPTS['uml'],
            'uml_expert': self.LORA_MOE_CKPTS['uml'],

            # General Expert
            'general': self.LORA_MOE_CKPTS['general'],
            'general_expert': self.LORA_MOE_CKPTS['general'],
        }

        self.TEXT_EXPERT_CKPT = self.LORA_MOE_CKPTS['text']
        self.IMAGE_EXPERT_CKPT = self.LORA_MOE_CKPTS['image']
        self.UML_EXPERT_CKPT = self.LORA_MOE_CKPTS['uml']

        self.OUTPUTS_DIR = self.PROJECT_ROOT / "outputs"
        self.GENERATED_INSTRUCTIONS_DIR = self.OUTPUTS_DIR / "generated_instructions"
        self.RECOGNITION_RESULTS_DIR = self.OUTPUTS_DIR / "recognition_results"
        self.IMAGE_RECOGNITION_DIR = self.RECOGNITION_RESULTS_DIR / "image"
        self.UML_RECOGNITION_DIR = self.RECOGNITION_RESULTS_DIR / "uml"
        self.EVALUATIONS_DIR = self.OUTPUTS_DIR / "evaluations"
        self.METRICS_DIR = self.EVALUATIONS_DIR / "metrics"
        self.COMPARISONS_DIR = self.EVALUATIONS_DIR / "comparisons"
        self.REPORTS_DIR = self.OUTPUTS_DIR / "reports"

        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        self.TRAINING_LOGS_DIR = self.LOGS_DIR / "training"
        self.INFERENCE_LOGS_DIR = self.LOGS_DIR / "inference"
        self.PREPROCESSING_LOGS_DIR = self.LOGS_DIR / "preprocessing"

    def get_text_model_path(self, version: str = None) -> Path:
        """Return text model path."""
        if version is None:
            model_cfg = get_model_config()
            version = model_cfg.version

        if version not in self.TEXT_MODEL_PATHS:
            raise ValueError(
                f"Unsupported text model version: {version}. "
                f"Supported versions: {list(self.TEXT_MODEL_PATHS.keys())}"
            )

        return self.TEXT_MODEL_PATHS[version]

    def get_vision_model_path(self, version: str = None) -> Path:
        """Return vision model path."""
        if version is None:
            vision_cfg = get_vision_model_config()
            version = vision_cfg.version

        if version not in self.VISION_MODEL_PATHS:
            raise ValueError(
                f"Unsupported vision model version: {version}. "
                f"Supported versions: {list(self.VISION_MODEL_PATHS.keys())}"
            )

        return self.VISION_MODEL_PATHS[version]

    def get_expert_weight_path(self, expert_name: str, method: str = 'lora_moe') -> Path:
        """Return expert weight path."""
        base_name = expert_name.replace('_expert', '')

        method_map = {
            'lora_moe': self.LORA_MOE_CKPTS,
            'p_tuning': self.PTUNING_CKPTS,
            'prompt_tuning': self.PROMPT_TUNING_CKPTS,
            'full_finetuning': self.FULL_FINETUNING_CKPTS,
        }

        if method == 'lora_single':
            return self.LORA_SINGLE_CKPT

        if method in method_map and base_name in method_map[method]:
            return method_map[method][base_name]

        if base_name in self.EXPERT_LORA_PATHS:
            return self.EXPERT_LORA_PATHS[base_name]

        return self.CHECKPOINTS_DIR / 'lora_moe' / f'{base_name}_expert'

    def get_checkpoint_path(self, expert_name: str) -> Path:
        """Return checkpoint path."""
        return self.CHECKPOINTS_DIR / f"{expert_name}_training"

    def create_directories(self):
        """Create directories."""
        dirs = [
            self.LORA_WEIGHTS_DIR,
            self.EXPERTS_DIR,
            self.CHECKPOINTS_DIR,
            self.TEXT_DATASET_DIR,
            self.IMAGE_DATASET_DIR,
            self.UML_DATASET_DIR,
            self.GENERAL_DATASET_DIR,
            self.INTERIM_IMAGE_DIR,
            self.INTERIM_TEXT_DIR,
            self.INTERIM_UML_DIR,
            self.INPUT_TEXT_DIR,
            self.INPUT_IMAGE_DIR,
            self.INPUT_UML_DIR,
            self.GENERATED_INSTRUCTIONS_DIR,
            self.IMAGE_RECOGNITION_DIR,
            self.UML_RECOGNITION_DIR,
            self.METRICS_DIR,
            self.COMPARISONS_DIR,
            self.REPORTS_DIR,
            self.TRAINING_LOGS_DIR,
            self.INFERENCE_LOGS_DIR,
            self.PREPROCESSING_LOGS_DIR,
        ]

        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

        print(f"✓ Created {len(dirs)} required directories")


@dataclass
class LoRAConfig:
    """Store LoRA adaptation configuration."""

    rank: int = 64

    alpha: int = 128

    dropout: float = 0.05

    target_modules: List[str] = None

    task_type: str = "CAUSAL_LM"

    bias: str = "none"

    def __post_init__(self):
        """Finalize dataclass initialization."""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]

    @classmethod
    def get_conservative_config(cls):
        """Return conservative config."""
        return cls(rank=64, alpha=128, dropout=0.05)

    @classmethod
    def get_aggressive_config(cls):
        """Return aggressive config."""
        return cls(rank=16, alpha=32, dropout=0.1)


@dataclass
class TrainingConfig:
    """Store shared training configuration."""

    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_epochs: int = 3
    learning_rate: float = 2e-4

    optimizer: str = "adamw_torch"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    lr_scheduler_type: str = "cosine"

    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3

    evaluation_strategy: str = "steps"
    eval_steps: int = 100

    fp16: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    max_seq_length: int = 2048

    text_train_ratio: float = 0.8
    text_val_ratio: float = 0.1
    text_test_ratio: float = 0.1

    image_train_ratio: float = 0.8
    image_val_ratio: float = 0.1
    image_test_ratio: float = 0.1

    uml_train_ratio: float = 0.8
    uml_val_ratio: float = 0.1
    uml_test_ratio: float = 0.1

@dataclass
class TrainingConfig4090:
    """Store RTX 4090-specific training configuration."""

    batch_size = 8
    gradient_accumulation_steps = 2
    num_epochs = 3
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    max_seq_length = 2048

    use_flash_attention = True
    bf16 = True
    tf32 = True

    dataloader_num_workers = 8
    dataloader_pin_memory = True
    dataloader_prefetch_factor = 4

    gradient_checkpointing = False
    max_grad_norm = 1.0

    save_strategy = "epoch"
    save_total_limit = 2
    evaluation_strategy = "epoch"
    logging_steps = 5

    optimizer_type = "adamw_torch_fused"
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8


@dataclass
class PTuningV2Config:
    """Store P-Tuning v2 configuration."""

    num_virtual_tokens: int = 20

    encoder_hidden_size: int = 64
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1

    task_type: str = "CAUSAL_LM"

    prefix_projection: bool = True

    @classmethod
    def get_default_config(cls):
        """Return default config."""
        return cls(num_virtual_tokens=20, encoder_hidden_size=64)

    @classmethod
    def get_large_config(cls):
        """Return large config."""
        return cls(num_virtual_tokens=30, encoder_hidden_size=128)

    @classmethod
    def get_emergency_config(cls):
        """Return emergency config."""
        return cls(num_virtual_tokens=15, encoder_hidden_size=32)


@dataclass
class PromptTuningConfig:
    """Store prompt-tuning configuration."""

    num_virtual_tokens: int = 10

    prompt_tuning_init: str = "RANDOM"
    prompt_tuning_init_text: Optional[str] = None

    task_type: str = "CAUSAL_LM"

    token_dim: Optional[int] = None

    @classmethod
    def get_default_config(cls):
        """Return default config."""
        return cls(num_virtual_tokens=10, prompt_tuning_init="RANDOM")

    @classmethod
    def get_large_config(cls):
        """Return large config."""
        return cls(num_virtual_tokens=20, prompt_tuning_init="RANDOM")


@dataclass
class FullFineTuningConfig:
    """Store full-finetuning configuration."""

    use_high_rank_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    target_modules: List[str] = None

    learning_rate: float = 1e-4
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    max_grad_norm: float = 0.5

    batch_size: int = 1
    gradient_accumulation_steps: int = 16

    max_seq_length: int = 2048

    def __post_init__(self):
        """Finalize dataclass initialization."""
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
            ]

    @classmethod
    def get_default_config(cls):
        """Return default config."""
        return cls(
            lora_rank=64,
            lora_alpha=128,
            max_seq_length=2048,
            gradient_accumulation_steps=16,
            num_epochs=3
        )

    @classmethod
    def get_memory_efficient_config(cls):
        """Return memory efficient config."""
        return cls(
            lora_rank=64,
            lora_alpha=128,
            max_seq_length=1536,
            gradient_accumulation_steps=16,
            num_epochs=3
        )

    @classmethod
    def get_balanced_config(cls):
        """Return balanced config."""
        return cls(
            lora_rank=12,
            lora_alpha=24,
            max_seq_length=1536,
            gradient_accumulation_steps=16,
            num_epochs=3
        )

    @classmethod
    def get_max_quality_config(cls):
        """Return max quality config."""
        return cls(
            lora_rank=32,
            lora_alpha=64,
            max_seq_length=1536,
            gradient_accumulation_steps=8,
            num_epochs=3
        )

@dataclass
class InferenceConfig:
    """Store inference configuration."""

    temperature: float = 0.3

    # Nucleus sampling
    top_p: float = 0.85

    # Top-k sampling
    top_k: int = 40

    repetition_penalty: float = 1.15

    max_new_tokens: int = 512


@dataclass
class DeviceConfig:
    """Store device and GPU-tier configuration."""

    device: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    is_high_end_gpu: bool = False
    enable_streaming: bool = False

    def __post_init__(self):
        """Finalize dataclass initialization."""
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                self.gpu_name = torch.cuda.get_device_name(0)
                self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

                self.is_high_end_gpu = self._detect_high_end_gpu()

                print(f"[Device] Using GPU: {self.gpu_name}")
                print(f"[Device] GPU memory: {self.gpu_memory_gb:.2f}GB")
                print(f"[Device] High-end GPU mode: {'enabled' if self.is_high_end_gpu else 'disabled'}")
            else:
                self.device = "cpu"
                print("[Device] Using CPU")

    def _detect_high_end_gpu(self) -> bool:
        """Detect high end GPU."""
        if not self.gpu_name:
            return False

        gpu_lower = self.gpu_name.lower()

        high_end_keywords = [
            '4090', '4080',
            'a100', 'h100', 'a6000',
            'v100',
        ]

        for keyword in high_end_keywords:
            if keyword in gpu_lower:
                return True

        if self.gpu_memory_gb and self.gpu_memory_gb >= 20.0:
            return True

        return False

    def get_device(self) -> str:
        """Return device."""
        return self.device

    def get_gpu_info(self) -> dict:
        """Return GPU info."""
        return {
            'device': self.device,
            'gpu_name': self.gpu_name,
            'gpu_memory_gb': self.gpu_memory_gb,
            'is_high_end_gpu': self.is_high_end_gpu
        }

    def should_use_quantization(self) -> bool:
        """Return whether quantization should be enabled."""
        return not self.is_high_end_gpu

    def get_gpu_tier(self) -> str:
        """Return GPU tier."""
        if not torch.cuda.is_available():
            return 'low'

        if self.is_high_end_gpu:
            return 'high'

        if self.gpu_memory_gb and 7.5 <= self.gpu_memory_gb <= 16.0:
            return 'mid'

        return 'low'

    def get_generation_config(self, task_type: str = 'uml') -> dict:
        """Return generation config."""
        tier = self.get_gpu_tier()

        uml_configs = {
            'high': {
                'max_new_tokens': 4096,
                'batch_size': 4,
                'temperature': 0.3,
                'top_p': 0.85,
                'use_cache': True,
            },
            'mid': {
                'max_new_tokens': 2048,
                'batch_size': 2,
                'temperature': 0.5,
                'top_p': 0.9,
                'use_cache': True,
            },
            'low': {
                'max_new_tokens': 1024,
                'batch_size': 1,
                'temperature': 0.6,
                'top_p': 0.95,
                'use_cache': True,
            }
        }

        image_configs = {
            'high': {
                'max_new_tokens': 512,
                'batch_size': 4,
                'temperature': 0.3,
                'top_p': 0.85,
                'use_cache': True,
            },
            'mid': {
                'max_new_tokens': 200,
                'batch_size': 2,
                'temperature': 0.5,
                'top_p': 0.9,
                'use_cache': True,
            },
            'low': {
                'max_new_tokens': 150,
                'batch_size': 1,
                'temperature': 0.6,
                'top_p': 0.95,
                'use_cache': True,
            }
        }

        if task_type == 'uml':
            return uml_configs.get(tier, uml_configs['mid'])
        elif task_type == 'image':
            return image_configs.get(tier, image_configs['mid'])
        else:
            return uml_configs.get(tier, uml_configs['mid'])

_path_config = None
_lora_config = None
_training_config = None
_inference_config = None
_device_config = None
_model_config = None
_vision_model_config = None
_ptuning_config = None
_prompt_tuning_config = None
_full_finetuning_config = None


def get_path_config() -> PathConfig:
    """Return path config."""
    global _path_config
    if _path_config is None:
        _path_config = PathConfig()
    return _path_config


def get_lora_config(config_type: str = "conservative") -> LoRAConfig:
    """Return LoRA config."""
    global _lora_config
    if _lora_config is None:
        if config_type == "aggressive":
            _lora_config = LoRAConfig.get_aggressive_config()
        else:
            _lora_config = LoRAConfig.get_conservative_config()
    return _lora_config


def get_training_config() -> TrainingConfig:
    """Return training config."""
    global _training_config
    if _training_config is None:
        _training_config = TrainingConfig()
    return _training_config


def get_ptuning_config(config_type: str = "default") -> PTuningV2Config:
    """Return ptuning config."""
    global _ptuning_config
    if _ptuning_config is None:
        if config_type == "large":
            _ptuning_config = PTuningV2Config.get_large_config()
        else:
            _ptuning_config = PTuningV2Config.get_default_config()
    return _ptuning_config


def get_prompt_tuning_config(config_type: str = "default") -> PromptTuningConfig:
    """Return prompt tuning config."""
    global _prompt_tuning_config
    if _prompt_tuning_config is None:
        if config_type == "large":
            _prompt_tuning_config = PromptTuningConfig.get_large_config()
        else:
            _prompt_tuning_config = PromptTuningConfig.get_default_config()
    return _prompt_tuning_config


def get_full_finetuning_config(config_type: str = "default") -> FullFineTuningConfig:
    """Return full finetuning config."""
    global _full_finetuning_config
    if _full_finetuning_config is None:
        if config_type == "memory_efficient":
            _full_finetuning_config = FullFineTuningConfig.get_memory_efficient_config()
        else:
            _full_finetuning_config = FullFineTuningConfig.get_default_config()
    return _full_finetuning_config


def get_inference_config() -> InferenceConfig:
    """Return inference config."""
    global _inference_config
    if _inference_config is None:
        _inference_config = InferenceConfig()
    return _inference_config


def get_device_config() -> DeviceConfig:
    """Return device config."""
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig()
    return _device_config


def set_streaming_mode(enable: bool):
    """Set streaming mode."""
    global _device_config
    if _device_config is None:
        _device_config = DeviceConfig()
    _device_config.enable_streaming = enable
    print(f"Streaming output mode: {'enabled' if enable else 'disabled'}")


def get_model_config(version: str = None) -> ModelConfig:
    """Return model config."""
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config





def get_vision_model_config(version: str = None) -> VisionModelConfig:
    """Return vision model config."""
    global _vision_model_config
    if _vision_model_config is None:
        _vision_model_config = VisionModelConfig()
    return _vision_model_config


def set_vision_model_version(version: str):
    """Set the vision-model version."""
    global _vision_model_config
    _vision_model_config = VisionModelConfig(version=version)
    print(f"✓ Switched vision model version: {version}")
    print(f"  Model: {_vision_model_config.get_model_name()}")


def validate_config() -> tuple:
    """Validate config."""
    messages = []
    is_valid = True

    print("\n" + "=" * 60)
    print("Validating configuration...")
    print("=" * 60)

    path_cfg = get_path_config()

    print("\n[1/5] Checking base model paths...")

    qwen3_8b_path = path_cfg.QWEN3_8B_PATH
    if not qwen3_8b_path.exists():
        messages.append(f"❌ Qwen3-8B模型未找到: {qwen3_8b_path}")
        is_valid = False
    else:
        print(f"✓ Qwen3-8B model path is valid")

    vision_path = path_cfg.get_vision_model_path('qwen3')
    if not vision_path.exists():
        messages.append(f"⚠ Qwen3-VL-8B 模型未找到: {vision_path}")
        print(f"⚠ Qwen3-VL-8B model not found (ignore if not downloaded yet)")
    else:
        print(f"✓ Qwen3-VL-8B vision model path is valid")

    print("\n[2/5] Checking datasets...")
    if path_cfg.IMAGE_DATASET_CSV.exists():
        print(f"✓ Image dataset found")
    else:
        messages.append(f"⚠ 图像数据集未找到: {path_cfg.IMAGE_DATASET_CSV}")

    text_dataset_count = sum(1 for f in path_cfg.TEXT_DATASET_FILES.values() if f.exists())
    print(f"✓ Found {text_dataset_count}/{len(path_cfg.TEXT_DATASET_FILES)} text dataset files")

    if not path_cfg.UML_DATASET_CSV.exists():
        messages.append(f"⚠ UML数据集未找到（可能尚未创建）: {path_cfg.UML_DATASET_CSV}")

    print("\n[3/5] Checking CUDA environment...")
    device_cfg = get_device_config()

    if device_cfg.device != "cuda":
        messages.append("⚠ CUDA不可用，将使用CPU模式（速度极慢）")
    else:
        print(f"✓ CUDA is available")

    print("\n[4/5] Checking dependencies...")
    required_packages = {
        'transformers': '模型加载',
        'torch': 'PyTorch框架',
        'peft': 'LoRA训练',
        'bitsandbytes': '4bit量化'
    }

    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package} ({description})")
        except ImportError:
            messages.append(f"❌ 缺少依赖: {package} - {description}")
            is_valid = False

    print("\n[5/5] Creating required directories...")
    try:
        path_cfg.create_directories()
    except Exception as e:
        messages.append(f"❌ 创建目录失败: {str(e)}")
        is_valid = False

    print("\n" + "=" * 60)
    print("Validation results")
    print("=" * 60)

    if is_valid:
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed; please fix the following issues:")

    for msg in messages:
        print(f"  {msg}")

    print("=" * 60 + "\n")

    return is_valid, messages
