"""Provide shared training orchestration for the supported adaptation methods."""

import os
import json
import math
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

from config.settings import (
    get_path_config,
    get_training_config,
    get_device_config,
    get_model_config
)
from peft import prepare_model_for_kbit_training
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    GeneralDatasetLoader,
    InstructionDataset,
    InstructionDataCollator,
    split_dataset_for_expert
)
from src.utils.logger import get_logger

logger = get_logger('training.base_trainer')


def _remove_qwen_special_tokens(text: str) -> str:
    """
    Remove Qwen model special tokens from text

    Removes tokens such as:
    - <|im_start|>
    - <|im_end|>
    - <think>
    - </think>

    Args:
        text: Input text containing special tokens

    Returns:
        str: Text with special tokens removed
    """
    if not text:
        return text

    # Remove Qwen chat format tokens
    text = text.replace('<|im_start|>', '')
    text = text.replace('<|im_end|>', '')

    # Remove thinking tags
    text = text.replace('<think>', '')
    text = text.replace('</think>', '')

    return text


def _get_transformers_version():
    """Return transformers version."""
    import transformers
    version_str = transformers.__version__
    major, minor = version_str.split('.')[:2]
    return int(major), int(minor)


def _should_use_eval_strategy():
    """Return whether evaluation should run."""
    try:
        major, minor = _get_transformers_version()
        return (major > 4) or (major == 4 and minor >= 46)
    except Exception:
        return False


class NaNAwareEarlyStoppingCallback(TrainerCallback):
    """Stop training safely when evaluation metrics contain NaN values."""

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.best_metric = None
        self.patience_counter = 0
        self.nan_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Handle the evaluation callback."""
        if metrics is None:
            return

        eval_loss = metrics.get('eval_loss')

        if eval_loss is None or (isinstance(eval_loss, float) and math.isnan(eval_loss)):
            self.nan_count += 1
            logger.warning(f"NaN validation loss detected (occurrence {self.nan_count}); skipping this early-stopping check")
            logger.warning("Possible causes of NaN: unstable training, excessive learning rate, or P-Tuning configuration issues")
            return control

        metric = eval_loss

        if self.best_metric is None:
            self.best_metric = metric
            logger.info(f"Initialized best validation loss: {metric:.4f}")
            return control

        if metric < (self.best_metric - self.early_stopping_threshold):
            self.best_metric = metric
            self.patience_counter = 0
            logger.info(f"Validation loss improved: {metric:.4f} (best: {self.best_metric:.4f})")
        else:
            self.patience_counter += 1
            logger.info(f"Validation loss did not improve: {metric:.4f} vs best {self.best_metric:.4f} "
                       f"(patience: {self.patience_counter}/{self.early_stopping_patience})")

            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {self.early_stopping_patience} consecutive evaluations without improvement")
                logger.info(f"Best validation loss: {self.best_metric:.4f}")
                logger.info(f"Current validation loss: {metric:.4f}")
                if self.nan_count > 0:
                    logger.info(f"Ignored {self.nan_count} NaN validation losses during training")
                control.should_training_stop = True

        return control


class TrainingHistoryCallback(TrainerCallback):
    """Record training and evaluation history."""

    def __init__(self):
        super().__init__()
        self.training_history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle the logging callback."""
        if logs is not None:
            log_entry = {
                'step': state.global_step,
                'epoch': logs.get('epoch', state.epoch),
            }

            for key, value in logs.items():
                if key not in ['step', 'epoch']:
                    log_entry[key] = value

            self.training_history.append(log_entry)

    def get_history(self):
        """Return history."""
        return self.training_history


class BaseTrainer(ABC):
    """Define shared behavior for training implementations."""

    def __init__(self,
                 expert_type: str,
                 method_name: str,
                 base_model_path: str | None = None,
                 output_dir: str | None = None,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False,
                 use_domain_templates: bool = False):
        """Initialize the instance."""
        valid_types = ['text', 'image', 'uml', 'general']
        if expert_type not in valid_types:
            raise ValueError(f"Unsupported expert type: {expert_type}. Supported types: {valid_types}")

        self.expert_type = expert_type
        self.method_name = method_name
        self.use_rtx4090_optimization = use_rtx4090_optimization
        self.debug_samples = debug_samples
        self.use_domain_templates = use_domain_templates

        self.path_cfg = get_path_config()
        self.train_cfg = get_training_config()
        self.device_cfg = get_device_config()
        self.model_cfg = get_model_config()

        self.epochs_from_env = False
        if 'TRAIN_EPOCHS' in os.environ:
            try:
                epochs = int(os.environ['TRAIN_EPOCHS'])
                self.train_cfg.num_epochs = epochs
                self.epochs_from_env = True
                logger.info(f"Training epochs read from environment variable: {epochs} (fixed value)")
            except ValueError:
                logger.warning(f"Invalid TRAIN_EPOCHS environment variable: {os.environ['TRAIN_EPOCHS']}")

        if base_model_path:
            self.base_model_path = base_model_path
        else:
            self.base_model_path = str(self.path_cfg.get_text_model_path())

        if 'Qwen3-8B' in self.base_model_path or 'qwen3-8B' in self.base_model_path:
            self.model_version = 'qwen3_8b'
        else:
            self.model_version = self.model_cfg.version
            logger.warning(f"Unable to infer model version from path; using configured version: {self.model_version}")

        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.path_cfg.PROJECT_ROOT / 'checkpoints' / method_name / f"{expert_type}_expert"

        self.checkpoint_dir = self.output_dir / 'training_checkpoints'

        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.history_callback = TrainingHistoryCallback()

        logger.info(f"Initializing {expert_type} expert trainer (method: {method_name})")
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"RTX 4090 optimization: {use_rtx4090_optimization}")

    def _print_training_config(self):
        """Print training config."""
        batch_size, gradient_accumulation_steps = self._get_batch_config()

        logger.info("Training configuration")
        logger.info(f"Expert type: {self.expert_type}")
        logger.info(f"Fine-tuning method: {self.method_name}")
        logger.info(f"Base model: {self.base_model_path}")
        logger.info(f"Model version: {self.model_version}")

        if self.use_rtx4090_optimization:
            logger.info(f"Batch size: {batch_size} (RTX 4090 optimization)")
            logger.info(f"Gradient accumulation: {gradient_accumulation_steps} (RTX 4090 optimization)")
        else:
            logger.info(f"Batch size: {batch_size}")
            logger.info(f"Gradient accumulation: {gradient_accumulation_steps}")

        logger.info(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"Training epochs: {self.train_cfg.num_epochs}")
        logger.info(f"Learning rate: {self.train_cfg.learning_rate}")
        logger.info(f"Maximum sequence length: {self.train_cfg.max_seq_length}")

    def _load_base_model(self, use_4bit: bool) -> bool:
        """Load base model."""
        try:
            logger.info("Loading base model...")

            quantization_config = None
            if use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Enabling 4-bit quantization")

            model_kwargs = {
                'pretrained_model_name_or_path': self.base_model_path,
                'trust_remote_code': True,
                'device_map': 'auto',
                'dtype': torch.bfloat16 if self.use_rtx4090_optimization else torch.float16,
                # Use PyTorch SDPA (memory-efficient attention) to reduce attention
                # activation memory from O(n^2) to O(n). This is critical for
                # p_tuning and prompt_tuning which disable gradient checkpointing,
                # causing full attention activations to remain in GPU memory.
                # SDPA is built into PyTorch >= 2.0, requires no extra packages,
                # and is compatible with 4bit quantization and all PEFT methods.
                'attn_implementation': 'sdpa',
            }
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config

            self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            if self.model_version == 'qwen3_8b':
                if hasattr(self.model.config, 'enable_thinking'):
                    self.model.config.enable_thinking = False
                    logger.info("Qwen3-8B: thinking mode disabled (enable_thinking=False)")
                else:
                    logger.info("Qwen3-8B: model does not support enable_thinking; skipping this setting")

            if use_4bit:
                # Certain PEFT methods (e.g. PrefixTuning) raise ValueError if gradient
                # checkpointing is already enabled on the base model when get_peft_model()
                # is called. When the subclass sets disable_gradient_checkpointing=True,
                # skip enabling GC here so that PEFT initialisation succeeds. The subclass
                # is then responsible for enabling GC on the PEFT-wrapped model afterwards.
                use_gc_for_kbit = not getattr(self, 'disable_gradient_checkpointing', False)
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=use_gc_for_kbit
                )

            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                padding_side='left'
            )

            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))

            logger.info(f"Tokenizer vocabulary size: {len(self.tokenizer)}")
            logger.info(f"PAD token: {self.tokenizer.pad_token}")
            return True

        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _get_batch_config(self) -> tuple[int, int]:
        """Return batch config."""
        if self.use_rtx4090_optimization:
            return 8, 2
        else:
            return self.train_cfg.batch_size, self.train_cfg.gradient_accumulation_steps

    def _get_max_seq_length(self) -> int:
        """Return max seq length."""
        return 2048

    def _get_num_epochs_from_data(self) -> int:
        """Return num epochs from data."""
        if not self.train_dataset:
            logger.warning("Training dataset is not prepared; using the default epoch count")
            return self.train_cfg.num_epochs

        data_size = len(self.train_dataset)

        if self.expert_type == 'image':
            base_epochs = 8
        elif self.expert_type == 'text':
            base_epochs = 5
        elif self.expert_type == 'uml':
            base_epochs = 6
        elif self.expert_type == 'general':
            base_epochs = 4
        else:
            base_epochs = 5

        if self.method_name in ['p_tuning', 'prompt_tuning']:
            method_epochs = base_epochs + 1
        elif self.method_name == 'full_finetuning':
            method_epochs = base_epochs
        else:
            method_epochs = base_epochs

        logger.info("Adaptive epoch calculation:")
        logger.info(f"  Records: {data_size}")
        logger.info(f"  Expert type: {self.expert_type} -> base epochs={base_epochs}")
        logger.info(f"  Fine-tuning method: {self.method_name} -> final epochs={method_epochs}")

        return method_epochs

    def prepare_data(self) -> bool:
        """Prepare data."""
        try:
            logger.info(f"Preparing training data for the {self.expert_type} expert...")

            if self.expert_type == 'text':
                loader = TextDatasetLoader()
                all_data = loader.load_csv_files()
            elif self.expert_type == 'image':
                loader = ImageDatasetLoader()
                all_data = loader.load_csv_file()
            elif self.expert_type == 'uml':
                loader = UMLDatasetLoader()
                all_data = loader.load_csv_file()
            elif self.expert_type == 'general':
                loader = GeneralDatasetLoader(use_domain_templates=self.use_domain_templates)
                all_data = loader.load_all_data()
            else:
                raise ValueError(f"Unsupported expert type: {self.expert_type}")

            if not all_data:
                logger.error("No data was loaded")
                return False

            logger.info(f"Loaded {len(all_data)} records")

            train_data, val_data, test_data = split_dataset_for_expert(
                all_data, self.expert_type
            )

            self._raw_train_data = train_data
            self._raw_val_data = val_data
            self._raw_test_data = test_data
            self.train_dataset = train_data
            self.val_dataset = val_data
            self.test_dataset = test_data

            logger.info("Dataset split complete:")
            logger.info(f"  Training set: {len(self.train_dataset)} records")
            logger.info(f"  Validation set: {len(self.val_dataset)} records")
            logger.info(f"  Test set: {len(self.test_dataset)} records")

            if not self.epochs_from_env:
                calculated_epochs = self._get_num_epochs_from_data()
                self.train_cfg.num_epochs = calculated_epochs
                logger.info(f"Training epochs set automatically from dataset size: {calculated_epochs}")
            else:
                logger.info(f"Using training epochs specified by environment variable: {self.train_cfg.num_epochs}")

            return True

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    @abstractmethod
    def setup_model(self) -> bool:
        """Configure the model."""
        pass

    def _get_early_stopping_patience(self) -> int:
        """Return early stopping patience."""
        if self.expert_type in ('text', 'general'):
            return 3
        elif self.expert_type == 'image':
            return 4
        elif self.expert_type == 'uml':
            return 4
        else:
            return 3

    def _get_eval_steps(self) -> int:
        """Return eval steps."""
        if not self.train_dataset:
            logger.warning("Training dataset is not prepared; using default eval_steps=50")
            return 50

        batch_size, gradient_accumulation_steps = self._get_batch_config()
        num_samples = len(self.train_dataset)

        steps_per_epoch = max(1, num_samples // (batch_size * gradient_accumulation_steps))
        total_steps = steps_per_epoch * self.train_cfg.num_epochs

        target_total_evals = 10
        eval_steps = max(1, round(total_steps / target_total_evals))

        eval_steps = min(eval_steps, steps_per_epoch)
        eval_steps = max(1, eval_steps)

        actual_total_evals = total_steps / eval_steps if eval_steps > 0 else 0
        actual_evals_per_epoch = steps_per_epoch / eval_steps if eval_steps > 0 else 0

        logger.info("Dynamic eval_steps configuration:")
        logger.info(f"  Training samples: {num_samples}")
        logger.info(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total training steps: {total_steps}")
        logger.info(f"  Target total evaluations: {target_total_evals}")
        logger.info(f"  Computed eval_steps: {eval_steps}")
        logger.info(f"  Actual total evaluations: {actual_total_evals:.1f}")
        logger.info(f"  Actual evaluations per epoch: {actual_evals_per_epoch:.1f}")

        return eval_steps

    def train(self) -> bool:
        """Run model training."""
        if self.model is None or self.tokenizer is None:
            logger.error("Model is not initialized; call setup_model() first")
            return False

        if self.train_dataset is None or self.val_dataset is None:
            logger.error("Data is not prepared; call prepare_data() first")
            return False

        try:
            logger.info(f"[train() entry] debug_samples={self.debug_samples}, "
                        f"train_samples={len(self.train_dataset) if self.train_dataset else 0}, "
                        f"method={self.method_name}, expert={self.expert_type}")

            if self.debug_samples and self.train_dataset is not None and len(self.train_dataset) > 0:
                logger.info("[Debug output] Showing prompts for the first three training samples")

                for i in range(min(3, len(self.train_dataset))):
                    sample = self.train_dataset.data[i]
                    logger.info(f"\nSample {i+1}:")
                    prompt_text = sample.get('input_with_prompt', 'N/A')
                    clean_prompt = _remove_qwen_special_tokens(prompt_text)
                    logger.info(f"Complete prompt:\n{clean_prompt}")
                    logger.info(f"Expected output:\n{sample['output']}")

                logger.info("[Debug output complete] Verify that the prompts above contain complete JSON structures")

            if torch.cuda.is_available():
                if self.method_name == 'full_finetuning':
                    for i in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    logger.info("Full fine-tuning: cleared the GPU cache three times")
                else:
                    torch.cuda.empty_cache()
                    if self.expert_type in ['uml', 'general']:
                        logger.info(f"Cleared GPU cache before training the {self.expert_type} expert to maximize available memory")

                allocated = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - allocated
                logger.info(f"[Before training] GPU memory: allocated={allocated:.2f} GB, available≈{free:.2f} GB, total={total:.2f} GB")

                if self.method_name == 'full_finetuning' and allocated > 10.0:
                    logger.error(f"Warning: {allocated:.2f} GB of GPU memory is already allocated before training, which may cause an out-of-memory error")
                    logger.error("Suggested actions:")
                    logger.error("  1. Check for other processes using the GPU (nvidia-smi)")
                    logger.error("  2. Restart the Python process to release GPU memory")
                    logger.error("  3. Close unnecessary programs")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(self, '_raw_train_data'):
                logger.info("Creating InstructionDataset instances (tokenizer ready)...")
                self.train_dataset = InstructionDataset(
                    self._raw_train_data, self.tokenizer, self.train_cfg.max_seq_length
                )
                self.val_dataset = InstructionDataset(
                    self._raw_val_data, self.tokenizer, self.train_cfg.max_seq_length
                )
                self.test_dataset = InstructionDataset(
                    self._raw_test_data, self.tokenizer, self.train_cfg.max_seq_length
                )
                logger.info(f"InstructionDataset creation complete: "
                            f"train={len(self.train_dataset)}, val={len(self.val_dataset)}")

            batch_size, gradient_accumulation_steps = self._get_batch_config()

            early_stopping_patience = self._get_early_stopping_patience()
            eval_steps = self._get_eval_steps()

            num_train_samples = len(self.train_dataset)
            steps_per_epoch = num_train_samples // (batch_size * gradient_accumulation_steps)
            total_steps = steps_per_epoch * self.train_cfg.num_epochs

            warmup_ratio = 0.1

            warmup_steps = int(total_steps * warmup_ratio)

            logger.info("Training-step configuration:")
            logger.info(f"  Steps per epoch: {steps_per_epoch}")
            logger.info(f"  Total training steps: {total_steps}")
            logger.info(f"  Warmup steps: {warmup_steps}")
            logger.info(f"  Evaluation frequency: every {eval_steps} steps")
            logger.info(f"  Early-stopping patience: {early_stopping_patience}")

            training_args_dict = {
                'output_dir': str(self.checkpoint_dir),
                'num_train_epochs': self.train_cfg.num_epochs,
                'per_device_train_batch_size': batch_size,
                'per_device_eval_batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'learning_rate': self.train_cfg.learning_rate,
                'weight_decay': 0.01,
                'lr_scheduler_type': 'cosine',
                'warmup_steps': warmup_steps,
                'logging_steps': 1,
                'eval_steps': eval_steps,
                'save_steps': eval_steps,
                'save_total_limit': 3,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'report_to': 'none',
                'remove_unused_columns': False,
            }

            if self.method_name == 'full_finetuning':
                training_args_dict['max_grad_norm'] = 0.8
                logger.info(f"{self.method_name} uses strict gradient clipping (0.8) for training stability")
            else:
                training_args_dict['max_grad_norm'] = 1.0
                logger.info(f"{self.method_name} uses standard gradient clipping (1.0)")

            if not getattr(self, 'disable_load_best_model', False):
                training_args_dict['load_best_model_at_end'] = True
            else:
                training_args_dict['load_best_model_at_end'] = False
                logger.info("load_best_model_at_end disabled because the current training method does not support it")

            if not getattr(self, 'disable_gradient_checkpointing', False):
                training_args_dict['gradient_checkpointing'] = True
            else:
                logger.info("Gradient checkpointing disabled because the current training method does not support it")

            if _should_use_eval_strategy():
                training_args_dict['eval_strategy'] = 'steps'
            else:
                training_args_dict['evaluation_strategy'] = 'steps'

            if self.use_rtx4090_optimization:
                if self.method_name == 'full_finetuning':
                    num_workers = 2
                    prefetch_factor = 1
                    logger.info("Full fine-tuning uses the minimum DataLoader configuration: workers=2, prefetch=1")
                elif self.expert_type in ['uml', 'general'] and getattr(self, 'reduced_workers', False):
                    num_workers = 2
                    prefetch_factor = 1
                    logger.info(f"The {self.expert_type} expert uses the minimum DataLoader configuration: workers=2, prefetch=1")
                elif getattr(self, 'reduced_workers', False):
                    num_workers = 4
                    prefetch_factor = 2
                else:
                    num_workers = 8
                    prefetch_factor = 4

                training_args_dict.update({
                    'bf16': True,
                    'tf32': True,
                    'optim': 'adamw_torch_fused',
                    'dataloader_num_workers': num_workers,
                    'dataloader_prefetch_factor': prefetch_factor,
                })

                # P-Tuning/Prompt Tuning: prefix encoder is float32 by default.
                # Training uses torch.autocast(bfloat16) which silently casts it,
                # but eval skips autocast when bf16_full_eval=False (the default),
                # causing a float32/bfloat16 dtype mismatch in attention that
                # produces NaN eval_loss. Setting bf16_full_eval=True makes eval
                # use the same precision context as training.
                if self.method_name in ['p_tuning', 'prompt_tuning']:
                    training_args_dict['bf16_full_eval'] = True
                    logger.info(f"{self.method_name}: enabled bf16_full_eval=True to match evaluation and training precision and prevent NaN validation loss")

                if getattr(self, 'reduced_workers', False) or self.method_name == 'full_finetuning':
                    logger.info(f"Using fewer DataLoader workers: {num_workers} (GPU-memory optimization)")
            elif torch.cuda.is_available():
                training_args_dict['fp16'] = True
                # Same precision consistency fix for fp16 mode
                if self.method_name in ['p_tuning', 'prompt_tuning']:
                    training_args_dict['fp16_full_eval'] = True
                    logger.info(f"{self.method_name}: enabled fp16_full_eval=True to match evaluation and training precision and prevent NaN validation loss")

            training_args = TrainingArguments(**training_args_dict)

            data_collator = InstructionDataCollator(
                tokenizer=self.tokenizer,
                pad_to_multiple_of=8
            )

            early_stopping_callback = NaNAwareEarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=0.0001
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=data_collator,
                callbacks=[self.history_callback, early_stopping_callback],
            )

            logger.info("Starting training loop...")
            train_result = trainer.train()

            logger.info("Saving final weights...")
            self._save_weights()

            metrics = train_result.metrics
            logger.info(f"Training complete; final loss: {metrics.get('train_loss', 'N/A')}")

            if early_stopping_callback.nan_count > 0:
                logger.warning(f"Detected {early_stopping_callback.nan_count} NaN validation losses during training")
                logger.warning("NaN validation loss may indicate:")
                logger.warning("  1. An excessive learning rate causing unstable training")
                logger.warning("  2. A P-Tuning v2 configuration that needs adjustment, such as encoder_hidden_size")
                logger.warning("  3. Abnormal samples in the dataset")
                logger.warning("  4. Improper virtual-token initialization")
                logger.warning("Suggested action: inspect eval_loss values in training_history.json")

            metrics_file = self.output_dir / "training_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

            training_history = self.history_callback.get_history()
            history_file = self.output_dir / "training_history.json"

            batch_size, gradient_accumulation_steps = self._get_batch_config()

            def clean_nan_values(obj):
                """Clean nan values."""
                if isinstance(obj, dict):
                    return {k: clean_nan_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_nan_values(item) for item in obj]
                elif isinstance(obj, float) and math.isnan(obj):
                    return None
                else:
                    return obj

            cleaned_history = clean_nan_values(training_history)

            history_data = {
                'expert_type': self.expert_type,
                'method_name': self.method_name,
                'total_steps': len(training_history),
                'num_epochs': self.train_cfg.num_epochs,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'effective_batch_size': batch_size * gradient_accumulation_steps,
                'learning_rate': self.train_cfg.learning_rate,
                'use_rtx4090_optimization': self.use_rtx4090_optimization,
                'history': cleaned_history
            }

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Training history saved to: {history_file}")
            logger.info(f"Recorded data for {len(training_history)} training steps")

            logger.info("Generating training-curve visualization...")
            try:
                self._plot_training_curves(training_history, self.expert_type)
                logger.info("Training-curve visualization generated")
            except Exception as e:
                logger.warning(f"Failed to generate training-curve visualization: {e}")
                logger.warning("Continuing without a visualization")

            logger.info(f"Weights saved to: {self.output_dir}")
            logger.info(f"Training metrics saved to: {metrics_file}")

            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_weights(self):
        """Save weights."""
        if self.model is None or self.tokenizer is None:
            logger.error("Model or tokenizer is not initialized; unable to save")
            return

        try:
            logger.info(f"Saving weights ({self.method_name})...")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(self.output_dir))
            self.tokenizer.save_pretrained(str(self.output_dir))
            logger.info(f"Weights saved to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def get_training_status(self) -> dict:
        """Return training status."""
        return {
            'expert_type': self.expert_type,
            'method_name': self.method_name,
            'base_model': self.base_model_path,
            'output_dir': str(self.output_dir),
            'model_loaded': self.model is not None,
            'data_prepared': self.train_dataset is not None,
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
        }

    def _plot_training_curves(self, training_history: list[dict], expert_type: str):
        """Plot training curves."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is not installed; skipping visualization")
            return

        curves_dir = self.path_cfg.PROJECT_ROOT / 'outputs' / 'training_curves'
        curves_dir.mkdir(parents=True, exist_ok=True)

        loss_steps = []
        losses = []
        eval_steps = []
        eval_losses = []
        grad_norm_steps = []
        grad_norms = []
        lr_steps = []
        learning_rates = []

        for entry in training_history:
            step = entry.get('step', 0)

            if 'loss' in entry:
                loss_val = entry['loss']
                if loss_val is not None and not (isinstance(loss_val, float) and math.isnan(loss_val)):
                    loss_steps.append(step)
                    losses.append(loss_val)

            if 'eval_loss' in entry:
                eval_val = entry['eval_loss']
                if eval_val is not None and not (isinstance(eval_val, float) and math.isnan(eval_val)):
                    eval_steps.append(step)
                    eval_losses.append(eval_val)

            if 'grad_norm' in entry:
                grad_val = entry['grad_norm']
                if grad_val is not None and not (isinstance(grad_val, float) and math.isnan(grad_val)):
                    grad_norm_steps.append(step)
                    grad_norms.append(grad_val)

            if 'learning_rate' in entry:
                lr_val = entry['learning_rate']
                if lr_val is not None and not (isinstance(lr_val, float) and math.isnan(lr_val)):
                    lr_steps.append(step)
                    learning_rates.append(lr_val)

        total_entries = len(training_history)
        nan_eval_count = sum(1 for e in training_history if 'eval_loss' in e and
                            (e['eval_loss'] is None or (isinstance(e['eval_loss'], float) and math.isnan(e['eval_loss']))))

        if total_entries < 10:
            logger.warning(f"Training history contains only {total_entries} entries, possibly because early stopping ended training")

        if len(losses) < 3:
            logger.warning(f"Training-loss series contains only {len(losses)} data points")
        if len(eval_losses) == 0:
            if nan_eval_count > 0:
                logger.warning(f"All {nan_eval_count} validation-loss values are NaN and were filtered; unable to plot the validation curve")
                logger.warning("This indicates unstable training. Suggested checks:")
                logger.warning("  1. Reduce the learning rate")
                logger.warning("  2. Adjust the P-Tuning or Prompt Tuning configuration")
                logger.warning("  3. Check dataset quality")
            else:
                logger.warning("No validation-loss data is available")
        elif len(eval_losses) < 3:
            if nan_eval_count > 0:
                logger.warning(f"Validation-loss series contains only {len(eval_losses)} valid points; {nan_eval_count} NaN values were filtered")
            else:
                logger.warning(f"Validation-loss series contains only {len(eval_losses)} data points")
        elif nan_eval_count > 0:
            logger.info(f"Filtered {nan_eval_count} NaN validation-loss values and retained {len(eval_losses)} valid values")

        logger.info(f"Curve data summary: Loss={len(losses)} points, EvalLoss={len(eval_losses)} points, GradNorm={len(grad_norms)} points, LR={len(learning_rates)} points")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Curves - {expert_type.upper()} Expert ({self.method_name})',
                     fontsize=16, fontweight='bold')

        # 1. Training Loss
        if losses:
            axes[0, 0].plot(loss_steps, losses, 'b-', linewidth=1.5, alpha=0.7)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'No training loss data',
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')

        # 2. Eval Loss
        if eval_losses:
            axes[0, 1].plot(eval_steps, eval_losses, 'r-', linewidth=2, marker='o', markersize=4)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Eval Loss')
            axes[0, 1].set_title('Validation Loss')
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation loss data',
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Eval Loss')
            axes[0, 1].set_title('Validation Loss')

        # 3. Gradient Norm
        if grad_norms:
            axes[1, 0].plot(grad_norm_steps, grad_norms, 'g-', linewidth=1, alpha=0.6)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No gradient norm data',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norm')

        # 4. Learning Rate
        if learning_rates:
            axes[1, 1].plot(lr_steps, learning_rates, 'm-', linewidth=1.5)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        else:
            axes[1, 1].text(0.5, 0.5, 'No learning rate data',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')

        plt.tight_layout()

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = curves_dir / f'{expert_type}_expert_{self.method_name}_training_curves_{timestamp}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training curves saved to: {plot_path}")
