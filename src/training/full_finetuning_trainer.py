"""Implement full-finetuning training."""

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.full_finetuning_trainer')


class FullFineTuningTrainer(BaseTrainer):
    """Train models with full finetuning."""

    def __init__(self,
                 expert_type: str,
                 base_model_path: str | None = None,
                 output_dir: str | None = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = True):
        """Initialize the instance."""
        super().__init__(
            expert_type=expert_type,
            method_name='full_finetuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )
        self.debug_samples = debug_samples

        self.use_4bit = use_4bit

        self.lora_rank = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        self.reduced_workers = True

        self.train_cfg.max_seq_length = self._get_max_seq_length()

        logger.info(f"4-bit quantization: {use_4bit}")
        logger.info(f"Full fine-tuning configuration: rank={self.lora_rank}, alpha={self.lora_alpha}")
        logger.info(f"Max seq length: {self.train_cfg.max_seq_length}")
        logger.info(f"Target modules: {self.target_modules}")
        logger.info("Training-stability configuration:")
        logger.info("  - Gradient clipping: max_grad_norm=0.8 (strict setting)")
        logger.info("  - Warmup ratio: 10% (standard setting)")
        logger.info("  - NaN-aware early stopping: automatically ignores NaN validation loss")

        self._print_training_config()

    def _get_batch_config(self):
        """Return batch config."""
        if self.use_rtx4090_optimization:
            if self.expert_type == 'image':
                return 2, 64
            elif self.expert_type in ['text', 'uml', 'general']:
                return 1, 128
            else:
                return 1, 128
        else:
            return 1, 128

    def setup_model(self) -> bool:
        """Configure the model."""
        try:
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

            if torch.cuda.is_available():
                for i in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                logger.info("Cleared the GPU cache three times")

                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"[Initial state] GPU memory: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, total={total:.2f} GB")

            if not self._load_base_model(self.use_4bit):
                return False

            if torch.cuda.is_available() and self.use_4bit:
                allocated = torch.cuda.memory_allocated() / 1024**3
                if allocated > 8.0:
                    logger.error(f"Warning: 4-bit quantization may not be active; model uses {allocated:.2f} GB, expected <6 GB")
                else:
                    logger.info(f"4-bit quantization is active: model uses {allocated:.2f} GB")

            logger.info(f"Configuring full fine-tuning LoRA (rank={self.lora_rank})...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("Full fine-tuning configuration complete")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"LoRA Rank: {self.lora_rank}")
            logger.info(f"LoRA Alpha: {self.lora_alpha}")
            logger.info(f"LoRA Dropout: {self.lora_dropout}")
            logger.info(f"Target Modules: {self.target_modules}")
            logger.info(f"Max Seq Length: {self.train_cfg.max_seq_length}")

            return True

        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
