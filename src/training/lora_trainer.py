"""Implement LoRA training for domain experts."""

import torch
from typing import Optional
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.lora_trainer')


class LoRATrainer(BaseTrainer):
    """Train domain experts with LoRA."""

    def __init__(self,
                 expert_type: str,
                 method_name: str = 'lora_moe',
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False,
                 lora_rank: int = 64,
                 lora_alpha: int = 128,
                 lora_dropout: float = 0.05,
                 use_domain_templates: bool = False):
        """Initialize the instance."""
        super().__init__(
            expert_type=expert_type,
            method_name=method_name,
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples,
            use_domain_templates=use_domain_templates
        )

        self.use_4bit = use_4bit

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        self.target_modules = self._get_target_modules()

        logger.info(f"4-bit quantization: {use_4bit}")
        logger.info(f"LoRA configuration: rank={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        logger.info(f"Target modules: {self.target_modules}")
        logger.info("Training-stability configuration:")
        logger.info("  - Gradient clipping: max_grad_norm=1.0 (standard setting)")
        logger.info("  - Warmup ratio: 10% (standard setting)")
        logger.info("  - NaN-aware early stopping: automatically ignores NaN validation loss")

        self._print_training_config()

    def _get_batch_config(self):
        """Return batch config."""
        if self.use_rtx4090_optimization:
            if self.expert_type in ['image', 'text']:
                return 2, 64
            elif self.expert_type in ['uml', 'general']:
                return 1, 128
            else:
                return 1, 128
        else:
            return self.train_cfg.batch_size, self.train_cfg.gradient_accumulation_steps

    def _get_target_modules(self) -> list:
        """Return target modules."""
        if self.model_version == 'qwen3_8b':
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            logger.warning(f"Unknown model version {self.model_version}; using the default Qwen3 configuration")
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def setup_model(self) -> bool:
        """Configure the model."""
        try:
            if not self._load_base_model(self.use_4bit):
                return False

            logger.info("Configuring LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=self.target_modules,
                bias="none",
            )

            self.model = get_peft_model(self.model, lora_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("LoRA configuration complete")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.2f}%)")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"LoRA Rank: {self.lora_rank}")
            logger.info(f"LoRA Alpha: {self.lora_alpha}")
            logger.info(f"LoRA Dropout: {self.lora_dropout}")
            logger.info(f"Target Modules: {self.target_modules}")

            return True

        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
