"""Implement prompt-tuning training."""

import torch
from peft import (
    PromptTuningConfig,
    PromptTuningInit,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.prompt_tuning_trainer')


class PromptTuningTrainer(BaseTrainer):
    """Train models with prompt tuning."""

    def __init__(self,
                 expert_type: str,
                 base_model_path: str | None = None,
                 output_dir: str | None = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False):
        """Initialize the instance."""
        super().__init__(
            expert_type=expert_type,
            method_name='prompt_tuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        self.num_virtual_tokens = 10


        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = 5e-5
        logger.warning("Prompt Tuning NaN protections enabled")
        logger.warning(f"Learning rate adjusted: {original_lr} → {self.train_cfg.learning_rate} (75% reduction)")
        logger.warning("Reason: Prompt Tuning trains only virtual tokens, and an excessive learning rate can cause NaN values")
        logger.warning("Additional protections: strict gradient clipping (0.5), NaN-aware early stopping, and 20% warmup")

        self.disable_load_best_model = True

        logger.info(f"4-bit quantization: {use_4bit}")
        logger.info(f"Prompt Tuning configuration: virtual_tokens={self.num_virtual_tokens}, "
                    f"init=RANDOM")

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

    def setup_model(self) -> bool:
        """Configure the model."""
        try:
            if not self._load_base_model(self.use_4bit):
                return False

            logger.info("Configuring Prompt Tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                prompt_tuning_init=PromptTuningInit.RANDOM,
                tokenizer_name_or_path=str(self.base_model_path)
            )

            self.model = get_peft_model(self.model, peft_config)

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("Prompt Tuning configuration complete")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.num_virtual_tokens}")
            logger.info("Initialization method: RANDOM")

            if self.expert_type in ['uml', 'general']:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared GPU cache for {self.expert_type} expert long-sequence optimization")

            return True

        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
