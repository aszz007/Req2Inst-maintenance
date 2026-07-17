"""Implement P-Tuning v2 training."""

import torch
import torch.utils.checkpoint
from typing import Optional
from peft import (
    PrefixTuningConfig,
    get_peft_model,
    TaskType,
)

from src.training.base_trainer import BaseTrainer
from src.utils.logger import get_logger

logger = get_logger('training.p_tuning_trainer')


class PTuningTrainer(BaseTrainer):
    """Train models with P-Tuning v2."""

    def __init__(self,
                 expert_type: str,
                 base_model_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 use_4bit: bool = True,
                 use_rtx4090_optimization: bool = True,
                 debug_samples: bool = False):
        """Initialize the instance."""
        super().__init__(
            expert_type=expert_type,
            method_name='p_tuning',
            base_model_path=base_model_path,
            output_dir=output_dir,
            use_rtx4090_optimization=use_rtx4090_optimization,
            debug_samples=debug_samples
        )

        self.use_4bit = use_4bit

        self.num_virtual_tokens = 20
        self.encoder_hidden_size = 128
        self.prefix_projection = True

        original_lr = self.train_cfg.learning_rate
        self.train_cfg.learning_rate = 1e-3
        logger.info(f"Learning rate: {original_lr} -> {self.train_cfg.learning_rate} (standard prefix-encoder setting)")

        self.disable_gradient_checkpointing = True

        self.disable_load_best_model = True

        self.train_cfg.max_seq_length = self._get_max_seq_length()
        logger.info(f"Maximum sequence length: {self.train_cfg.max_seq_length} (managed by base_trainer)")

        self.reduced_workers = True

        logger.info(f"4-bit quantization: {use_4bit}")
        logger.info(f"P-Tuning v2 configuration: virtual_tokens={self.num_virtual_tokens}, "
                    f"encoder_hidden_size={self.encoder_hidden_size}, "
                    f"prefix_projection={self.prefix_projection}")
        logger.info(f"Maximum sequence length: {self.train_cfg.max_seq_length}")
        logger.info("GPU-memory optimization strategy:")
        logger.info("  1. encoder_hidden_size=128 (balances performance and stability)")
        logger.info("  2. Sequence length: fixed at 2048 (managed by base_trainer)")
        logger.info("  3. Enable expandable_segments to reduce fragmentation")
        logger.info("  4. batch_size=1 with gradient accumulation of 128")
        logger.info("  5. Learning rate=1e-3 (standard prefix-encoder setting)")
        logger.info("  6. Memory-efficient SDPA attention (enabled when base_trainer loads the model)")
        logger.info("  7. MLP-level activation checkpointing (enabled in setup_model)")
        logger.info("Note: layer-level gradient checkpointing is disabled because P-Tuning v2 does not support it")

        self._print_training_config()

    def _get_batch_config(self):
        """Return batch config."""
        return 1, 128

    def _enable_mlp_activation_checkpointing(self):
        """Enable activation checkpointing for MLP layers."""
        try:
            base_model = self.model.get_base_model()
            if not (hasattr(base_model, 'model') and hasattr(base_model.model, 'layers')):
                logger.warning("Unable to access the decoder-layer list; MLP activation checkpointing was not enabled")
                return

            num_layers = len(base_model.model.layers)
            patched = 0
            for layer in base_model.model.layers:
                if not hasattr(layer, 'mlp'):
                    continue

                original_forward = layer.mlp.forward

                def make_ckpt_forward(fwd):
                    def checkpointed_mlp_forward(hidden_states):
                        return torch.utils.checkpoint.checkpoint(
                            fwd, hidden_states, use_reentrant=False
                        )
                    return checkpointed_mlp_forward

                layer.mlp.forward = make_ckpt_forward(original_forward)
                patched += 1

            logger.info(f"MLP activation checkpointing enabled for {patched}/{num_layers} decoder layers")
            logger.info("Effect: MLP intermediate activations are recomputed during backward instead of stored, saving approximately 4 GB of GPU memory")

        except Exception as e:
            logger.warning(f"Failed to enable MLP activation checkpointing; continuing without it: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def setup_model(self) -> bool:
        """Configure the model."""
        try:
            import os
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce memory fragmentation")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared GPU cache")

            if not self._load_base_model(self.use_4bit):
                return False

            logger.info("Configuring P-Tuning v2...")

            # use_cache must be False during training to avoid storing KV cache for every
            # layer, which wastes significant GPU memory. (base_trainer already guarantees
            # prepare_model_for_kbit_training is called without GC when
            # disable_gradient_checkpointing=True, so no explicit disable needed here.)
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
                logger.info("Disabled use_cache during training to reduce GPU memory usage")

            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.num_virtual_tokens,
                encoder_hidden_size=self.encoder_hidden_size,
                prefix_projection=self.prefix_projection
            )

            self.model = get_peft_model(self.model, peft_config)

            # Cast prefix encoder to match the base model dtype (bfloat16 or float16).
            # The PrefixEncoder MLP is initialized in float32 by default. Without this
            # cast, there is a dtype mismatch between the float32 prefix key/values and
            # the bfloat16 attention layers, which causes NaN in eval (where autocast
            # is not active) while training loss stays valid (autocast bridges the gap).
            model_dtype = torch.bfloat16 if self.use_rtx4090_optimization else torch.float16
            if hasattr(self.model, 'prompt_encoder'):
                self.model.prompt_encoder.to(model_dtype)
                logger.info(f"Converted prefix encoder to {model_dtype} to match the base-model dtype")

            # NOTE: Gradient checkpointing is intentionally NOT enabled for PrefixTuning.
            # Qwen3's gradient checkpointing implementation forces `past_key_values=None`
            # in every decoder layer during the forward pass. PrefixTuning injects the
            # learned prefix representations via `past_key_values`, so enabling gradient
            # checkpointing silently discards all prefix key-values, making the prefix
            # encoder unreachable by gradients (grad_norm stays 0.0) and producing a
            # frozen eval_loss that never improves.
            # disable_gradient_checkpointing=True is set so TrainingArguments also does
            # not call gradient_checkpointing_enable() via the Trainer.
            #
            # Instead, activation memory is reduced via MLP-level checkpointing below,
            # which only touches the FFN path and leaves past_key_values untouched.
            self._enable_mlp_activation_checkpointing()

            logger.info("Model dtype diagnostics")

            if hasattr(self.model, 'prompt_encoder'):
                for name, param in self.model.prompt_encoder.named_parameters():
                    logger.info(f"  prompt_encoder.{name}: dtype={param.dtype}, shape={param.shape}")

            base_model = self.model.get_base_model()
            if hasattr(base_model, 'model'):
                if hasattr(base_model.model, 'layers') and len(base_model.model.layers) > 0:
                    first_layer = base_model.model.layers[0]
                    if hasattr(first_layer, 'self_attn'):
                        attn = first_layer.self_attn
                        if hasattr(attn, 'q_proj'):
                            logger.info(f"  base_model.layers[0].self_attn.q_proj.weight: dtype={attn.q_proj.weight.dtype}")

            logger.info(f"  Target dtype: {model_dtype} ({'bfloat16' if self.use_rtx4090_optimization else 'float16'})")

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_ratio = 100 * trainable_params / total_params

            logger.info("P-Tuning v2 configuration complete")
            logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_ratio:.4f}%)")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Virtual Tokens: {self.num_virtual_tokens}")
            logger.info(f"Encoder Hidden Size: {self.encoder_hidden_size}")
            logger.info(f"Prefix Projection: {self.prefix_projection}")

            return True

        except Exception as e:
            logger.error(f"Model setup failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
