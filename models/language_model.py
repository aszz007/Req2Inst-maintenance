"""Provide the Qwen3-8B language-model wrapper with quantization and LoRA support."""

import time
import threading
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
)
from tqdm import tqdm

from peft import PeftModel
from pathlib import Path
import warnings
from concurrent.futures import ThreadPoolExecutor
import os
import json
import tempfile
import shutil
warnings.filterwarnings('ignore')

from config.settings import get_path_config, get_device_config, get_model_config, get_inference_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


class SanitizeLogitsProcessor(LogitsProcessor):
    """Replace invalid logits before sampling."""

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Sanitize one logits tensor before sampling."""
        # Replace NaN and +inf with -inf so these tokens are never sampled
        bad_mask = torch.isnan(scores) | (scores == float('inf'))
        if bad_mask.any():
            scores = scores.masked_fill(bad_mask, float('-inf'))
        # If all logits in a row are -inf, multinomial sampling will fail.
        # Fall back to uniform distribution over the full vocabulary for that row.
        all_invalid = (scores == float('-inf')).all(dim=-1, keepdim=True)
        if all_invalid.any():
            scores = scores.masked_fill(all_invalid.expand_as(scores), 0.0)
        return scores


class LanguageModel:
    """Load Qwen3-8B and manage generation and LoRA adapters."""

    def __init__(self, model_path: str | None = None, use_4bit: bool = True):
        """Initialize the instance."""
        path_cfg = get_path_config()
        device_cfg = get_device_config()
        model_cfg = get_model_config()

        if model_path is None:
            self.model_path = str(path_cfg.get_text_model_path())
            self.model_version = model_cfg.version
        else:
            self.model_path = model_path
            self.model_version = 'qwen3_8b'
            if 'Qwen3-8B' not in model_path and 'qwen3-8B' not in model_path:
                logger.warning(f"Path does not contain the Qwen3-8B identifier; continuing with the qwen3_8b configuration: {model_path}")

        self.device = device_cfg.get_device()
        self.device_cfg = device_cfg

        if self.device != "cuda":
            self.use_4bit = False
        elif use_4bit:
            self.use_4bit = True
        else:
            self.use_4bit = device_cfg.should_use_quantization()
        self.gpu_tier = device_cfg.get_gpu_tier()
        self.is_high_end_gpu = device_cfg.is_high_end_gpu

        self.model = None
        self.tokenizer = None
        self.current_lora_path = None
        self.is_lora_loaded = False

        logger.info("Initializing language model")
        logger.info(f"Model version: {self.model_version}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU information: {device_cfg.get_gpu_info()}")
        logger.info(f"Quantization: {'4-bit' if self.use_4bit else 'FP16 (no quantization)'}")
        logger.info(f"GPU profile: {self.gpu_tier.upper()} tier")

        self._load_base_model()

    def _load_base_model(self):
        """Load base model."""
        try:
            logger.info("Loading base model...")

            if self.use_4bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                quantization_config = None

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side='left',
                use_fast=True,
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = '<|endoftext|>'
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = '<|im_end|>'

            device_map = {"": 0} if self.device == "cuda" else "auto"

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.float16 if not self.use_4bit else None,
                low_cpu_mem_usage=True
            )

            self.model.eval()
            logger.info("Base model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_target_modules(self) -> list:
        """Return target modules."""
        return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def _clean_lora_config(self, lora_path: Path) -> Path | None:
        """Clean LoRA config."""
        try:
            config_file = lora_path / "adapter_config.json"
            if not config_file.exists():
                logger.warning(f"Configuration file does not exist: {config_file}")
                return None

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            incompatible_params = [
                'alora_invocation_tokens',
                'alora_prefix',
                'alora_suffix',
                'arrow_config',
            ]

            needs_cleaning = any(param in config for param in incompatible_params)

            if not needs_cleaning:
                return lora_path

            temp_dir = Path(tempfile.mkdtemp(prefix="lora_cleaned_"))
            logger.info(f"Created temporary directory: {temp_dir}")

            for item in lora_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, temp_dir / item.name)

            for param in incompatible_params:
                if param in config:
                    logger.info(f"Removed incompatible parameter: {param}")
                    del config[param]

            cleaned_config_file = temp_dir / "adapter_config.json"
            with open(cleaned_config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logger.info("LoRA configuration cleaned")
            return temp_dir

        except Exception as e:
            logger.error(f"Failed to clean LoRA configuration: {e}")
            return None

    def load_lora_from_path(self, lora_path: str) -> bool:
        """Load LoRA from path."""
        temp_dir = None
        try:
            lora_path = Path(lora_path)

            if not lora_path.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return False

            if self.current_lora_path == str(lora_path):
                logger.info(f"LoRA adapter is already loaded: {lora_path}")
                return True

            if self.is_lora_loaded:
                logger.info("Unloading the previous LoRA adapter...")
                self.unload_lora()

            logger.info(f"Loading LoRA weights: {lora_path}")

            cleaned_path = self._clean_lora_config(lora_path)
            if cleaned_path is None:
                logger.warning("Configuration cleanup failed; attempting to load the adapter directly")
                cleaned_path = lora_path
            elif cleaned_path != lora_path:
                temp_dir = cleaned_path
                logger.info(f"Using cleaned configuration: {cleaned_path}")

            self.model = PeftModel.from_pretrained(
                self.model,
                str(cleaned_path),
                is_trainable=False
            )

            self.current_lora_path = str(lora_path)
            self.is_lora_loaded = True

            logger.info("LoRA adapter loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            return False

        finally:
            if temp_dir and temp_dir != lora_path:
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Removed temporary directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory: {e}")

    def unload_lora(self) -> bool:
        """Unload the active LoRA adapter."""
        try:
            if not self.is_lora_loaded:
                logger.info("No LoRA adapter is currently loaded")
                return True

            logger.info("Unloading LoRA adapter...")

            if hasattr(self.model, 'unload'):
                self.model = self.model.unload()
            elif hasattr(self.model, 'get_base_model'):
                self.model = self.model.get_base_model()
            else:
                logger.warning("The model does not support unload(); resetting only the adapter state flag")

            self.current_lora_path = None
            self.is_lora_loaded = False

            logger.info("LoRA adapter unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter: {e}")
            self.current_lora_path = None
            self.is_lora_loaded = False
            return False

    def _get_stop_tokens(self) -> list:
        """Return stop tokens."""
        stop_tokens = []
        if self.tokenizer.eos_token_id is not None:
            stop_tokens.append(self.tokenizer.eos_token_id)

        im_end_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
        if im_end_id is not None and im_end_id != self.tokenizer.unk_token_id:
            stop_tokens.append(im_end_id)

        endoftext_id = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        if endoftext_id is not None and endoftext_id != self.tokenizer.unk_token_id:
            stop_tokens.append(endoftext_id)

        return list(set(stop_tokens))

    def _build_generation_config(self, max_new_tokens: int, temperature: float,
                                  top_p: float, top_k: int,
                                  repetition_penalty: float) -> dict:
        """Build generation config."""
        return {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "do_sample": True if temperature > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self._get_stop_tokens(),
            "use_cache": True,
            "logits_processor": LogitsProcessorList([SanitizeLogitsProcessor()]),
        }

    def _model_generate(self, inputs: dict, generation_config: dict) -> torch.Tensor:
        """Generate text with the model."""
        with torch.no_grad():
            if self.use_4bit:
                return self.model.generate(**inputs, **generation_config)
            else:
                with torch.cuda.amp.autocast():
                    return self.model.generate(**inputs, **generation_config)

    def _suppress_thinking(self, prompt: str) -> str:
        """
        For Qwen3-8B, append an empty think block to the prompt to disable
        chain-of-thought generation. This mirrors what
        tokenizer.apply_chat_template(..., enable_thinking=False) produces:
        the pre-filled empty <think></think> block tells the model to skip
        reasoning and output the answer directly.
        Has no effect on other model versions or if already suppressed.
        """
        if self.model_version != 'qwen3_8b':
            return prompt
        if '<think>' in prompt:
            return prompt
        return prompt + '<think>\n\n</think>\n'

    def generate(self, prompt: str, max_new_tokens: int = 2048,
                 temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                 repetition_penalty: float = 1.1) -> str:
        """Generate output."""
        try:
            # For Qwen3-8B: pre-fill empty think block to disable thinking mode
            _gen_start = time.perf_counter()
            logger.info(
                f"[TIMING][generate] called | torch_threads={torch.get_num_threads()} | interop={torch.get_num_interop_threads()} | thread={threading.get_ident()}")
            prompt = self._suppress_thinking(prompt)

            _tok_start = time.perf_counter()
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            input_length = inputs['input_ids'].shape[1]
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            _tok_end = time.perf_counter()
            logger.info(f"[TIMING][generate] tokenize 1 sample: {_tok_end - _tok_start:.3f}s | input_len={input_length}")

            generation_config = self._build_generation_config(
                max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )
            outputs = self._model_generate(inputs, generation_config)

            generated_ids = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            _gpu_done = time.perf_counter()
            logger.info(f"[TIMING][generate] GPU done: {_gpu_done - _gen_start:.3f}s | new_tokens={len(generated_ids)}")
            generated_text = self._post_process_text(generated_text)

            _post_done = time.perf_counter()
            logger.info(
                f"[TIMING][generate] post-process: {_post_done - _gpu_done:.3f}s | total: {_post_done - _gen_start:.3f}s")
            return generated_text

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def generate_batch(self, prompts: list, max_new_tokens: int = 2048,
                      temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                      repetition_penalty: float = 1.1, batch_size: int = None) -> list:
        """Generate batch."""
        if not prompts:
            return []

        if batch_size is None:
            if self.gpu_tier == 'high':
                batch_size = 16
            elif self.gpu_tier == 'mid':
                batch_size = 2
            else:
                batch_size = 1

        logger.info(f"Batch inference: {len(prompts)} samples, batch_size={batch_size}")

        results = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        all_batches = []
        for i in range(0, len(prompts), batch_size):
            bp = [self._suppress_thinking(p) for p in prompts[i:i + batch_size]]
            all_batches.append(bp)

        generation_config = self._build_generation_config(
            max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

        def _tokenize(batch):
            return self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )

        pbar = tqdm(total=len(prompts), desc="Batch generation", unit="samples", ncols=100)

        tok_executor = ThreadPoolExecutor(max_workers=2)
        next_tok_future = tok_executor.submit(_tokenize, all_batches[0]) if all_batches else None

        for batch_idx, batch_prompts in enumerate(all_batches):
            try:
                _tok_fetch_start = time.perf_counter()
                inputs_raw = next_tok_future.result()

                if batch_idx + 1 < len(all_batches):
                    next_tok_future = tok_executor.submit(_tokenize, all_batches[batch_idx + 1])

                input_lengths = inputs_raw['input_ids'].shape[1]
                inputs = {k: v.to(self.model.device) for k, v in inputs_raw.items()}
                _tok_end = time.perf_counter()
                logger.info(
                    f"[TIMING][batch] tok_fetch+H2D {len(batch_prompts)} samples: {_tok_end - _tok_fetch_start:.3f}s | seq_len={input_lengths}")

                _infer_start = time.perf_counter()
                batch_results = self._generate_and_decode_batch(
                    inputs, input_lengths, generation_config
                )
                _infer_end = time.perf_counter()
                logger.info(
                    f"[TIMING][batch] model.generate {len(batch_prompts)} samples: {_infer_end - _infer_start:.3f}s")

                results.extend(batch_results)
                pbar.update(len(batch_prompts))

            except Exception as e:
                error_str = str(e)
                i = batch_idx * batch_size
                if 'out of memory' in error_str.lower() and len(batch_prompts) > 1:
                    logger.warning(
                        f"Out of memory during batch generation (batch {i//batch_size + 1}/{num_batches}); "
                        f"current batch_size={len(batch_prompts)}. Retrying with smaller batches..."
                    )
                    torch.cuda.empty_cache()
                    retry_results = self._retry_batch_with_smaller_size(
                        batch_prompts, generation_config
                    )
                    results.extend(retry_results)
                    if all(r != "" for r in retry_results):
                        logger.info(
                            f"Reduced-size retry succeeded for batch {i//batch_size + 1}/{num_batches}"
                        )
                else:
                    logger.error(f"Batch generation failed (batch {i//batch_size + 1}/{num_batches}): {e}")
                    results.extend([""] * len(batch_prompts))
                pbar.update(len(batch_prompts))

        pbar.close()
        tok_executor.shutdown(wait=False)
        return results

    def _generate_and_decode_batch(self, inputs: dict, input_lengths: int,
                                    generation_config: dict) -> list:
        """Generate and decode batch."""
        outputs = self._model_generate(inputs, generation_config)

        generated_ids_list = [output[input_lengths:] for output in outputs]
        decoded_texts = self.tokenizer.batch_decode(generated_ids_list, skip_special_tokens=True)

        with ThreadPoolExecutor(max_workers=min(len(decoded_texts), os.cpu_count() or 16)) as _exec:
            return list(_exec.map(self._post_process_text, decoded_texts))

    def _retry_batch_with_smaller_size(self, batch_prompts: list,
                                        generation_config: dict) -> list:
        """Retry a failed batch with a smaller batch size."""
        retry_size = max(1, len(batch_prompts) // 2)
        retry_results = []

        for r in range(0, len(batch_prompts), retry_size):
            retry_batch = batch_prompts[r:r + retry_size]
            try:
                retry_inputs = self.tokenizer(
                    retry_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=8192
                )
                retry_input_lengths = retry_inputs['input_ids'].shape[1]
                retry_inputs = {k: v.to(self.model.device) for k, v in retry_inputs.items()}

                chunk_results = self._generate_and_decode_batch(
                    retry_inputs, retry_input_lengths, generation_config
                )
                retry_results.extend(chunk_results)
                torch.cuda.empty_cache()

            except Exception as retry_e:
                logger.error(f"Reduced-size retry failed (retry_size={retry_size}): {retry_e}")
                retry_results.extend([""] * len(retry_batch))

        return retry_results

    def _post_process_text(self, text: str) -> str:
        """Post-process generated text."""
        text = self._clean_generated_text(text)
        return self._truncate_after_three_parts(text)

    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text."""
        text = text.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()

        import re

        chinese_match = re.search(r'[\u4e00-\u9fff]', text)
        if chinese_match:
            idx = chinese_match.start()
            truncate_pos = idx
            for i in range(idx - 1, max(0, idx - 50), -1):
                if text[i] in '.!?\n':
                    truncate_pos = i + 1
                    break
            text = text[:truncate_pos].strip()

        return text

    def _truncate_after_three_parts(self, text: str) -> str:
        """Truncate output after the three required sections."""
        import re

        MAX_DO_NOT = 6

        def _limit_do_not(content: str) -> str:
            """Limit the ?Things to Avoid? section."""
            sentences = re.split(r'(?<=[.!?])\s+', content.strip())
            do_not_count = 0
            kept = []
            for s in sentences:
                if re.match(r'Do not\b', s.strip(), re.IGNORECASE):
                    do_not_count += 1
                    if do_not_count > MAX_DO_NOT:
                        break
                kept.append(s)
            return ' '.join(kept).strip()

        lines = text.split('\n')

        definition_idx = emphasis_idx = avoid_idx = None
        for i, line in enumerate(lines):
            ls = line.strip()
            if ls.startswith('Definition:') and definition_idx is None:
                definition_idx = i
            elif (ls.startswith('Emphasis & Caution:') or ls.startswith('Emphasis and Caution:')) \
                    and emphasis_idx is None:
                emphasis_idx = i
            elif ls.startswith('Things to Avoid:') and avoid_idx is None:
                avoid_idx = i

        if definition_idx is not None and emphasis_idx is not None and avoid_idx is not None:
            avoid_header_line = lines[avoid_idx]
            inline = avoid_header_line[avoid_header_line.index('Things to Avoid:') + len('Things to Avoid:'):].strip()

            extra = []
            for line in lines[avoid_idx + 1:]:
                s = line.strip()
                if s == '' or re.match(
                    r'^(Definition:|Emphasis\s*(?:&|and)\s*Caution:|Things to Avoid:)',
                    s, re.IGNORECASE
                ):
                    break
                extra.append(s)

            full_avoid = (inline + ' ' + ' '.join(extra)).strip()
            limited_avoid = _limit_do_not(full_avoid)

            result_lines = lines[definition_idx:avoid_idx]
            result_lines.append(
                f"Things to Avoid: {limited_avoid}" if limited_avoid else avoid_header_line.rstrip()
            )
            return '\n'.join(result_lines)

        flat = ' '.join(lines)
        m = re.search(
            r'(Definition:.*?)'
            r'(Emphasis\s*(?:&|and)\s*Caution:.*?)'
            r'(Things to Avoid:\s*)(.*)',
            flat,
            re.DOTALL | re.IGNORECASE
        )
        if m:
            def_part    = m.group(1).strip()
            emph_part   = m.group(2).strip()
            avoid_body  = m.group(4).strip()
            limited_avoid = _limit_do_not(avoid_body)
            parts = [def_part, emph_part, f"Things to Avoid: {limited_avoid}"]
            return '\n'.join(parts)

        return text

    def get_lora_status(self) -> dict:
        """Return LoRA status."""
        return {
            'is_loaded': self.is_lora_loaded,
            'current_path': self.current_lora_path,
            'base_model': self.model_path
        }


class InstructionGenerator:
    """Generate instructions with the configured model interface."""

    def __init__(self, model_path: str | None = None, use_4bit: bool = True):
        """Initialize the instance."""
        self.language_model = LanguageModel(
            model_path=model_path,
            use_4bit=use_4bit
        )
        self._inference_cfg = get_inference_config()
        logger.info("Instruction generator initialized")

    def load_expert(self, expert_name_or_path: str) -> bool:
        """Load expert."""
        path_cfg = get_path_config()

        # First try checkpoints/lora_moe/{expert_name}/ (framework standard path)
        lora_path = path_cfg.PROJECT_ROOT / 'checkpoints' / 'lora_moe' / expert_name_or_path
        if not lora_path.exists():
            # Fallback: try via get_expert_weight_path (legacy compatibility)
            lora_path = path_cfg.get_expert_weight_path(expert_name_or_path)

        if not lora_path.exists():
            lora_path = Path(expert_name_or_path)

        return self.language_model.load_lora_from_path(str(lora_path))

    def unload_expert(self) -> bool:
        """Unload an expert."""
        return self.language_model.unload_lora()

    def generate(self, prompt: str, max_new_tokens: int = None,
                 temperature: float = None, top_p: float = None,
                 top_k: int = None, repetition_penalty: float = None) -> str:
        """Generate output."""
        cfg = self._inference_cfg
        return self.language_model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
        )

    def get_expert_status(self) -> dict:
        """Return expert status."""
        return self.language_model.get_lora_status()

    def generate_batch(self, prompts: list, max_new_tokens: int = None,
                      temperature: float = None, top_p: float = None,
                      top_k: int = None, repetition_penalty: float = None,
                      batch_size: int = None) -> list:
        """Generate batch."""
        cfg = self._inference_cfg
        return self.language_model.generate_batch(
            prompts=prompts,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else cfg.max_new_tokens,
            temperature=temperature if temperature is not None else cfg.temperature,
            top_p=top_p if top_p is not None else cfg.top_p,
            top_k=top_k if top_k is not None else cfg.top_k,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty,
            batch_size=batch_size
        )
