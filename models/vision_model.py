"""Provide Qwen3-VL-8B-Instruct image and FlowChart recognition."""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from peft import PeftModel
import json
import re
import gc
from pathlib import Path
from threading import Thread

from config.settings import get_path_config, get_device_config, get_vision_model_config
from models.prompt_templates.image_template import ImageInstructionTemplate
from models.prompt_templates.uml_template import UMLInstructionTemplate
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisionModel:
    """Recognize image and FlowChart inputs with Qwen3-VL."""

    def __init__(self, model_path: str | None = None, version: str = None):
        """Initialize the instance."""
        path_cfg = get_path_config()
        device_cfg = get_device_config()

        self.version = version or get_vision_model_config().version
        configured_model_path = path_cfg.get_vision_model_path(self.version)
        if model_path is None:
            self.model_path = str(configured_model_path)
        else:
            self.model_path = model_path
        self.model_name = "Qwen3-VL-8B-Instruct"

        self.device = device_cfg.get_device()
        self.device_cfg = device_cfg
        self.model = None
        self.processor = None
        self.current_lora_path = None
        self.is_lora_loaded = False

        self.use_quantization = device_cfg.should_use_quantization()

        self.gpu_tier = device_cfg.get_gpu_tier()
        self.uml_gen_config = device_cfg.get_generation_config('uml')
        self.image_gen_config = device_cfg.get_generation_config('image')

        self.enable_streaming = device_cfg.enable_streaming

        logger.info(f"Initializing vision model: {self.model_name}")
        logger.info(f"Model version: {self.version}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"GPU information: {device_cfg.get_gpu_info()}")
        logger.info(f"Quantization: {'4-bit' if self.use_quantization else 'FP16 (no quantization)'}")
        logger.info(f"GPU profile: {self.gpu_tier.upper()} tier")
        logger.info(f"FlowChart generation tokens: {self.uml_gen_config['max_new_tokens']}, image generation tokens: {self.image_gen_config['max_new_tokens']}")
        logger.info(f"Streaming output: {'enabled' if self.enable_streaming else 'disabled'}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        self._load_base_model()

    def get_model_info(self) -> dict:
        """Return model info."""
        return {
            'version': self.version,
            'model_name': self.model_name,
            'model_path': self.model_path,
            'lora_loaded': self.is_lora_loaded,
            'lora_path': self.current_lora_path,
            'device': self.device
        }

    def _load_base_model(self):
        """Load base model."""
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            if self.processor.tokenizer.pad_token is None:
                logger.info("Tokenizer has no pad_token; using eos_token instead")
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
                self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

            if not hasattr(self.processor.tokenizer, 'padding_side'):
                self.processor.tokenizer.padding_side = 'left'

            if self.use_quantization:
                logger.info("Using 4-bit quantization to reduce GPU memory usage...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

                logger.info("Loading model with 4-bit quantization...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            else:
                logger.info("Using the FP16 configuration optimized for high-end GPUs...")
                logger.info("Loading model in FP16 without quantization...")
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

            logger.info("Model loaded successfully")

            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Allocated GPU memory: {memory_allocated:.2f} GB")
                logger.info(f"Reserved GPU memory: {memory_reserved:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


    def _build_messages(self, prompt: str, image_path: str | None = None) -> list:
        """Build messages."""
        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages: list):
        """Prepare inputs."""
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        return inputs

    def _model_generate_vision(self, inputs, gen_kwargs: dict):
        """Generate vision-model output."""
        with torch.no_grad():
            if self.use_quantization:
                return self.model.generate(**inputs, **gen_kwargs)
            else:
                with torch.amp.autocast('cuda'):
                    return self.model.generate(**inputs, **gen_kwargs)

    def _decode_output(self, inputs, generated_ids) -> str:
        """Decode output."""
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response

    def _cleanup_gpu(self, *tensors):
        """Release GPU resources."""
        for t in tensors:
            del t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _build_gen_kwargs(self, gen_config: dict, extra_kwargs: dict = None) -> dict:
        """Build gen kwargs."""
        eos_token_id = self.processor.tokenizer.eos_token_id
        kwargs = {
            'max_new_tokens': gen_config['max_new_tokens'],
            'min_new_tokens': 1,
            'temperature': gen_config['temperature'],
            'do_sample': True,
            'top_p': gen_config['top_p'],
            'use_cache': gen_config['use_cache'],
            'num_beams': 1,
            'pad_token_id': self.processor.tokenizer.pad_token_id,
            'eos_token_id': eos_token_id,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs

    def _extract_json(self, response: str) -> str | None:
        """Extract JSON."""
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json_match.group(0)
        return None

    def load_lora_from_path(self, lora_path: str) -> bool:
        """Load LoRA from path."""
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

            self.model = PeftModel.from_pretrained(
                self.model,
                str(lora_path),
                is_trainable=False
            )

            self.current_lora_path = str(lora_path)
            self.is_lora_loaded = True

            logger.info("LoRA adapter loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load LoRA adapter: {e}")
            return False

    def unload_lora(self) -> bool:
        """Unload the active LoRA adapter."""
        try:
            if not self.is_lora_loaded:
                logger.info("No LoRA adapter is currently loaded")
                return True

            logger.info("Unloading LoRA adapter...")

            self.model = self.model.unload()

            self.current_lora_path = None
            self.is_lora_loaded = False

            logger.info("LoRA adapter unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unload LoRA adapter: {e}")
            return False

    def generate(self, prompt: str, image_path: str | None = None,
                 max_new_tokens: int = 1024, temperature: float = 0.3,
                 top_p: float = 0.8, do_sample: bool = True) -> str:
        """Generate output."""
        try:
            messages = self._build_messages(prompt, image_path)
            inputs = self._prepare_inputs(messages)

            gen_kwargs = self._build_gen_kwargs(
                {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature if do_sample else 1.0,
                    'top_p': top_p if do_sample else 1.0,
                    'use_cache': True,
                },
                extra_kwargs={'do_sample': do_sample}
            )

            generated_ids = self._model_generate_vision(inputs, gen_kwargs)
            response = self._decode_output(inputs, generated_ids)

            self._cleanup_gpu(inputs, generated_ids)

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def recognize_image(self, image_path: str, prompt: str | None = None) -> dict:
        """Recognize image."""
        if prompt is None:
            prompt = ImageInstructionTemplate.get_recognition_prompt()

        logger.info(f"Recognizing image: {Path(image_path).name}")

        try:
            messages = self._build_messages(prompt, image_path)
            inputs = self._prepare_inputs(messages)

            response, confidence = self._generate_with_confidence(inputs)

            self._cleanup_gpu(inputs)

            result = self._parse_image_response(response, image_path)
            result["confidence"] = confidence
            result["recognition_status"] = "success"

            logger.info(f"Recognition succeeded; confidence: {confidence:.3f}")
            return result

        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            return {
                "description": "",
                "details": {
                    "objects": [],
                    "scene": "unknown",
                    "spatial_info": ""
                },
                "confidence": 0.0,
                "recognition_status": "failed",
                "error": str(e)
            }

    def recognize_uml(self, uml_path: str, max_retries: int = 2, prompt: str | None = None,
                      streaming: bool | None = None) -> dict:
        """Recognize FlowChart."""
        if prompt is None:
            prompt = UMLInstructionTemplate.get_recognition_prompt()

        use_streaming = streaming if streaming is not None else self.enable_streaming

        logger.info(f"Recognizing FlowChart diagram: {Path(uml_path).name}")
        logger.info(f"Generation configuration: max_tokens={self.uml_gen_config['max_new_tokens']}, temp={self.uml_gen_config['temperature']}")
        logger.info(f"Streaming output: {'enabled' if use_streaming else 'disabled'}")

        for attempt in range(max_retries):
            try:
                messages = self._build_messages(prompt, uml_path)
                inputs = self._prepare_inputs(messages)

                if use_streaming:
                    response = self._generate_streaming(inputs, task_type='uml')
                else:
                    response = self._generate_standard(inputs, task_type='uml')

                self._cleanup_gpu(inputs)

                result = self._parse_uml_response(response, uml_path)

                if result['success'] or attempt == max_retries - 1:
                    if result['success']:
                        logger.info("FlowChart recognition succeeded")
                    else:
                        logger.warning("FlowChart recognition failed and the maximum number of retries has been reached")
                    return result
                else:
                    logger.warning(f"Attempt {attempt + 1} failed; retrying...")
                    continue

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"FlowChart recognition failed: {e}")
                    return {
                        'description': f"Recognition failed: {str(e)}",
                        'success': False,
                        'error': str(e)
                    }
                else:
                    logger.warning(f"Attempt {attempt + 1} raised an error: {e}; retrying...")
                    continue

    def _generate_standard(self, inputs, task_type: str = 'uml') -> str:
        """Generate standard."""
        gen_config = self.uml_gen_config if task_type == 'uml' else self.image_gen_config

        logger.info(f"[Standard generation] pad_token_id: {self.processor.tokenizer.pad_token_id}")
        logger.info(f"[Standard generation] Model in eval mode: {not self.model.training}")
        logger.info(f"[Standard generation] Input input_ids length: {inputs.input_ids.shape[1]}")
        logger.info(f"[Standard generation] max_new_tokens: {gen_config['max_new_tokens']}")
        logger.info(f"[Standard generation] Quantization enabled: {self.use_quantization}")

        gen_kwargs = self._build_gen_kwargs(gen_config)

        logger.info("[Standard generation] Calling model.generate()...")
        generated_ids = self._model_generate_vision(inputs, gen_kwargs)
        logger.info("[Standard generation] model.generate() completed")

        logger.info(f"[Standard generation] Generation complete; generated_ids shape: {generated_ids.shape}")
        logger.info(f"[Standard generation] Input length: {inputs.input_ids.shape[1]}, output length: {generated_ids.shape[1]}")
        logger.info(f"[Standard generation] New tokens: {generated_ids.shape[1] - inputs.input_ids.shape[1]}")

        response = self._decode_output(inputs, generated_ids)

        logger.info(f"[Standard generation] Decoding complete; generated text length: {len(response)}")
        if len(response) > 0:
            logger.info(f"[Standard generation] Generated text preview: {response[:100]}...")
        else:
            logger.error("[Standard generation] Generated text is empty")

        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    def _consume_streamer(self, streamer, thread, thread_error) -> str:
        """Collect streamed text while monitoring the generation worker."""
        import queue

        chunks = []
        iteration_count = 0

        while True:
            try:
                new_text = next(streamer)
            except StopIteration:
                break
            except queue.Empty:
                if thread.is_alive():
                    continue

                error = thread_error.get('error')
                if error:
                    raise RuntimeError(error)
                raise RuntimeError(
                    "Streaming generation stopped before signaling completion"
                )

            iteration_count += 1
            logger.debug(
                f"[Streaming generation - iteration] Iteration {iteration_count}; "
                f"received text length: {len(new_text) if new_text else 0}"
            )

            if new_text:
                print(new_text, end='', flush=True)
                chunks.append(new_text)

        thread.join()

        error = thread_error.get('error')
        if error:
            raise RuntimeError(error)

        generated_text = ''.join(chunks)
        if not generated_text.strip():
            raise ValueError("Streaming generation produced no output")

        return generated_text

    def _generate_streaming(self, inputs, task_type: str = 'uml') -> str:
        """Generate text with real-time console output and a safe fallback."""
        gen_config = self.uml_gen_config if task_type == 'uml' else self.image_gen_config
        thread_error = {'error': None}
        thread = None

        try:
            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=5.0
            )
            gen_kwargs = self._build_gen_kwargs(
                gen_config,
                extra_kwargs={'streamer': streamer}
            )

            def generate_with_error_capture():
                try:
                    self._model_generate_vision(inputs, gen_kwargs)
                except Exception as e:
                    import traceback
                    error_msg = (
                        f"Exception in generation thread: {str(e)}\n"
                        f"{traceback.format_exc()}"
                    )
                    logger.error(f"[Streaming generation - thread] {error_msg}")
                    thread_error['error'] = error_msg

            thread = Thread(
                target=generate_with_error_capture,
                daemon=True
            )
            thread.start()

            print("\n" + "="*80)
            print("Streaming generated content:")
            print("="*80)
            print("", flush=True)

            generated_text = self._consume_streamer(
                streamer,
                thread,
                thread_error
            )
            print("\n" + "="*80)
            return generated_text

        except Exception as e:
            import traceback

            if thread is not None and thread.is_alive():
                logger.warning(
                    "[Streaming generation] Waiting for the active generation "
                    "worker before using the standard fallback"
                )
                thread.join()

            logger.error(f"[Streaming generation] Failed: {str(e)}")
            logger.error(
                f"[Streaming generation] Exception details:\n"
                f"{traceback.format_exc()}"
            )
            logger.info("[Streaming generation] Falling back to standard generation")

            try:
                return self._generate_standard(inputs, task_type)
            except Exception as fallback_error:
                logger.error(f"[Standard generation] Fallback failed: {str(fallback_error)}")
                raise RuntimeError(
                    "Both streaming and standard generation failed: "
                    f"streaming={str(e)}, standard={str(fallback_error)}"
                )

    def _generate_with_confidence(self, inputs) -> tuple[str, float]:
        """Generate with confidence."""
        gen_kwargs = self._build_gen_kwargs(
            self.image_gen_config,
            extra_kwargs={'return_dict_in_generate': True, 'output_scores': True}
        )

        outputs = self._model_generate_vision(inputs, gen_kwargs)

        scores = outputs.scores
        entropies = []

        for score in scores:
            probs = F.softmax(score[0], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
            normalized_entropy = min(entropy / 10.0, 1.0)
            entropies.append(normalized_entropy)

        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.5
        confidence = 1.0 - avg_entropy

        response = self._decode_output(inputs, outputs.sequences)

        return response, float(confidence)

    def _parse_image_response(self, response: str, image_path: str) -> dict:
        """Parse image response."""
        try:
            json_str = self._extract_json(response) or response
            result = json.loads(json_str)

            if 'description' not in result:
                result['description'] = response[:200]

            if 'details' not in result:
                result['details'] = {
                    "objects": [],
                    "scene": "unknown scene",
                    "spatial_info": ""
                }

            return result

        except json.JSONDecodeError:
            logger.warning("JSON parsing failed; using the fallback parser")
            return {
                "description": response[:200] if response else "",
                "details": {
                    "objects": [],
                    "scene": "unknown",
                    "spatial_info": ""
                }
            }

    def _parse_uml_response(self, response: str, uml_path: str) -> dict:
        """Parse FlowChart response."""
        try:
            json_str = self._extract_json(response) or response

            json_str = self._fix_truncated_json(json_str)

            result = json.loads(json_str)

            result.setdefault('actors', [])
            result.setdefault('use_cases', [])
            result.setdefault('system_boundary', {"name": "Not Recognized", "is_present": False})
            result.setdefault('relationships', [])
            result.setdefault('overall_description', "")

            result['success'] = True
            return {"description": json.dumps(result, ensure_ascii=False), "success": True}

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FlowChart JSON: {e}")
            return {
                'description': response[:500] if response else "",
                'success': False,
                'error': str(e)
            }

    def _fix_truncated_json(self, json_str: str) -> str:
        """Repair truncated JSON output."""
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')

        if open_braces > close_braces or open_brackets > close_brackets:
            last_complete = max(
                json_str.rfind('},'),
                json_str.rfind('],'),
                json_str.rfind('}')
            )

            if last_complete > 0:
                json_str = json_str[:last_complete + 1]

            json_str += ']' * (open_brackets - json_str.count(']'))
            json_str += '}' * (open_braces - json_str.count('}'))

        return json_str

    def get_lora_status(self) -> dict:
        """Return LoRA status."""
        return {
            'is_loaded': self.is_lora_loaded,
            'current_path': self.current_lora_path,
            'base_model': self.model_path
        }
