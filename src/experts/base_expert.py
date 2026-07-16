"""Define the common interface and model lifecycle for domain experts."""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
_cpu_count = os.cpu_count() or 25
torch.set_num_threads(min(16, _cpu_count))
torch.set_num_interop_threads(min(8, _cpu_count // 3))
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
import torch

from models.language_model import LanguageModel
from src.utils.logger import get_logger

logger = get_logger('experts.base')


class BaseExpert(ABC):
    """Define the shared expert interface and model lifecycle."""

    _shared_base_model = None
    _shared_base_model_path = None

    def __init__(self,
                 expert_name: str,
                 base_model_path: str,
                 lora_path: Optional[str] = None,
                 use_4bit: bool = True,
                 version: Optional[str] = None):
        """Initialize the instance."""
        self.expert_name = expert_name
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.use_4bit = use_4bit
        self.version = version

        self.model = None
        self.is_model_loaded = False

        logger.info(f"初始化专家: {expert_name}")
        logger.info(f"基础模型: {base_model_path}")
        if version:
            logger.info(f"模型版本: {version}")
        if lora_path:
            logger.info(f"LoRA路径: {lora_path}")

    def load_model(self) -> bool:
        """Load model."""
        try:
            logger.info(f"加载{self.expert_name}的模型...")

            self.model = LanguageModel(
                model_path=self.base_model_path,
                use_4bit=self.use_4bit
            )

            if self.lora_path:
                lora_path = Path(self.lora_path)
                if lora_path.exists():
                    logger.info(f"加载LoRA权重: {self.lora_path}")
                    success = self.model.load_lora_from_path(str(self.lora_path))
                    if not success:
                        logger.warning("LoRA加载失败,使用基础模型")
                else:
                    logger.warning(f"LoRA路径不存在: {self.lora_path}")
                    logger.warning("使用基础模型(未微调)")

            self.is_model_loaded = True
            logger.info("模型加载完成")
            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.is_model_loaded = False
            return False

    def unload_model(self) -> bool:
        """Unload model."""
        try:
            if self.model:
                if self.model.is_lora_loaded:
                    self.model.unload_lora()

                del self.model
                self.model = None

                import gc
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.is_model_loaded = False
                logger.info("模型已卸载")

            return True

        except Exception as e:
            logger.error(f"模型卸载失败: {e}")
            return False

    @abstractmethod
    def generate_instruction(self, input_data: Any) -> str:
        """Generate instruction."""
        pass

    @abstractmethod
    def validate_output(self, instruction: str) -> bool:
        """Validate output."""
        pass

    def _generate_with_model(self,
                            prompt: str,
                            max_new_tokens: int = 2048,
                            temperature: float = 0.7,
                            top_p: float = 0.9,
                            top_k: int = 50,
                            repetition_penalty: float = 1.1,
                            sample_index: int = None,
                            verbose: bool = True) -> str:
        """Generate with model."""
        logger.info(f"[ROUTE] _generate_with_model called (SINGLE) | expert={self.expert_name}")
        if not self.is_model_loaded:
            logger.error("模型未加载,无法生成")
            return ""

        try:
            show_debug = verbose and (sample_index is None or sample_index < 3)

            if show_debug:
                logger.info("=" * 80)
                logger.info(f"[调试] 样本 {sample_index + 1 if sample_index is not None else 'N/A'} - 完整Prompt内容:")
                logger.info("-" * 80)
                logger.info(prompt)
                logger.info("=" * 80)
                logger.info(f"[调试] 生成参数: temp={temperature}, top_p={top_p}, top_k={top_k}, rep_penalty={repetition_penalty}")

            generated_text = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )

            if show_debug:
                logger.info(f"[调试] 原始生成内容长度: {len(generated_text)} 字符")
                logger.info(f"[调试] 原始生成内容（前500字符）：\n{generated_text[:500]}")
                if len(generated_text) > 500:
                    logger.info(f"[调试] 原始生成内容（后200字符）：\n{generated_text[-200:]}")

            return generated_text

        except Exception as e:
            logger.error(f"生成失败: {e}")
            return ""

    def _generate_batch_with_model(self,
                                   prompts: list,
                                   max_new_tokens: int = 2048,
                                   temperature: float = 0.7,
                                   top_p: float = 0.9,
                                   top_k: int = 50,
                                   repetition_penalty: float = 1.1,
                                   batch_size: int = None,
                                   start_index: int = 0,
                                   verbose: bool = True) -> list:
        """Generate batch with model."""
        logger.info(
            f"[ROUTE] _generate_batch_with_model called (BATCH) | expert={self.expert_name} | prompts={len(prompts)} | batch_size={batch_size}")
        if not self.is_model_loaded:
            logger.error("模型未加载,无法生成")
            return [""] * len(prompts)

        try:
            show_debug = verbose and start_index < 3

            if show_debug:
                logger.info(f"批量生成 - 共{len(prompts)}个样本，起始索引{start_index}")

            if hasattr(self.model, 'generate_batch'):
                results = self.model.generate_batch(
                    prompts=prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    batch_size=batch_size
                )

                return results
            else:
                logger.warning("模型不支持批量生成，降级到逐个生成")
                results = []
                for i, prompt in enumerate(prompts):
                    result = self._generate_with_model(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        sample_index=start_index + i,
                        verbose=verbose
                    )
                    results.append(result)
                return results

        except Exception as e:
            logger.error(f"批量生成失败: {e}")
            return [""] * len(prompts)

    def _normalize_instruction(self, instruction: str) -> str:
        """Normalize instruction."""
        import re

        if not instruction:
            return instruction

        text = instruction.strip()

        header_normalizations = [
            (r'^DEFINITION:', 'Definition:'),
            (r'^definition:', 'Definition:'),
            (r'^EMphasis\s*&\s*Caution:', 'Emphasis & Caution:'),
            (r'^EMPHASIS\s*&\s*CAUTION:', 'Emphasis & Caution:'),
            (r'^emphasis\s*&\s*caution:', 'Emphasis & Caution:'),
            (r'^Emphasis\s*and\s*Caution:', 'Emphasis & Caution:'),
            (r'^EMPHASIS\s*AND\s*CAUTION:', 'Emphasis & Caution:'),
            (r'^EMphasis\s*and\s*Caution:', 'Emphasis & Caution:'),
            (r'^THINGS\s*TO\s*AVOID:', 'Things to Avoid:'),
            (r'^things\s*to\s*avoid:', 'Things to Avoid:'),
        ]
        normalized_lines = []
        for line in text.split('\n'):
            line_stripped = line.strip()
            matched = False
            for pattern, replacement in header_normalizations:
                if re.match(pattern, line_stripped):
                    line_stripped = re.sub(pattern, replacement, line_stripped, count=1)
                    matched = True
                    break
            normalized_lines.append(line_stripped if matched else line)
        text = '\n'.join(normalized_lines)

        if re.match(r'^in this task\b', text, re.IGNORECASE) and not text.startswith('Definition:'):
            text = 'Definition: ' + text
            logger.debug("自动补全 'Definition:' 标签（原始以 'In this task' 开头）")

        text = re.sub(r'(\s*[-_]\s*){2,}\s*$', '', text).strip()
        lines_clean = []
        for line in text.split('\n'):
            line = re.sub(r'(\s*_\s*){2,}\s*$', '', line).rstrip()
            lines_clean.append(line)
        text = '\n'.join(lines_clean).strip()

        return text

    def _extract_three_part_instruction(self, text: str) -> str:
        """Extract three part instruction."""
        # logger.info("=" * 80)
        # logger.info("-" * 80)
        # logger.info(text)
        # logger.info("=" * 80)

        if not text:
            logger.warning("[提取结束] 输入文本为空")
            return ""

        lines = text.split('\n')

        # for i, line in enumerate(lines):

        definition_line = None
        emphasis_line = None
        avoid_line = None

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if line_stripped.startswith('Definition:'):
                definition_line = i
            elif line_stripped.startswith('Emphasis & Caution:') or line_stripped.startswith('Emphasis and Caution:'):
                emphasis_line = i
            elif line_stripped.startswith('Things to Avoid:'):
                avoid_line = i


        if definition_line is not None and emphasis_line is not None and avoid_line is not None:
            if definition_line < emphasis_line < avoid_line:
                extracted_lines = []
                for idx, line_idx in enumerate([definition_line, emphasis_line, avoid_line]):
                    line = lines[line_idx].strip()
                    cleaned_line = self._clean_instruction_line(line)
                    extracted_lines.append(cleaned_line)

                extracted_text = '\n'.join(extracted_lines)
                def_content = extracted_lines[0].split(':', 1)[1].strip() if ':' in extracted_lines[0] else ''

                invalid_keywords = ['Definition:', 'Emphasis', 'Things to Avoid', 'Caution']
                has_invalid_keyword = any(keyword in def_content for keyword in invalid_keywords)

                if def_content and def_content != '-' and not has_invalid_keyword:
                    extracted_lines = self._ensure_definition_format(extracted_lines)
                    extracted_text = '\n'.join(extracted_lines)

                    # logger.info("=" * 80)
                    # logger.info("-" * 80)
                    # logger.info(extracted_text)
                    # logger.info("=" * 80)
                    return extracted_text
                else:
                    if has_invalid_keyword:
                        logger.warning(f"[提取] 检测到重复标签，Definition内容无效: {def_content}")
                    else:
                        logger.warning("[提取] 提取的Definition内容为空，尝试智能分割")
            else:
                logger.warning(f"[提取] 行号顺序不正确: {definition_line}, {emphasis_line}, {avoid_line}")
        else:
            logger.warning("[提取] 未找到完整的三段式标签")

        return self._smart_split_to_three_parts(text)

    def _smart_split_to_three_parts(self, text: str) -> str:
        """Split generated text into the three required sections."""
        text = text.replace('Task Instructions:', '').strip()

        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        definition = None
        emphasis = None
        avoid = None

        for sentence in sentences:
            sentence_lower = sentence.lower()

            if not definition and ('in this task' in sentence_lower or
                                  ('draw' in sentence_lower and 'box' in sentence_lower) or
                                  ('annotate' in sentence_lower and 'this task' not in sentence_lower)):
                if not sentence.startswith('In this task'):
                    sentence = 'In this task, ' + sentence[0].lower() + sentence[1:]
                definition = sentence

            elif not emphasis and any(kw in sentence_lower for kw in
                                     ['focus', 'ensure', 'pay attention', 'must', 'should',
                                      'important', 'critical', 'key']):
                emphasis = sentence

            elif not avoid and any(kw in sentence_lower for kw in
                                   ['do not', "don't", 'avoid', 'never', 'not', 'skip']):
                avoid = sentence

        if len(sentences) >= 3:
            if not definition:
                definition = sentences[0]
                if not definition.startswith('In this task'):
                    definition = 'In this task, ' + definition[0].lower() + definition[1:]
            if not emphasis:
                emphasis = sentences[1] if len(sentences) > 1 else '-'
            if not avoid:
                avoid = sentences[2] if len(sentences) > 2 else '-'
        elif len(sentences) == 2:
            if not definition:
                definition = sentences[0]
            if not emphasis:
                emphasis = sentences[1]
            if not avoid:
                avoid = '-'
        elif len(sentences) == 1:
            if not definition:
                definition = sentences[0]
            emphasis = '-'
            avoid = '-'

        definition = definition or 'In this task, complete the required task.'
        emphasis = emphasis or '-'
        avoid = avoid or '-'

        formatted_instruction = f"""Definition: {definition}
Emphasis & Caution: {emphasis}
Things to Avoid: {avoid}"""

        logger.debug(f"智能分割完成：\n{formatted_instruction}")

        return formatted_instruction

    def _clean_instruction_line(self, line: str) -> str:
        """Clean instruction line."""
        import re


        prefixes = [
            'Definition:',
            'Emphasis & Caution:',
            'Emphasis and Caution:',
            'Things to Avoid:'
        ]

        current_prefix = None
        for prefix in prefixes:
            if line.startswith(prefix):
                current_prefix = prefix
                break

        if current_prefix is None:
            return line

        content = line[len(current_prefix):].strip()

        original_content = content
        max_iterations = 10
        for iteration in range(max_iterations):
            found_duplicate = False
            for check_prefix in prefixes:
                if content.startswith(check_prefix):
                    content = content[len(check_prefix):].strip()
                    found_duplicate = True
                    break
            if not found_duplicate:
                break

        # if content != original_content:

        if not content or content == '-':
            return f"{current_prefix} -"

        garbage_patterns = [
            r'\.is a (list|type|kind|form|way|computer program|software|document|method|system|tool)',
            r'\.is (often used|used|typically|commonly|generally|usually|one of|the)',
            r'\.is (a type|an|the|one of|part of)',
            r'\.(document|software|system|program|application|tool|platform|service)',
            r'\.(it|this|that|these|those) (is|are|was|were|can|could|will|would|may|might)',
            r'\.the (purpose|goal|aim|objective|main|primary|key|first)',
            r'\.(in order|to ensure|for|with|by|through|via)',
            r'\.[a-z]{2,}',
        ]

        original_content = content
        for pattern in garbage_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                idx = match.start()
                content = content[:idx + 1].strip()
                break

        # if content != original_content:

        chinese_match = re.search(r'[\u4e00-\u9fff]', content)
        if chinese_match:
            idx = chinese_match.start()
            truncate_pos = idx
            for i in range(idx - 1, max(0, idx - 50), -1):
                if content[i] in '.!?':
                    truncate_pos = i + 1
                    break
            content = content[:truncate_pos].strip()

        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) > 1:
            last_sentence = sentences[-1].strip()
            if last_sentence and len(last_sentence) > 0 and last_sentence[0].islower():
                content = ' '.join(sentences[:-1]).strip()

        if content and not content.endswith(('.', '!', '?', '-')):
            content += '.'

        cleaned_line = f"{current_prefix} {content}"

        return cleaned_line

    def _ensure_definition_format(self, lines: list) -> list:
        """Normalize the Definition section format."""
        if not lines or len(lines) < 1:
            return lines

        definition_line = lines[0]

        if not definition_line.startswith('Definition:'):
            return lines

        content = definition_line[len('Definition:'):].strip()

        if content.lower().startswith('in this task,'):
            return lines

        if content:
            content = content[0].lower() + content[1:] if len(content) > 1 else content.lower()

        new_definition = f"Definition: In this task, {content}"

        lines[0] = new_definition

        return lines


    def get_expert_info(self) -> Dict[str, Any]:
        """Return expert info."""
        info = {
            'expert_name': self.expert_name,
            'base_model': self.base_model_path,
            'lora_path': self.lora_path,
            'is_model_loaded': self.is_model_loaded,
            'use_4bit': self.use_4bit,
            'version': self.version
        }

        if self.model and self.is_model_loaded:
            info['lora_status'] = self.model.get_lora_status()

        return info

    def __repr__(self) -> str:
        """Return a readable representation."""
        version_str = f", version={self.version}" if self.version else ""
        return f"<{self.__class__.__name__}: {self.expert_name}{version_str}, loaded={self.is_model_loaded}>"

    def __enter__(self):
        """Enter the context manager."""
        if not self.is_model_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and release resources."""
        self.unload_model()

    @classmethod
    def load_shared_base_model(cls, base_model_path: str, use_4bit: bool = True) -> bool:
        """Load shared base model."""
        try:
            if cls._shared_base_model is not None and cls._shared_base_model_path == base_model_path:
                logger.info(f"共享基础模型已加载: {base_model_path}")
                return True

            logger.info(f"加载共享基础模型: {base_model_path}")
            cls._shared_base_model = LanguageModel(
                model_path=base_model_path,
                use_4bit=use_4bit
            )
            cls._shared_base_model_path = base_model_path
            logger.info("共享基础模型加载成功")
            return True

        except Exception as e:
            logger.error(f"共享基础模型加载失败: {e}")
            cls._shared_base_model = None
            cls._shared_base_model_path = None
            return False

    @classmethod
    def unload_shared_base_model(cls) -> bool:
        """Unload the shared base model."""
        try:
            if cls._shared_base_model:
                if cls._shared_base_model.is_lora_loaded:
                    cls._shared_base_model.unload_lora()

                del cls._shared_base_model
                cls._shared_base_model = None
                cls._shared_base_model_path = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("共享基础模型已卸载")

            return True

        except Exception as e:
            logger.error(f"共享基础模型卸载失败: {e}")
            return False

    def load_model_with_shared_base(self) -> bool:
        """Load model with shared base."""
        try:
            if self.__class__._shared_base_model is None:
                logger.error("共享基础模型未加载，请先调用load_shared_base_model")
                return False

            if self.base_model_path != self.__class__._shared_base_model_path:
                logger.warning(f"基础模型路径不匹配：专家期望{self.base_model_path}，共享模型是{self.__class__._shared_base_model_path}")
                logger.warning("将使用共享模型")

            logger.info(f"使用共享基础模型加载{self.expert_name}...")

            self.model = self.__class__._shared_base_model

            if self.lora_path:
                lora_path = Path(self.lora_path)
                if lora_path.exists():
                    logger.info(f"加载LoRA权重: {self.lora_path}")
                    success = self.model.load_lora_from_path(str(self.lora_path))
                    if not success:
                        logger.warning("LoRA加载失败，使用基础模型")
                else:
                    logger.warning(f"LoRA路径不存在: {self.lora_path}")
                    logger.warning("使用基础模型（未微调）")

            self.is_model_loaded = True
            logger.info(f"{self.expert_name}加载完成（使用共享基础模型）")
            return True

        except Exception as e:
            logger.error(f"使用共享基础模型加载失败: {e}")
            self.is_model_loaded = False
            return False

    def unload_model_keep_shared_base(self) -> bool:
        """Unload the expert while retaining the shared base model."""
        try:
            if self.model:
                if self.model.is_lora_loaded:
                    self.model.unload_lora()

                self.model = None
                self.is_model_loaded = False
                logger.info(f"{self.expert_name}已卸载（保留共享基础模型）")

            return True

        except Exception as e:
            logger.error(f"卸载失败: {e}")
            return False
