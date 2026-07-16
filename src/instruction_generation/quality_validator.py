"""Validate the three-part instruction format and basic content quality."""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger('instruction_generation.quality_validator')


@dataclass
class ValidationResult:
    """Store instruction-validation results."""
    is_valid: bool
    has_definition: bool
    has_emphasis: bool
    has_things_to_avoid: bool
    definition_has_content: bool
    definition_starts_with_in_this_task: bool
    emphasis_is_valid: bool
    avoid_is_valid: bool
    format_score: float
    errors: List[str]
    warnings: List[str]


class QualityValidator:
    """Validate instruction structure and content quality."""

    def __init__(self, strict_mode: bool = False):
        """Initialize the instance."""
        self.strict_mode = strict_mode
        logger.info(f"质量验证器初始化完成 - 严格模式: {strict_mode}")

    def validate_instruction(self, instruction: str) -> ValidationResult:
        """Validate instruction."""
        errors = []
        warnings = []

        result = {
            'is_valid': False,
            'has_definition': False,
            'has_emphasis': False,
            'has_things_to_avoid': False,
            'definition_has_content': False,
            'definition_starts_with_in_this_task': False,
            'emphasis_is_valid': False,
            'avoid_is_valid': False,
            'format_score': 0.0
        }

        if not instruction or len(instruction.strip()) < 20:
            errors.append("指令内容过短或为空")
            return ValidationResult(**result, errors=errors, warnings=warnings)

        lines = instruction.split('\n')

        definition_line = None
        emphasis_line = None
        avoid_line = None

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue

            if line_stripped.startswith('Definition:'):
                definition_line = line_stripped
                result['has_definition'] = True
            elif line_stripped.startswith('Emphasis & Caution:') or line_stripped.startswith('Emphasis and Caution:'):
                emphasis_line = line_stripped
                result['has_emphasis'] = True
            elif line_stripped.startswith('Things to Avoid:'):
                avoid_line = line_stripped
                result['has_things_to_avoid'] = True

        if definition_line:
            content = definition_line.split('Definition:', 1)[1].strip()

            if content and content != '-':
                result['definition_has_content'] = True
            else:
                errors.append("Definition没有实际内容(不能只是'-')")

            if content.lower().startswith('in this task'):
                result['definition_starts_with_in_this_task'] = True
            else:
                warnings.append("Definition建议以'In this task'开头")

            if len(content) < 10:
                warnings.append("Definition内容过短")
        else:
            errors.append("缺少Definition部分")

        if emphasis_line:
            content = emphasis_line.split(':', 1)[1].strip()

            if self.strict_mode:
                if content and content != '-':
                    result['emphasis_is_valid'] = True
                else:
                    errors.append("Emphasis & Caution必须有实际内容(严格模式)")
            else:
                if content:
                    result['emphasis_is_valid'] = True
                    if content == '-':
                        warnings.append("Emphasis & Caution为'-',建议提供具体内容")
        else:
            errors.append("缺少Emphasis & Caution部分")

        if avoid_line:
            content = avoid_line.split(':', 1)[1].strip()

            if self.strict_mode:
                if content and content != '-':
                    result['avoid_is_valid'] = True
                else:
                    errors.append("Things to Avoid必须有实际内容(严格模式)")
            else:
                if content:
                    result['avoid_is_valid'] = True
                    if content == '-':
                        warnings.append("Things to Avoid为'-',建议提供具体内容")
        else:
            errors.append("缺少Things to Avoid部分")

        score_components = [
            result['has_definition'],
            result['definition_has_content'],
            result['definition_starts_with_in_this_task'],
            result['has_emphasis'],
            result['emphasis_is_valid'],
            result['has_things_to_avoid'],
            result['avoid_is_valid']
        ]
        result['format_score'] = sum(score_components) / len(score_components)

        if self.strict_mode:
            result['is_valid'] = (
                    result['definition_has_content'] and
                    result['has_emphasis'] and
                    result['emphasis_is_valid'] and
                    result['has_things_to_avoid'] and
                    result['avoid_is_valid']
            )
        else:
            result['is_valid'] = (
                    result['definition_has_content'] and
                    result['has_emphasis'] and
                    result['has_things_to_avoid']
            )

        return ValidationResult(**result, errors=errors, warnings=warnings)

    def batch_validate(
            self,
            instructions: List[str]
    ) -> Tuple[List[ValidationResult], Dict]:
        """Validate instructions in batches."""
        logger.info(f"批量验证 - 共{len(instructions)}条指令")

        results = []
        for i, instruction in enumerate(instructions, 1):
            result = self.validate_instruction(instruction)
            results.append(result)

            if not result.is_valid:
                logger.debug(f"指令{i}验证失败: {result.errors}")

        summary = self._generate_summary(results)

        logger.info(f"验证完成 - 通过率: {summary['pass_rate']:.2%}")

        return results, summary

    def _generate_summary(self, results: List[ValidationResult]) -> Dict:
        """Generate summary."""
        total = len(results)

        if total == 0:
            return {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }

        passed = sum(1 for r in results if r.is_valid)
        failed = total - passed

        summary = {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total,

            'definition_present_rate': sum(1 for r in results if r.has_definition) / total,
            'definition_has_content_rate': sum(1 for r in results if r.definition_has_content) / total,
            'definition_starts_with_in_this_task_rate': sum(
                1 for r in results if r.definition_starts_with_in_this_task) / total,

            'emphasis_present_rate': sum(1 for r in results if r.has_emphasis) / total,
            'emphasis_valid_rate': sum(1 for r in results if r.emphasis_is_valid) / total,

            'avoid_present_rate': sum(1 for r in results if r.has_things_to_avoid) / total,
            'avoid_valid_rate': sum(1 for r in results if r.avoid_is_valid) / total,

            'avg_format_score': sum(r.format_score for r in results) / total,
            'min_format_score': min(r.format_score for r in results),
            'max_format_score': max(r.format_score for r in results),

            'total_errors': sum(len(r.errors) for r in results),
            'total_warnings': sum(len(r.warnings) for r in results),

            'common_errors': self._count_common_errors(results)
        }

        return summary

    def _count_common_errors(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Count common validation errors."""
        error_counts = {}

        for result in results:
            for error in result.errors:
                error_counts[error] = error_counts.get(error, 0) + 1

        sorted_errors = dict(
            sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        )

        return sorted_errors

    def print_validation_report(
            self,
            results: List[ValidationResult],
            summary: Dict,
            show_details: bool = False
    ):
        """Print validation report."""
        print("\n" + "=" * 80)
        print("指令质量验证报告")
        print("=" * 80)

        print(f"\n[总体统计]")
        print(f"  总计:     {summary['total']} 条")
        print(f"  通过:     {summary['passed']} 条")
        print(f"  失败:     {summary['failed']} 条")
        print(f"  通过率:   {summary['pass_rate']:.2%}")

        print(f"\n[格式分数]")
        print(f"  平均分数: {summary['avg_format_score']:.4f}")
        print(f"  最高分数: {summary['max_format_score']:.4f}")
        print(f"  最低分数: {summary['min_format_score']:.4f}")

        print(f"\n[分项统计]")
        print(f"  Definition存在率:   {summary['definition_present_rate']:.2%}")
        print(f"  Definition有效率:   {summary['definition_has_content_rate']:.2%}")
        print(f"  Emphasis存在率:     {summary['emphasis_present_rate']:.2%}")
        print(f"  Emphasis有效率:     {summary['emphasis_valid_rate']:.2%}")
        print(f"  Avoid存在率:        {summary['avoid_present_rate']:.2%}")
        print(f"  Avoid有效率:        {summary['avoid_valid_rate']:.2%}")

        print(f"\n[错误统计]")
        print(f"  总错误数:   {summary['total_errors']}")
        print(f"  总警告数:   {summary['total_warnings']}")

        if summary['common_errors']:
            print(f"\n[常见错误Top 5]")
            for i, (error, count) in enumerate(list(summary['common_errors'].items())[:5], 1):
                print(f"  {i}. {error}: {count}次")

        if show_details and results:
            print(f"\n[详细信息]")
            for i, result in enumerate(results[:10], 1):
                print(f"\n指令 {i}:")
                print(f"  有效: {result.is_valid}")
                print(f"  分数: {result.format_score:.4f}")
                if result.errors:
                    print(f"  错误: {', '.join(result.errors)}")
                if result.warnings:
                    print(f"  警告: {', '.join(result.warnings)}")

            if len(results) > 10:
                print(f"\n  ... (还有 {len(results) - 10} 条)")

        print("=" * 80 + "\n")

    def filter_valid_instructions(
            self,
            instructions: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Filter valid instructions."""
        valid = []
        invalid = []

        for instruction in instructions:
            result = self.validate_instruction(instruction)
            if result.is_valid:
                valid.append(instruction)
            else:
                invalid.append(instruction)

        logger.info(f"过滤完成 - 有效: {len(valid)}, 无效: {len(invalid)}")

        return valid, invalid
