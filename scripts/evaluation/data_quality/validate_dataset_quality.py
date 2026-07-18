"""Validate dataset structure, completeness, and instruction quality."""

import os
import sys
import json
import pandas as pd
import chardet
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any


class UMLDatasetValidator:
    """Validate FlowChart dataset records and instructions."""

    def __init__(self, dataset_path: str, enable_period_check: bool = False):
        """Initialize the instance."""
        self.dataset_path = dataset_path
        self.enable_period_check = enable_period_check
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0

    def detect_encoding(self, filepath: str) -> str:
        """Detect encoding."""
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read(100000)
                result = chardet.detect(raw_data)
                return result['encoding']
        except Exception as e:
            print(f"Encoding detection error: {e}")
            return 'utf-8'

    def load_dataset(self) -> pd.DataFrame:
        """Load dataset."""
        print(f"\nLoading dataset: {os.path.basename(self.dataset_path)}")

        encoding = self.detect_encoding(self.dataset_path)
        print(f"Detected encoding: {encoding}")

        try:
            df = pd.read_csv(self.dataset_path, encoding=encoding)
            print(f"Successfully loaded {len(df)} rows\n")
            return df
        except Exception as e:
            print(f"Failed to load with {encoding}; trying other encodings...")
            for enc in ['utf-8', 'gbk', 'gb18030', 'latin1']:
                try:
                    df = pd.read_csv(self.dataset_path, encoding=enc)
                    print(f"Successfully loaded with {enc} encoding")
                    print(f"Loaded {len(df)} rows\n")
                    return df
                except:
                    continue
            raise Exception(f"Failed to load dataset: {e}")

    def validate_json_description(self, description: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate JSON description."""
        errors = []

        if not description or pd.isna(description):
            errors.append("Description为空")
            return False, errors

        try:
            desc_json = json.loads(description)

            required_fields = ['actors', 'use_cases', 'relationships', 'overall_description']
            for field in required_fields:
                if field not in desc_json:
                    errors.append(f"缺少必需字段: {field}")

            if 'actors' in desc_json:
                if not isinstance(desc_json['actors'], list):
                    errors.append("'actors'应该是列表")
                else:
                    for idx, actor in enumerate(desc_json['actors']):
                        if not isinstance(actor, dict) or 'name' not in actor:
                            errors.append(f"actors索引{idx}结构无效")

            if 'use_cases' in desc_json:
                if not isinstance(desc_json['use_cases'], list):
                    errors.append("'use_cases'应该是列表")
                else:
                    for idx, uc in enumerate(desc_json['use_cases']):
                        if not isinstance(uc, dict):
                            errors.append(f"use_cases索引{idx}结构无效")
                        elif 'name' not in uc:
                            errors.append(f"use_cases索引{idx}缺少'name'字段")

            if 'relationships' in desc_json:
                if not isinstance(desc_json['relationships'], list):
                    errors.append("'relationships'应该是列表")
                else:
                    for idx, rel in enumerate(desc_json['relationships']):
                        if not isinstance(rel, dict):
                            errors.append(f"relationships索引{idx}结构无效")
                        else:
                            required_rel_fields = ['type', 'from', 'to']
                            for field in required_rel_fields:
                                if field not in rel:
                                    errors.append(f"relationships索引{idx}缺少'{field}'字段")

        except json.JSONDecodeError as e:
            errors.append(f"JSON格式无效: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"未预期的错误: {str(e)}")
            return False, errors

        return len(errors) == 0, errors

    def validate_three_part_format(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate three part format."""
        errors = []

        if not instruction or pd.isna(instruction) or instruction.strip() == '':
            errors.append("Instruction为空")
            return False, errors

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        if len(lines) < 3:
            errors.append(f"行数不足(期望3行，实际{len(lines)}行)")

        has_definition = False
        has_emphasis = False
        has_avoid = False

        for line in lines:
            if line.startswith('Definition:'):
                has_definition = True
                content = line[len('Definition:'):].strip()
                if not content.lower().startswith('in this task'):
                    errors.append("Definition未以'In this task'开头")
                if self.enable_period_check and not content.endswith('.'):
                    errors.append("Definition缺少结尾句号")

            elif line.startswith('Emphasis & Caution:') or line.startswith('Emphasis and Caution:'):
                has_emphasis = True
                content = line.split(':', 1)[1].strip() if ':' in line else ""
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Emphasis & Caution缺少结尾句号")

            elif line.startswith('Things to Avoid:'):
                has_avoid = True
                content = line[len('Things to Avoid:'):].strip()
                if self.enable_period_check and content and content != '-' and not content.endswith('.'):
                    errors.append("Things to Avoid缺少结尾句号")

        if not has_definition:
            errors.append("缺少Definition部分")
        if not has_emphasis:
            errors.append("缺少Emphasis & Caution部分")
        if not has_avoid:
            errors.append("缺少Things to Avoid部分")

        is_valid = (has_definition and has_emphasis and has_avoid and len(errors) == 0)
        return is_valid, errors

    def check_things_to_avoid_completeness(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Check things to avoid completeness."""
        warnings = []

        if not instruction or pd.isna(instruction):
            return False, ["Instruction为空"]

        avoid_pattern = r'Things to Avoid:\s*(.+?)(?:\n|$)'
        match = re.search(avoid_pattern, instruction, re.DOTALL)

        if not match:
            warnings.append("无法找到Things to Avoid部分")
            return False, warnings

        avoid_content = match.group(1).strip()

        if avoid_content == '-':
            return True, []

        if not avoid_content:
            warnings.append("Things to Avoid内容为空")
            return False, warnings

        incomplete_patterns = [
            r'^TBD\s*$',
            r'^TODO\s*$',
            r'^N/A\s*$',
        ]

        for pattern in incomplete_patterns:
            if re.match(pattern, avoid_content, re.IGNORECASE):
                warnings.append(f"Things to Avoid看起来不完整: '{avoid_content}'")
                return False, warnings

        return True, []

    def validate_description_instruction_correspondence(
        self,
        description: str,
        instruction: str,
        row_num: int
    ) -> Tuple[bool, List[str]]:
        """Validate description instruction correspondence."""
        warnings = []

        if not description or not instruction:
            warnings.append("Description或Instruction为空")
            return False, warnings

        try:
            desc_json = json.loads(description)

            actors = [actor.get('name', '').lower() for actor in desc_json.get('actors', [])]
            use_cases = [uc.get('name', '').lower() for uc in desc_json.get('use_cases', [])]

            instruction_lower = instruction.lower()

            use_cases_mentioned = sum(1 for uc in use_cases if uc and uc in instruction_lower)
            if len(use_cases) > 0 and use_cases_mentioned == 0:
                warnings.append("Description中的use cases似乎没有在Instruction中提及")

            actors_mentioned = sum(1 for actor in actors if actor and actor in instruction_lower)
            if len(actors) > 0 and actors_mentioned == 0:
                warnings.append("Description中的actors似乎没有在Instruction中提及")

            relationships = desc_json.get('relationships', [])
            has_include = any(rel.get('type') == 'include' for rel in relationships)
            has_extend = any(rel.get('type') == 'extend' for rel in relationships)

            if has_include:
                include_keywords = ['include', 'required', 'mandatory', 'must', 'prerequisite']
                if not any(keyword in instruction_lower for keyword in include_keywords):
                    warnings.append("Description有'include'关系但Instruction中可能未体现")

            if has_extend:
                extend_keywords = ['extend', 'optional', 'conditional', 'may', 'can']
                if not any(keyword in instruction_lower for keyword in extend_keywords):
                    warnings.append("Description有'extend'关系但Instruction中可能未体现")

        except json.JSONDecodeError:
            warnings.append("无法解析Description的JSON进行对应关系检查")
            return False, warnings
        except Exception as e:
            warnings.append(f"检查对应关系时出错: {str(e)}")
            return False, warnings

        return len(warnings) == 0, warnings

    def check_error_markers(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Check error markers."""
        errors = []

        if not instruction or pd.isna(instruction):
            errors.append("Instruction为空")
            return False, errors

        error_patterns = [
            r'ERROR\s*:',
            r'error\s*:',
            r'生成失败',
            r'generation failed',
            r'failed to generate',
        ]

        for pattern in error_patterns:
            if re.search(pattern, instruction, re.IGNORECASE):
                errors.append(f"包含ERROR标记: 匹配模式'{pattern}'")
                return False, errors

        return True, []

    def validate_keyword_density(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate keyword density."""
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        instruction_lower = instruction.lower()

        uml_keywords = {
            'relationships': ['include', 'extend', 'association', 'generalization', 'dependency'],
            'elements': ['actor', 'use case', 'usecase', 'use-case', 'system', 'boundary'],
            'qualifiers': ['required', 'optional', 'mandatory', 'conditional', 'prerequisite'],
            'workflow': ['workflow', 'process', 'interaction', 'execute', 'implement', 'trigger']
        }

        category_counts = {}
        total_keywords = 0

        for category, keywords in uml_keywords.items():
            count = sum(1 for keyword in keywords if keyword in instruction_lower)
            category_counts[category] = count
            total_keywords += count

        if total_keywords < 3:
            warnings.append(f"UML关键术语过少(仅{total_keywords}个)，可能缺乏专业性")

        categories_with_keywords = sum(1 for count in category_counts.values() if count > 0)
        if categories_with_keywords < 2:
            warnings.append(f"UML关键术语类别单一(仅{categories_with_keywords}类)，建议增加多样性")

        return len(warnings) == 0, warnings

    def validate_content_length(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate content length."""
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        definition_match = re.search(r'Definition:\s*(.+?)(?=\n(?:Emphasis|$))', instruction, re.DOTALL)
        emphasis_match = re.search(r'Emphasis & Caution:\s*(.+?)(?=\nThings to Avoid:|$)', instruction, re.DOTALL)
        avoid_match = re.search(r'Things to Avoid:\s*(.+?)$', instruction, re.DOTALL)

        min_lengths = {
            'Definition': 50,
            'Emphasis & Caution': 10,
            'Things to Avoid': 10
        }

        if definition_match:
            definition_content = definition_match.group(1).strip()
            def_len = len(definition_content)
            min_len = min_lengths['Definition']

            if def_len < min_len:
                warnings.append(f"Definition过短({def_len}字符)，可能不完整")

        if emphasis_match:
            emphasis_content = emphasis_match.group(1).strip()
            if emphasis_content != '-':
                emp_len = len(emphasis_content)
                min_len = min_lengths['Emphasis & Caution']

                if emp_len < min_len:
                    warnings.append(f"Emphasis & Caution过短({emp_len}字符)，可能不完整")

        if avoid_match:
            avoid_content = avoid_match.group(1).strip()
            if avoid_content != '-':
                avoid_len = len(avoid_content)
                min_len = min_lengths['Things to Avoid']

                if avoid_len < min_len:
                    warnings.append(f"Things to Avoid过短({avoid_len}字符)，可能不完整")

        return len(warnings) == 0, warnings

    def validate_content_duplication(self, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate content duplication."""
        warnings = []

        if not instruction or pd.isna(instruction):
            warnings.append("Instruction为空")
            return False, warnings

        definition_match = re.search(r'Definition:\s*(.+?)(?=\n(?:Emphasis|$))', instruction, re.DOTALL)
        emphasis_match = re.search(r'Emphasis & Caution:\s*(.+?)(?=\nThings to Avoid:|$)', instruction, re.DOTALL)
        avoid_match = re.search(r'Things to Avoid:\s*(.+?)$', instruction, re.DOTALL)

        if not (definition_match and emphasis_match and avoid_match):
            return True, []

        definition_content = definition_match.group(1).strip().lower()
        emphasis_content = emphasis_match.group(1).strip().lower()
        avoid_content = avoid_match.group(1).strip().lower()

        if emphasis_content == '-' or avoid_content == '-':
            return True, []

        def get_words(text):
            words = re.findall(r'\b[a-z]{4,}\b', text)
            return set(words)

        def_words = get_words(definition_content)
        emp_words = get_words(emphasis_content)
        avoid_words = get_words(avoid_content)

        if len(def_words) > 0 and len(emp_words) > 0:
            overlap_def_emp = len(def_words & emp_words)
            overlap_ratio = overlap_def_emp / min(len(def_words), len(emp_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Definition和Emphasis & Caution内容重复度过高({overlap_ratio:.1%})")

        if len(def_words) > 0 and len(avoid_words) > 0:
            overlap_def_avoid = len(def_words & avoid_words)
            overlap_ratio = overlap_def_avoid / min(len(def_words), len(avoid_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Definition和Things to Avoid内容重复度过高({overlap_ratio:.1%})")

        if len(emp_words) > 0 and len(avoid_words) > 0:
            overlap_emp_avoid = len(emp_words & avoid_words)
            overlap_ratio = overlap_emp_avoid / min(len(emp_words), len(avoid_words))

            if overlap_ratio > 0.7:
                warnings.append(f"Emphasis & Caution和Things to Avoid内容重复度过高({overlap_ratio:.1%})")

        return len(warnings) == 0, warnings

    def validate_coverage(self, description: str, instruction: str, row_num: int) -> Tuple[bool, List[str]]:
        """Validate coverage."""
        warnings = []

        if not description or not instruction:
            warnings.append("Description或Instruction为空")
            return False, warnings

        try:
            desc_json = json.loads(description)

            actors = desc_json.get('actors', [])
            use_cases = desc_json.get('use_cases', [])

            instruction_lower = instruction.lower()

            if actors:
                actor_names = [actor.get('name', '').lower() for actor in actors if actor.get('name')]
                mentioned_actors = [name for name in actor_names if name and name in instruction_lower]
                coverage_rate = len(mentioned_actors) / len(actor_names) if actor_names else 0

                if coverage_rate < 0.5:
                    warnings.append(f"Actors覆盖率过低({coverage_rate:.0%})，仅提及{len(mentioned_actors)}/{len(actor_names)}个")

            if use_cases:
                uc_names = [uc.get('name', '').lower() for uc in use_cases if uc.get('name')]
                mentioned_ucs = [name for name in uc_names if name and name in instruction_lower]
                coverage_rate = len(mentioned_ucs) / len(uc_names) if uc_names else 0

                if coverage_rate < 0.5:
                    warnings.append(f"Use Cases覆盖率过低({coverage_rate:.0%})，仅提及{len(mentioned_ucs)}/{len(uc_names)}个")

        except json.JSONDecodeError:
            warnings.append("无法解析Description的JSON进行覆盖度检查")
            return False, warnings
        except Exception as e:
            warnings.append(f"检查覆盖度时出错: {str(e)}")
            return False, warnings

        return len(warnings) == 0, warnings

    def validate_row(self, row: pd.Series, row_num: int) -> Dict[str, Any]:
        """Validate row."""
        result = {
            'row_num': row_num,
            'header': row.get('Header', 'N/A'),
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        description = str(row.get('Description', ''))
        instruction = str(row.get('Instruction', ''))

        json_valid, json_errors = self.validate_json_description(description, row_num)
        if not json_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[JSON] {err}" for err in json_errors])

        error_clean, error_messages = self.check_error_markers(instruction, row_num)
        if not error_clean:
            result['is_valid'] = False
            result['errors'].extend([f"[ERROR] {err}" for err in error_messages])

        format_valid, format_errors = self.validate_three_part_format(instruction, row_num)
        if not format_valid:
            result['is_valid'] = False
            result['errors'].extend([f"[FORMAT] {err}" for err in format_errors])

        avoid_complete, avoid_warnings = self.check_things_to_avoid_completeness(instruction, row_num)
        if not avoid_complete:
            result['warnings'].extend([f"[AVOID] {warn}" for warn in avoid_warnings])

        corr_valid, corr_warnings = self.validate_description_instruction_correspondence(
            description, instruction, row_num
        )
        if not corr_valid:
            result['warnings'].extend([f"[CORRESPONDENCE] {warn}" for warn in corr_warnings])

        keyword_valid, keyword_warnings = self.validate_keyword_density(instruction, row_num)
        if not keyword_valid:
            result['warnings'].extend([f"[KEYWORD] {warn}" for warn in keyword_warnings])

        length_valid, length_warnings = self.validate_content_length(instruction, row_num)
        if not length_valid:
            result['warnings'].extend([f"[LENGTH] {warn}" for warn in length_warnings])

        dup_valid, dup_warnings = self.validate_content_duplication(instruction, row_num)
        if not dup_valid:
            result['warnings'].extend([f"[DUPLICATION] {warn}" for warn in dup_warnings])

        cov_valid, cov_warnings = self.validate_coverage(description, instruction, row_num)
        if not cov_valid:
            result['warnings'].extend([f"[COVERAGE] {warn}" for warn in cov_warnings])

        return result

    def validate_dataset(self) -> List[Dict[str, Any]]:
        """Validate dataset."""
        print("=" * 80)
        print("FlowChart Dataset Quality Validation".center(80))
        print("=" * 80)
        print(f"Dataset: {os.path.basename(self.dataset_path)}")
        print(f"Period check: {'enabled' if self.enable_period_check else 'disabled'}")
        print("=" * 80)
        print()

        df = self.load_dataset()

        print("Starting validation...\n")

        results = []
        for idx, row in df.iterrows():
            row_num = idx + 1
            if row_num % 100 == 0:
                print(f"Progress: {row_num}/{len(df)} rows validated")

            result = self.validate_row(row, row_num)
            results.append(result)

            if not result['is_valid']:
                self.error_count += 1
            if result['warnings']:
                self.warning_count += 1

        self.validation_results = results
        return results

    def generate_report(self, save_path: str = None) -> str:
        """Generate report."""
        if not self.validation_results:
            return "无验证结果。请先运行validate_dataset()。"

        print("\n" + "=" * 80)
        print("Validation Report".center(80))
        print("=" * 80)

        total_rows = len(self.validation_results)
        valid_rows = sum(1 for r in self.validation_results if r['is_valid'])
        invalid_rows = total_rows - valid_rows
        rows_with_warnings = sum(1 for r in self.validation_results if r['warnings'])

        summary = f"""
总行数: {total_rows}
有效行数: {valid_rows} ({valid_rows/total_rows*100:.1f}%)
无效行数: {invalid_rows} ({invalid_rows/total_rows*100:.1f}%)
有警告的行数: {rows_with_warnings} ({rows_with_warnings/total_rows*100:.1f}%)
"""

        print(summary)

        if invalid_rows > 0:
            print("\n" + "-" * 80)
            print("Detailed errors:")
            print("-" * 80)

            for result in self.validation_results:
                if not result['is_valid']:
                    print(f"\nRow {result['row_num']} [{result['header'][:40]}...]:")
                    for error in result['errors']:
                        print(f"  Error: {error}")
                    for warning in result['warnings']:
                        print(f"  Warning: {warning}")

        if rows_with_warnings > 0:
            print("\n" + "-" * 80)
            print("Detailed warnings:")
            print("-" * 80)

            warning_count = 0
            for result in self.validation_results:
                if result['warnings'] and result['is_valid']:
                    warning_count += 1
                    if warning_count <= 20:
                        print(f"\nRow {result['row_num']} [{result['header'][:40]}...]:")
                        for warning in result['warnings']:
                            print(f"  Warning: {warning}")

            if warning_count > 20:
                print(f"\n... {warning_count - 20} more rows have warnings")

        if save_path:
            self.save_report_csv(save_path)
            print(f"\nDetailed report saved to: {save_path}")

        print("=" * 80)

        return summary

    def save_report_csv(self, save_path: str):
        """Save report CSV."""
        report_data = []

        for result in self.validation_results:
            report_data.append({
                '行号': result['row_num'],
                'Header': result['header'],
                '是否有效': result['is_valid'],
                '错误': ' | '.join(result['errors']),
                '警告': ' | '.join(result['warnings'])
            })

        df_report = pd.DataFrame(report_data)
        df_report.to_csv(save_path, index=False, encoding='utf-8-sig')

    def save_problematic_instructions(self, output_dir: str, df: pd.DataFrame):
        """Save problematic instructions."""
        os.makedirs(output_dir, exist_ok=True)

        problematic_rows = []
        for result in self.validation_results:
            if not result['is_valid'] or result['warnings']:
                row_num = result['row_num']
                row_data = df.iloc[row_num - 1]

                problematic_rows.append({
                    '行号': result['row_num'],
                    'Header': result['header'],
                    'Description': row_data.get('Description', ''),
                    'Instruction': row_data.get('Instruction', ''),
                    '错误': ' | '.join(result['errors']) if result['errors'] else '',
                    '警告': ' | '.join(result['warnings']) if result['warnings'] else '',
                    '是否有效': result['is_valid']
                })

        if problematic_rows:
            df_problematic = pd.DataFrame(problematic_rows)
            output_path = os.path.join(output_dir, 'problematic_instructions_for_llm_review.csv')
            df_problematic.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"\nProblematic instructions saved to: {output_path}")
            print(f"{len(problematic_rows)} entries require manual or LLM review")

            error_only = sum(1 for r in problematic_rows if r['错误'] and not r['警告'])
            warning_only = sum(1 for r in problematic_rows if r['警告'] and not r['错误'])
            both = sum(1 for r in problematic_rows if r['错误'] and r['警告'])

            print(f"  - Errors only: {error_only}")
            print(f"  - Warnings only: {warning_only}")
            print(f"  - Both errors and warnings: {both}")
        else:
            print("\nNo problematic instructions found")

    def get_error_rows(self) -> List[int]:
        """Return error rows."""
        return [r['row_num'] for r in self.validation_results if not r['is_valid']]

    def get_warning_rows(self) -> List[int]:
        """Return warning rows."""
        return [r['row_num'] for r in self.validation_results if r['warnings']]


def main():
    """Run the command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate FlowChart dataset quality')
    parser.add_argument('--dataset', type=str,
                       default='data/dataset/uml/uml_dataset.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--enable-period-check', action='store_true',
                       help='Enable sentence-ending period checks')
    parser.add_argument('--report-output', type=str,
                       default=None,
                       help='Output path for the validation report CSV')

    args = parser.parse_args()

    validator = UMLDatasetValidator(
        dataset_path=args.dataset,
        enable_period_check=args.enable_period_check
    )

    start_time = datetime.now()
    results = validator.validate_dataset()
    end_time = datetime.now()

    df = validator.load_dataset()

    if args.report_output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.report_output = f'outputs/validation/uml_validation_report_{timestamp}.csv'

    os.makedirs(os.path.dirname(args.report_output), exist_ok=True)

    summary = validator.generate_report(save_path=args.report_output)

    problematic_output_dir = os.path.join(os.path.dirname(args.report_output), 'problematic_instructions')
    validator.save_problematic_instructions(problematic_output_dir, df)

    duration = end_time - start_time
    print(f"\nValidation completed in: {duration}")

    error_count = validator.error_count
    warning_count = validator.warning_count

    print(f"\n{'=' * 80}")
    print(f"Validation Summary".center(80))
    print(f"{'=' * 80}")
    print(f"Total errors: {error_count}")
    print(f"Total warnings: {warning_count}")
    print(f"Validation report: {args.report_output}")
    if error_count > 0 or warning_count > 0:
        print(f"Problematic instructions: {problematic_output_dir}/problematic_instructions_for_llm_review.csv")
    print(f"{'=' * 80}")

    if error_count > 0:
        print(f"\nError row numbers: {validator.get_error_rows()[:20]}")
        if len(validator.get_error_rows()) > 20:
            print(f"... {len(validator.get_error_rows()) - 20} more rows have errors")
        print(f"\nTo repair errors, run:")
        print(f"python scripts/data_preparation/uml_dataset_regenerate.py")

    if warning_count > 0:
        print(f"\nWarning row numbers: {validator.get_warning_rows()[:20]}")
        if len(validator.get_warning_rows()) > 20:
            print(f"... {len(validator.get_warning_rows()) - 20} more rows have warnings")
        print(f"\nWarnings may be false positives; review the problematic-instructions file manually or with an LLM")

    if error_count == 0 and warning_count == 0:
        print("\nDataset quality is excellent; no issues found.")

    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
