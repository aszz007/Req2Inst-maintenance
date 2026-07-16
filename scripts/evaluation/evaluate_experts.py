"""Evaluate expert checkpoints and write aggregate reports."""

import sys
import json
import argparse
import gc
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_path_config
from src.utils.enhanced_metrics import EnhancedMetrics
from src.instruction_generation.quality_validator import QualityValidator
from src.training.data_loader import (
    TextDatasetLoader,
    ImageDatasetLoader,
    UMLDatasetLoader,
    split_dataset_for_expert
)
from src.experts import TextExpert, ImageExpert, UMLExpert, GeneralExpert
from src.utils.logger import get_logger
import traceback

logger = get_logger('evaluation.evaluate_experts')


class ExpertEvaluator:
    """Evaluate expert checkpoints and aggregate metrics."""

    def __init__(
            self,
            use_bertscore: bool = False,
            strict_validation: bool = False
    ):
        """Initialize the instance."""
        self.metrics = EnhancedMetrics(use_bertscore=use_bertscore)
        self.validator = QualityValidator(strict_mode=strict_validation)
        self.path_cfg = get_path_config()
        self.show_samples = False

        logger.info("专家评估器初始化完成")
        logger.info(f"使用BERTScore: {use_bertscore}")
        logger.info(f"严格验证模式: {strict_validation}")

    def _force_cleanup_gpu(self):
        """Force GPU resource cleanup."""
        import torch

        logger.info("强制清理GPU显存...")

        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        time.sleep(2)

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存状态 - 已分配: {memory_allocated:.2f}GB, 已保留: {memory_reserved:.2f}GB")


    def _display_samples(self, test_data: List[Dict], expert_type: str, num_display: int = 5):
        """Display representative samples."""
        if not self.show_samples:
            return

        logger.info("=" * 80)
        logger.info(f"[{expert_type}] 测试数据样本预览 (前{num_display}条)")
        logger.info("=" * 80)

        for i in range(min(num_display, len(test_data))):
            input_text = test_data[i]['input']
            if len(input_text) > 100:
                input_text = input_text[:100] + "..."
            logger.info(f"样本 {i+1}: {input_text}")

        logger.info("=" * 80)

    def evaluate_text_expert(
            self,
            num_samples: Optional[int] = None,
            save_predictions: bool = True,
            batch_size: int = 16
    ) -> Dict:
        """Evaluate text expert."""
        logger.info("=" * 80)
        logger.info("评估文本专家")
        logger.info("=" * 80)

        loader = TextDatasetLoader()
        data = loader.load_csv_files()
        _, _, test_data = split_dataset_for_expert(data, 'text')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")
        logger.info(f"批处理大小: {batch_size}")

        self._display_samples(test_data, "Text Expert")

        expert = TextExpert()
        if not expert.load_model():
            logger.error("文本专家加载失败")
            return {}

        inputs = [item['input'] for item in test_data]
        references = [item['output'] for item in test_data]

        logger.info("开始批量生成指令...")
        predictions = expert.batch_generate_instruction(inputs, batch_size=batch_size)
        logger.info("批量生成完成")

        expert.unload_model()

        del expert
        self._force_cleanup_gpu()

        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='text_expert',
            inputs=inputs,
            save_predictions=save_predictions,
            save_dir=str(self.path_cfg.METRICS_DIR)
        )

        return results

    def evaluate_image_expert(
            self,
            num_samples: Optional[int] = None,
            save_predictions: bool = True,
            batch_size: int = 16
    ) -> Dict:
        """Evaluate image expert."""
        logger.info("=" * 80)
        logger.info("评估图像专家")
        logger.info("=" * 80)

        loader = ImageDatasetLoader()
        data = loader.load_csv_file()
        _, _, test_data = split_dataset_for_expert(data, 'image')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")
        logger.info(f"批处理大小: {batch_size}")

        self._display_samples(test_data, "Image Expert")

        expert = ImageExpert()
        if not expert.load_model():
            logger.error("图像专家加载失败")
            return {}

        inputs = [item['input'] for item in test_data]
        references = [item['output'] for item in test_data]

        logger.info("开始批量生成指令...")
        predictions = expert.batch_generate_instruction(inputs, batch_size=batch_size)
        logger.info("批量生成完成")

        expert.unload_model()

        del expert
        self._force_cleanup_gpu()

        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='image_expert',
            inputs=inputs,
            save_predictions=save_predictions,
            save_dir=str(self.path_cfg.METRICS_DIR)
        )

        return results

    def evaluate_uml_expert(
            self,
            num_samples: Optional[int] = None,
            save_predictions: bool = True,
            batch_size: int = 8
    ) -> Dict:
        """Evaluate UML expert."""
        logger.info("=" * 80)
        logger.info("评估UML专家")
        logger.info("=" * 80)

        loader = UMLDatasetLoader()
        data = loader.load_csv_file()
        _, _, test_data = split_dataset_for_expert(data, 'uml')

        if num_samples:
            test_data = test_data[:num_samples]

        logger.info(f"测试样本数: {len(test_data)}")
        logger.info(f"批处理大小: {batch_size}")
        logger.info(f"UML数据集: uml_dataset.csv")

        self._display_samples(test_data, "UML Expert")

        expert = UMLExpert()
        if not expert.load_model():
            logger.error("UML专家加载失败")
            return {}

        inputs = [item['input'] for item in test_data]
        references = [item['output'] for item in test_data]

        logger.info("开始批量生成指令...")
        predictions = expert.batch_generate_instruction(inputs, batch_size=batch_size)
        logger.info("批量生成完成")

        expert.unload_model()

        del expert
        self._force_cleanup_gpu()

        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='uml_expert',
            inputs=inputs,
            save_predictions=save_predictions,
            save_dir=str(self.path_cfg.METRICS_DIR)
        )

        return results

    def evaluate_general_expert(
            self,
            num_samples: Optional[int] = None,
            save_predictions: bool = True,
            batch_size: int = 8
    ) -> Dict:
        """Evaluate general expert."""
        logger.info("=" * 80)
        logger.info("评估通用专家")
        logger.info("=" * 80)

        text_loader = TextDatasetLoader()
        image_loader = ImageDatasetLoader()
        uml_loader = UMLDatasetLoader()

        text_data = text_loader.load_csv_files()
        image_data = image_loader.load_csv_file()
        uml_data = uml_loader.load_csv_file()

        _, _, text_test = split_dataset_for_expert(text_data, 'text')
        _, _, image_test = split_dataset_for_expert(image_data, 'image')
        _, _, uml_test = split_dataset_for_expert(uml_data, 'uml')

        if num_samples:
            text_ratio = len(text_test) / (len(text_test) + len(image_test) + len(uml_test))
            image_ratio = len(image_test) / (len(text_test) + len(image_test) + len(uml_test))
            uml_ratio = len(uml_test) / (len(text_test) + len(image_test) + len(uml_test))

            text_samples = min(int(num_samples * text_ratio), len(text_test))
            image_samples = min(int(num_samples * image_ratio), len(image_test))
            uml_samples = min(int(num_samples * uml_ratio), len(uml_test))

            test_data = (
                text_test[:text_samples] +
                image_test[:image_samples] +
                uml_test[:uml_samples]
            )
            logger.info(f"测试样本数: {len(test_data)} (text: {text_samples}, image: {image_samples}, uml: {uml_samples})")
        else:
            test_data = text_test + image_test + uml_test
            logger.info(f"测试样本数: {len(test_data)} (text: {len(text_test)}, image: {len(image_test)}, uml: {len(uml_test)})")

        logger.info(f"批处理大小: {batch_size}")
        logger.info(f"UML数据集: uml_dataset.csv")

        self._display_samples(test_data, "General Expert")

        expert = GeneralExpert()
        if not expert.load_model():
            logger.error("通用专家加载失败")
            return {}

        inputs = [item['input'] for item in test_data]
        references = [item['output'] for item in test_data]

        logger.info("开始批量生成指令...")
        predictions = expert.batch_generate_instruction(inputs, batch_size=batch_size)
        logger.info("批量生成完成")

        expert.unload_model()

        del expert
        self._force_cleanup_gpu()

        results = self._evaluate_predictions(
            predictions=predictions,
            references=references,
            expert_name='general_expert',
            inputs=inputs,
            save_predictions=save_predictions,
            save_dir=str(self.path_cfg.METRICS_DIR)
        )

        return results

    def _evaluate_predictions(
            self,
            predictions: List[str],
            references: List[str],
            expert_name: str,
            inputs: Optional[List[str]] = None,
            save_predictions: bool = False,
            save_dir: Optional[str] = None
    ) -> Dict:
        """Evaluate predictions."""
        logger.info("开始评估指标计算...")

        valid_pairs = [
            (pred, ref) for pred, ref in zip(predictions, references)
            if pred.strip()
        ]

        if not valid_pairs:
            logger.error("没有有效的预测结果")
            return {}

        valid_predictions = [pair[0] for pair in valid_pairs]
        valid_references = [pair[1] for pair in valid_pairs]

        logger.info(f"有效样本数: {len(valid_predictions)}/{len(predictions)}")

        if save_predictions and save_dir and inputs:
            self._save_predictions_json(
                inputs=inputs,
                predictions=predictions,
                references=references,
                expert_name=expert_name,
                save_dir=save_dir
            )

        quality_metrics = self.metrics.calculate_generation_quality(
            predictions=valid_predictions,
            references=valid_references
        )

        format_metrics = self.metrics.calculate_format_metrics(
            instructions=valid_predictions
        )

        binary_metrics = self.metrics.calculate_binary_classification_metrics(
            predictions=valid_predictions,
            references=valid_references
        )

        statistical_metrics = self.metrics.calculate_statistical_metrics(
            instructions=valid_predictions
        )

        validation_results, validation_summary = self.validator.batch_validate(
            instructions=valid_predictions
        )

        results = {
            'expert_name': expert_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': len(predictions),
            'valid_samples': len(valid_predictions),
            'generation_quality': quality_metrics,
            'format_metrics': format_metrics,
            'binary_classification': binary_metrics,
            'statistical_metrics': statistical_metrics,
            'validation_summary': validation_summary
        }

        return results

    def _save_predictions_json(
            self,
            inputs: List[str],
            predictions: List[str],
            references: List[str],
            expert_name: str,
            save_dir: str
    ):
        """Save predictions JSON."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{expert_name}_predictions_{timestamp}.json'
        filepath = save_dir / filename

        data = {
            'expert_name': expert_name,
            'timestamp': timestamp,
            'total_samples': len(inputs),
            'samples': [
                {
                    'index': i,
                    'input': inp,
                    'prediction': pred,
                    'reference': ref
                }
                for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references))
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"预测数据已保存至: {filepath}")
        logger.info(f"可使用 calculate_metrics_from_json.py 脚本快速重新计算指标")

    def evaluate_all_experts(
            self,
            num_samples: Optional[int] = None,
            save_dir: Optional[str] = None
    ) -> Dict[str, Dict]:
        """Evaluate all experts."""
        logger.info("=" * 80)
        logger.info("评估所有专家")
        logger.info("=" * 80)

        all_results = {}

        try:
            all_results['text_expert'] = self.evaluate_text_expert(num_samples)
        except Exception as e:
            logger.error(f"文本专家评估失败: {e}")
            self._force_cleanup_gpu()

        try:
            all_results['image_expert'] = self.evaluate_image_expert(num_samples)
        except Exception as e:
            logger.error(f"图像专家评估失败: {e}")
            self._force_cleanup_gpu()

        try:
            all_results['uml_expert'] = self.evaluate_uml_expert(num_samples)
        except Exception as e:
            logger.error(f"UML专家评估失败: {e}")
            self._force_cleanup_gpu()

        try:
            all_results['general_expert'] = self.evaluate_general_expert(num_samples)
        except Exception as e:
            logger.error(f"通用专家评估失败: {e}")
            self._force_cleanup_gpu()

        if save_dir:
            self._save_all_results(all_results, save_dir)

        self._print_comparison_summary(all_results)

        return all_results

    def _save_all_results(self, results: Dict, save_dir: str):
        """Save all results."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        full_path = save_dir / f'evaluation_results_{timestamp}.json'
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"评估结果已保存至: {full_path}")

        summary = self._create_summary(results)
        summary_path = save_dir / f'evaluation_summary_{timestamp}.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"评估摘要已保存至: {summary_path}")

    def _create_summary(self, results: Dict) -> Dict:
        """Create summary."""
        summary = {}

        for expert_name, result in results.items():
            if not result:
                continue

            summary[expert_name] = {
                'bleu': result['generation_quality'].get('bleu', 0),
                'rouge_l': result['generation_quality'].get('rougeL', 0),
                'meteor': result['generation_quality'].get('meteor', 0),
                'bertscore_f1': result['generation_quality'].get('bertscore_f1', 0),
                'format_score': result['format_metrics'].get('avg_format_score', 0),
                'valid_rate': result['format_metrics'].get('valid_rate', 0),
                'avg_length': result['statistical_metrics']['char_length'].get('mean', 0),
                'precision': result.get('binary_classification', {}).get('precision', 0),
                'recall': result.get('binary_classification', {}).get('recall', 0),
                'f1_score': result.get('binary_classification', {}).get('f1_score', 0),
                'tp': result.get('binary_classification', {}).get('TP', 0),
                'fp': result.get('binary_classification', {}).get('FP', 0),
                'fn': result.get('binary_classification', {}).get('FN', 0)
            }

        return summary

    def _print_comparison_summary(self, results: Dict):
        """Print comparison summary."""
        print("\n" + "=" * 80)
        print("专家评估对比摘要")
        print("=" * 80)

        print(f"\n{'专家':<20} {'BLEU':<10} {'ROUGE-L':<10} {'METEOR':<10} {'BERTScore':<10}")
        print("-" * 80)

        for expert_name, result in results.items():
            if not result:
                continue

            bleu = result['generation_quality'].get('bleu', 0)
            rouge_l = result['generation_quality'].get('rougeL', 0)
            meteor = result['generation_quality'].get('meteor', 0)
            bertscore = result['generation_quality'].get('bertscore_f1', 0)

            print(f"{expert_name:<20} {bleu:<10.4f} {rouge_l:<10.4f} {meteor:<10.4f} {bertscore:<10.4f}")

        print(f"\n{'专家':<20} {'格式分数':<12} {'通过率':<12}")
        print("-" * 80)

        for expert_name, result in results.items():
            if not result:
                continue

            format_score = result['format_metrics'].get('avg_format_score', 0)
            valid_rate = result['format_metrics'].get('valid_rate', 0)

            print(f"{expert_name:<20} {format_score:<12.4f} {valid_rate:<12.2%}")

        print(f"\n{'专家':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'TP':<6} {'FP':<6} {'FN':<6}")
        print("-" * 80)

        for expert_name, result in results.items():
            if not result:
                continue

            binary = result.get('binary_classification', {})
            precision = binary.get('precision', 0)
            recall = binary.get('recall', 0)
            f1 = binary.get('f1_score', 0)
            tp = binary.get('TP', 0)
            fp = binary.get('FP', 0)
            fn = binary.get('FN', 0)

            print(f"{expert_name:<20} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {tp:<6d} {fp:<6d} {fn:<6d}")

        print("=" * 80 + "\n")


def main():
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description='评估专家性能')
    parser.add_argument('--expert', type=str, choices=['text', 'image', 'uml', 'general', 'all'],
                        default='all', help='要评估的专家')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='使用的样本数(None表示全部)')
    parser.add_argument('--test-mode', action='store_true',
                        help='测试模式:每个数据集只使用10条数据快速验证流程')
    parser.add_argument('--show-samples', action='store_true',
                        help='显示测试数据样本(前5条)')
    parser.add_argument('--use-bertscore', action='store_true', default=True,
                        help='使用BERTScore评估语义相似度（默认启用）')
    parser.add_argument('--no-bertscore', dest='use_bertscore', action='store_false',
                        help='禁用BERTScore（加快评估速度）')
    parser.add_argument('--strict-validation', action='store_true',
                        help='使用严格的格式验证')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='保存目录')

    args = parser.parse_args()

    try:
        if args.test_mode:
            if args.num_samples is None:
                args.num_samples = 10
                logger.info("=" * 80)
                logger.info("测试模式已启用 - 每个数据集使用10条数据")
                logger.info("=" * 80)
            else:
                logger.warning(f"测试模式已启用，但--num-samples已设置为{args.num_samples}，将使用该值")

        evaluator = ExpertEvaluator(
            use_bertscore=args.use_bertscore,
            strict_validation=args.strict_validation
        )

        evaluator.show_samples = args.show_samples

        if args.save_dir is None:
            path_cfg = get_path_config()
            args.save_dir = str(path_cfg.METRICS_DIR)

        if args.expert == 'all':
            results = evaluator.evaluate_all_experts(
                num_samples=args.num_samples,
                save_dir=args.save_dir
            )
        elif args.expert == 'text':
            results = evaluator.evaluate_text_expert(num_samples=args.num_samples)
            if args.save_dir:
                evaluator._save_all_results({'text_expert': results}, args.save_dir)
        elif args.expert == 'image':
            results = evaluator.evaluate_image_expert(num_samples=args.num_samples)
            if args.save_dir:
                evaluator._save_all_results({'image_expert': results}, args.save_dir)
        elif args.expert == 'uml':
            results = evaluator.evaluate_uml_expert(num_samples=args.num_samples)
            if args.save_dir:
                evaluator._save_all_results({'uml_expert': results}, args.save_dir)
        elif args.expert == 'general':
            results = evaluator.evaluate_general_expert(num_samples=args.num_samples)
            if args.save_dir:
                evaluator._save_all_results({'general_expert': results}, args.save_dir)

        logger.info("=" * 80)
        logger.info("所有评估任务完成!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("检测到用户中断 (Ctrl+C)")
        logger.warning("评估任务已停止")
        logger.warning("=" * 80)
        sys.exit(1)

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"评估过程中发生严重错误: {e}")
        logger.error(f"异常详情:\n{traceback.format_exc()}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
