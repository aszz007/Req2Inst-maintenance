"""Calculate generation, format, and binary evaluation metrics."""

from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')

from src.utils.logger import get_logger  # noqa: E402

logger = get_logger('metrics.enhanced')


class EvaluationThresholds:
    """Store metric decision thresholds."""

    ROUGE_L_THRESHOLD = 0.5
    BERTSCORE_F1_THRESHOLD = 0.85

    USE_AND_LOGIC = True

    FORMAT_SCORE_THRESHOLD = 1.0

    @classmethod
    def get_config(cls) -> dict:
        """Return config."""
        return {
            'rouge_l_threshold': cls.ROUGE_L_THRESHOLD,
            'bertscore_f1_threshold': cls.BERTSCORE_F1_THRESHOLD,
            'use_and_logic': cls.USE_AND_LOGIC,
            'format_score_threshold': cls.FORMAT_SCORE_THRESHOLD
        }

    @classmethod
    def update_config(cls, rouge_l: float = None, bertscore_f1: float = None,
                     use_and: bool = None, format_score: float = None):
        """Update config."""
        if rouge_l is not None:
            cls.ROUGE_L_THRESHOLD = rouge_l
        if bertscore_f1 is not None:
            cls.BERTSCORE_F1_THRESHOLD = bertscore_f1
        if use_and is not None:
            cls.USE_AND_LOGIC = use_and
        if format_score is not None:
            cls.FORMAT_SCORE_THRESHOLD = format_score


class EnhancedMetrics:
    """Calculate generation and quality metrics."""

    def __init__(self, use_bertscore: bool = True):
        """Initialize the instance."""
        self.use_bertscore = use_bertscore

        self.bleu_metric = None
        self.rouge_metric = None
        self.meteor_metric = None
        self.bertscore_metric = None

        logger.info("Initializing enhanced evaluation metrics")
        if use_bertscore:
            logger.info("BERTScore enabled by default for semantic-similarity evaluation")

    def cleanup(self):
        """Release temporary resources."""
        import gc
        for attr in ('bertscore_metric', 'bleu_metric', 'rouge_metric', 'meteor_metric'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    del obj
                except Exception:
                    pass
                setattr(self, attr, None)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("GPU memory used by evaluation metrics has been released")
        except ImportError:
            pass

    def _lazy_load_metrics(self):
        """Lazily load metric implementations."""
        if self.bleu_metric is None:
            try:
                from evaluate import load

                self._ensure_nltk_data()

                logger.info("Loading BLEU metric...")
                self.bleu_metric = load('bleu')

                logger.info("Loading ROUGE metric...")
                self.rouge_metric = load('rouge')

                logger.info("Loading METEOR metric...")
                self.meteor_metric = load('meteor')

                if self.use_bertscore:
                    try:
                        logger.info("Loading BERTScore metric...")
                        self.bertscore_metric = load('bertscore')
                        logger.info("BERTScore metric loaded successfully")
                    except Exception as e:
                        logger.warning(f"Failed to load BERTScore metric: {e}")
                        logger.warning("BERTScore computation will be skipped")
                        self.use_bertscore = False

                logger.info("Evaluation metrics loaded")
            except Exception as e:
                logger.error(f"Failed to load evaluation metrics: {e}")
                raise

    def _ensure_nltk_data(self):
        """Ensure required NLTK resources are available."""
        try:
            import nltk
            from nltk.data import find

            required_data = [
                ('corpora/wordnet', 'wordnet'),
                ('corpora/omw-1.4', 'omw-1.4'),
                ('tokenizers/punkt', 'punkt'),
                ('tokenizers/punkt_tab', 'punkt_tab')
            ]

            logger.info("Checking NLTK resources...")

            for data_path, data_name in required_data:
                try:
                    find(data_path)
                    logger.debug(f"NLTK resource is already available: {data_name}")
                except LookupError:
                    logger.warning(f"NLTK resource is missing: {data_name}; attempting download...")
                    try:
                        nltk.download(data_name, quiet=True)
                        logger.info(f"NLTK resource downloaded successfully: {data_name}")
                    except Exception as e:
                        logger.warning(f"Failed to download NLTK resource {data_name}: {e}")
                        logger.warning("METEOR computation may fail or run slowly")

            logger.info("NLTK resource check complete")

        except ImportError:
            logger.warning("NLTK is not installed; METEOR computation may fail")
        except Exception as e:
            logger.warning(f"NLTK resource check failed: {e}")
            logger.warning("Continuing, but METEOR computation may fail")

    def calculate_generation_quality(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """Calculate generation quality."""
        self._lazy_load_metrics()

        if len(predictions) != len(references):
            raise ValueError(
                f"Prediction and reference counts do not match: {len(predictions)} vs {len(references)}"
            )

        logger.info(f"Computing generation-quality metrics - samples: {len(predictions)}")

        results = {}

        # BLEU
        try:
            logger.info("Computing BLEU...")
            bleu_result = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            results['bleu'] = bleu_result['bleu']
            logger.info(f"BLEU computation complete: {results['bleu']:.4f}")
        except Exception as e:
            logger.error(f"BLEU computation failed: {e}")
            results['bleu'] = 0.0

        # ROUGE
        try:
            logger.info("Computing ROUGE...")
            rouge_result = self.rouge_metric.compute(
                predictions=predictions,
                references=references
            )
            results['rouge1'] = rouge_result['rouge1']
            results['rouge2'] = rouge_result['rouge2']
            results['rougeL'] = rouge_result['rougeL']
            logger.info(f"ROUGE computation complete - ROUGE-L: {results['rougeL']:.4f}")
        except Exception as e:
            logger.error(f"ROUGE computation failed: {e}")
            results['rouge1'] = results['rouge2'] = results['rougeL'] = 0.0

        # METEOR
        try:
            logger.info("Computing METEOR...")
            logger.info(f"METEOR computation in progress - samples: {len(predictions)}; this may take a while...")

            meteor_result = self.meteor_metric.compute(
                predictions=predictions,
                references=references
            )
            results['meteor'] = meteor_result['meteor']
            logger.info(f"METEOR computation complete: {results['meteor']:.4f}")
        except Exception as e:
            logger.error(f"METEOR computation failed: {e}")
            logger.error("Possible cause: missing NLTK resources or a network problem")
            logger.error("Suggested action: download the required NLTK resources manually or disable METEOR")
            results['meteor'] = 0.0

        # BERTScore
        if self.use_bertscore and self.bertscore_metric is not None:
            try:
                logger.info("Computing BERTScore...")
                logger.info("BERTScore computation in progress; this may take several minutes...")

                bertscore_result = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang='en'
                )
                results['bertscore_precision'] = sum(bertscore_result['precision']) / len(predictions)
                results['bertscore_recall'] = sum(bertscore_result['recall']) / len(predictions)
                results['bertscore_f1'] = sum(bertscore_result['f1']) / len(predictions)
                results['bertscore_f1_scores'] = list(bertscore_result['f1'])
                logger.info(f"BERTScore computation complete - F1: {results['bertscore_f1']:.4f}")
            except Exception as e:
                logger.error(f"BERTScore computation failed: {e}")
                results['bertscore_precision'] = 0.0
                results['bertscore_recall'] = 0.0
                results['bertscore_f1'] = 0.0
                results['bertscore_f1_scores'] = []

        logger.info("All generation-quality metrics computed")
        return results

    def calculate_format_metrics(
        self,
        instructions: List[str]
    ) -> Dict[str, Any]:
        """Calculate format metrics."""
        logger.info(f"Computing format metrics - samples: {len(instructions)}")

        format_results = []

        for instruction in instructions:
            result = self._check_single_format(instruction)
            format_results.append(result)

        total = len(format_results)

        summary = {
            'total_samples': total,
            'valid_count': sum(1 for r in format_results if r['is_valid']),
            'valid_rate': sum(1 for r in format_results if r['is_valid']) / total if total > 0 else 0,

            'definition_present': sum(1 for r in format_results if r['has_definition']) / total if total > 0 else 0,
            'definition_has_content': sum(1 for r in format_results if r['definition_has_content']) / total if total > 0 else 0,
            'definition_in_this_task': sum(1 for r in format_results if r['definition_starts_with_in_this_task']) / total if total > 0 else 0,

            'emphasis_present': sum(1 for r in format_results if r['has_emphasis']) / total if total > 0 else 0,
            'emphasis_valid': sum(1 for r in format_results if r['emphasis_valid']) / total if total > 0 else 0,

            'avoid_present': sum(1 for r in format_results if r['has_avoid']) / total if total > 0 else 0,
            'avoid_valid': sum(1 for r in format_results if r['avoid_valid']) / total if total > 0 else 0,

            'avg_format_score': sum(r['format_score'] for r in format_results) / total if total > 0 else 0,

            'detailed_results': format_results
        }

        logger.info(f"Format-validation pass rate: {summary['valid_rate']:.2%}")
        logger.info(f"Average format score: {summary['avg_format_score']:.4f}")

        return summary

    def calculate_statistical_metrics(
        self,
        instructions: List[str],
        expert_usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Calculate statistical metrics."""
        logger.info(f"Computing statistical metrics - samples: {len(instructions)}")

        lengths = [len(inst) for inst in instructions]
        word_counts = [len(inst.split()) for inst in instructions]
        line_counts = [len(inst.split('\n')) for inst in instructions]

        stats = {
            'char_length': {
                'mean': sum(lengths) / len(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'median': sorted(lengths)[len(lengths)//2] if lengths else 0
            },

            'word_count': {
                'mean': sum(word_counts) / len(word_counts) if word_counts else 0,
                'min': min(word_counts) if word_counts else 0,
                'max': max(word_counts) if word_counts else 0,
                'median': sorted(word_counts)[len(word_counts)//2] if word_counts else 0
            },

            'line_count': {
                'mean': sum(line_counts) / len(line_counts) if line_counts else 0,
                'min': min(line_counts) if line_counts else 0,
                'max': max(line_counts) if line_counts else 0,
                'median': sorted(line_counts)[len(line_counts)//2] if line_counts else 0
            }
        }

        if expert_usage:
            total_usage = sum(expert_usage.values())
            stats['expert_usage'] = {
                'total_calls': total_usage,
                'usage_by_expert': expert_usage,
                'usage_percentage': {
                    expert: (count / total_usage * 100) if total_usage > 0 else 0
                    for expert, count in expert_usage.items()
                }
            }

        logger.info(f"Average character length: {stats['char_length']['mean']:.1f}")
        logger.info(f"Average word count: {stats['word_count']['mean']:.1f}")

        return stats

    def calculate_binary_classification_metrics(
        self,
        predictions: List[str],
        references: List[str],
        format_threshold: float = None,
        rouge_threshold: float = None,
        bertscore_threshold: float = None,
        use_and_logic: bool = None,
        precomputed_bertscore_f1: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Calculate binary classification metrics."""
        if format_threshold is None:
            format_threshold = EvaluationThresholds.FORMAT_SCORE_THRESHOLD
        if rouge_threshold is None:
            rouge_threshold = EvaluationThresholds.ROUGE_L_THRESHOLD
        if bertscore_threshold is None:
            bertscore_threshold = EvaluationThresholds.BERTSCORE_F1_THRESHOLD
        if use_and_logic is None:
            use_and_logic = EvaluationThresholds.USE_AND_LOGIC

        logger.info(f"Computing binary-classification metrics - samples: {len(predictions)}")
        logger.info("Threshold configuration:")
        logger.info(f"  Format-score threshold: {format_threshold}")
        logger.info(f"  ROUGE-L threshold: {rouge_threshold}")
        logger.info(f"  BERTScore F1 threshold: {bertscore_threshold}")
        logger.info(f"  Combination rule: {'AND (both must pass)' if use_and_logic else 'OR (either may pass)'}")

        if len(predictions) != len(references):
            raise ValueError(
                f"Prediction and reference counts do not match: {len(predictions)} vs {len(references)}"
            )

        self._lazy_load_metrics()
        try:
            per_sample_rouge = self.rouge_metric.compute(
                predictions=predictions,
                references=references,
                use_aggregator=False
            )
            rouge_l_scores = per_sample_rouge['rougeL']
        except Exception as e:
            logger.error(f"ROUGE-L computation failed: {e}")
            rouge_l_scores = [0.0] * len(predictions)

        bertscore_f1_scores = []
        if precomputed_bertscore_f1 is not None and len(precomputed_bertscore_f1) == len(predictions):
            bertscore_f1_scores = precomputed_bertscore_f1
            logger.info(f"Using precomputed BERTScore - mean F1: {sum(bertscore_f1_scores)/len(bertscore_f1_scores):.4f}")
        elif self.use_bertscore and self.bertscore_metric is not None:
            try:
                logger.info("Computing semantic similarity with BERTScore...")
                bertscore_result = self.bertscore_metric.compute(
                    predictions=predictions,
                    references=references,
                    lang='en'
                )
                bertscore_f1_scores = bertscore_result['f1']
                logger.info(f"Mean BERTScore F1: {sum(bertscore_f1_scores)/len(bertscore_f1_scores):.4f}")
            except Exception as e:
                logger.error(f"BERTScore computation failed: {e}")
                bertscore_f1_scores = [0.0] * len(predictions)

        tp = 0  # True Positive
        fp = 0  # False Positive
        fn = 0  # False Negative
        tn = 0

        valid_samples = []
        invalid_samples = []

        for i, (pred, ref) in enumerate(zip(predictions, references)):
            format_check = self._check_single_format(pred)
            is_format_valid = (
                format_check['has_definition'] and
                format_check['has_emphasis'] and
                format_check['has_avoid'] and
                format_check['format_score'] >= format_threshold
            )

            rouge_l_score = rouge_l_scores[i]
            rouge_valid = rouge_l_score >= rouge_threshold

            if bertscore_f1_scores:
                bertscore_f1 = bertscore_f1_scores[i]
                bertscore_valid = bertscore_f1 >= bertscore_threshold

                if use_and_logic:
                    is_semantic_valid = rouge_valid and bertscore_valid
                else:
                    is_semantic_valid = rouge_valid or bertscore_valid
            else:
                is_semantic_valid = rouge_valid

            if is_format_valid and is_semantic_valid:
                tp += 1
                valid_samples.append(i)
            elif is_format_valid and not is_semantic_valid:
                fp += 1
                invalid_samples.append(i)
            else:
                fn += 1
                invalid_samples.append(i)

        total = len(predictions)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        results = {
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,

            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,

            'total_samples': total,
            'valid_samples': valid_samples,
            'invalid_samples': invalid_samples,

            'format_threshold': format_threshold,
            'rouge_threshold': rouge_threshold,
            'bertscore_threshold': bertscore_threshold,
            'use_and_logic': use_and_logic,
            'use_bertscore': self.use_bertscore and len(bertscore_f1_scores) > 0
        }

        logger.info("Binary-classification metrics computed:")
        logger.info(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        logger.info(f"  Precision: {precision:.4f}, recall: {recall:.4f}")
        logger.info(f"  F1 score: {f1_score:.4f}, accuracy: {accuracy:.4f}")

        return results

    def _check_single_format(self, instruction: str) -> Dict[str, Any]:
        """Check the manuscript-required three-part instruction format."""
        result = {
            'is_valid': False,
            'has_definition': False,
            'definition_has_content': False,
            'definition_starts_with_in_this_task': False,
            'has_emphasis': False,
            'emphasis_valid': False,
            'emphasis_is_valid': False,
            'has_avoid': False,
            'avoid_valid': False,
            'avoid_is_valid': False,
            'format_score': 0.0,
            'errors': []
        }

        if not instruction:
            result['errors'] = [
                '缺少Definition',
                '缺少Emphasis & Caution',
                '缺少Things to Avoid'
            ]
            return result

        lines = [line.strip() for line in instruction.strip().split('\n') if line.strip()]

        section_headers = [
            ('Definition:', 'definition', len('Definition:')),
            ('Emphasis & Caution:', 'emphasis', len('Emphasis & Caution:')),
            ('Emphasis and Caution:', 'emphasis', len('Emphasis and Caution:')),
            ('Things to Avoid:', 'avoid', len('Things to Avoid:')),
        ]

        def _match_header(line):
            for prefix, key, offset in section_headers:
                if line.startswith(prefix):
                    return key, line[offset:].strip()
            return None, None

        sections = {}
        current_key = None
        current_lines = []

        for line in lines:
            key, inline_content = _match_header(line)
            if key is not None:
                if current_key is not None:
                    sections[current_key] = '\n'.join(current_lines).strip()
                current_key = key
                current_lines = [inline_content] if inline_content else []
            elif current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            sections[current_key] = '\n'.join(current_lines).strip()

        if 'definition' in sections:
            result['has_definition'] = True
            content = sections['definition']
            if content and content != '-':
                result['definition_has_content'] = True
            else:
                result['errors'].append('Definition没有实际内容')

            if content.lower().startswith('in this task'):
                result['definition_starts_with_in_this_task'] = True
        else:
            result['errors'].append('缺少Definition')

        if 'emphasis' in sections:
            result['has_emphasis'] = True
            if sections['emphasis']:
                result['emphasis_valid'] = True
                result['emphasis_is_valid'] = True
        else:
            result['errors'].append('缺少Emphasis & Caution')

        if 'avoid' in sections:
            result['has_avoid'] = True
            if sections['avoid']:
                result['avoid_valid'] = True
                result['avoid_is_valid'] = True
        else:
            result['errors'].append('缺少Things to Avoid')

        score = 0.0
        if result['definition_has_content']:
            score += 0.4
        if result['has_emphasis']:
            score += 0.3
        if result['has_avoid']:
            score += 0.3

        result['format_score'] = score
        result['is_valid'] = (
            result['definition_has_content'] and
            result['has_emphasis'] and
            result['has_avoid']
        )

        return result

    def generate_comprehensive_report(
        self,
        predictions: List[str],
        references: List[str],
        expert_usage: Optional[Dict[str, int]] = None,
        save_path: Optional[str] = None,
        include_binary_metrics: bool = True,
        format_threshold: float = None,
        rouge_threshold: float = None,
        bertscore_threshold: float = None,
        use_and_logic: bool = None
    ) -> Dict[str, Any]:
        """Generate one canonical report for valid prediction/reference pairs."""
        logger.info("Generating comprehensive evaluation report")

        if len(predictions) != len(references):
            raise ValueError(
                f"Prediction and reference counts do not match: {len(predictions)} vs {len(references)}"
            )

        valid_pairs = [
            (prediction, reference)
            for prediction, reference in zip(predictions, references)
            if prediction and prediction.strip()
        ]
        skipped_samples = len(predictions) - len(valid_pairs)

        if skipped_samples:
            logger.warning(
                f"Skipped {skipped_samples} empty predictions out of {len(predictions)}"
            )

        if not valid_pairs:
            logger.error("No valid predictions are available for evaluation")
            return {}

        valid_predictions = [pair[0] for pair in valid_pairs]
        valid_references = [pair[1] for pair in valid_pairs]

        effective_format_threshold = (
            EvaluationThresholds.FORMAT_SCORE_THRESHOLD
            if format_threshold is None else format_threshold
        )
        effective_rouge_threshold = (
            EvaluationThresholds.ROUGE_L_THRESHOLD
            if rouge_threshold is None else rouge_threshold
        )
        effective_bertscore_threshold = (
            EvaluationThresholds.BERTSCORE_F1_THRESHOLD
            if bertscore_threshold is None else bertscore_threshold
        )
        effective_use_and_logic = (
            EvaluationThresholds.USE_AND_LOGIC
            if use_and_logic is None else use_and_logic
        )

        report = {
            'metadata': {
                'total_samples': len(predictions),
                'valid_samples': len(valid_predictions),
                'skipped_samples': skipped_samples,
                'timestamp': self._get_timestamp()
            },
            'total_samples': len(predictions),
            'valid_samples': len(valid_predictions),
            'skipped_samples': skipped_samples,
            'threshold_config': {
                'rouge_l_threshold': effective_rouge_threshold,
                'bertscore_f1_threshold': effective_bertscore_threshold,
                'use_and_logic': effective_use_and_logic,
                'format_score_threshold': effective_format_threshold
            }
        }

        logger.info("\n[1/4] Computing generation-quality metrics...")
        report['generation_quality'] = self.calculate_generation_quality(
            valid_predictions, valid_references
        )

        logger.info("\n[2/4] Computing format metrics...")
        report['format_metrics'] = self.calculate_format_metrics(valid_predictions)

        if include_binary_metrics:
            logger.info("\n[3/4] Computing binary-classification metrics (TP/TN/FP/FN)...")
            precomputed_bs = report['generation_quality'].get('bertscore_f1_scores')
            report['binary_classification'] = self.calculate_binary_classification_metrics(
                valid_predictions,
                valid_references,
                format_threshold=effective_format_threshold,
                rouge_threshold=effective_rouge_threshold,
                bertscore_threshold=effective_bertscore_threshold,
                use_and_logic=effective_use_and_logic,
                precomputed_bertscore_f1=precomputed_bs
            )
        else:
            logger.info("\n[3/4] Skipping binary-classification metrics")

        logger.info("\n[4/4] Computing statistical metrics...")
        report['statistical_metrics'] = self.calculate_statistical_metrics(
            valid_predictions, expert_usage
        )

        if save_path:
            self._save_report(report, save_path)

        logger.info("Comprehensive evaluation report generated")

        return report

    def _get_timestamp(self) -> str:
        """Return timestamp."""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _save_report(self, report: Dict, save_path: str):
        """Save report."""
        import json
        from pathlib import Path

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation report saved to: {save_path}")

    def print_report_summary(self, report: Dict):
        """Print report summary."""
        print("\n" + "=" * 80)
        print("Evaluation Report Summary")
        print("=" * 80)

        print("\n[Generation Quality Metrics]")
        quality = report['generation_quality']
        print(f"  BLEU:      {quality['bleu']:.4f}")
        print(f"  ROUGE-1:   {quality['rouge1']:.4f}")
        print(f"  ROUGE-2:   {quality['rouge2']:.4f}")
        print(f"  ROUGE-L:   {quality['rougeL']:.4f}")
        print(f"  METEOR:    {quality['meteor']:.4f}")
        if 'bertscore_f1' in quality:
            print(f"  BERTScore P: {quality['bertscore_precision']:.4f}")
            print(f"  BERTScore R: {quality['bertscore_recall']:.4f}")
            print(f"  BERTScore F1: {quality['bertscore_f1']:.4f}")

        print("\n[Format Metrics]")
        format_m = report['format_metrics']
        print(f"  Format validation pass rate: {format_m['valid_rate']:.2%}")
        print(f"  Average format score:        {format_m['avg_format_score']:.4f}")
        print(f"  Definition valid:            {format_m['definition_has_content']:.2%}")
        print(f"  Emphasis valid:              {format_m['emphasis_valid']:.2%}")
        print(f"  Avoid valid:                 {format_m['avoid_valid']:.2%}")

        if 'binary_classification' in report:
            print("\n[Binary Classification Metrics (TP/TN/FP/FN)]")
            binary = report['binary_classification']
            print(f"  TP (True Positive):  {binary['TP']:4d}  - format correct and semantically adequate")
            print(f"  FP (False Positive): {binary['FP']:4d}  - format correct but semantically inadequate")
            print(f"  FN (False Negative): {binary['FN']:4d}  - format incorrect or semantically inadequate")
            print(f"  TN (True Negative):  {binary['TN']:4d}  - not applicable")
            print("  ---")
            print(f"  Precision:           {binary['precision']:.4f}")
            print(f"  Recall:              {binary['recall']:.4f}")
            print(f"  F1 Score:           {binary['f1_score']:.4f}")
            print(f"  Accuracy:            {binary['accuracy']:.4f}")

        print("\n[Statistical Metrics]")
        stats = report['statistical_metrics']
        print(f"  Average character length: {stats['char_length']['mean']:.1f}")
        print(f"  Average word count:       {stats['word_count']['mean']:.1f}")
        print(f"  Average line count:       {stats['line_count']['mean']:.1f}")

        if 'expert_usage' in stats:
            print("\n[Expert Usage Statistics]")
            for expert, pct in stats['expert_usage']['usage_percentage'].items():
                print(f"  {expert}: {pct:.1f}%")

        print("=" * 80 + "\n")
