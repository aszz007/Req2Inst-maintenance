"""Configure project logging and structured runtime diagnostics."""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any
import torch


class LoggerManager:
    """Create and manage project loggers."""

    _instance = None
    _loggers = {}

    def __new__(cls):
        """Return the singleton logger manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the instance."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_log_dirs()

    def _setup_log_dirs(self):
        """Create log directories."""
        try:
            from config import get_path_config
            path_cfg = get_path_config()
            self.logs_dir = path_cfg.LOGS_DIR
            self.training_logs_dir = path_cfg.TRAINING_LOGS_DIR
            self.inference_logs_dir = path_cfg.INFERENCE_LOGS_DIR
            self.preprocessing_logs_dir = path_cfg.PREPROCESSING_LOGS_DIR
        except ImportError:
            project_root = Path(__file__).parent.parent.parent
            self.logs_dir = project_root / "logs"
            self.training_logs_dir = self.logs_dir / "training"
            self.inference_logs_dir = self.logs_dir / "inference"
            self.preprocessing_logs_dir = self.logs_dir / "preprocessing"

        for log_dir in [self.training_logs_dir, self.inference_logs_dir,
                        self.preprocessing_logs_dir]:
            log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_dir(self, module_name: str) -> Path:
        """Return log dir."""
        if 'training' in module_name:
            return self.training_logs_dir
        elif 'inference' in module_name or 'generation' in module_name:
            return self.inference_logs_dir
        elif 'preprocessing' in module_name or 'data' in module_name:
            return self.preprocessing_logs_dir
        else:
            return self.logs_dir

    def _create_formatter(self, detailed: bool = True) -> logging.Formatter:
        """Create formatter."""
        if detailed:
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s'
        else:
            fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

        return logging.Formatter(
            fmt=fmt,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    def setup_logger(
            self,
            module_name: str,
            level: int = logging.INFO,
            console_output: bool = True,
            file_output: bool = True
    ) -> logging.Logger:
        """Configure a logger."""
        if module_name in self._loggers:
            return self._loggers[module_name]

        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        logger.propagate = False

        logger.handlers.clear()

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(self._create_formatter(detailed=False))
            logger.addHandler(console_handler)

        if file_output:
            log_dir = self._get_log_dir(module_name)
            date_str = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"{module_name.replace('.', '_')}_{date_str}.log"

            file_handler = RotatingFileHandler(
                filename=log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(self._create_formatter(detailed=True))
            logger.addHandler(file_handler)

        self._loggers[module_name] = logger

        return logger

    def get_logger(self, module_name: str) -> logging.Logger:
        """Return logger."""
        if module_name not in self._loggers:
            return self.setup_logger(module_name)
        return self._loggers[module_name]


_logger_manager = LoggerManager()


def setup_logger(
        module_name: str,
        level: int = logging.INFO,
        console_output: bool = True,
        file_output: bool = True
) -> logging.Logger:
    """Configure a logger."""
    return _logger_manager.setup_logger(module_name, level, console_output, file_output)


def get_logger(module_name: str) -> logging.Logger:
    """Return logger."""
    return _logger_manager.get_logger(module_name)



def log_model_info(logger: logging.Logger, model: Any, model_name: str = "模型"):
    """Log model metadata."""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info(f"{model_name}信息")
        logger.info("=" * 60)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"可训练比例: {100 * trainable_params / total_params:.2f}%")

        if next(model.parameters()).is_cuda:
            device_id = next(model.parameters()).get_device()
            logger.info(f"GPU显存使用: {torch.cuda.memory_allocated(device_id) / 1024 ** 3:.2f} GB")
            logger.info(f"GPU显存缓存: {torch.cuda.memory_reserved(device_id) / 1024 ** 3:.2f} GB")

        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"无法记录模型信息: {str(e)}")


def log_training_metrics(
        logger: logging.Logger,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        prefix: str = ""
):
    """Log training metrics."""
    prefix_str = f"[{prefix}] " if prefix else ""
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix_str}Epoch {epoch} Step {step} | {metric_str}")


def log_gpu_memory(logger: logging.Logger, device_id: int = 0):
    """Log GPU memory use."""
    if not torch.cuda.is_available():
        logger.warning("CUDA不可用，无法记录GPU显存")
        return

    try:
        allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024 ** 3

        logger.info(f"GPU显存: 已分配={allocated:.2f}GB, 已缓存={reserved:.2f}GB, 峰值={max_allocated:.2f}GB")
    except Exception as e:
        logger.warning(f"无法记录GPU显存: {str(e)}")


def log_data_info(
        logger: logging.Logger,
        dataset_name: str,
        train_size: int,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None
):
    """Log dataset metadata."""
    logger.info("=" * 60)
    logger.info(f"数据集: {dataset_name}")
    logger.info("=" * 60)
    logger.info(f"训练集样本数: {train_size}")

    if val_size is not None:
        logger.info(f"验证集样本数: {val_size}")

    if test_size is not None:
        logger.info(f"测试集样本数: {test_size}")

    total = train_size + (val_size or 0) + (test_size or 0)
    logger.info(f"总样本数: {total}")
    logger.info("=" * 60)


def log_config(logger: logging.Logger, config: Dict[str, Any], config_name: str = "配置"):
    """Log configuration values."""
    logger.info("=" * 60)
    logger.info(f"{config_name}")
    logger.info("=" * 60)

    for key, value in config.items():
        logger.info(f"{key}: {value}")

    logger.info("=" * 60)

def log_recognition_failure(
        logger: logging.Logger,
        file_path: str,
        error: str,
        retry_count: int = 0
):
    """Log an input-recognition failure."""
    retry_info = f"(重试{retry_count}次后)" if retry_count > 0 else ""
    logger.error(f"识别失败{retry_info}: {file_path}")
    logger.error(f"  错误详情: {error}")

if __name__ == "__main__":
    print("=" * 60)
    print("日志系统测试")
    print("=" * 60)

    print("\n【测试1】创建多个logger")
    print("-" * 60)

    logger_train = setup_logger('training.text_expert', level=logging.DEBUG)
    logger_inference = setup_logger('inference.generation', level=logging.INFO)
    logger_data = setup_logger('preprocessing.data_loader', level=logging.INFO)

    print("\n【测试2】不同级别的日志输出")
    print("-" * 60)

    logger_train.debug("这是DEBUG级别的消息（详细调试信息）")
    logger_train.info("这是INFO级别的消息（关键流程信息）")
    logger_train.warning("这是WARNING级别的消息（警告信息）")
    logger_train.error("这是ERROR级别的消息（错误信息）")

    print("\n【测试3】记录模型信息")
    print("-" * 60)


    class MockModel:
        """Provide a minimal model for logger smoke tests."""

        def __init__(self):
            self.param1 = torch.nn.Parameter(torch.randn(1000, 1000))
            self.param2 = torch.nn.Parameter(torch.randn(500, 500))

        def parameters(self):
            """Return the mock model parameters."""
            return [self.param1, self.param2]


    mock_model = MockModel()
    log_model_info(logger_train, mock_model, "测试模型")

    print("\n【测试4】记录训练指标")
    print("-" * 60)

    metrics = {
        'loss': 0.5234,
        'accuracy': 0.8765,
        'learning_rate': 2e-4
    }
    log_training_metrics(logger_train, epoch=1, step=100, metrics=metrics, prefix="train")

    print("\n【测试5】记录数据集信息")
    print("-" * 60)

    log_data_info(logger_data, "CCHIT数据集", train_size=800, val_size=100, test_size=100)

    print("\n【测试6】记录配置信息")
    print("-" * 60)

    config = {
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 3,
        'lora_rank': 8
    }
    log_config(logger_train, config, "训练配置")

    print("\n【测试7】GPU显存记录")
    print("-" * 60)

    log_gpu_memory(logger_train)

    print("\n【测试8】记录识别失败")
    print("-" * 60)
    log_recognition_failure(logger_data, "/path/to/image.jpg", "JSON解析错误", retry_count=2)

    print("\n日志系统测试完成！")
    print("请检查 logs/ 目录下的日志文件")
