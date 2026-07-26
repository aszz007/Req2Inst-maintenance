"""Configure project logging and structured runtime diagnostics."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

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



def log_model_info(logger: logging.Logger, model: Any, model_name: str = "Model"):
    """Log model metadata."""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"{model_name} information")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

        if next(model.parameters()).is_cuda:
            device_id = next(model.parameters()).get_device()
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(device_id) / 1024 ** 3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(device_id) / 1024 ** 3:.2f} GB")


    except Exception as e:
        logger.warning(f"Unable to log model information: {str(e)}")


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
        logger.warning("CUDA is unavailable; unable to log GPU memory usage")
        return

    try:
        allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 3
        max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024 ** 3

        logger.info(f"GPU memory: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB, peak={max_allocated:.2f} GB")
    except Exception as e:
        logger.warning(f"Unable to log GPU memory usage: {str(e)}")


def log_data_info(
        logger: logging.Logger,
        dataset_name: str,
        train_size: int,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None
):
    """Log dataset metadata."""
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Training samples: {train_size}")

    if val_size is not None:
        logger.info(f"Validation samples: {val_size}")

    if test_size is not None:
        logger.info(f"Test samples: {test_size}")

    total = train_size + (val_size or 0) + (test_size or 0)
    logger.info(f"Total samples: {total}")


def log_config(logger: logging.Logger, config: Dict[str, Any], config_name: str = "Configuration"):
    """Log configuration values."""
    logger.info(f"{config_name}")

    for key, value in config.items():
        logger.info(f"{key}: {value}")


def log_recognition_failure(
        logger: logging.Logger,
        file_path: str,
        error: str,
        retry_count: int = 0
):
    """Log an input-recognition failure."""
    retry_info = f" (after {retry_count} retries)" if retry_count > 0 else ""
    logger.error(f"Recognition failed{retry_info}: {file_path}")
    logger.error(f"  Error details: {error}")

if __name__ == "__main__":
    print("=" * 60)
    print("Logging system test")
    print("=" * 60)

    print("\n[Test 1] Creating multiple loggers")
    print("-" * 60)

    logger_train = setup_logger('training.text_expert', level=logging.DEBUG)
    logger_inference = setup_logger('inference.generation', level=logging.INFO)
    logger_data = setup_logger('preprocessing.data_loader', level=logging.INFO)

    print("\n[Test 2] Logging at different levels")
    print("-" * 60)

    logger_train.debug("DEBUG-level message with detailed diagnostic information")
    logger_train.info("INFO-level message for a key workflow event")
    logger_train.warning("WARNING-level message")
    logger_train.error("ERROR-level message")

    print("\n[Test 3] Recording model information")
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
    log_model_info(logger_train, mock_model, "Test model")

    print("\n[Test 4] Recording training metrics")
    print("-" * 60)

    metrics = {
        'loss': 0.5234,
        'accuracy': 0.8765,
        'learning_rate': 2e-4
    }
    log_training_metrics(logger_train, epoch=1, step=100, metrics=metrics, prefix="train")

    print("\n[Test 5] Recording dataset information")
    print("-" * 60)

    log_data_info(logger_data, "CCHIT dataset", train_size=800, val_size=100, test_size=100)

    print("\n[Test 6] Recording configuration information")
    print("-" * 60)

    config = {
        'batch_size': 4,
        'learning_rate': 2e-4,
        'epochs': 3,
        'lora_rank': 8
    }
    log_config(logger_train, config, "Training configuration")

    print("\n[Test 7] Recording GPU memory")
    print("-" * 60)

    log_gpu_memory(logger_train)

    print("\n[Test 8] Recording recognition failure")
    print("-" * 60)
    log_recognition_failure(logger_data, "/path/to/image.jpg", "JSON parsing error", retry_count=2)

    print("\nLogging system test completed!")
    print("Check the log files in the logs/ directory")
