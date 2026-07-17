"""Initialize the utils package."""

from .logger import (
    setup_logger,
    get_logger,
    log_model_info,
    log_training_metrics,
    log_gpu_memory,
    log_data_info,
    log_config,
    log_recognition_failure
)

from .file_utils import (
    ensure_dir,
    safe_path_join,
    get_relative_path,
    validate_path_exists,

    load_json,
    save_json,
    update_json,

    load_csv,
    load_csv_chunks,
    save_csv,

    load_lora_weights,
    save_lora_weights,
    list_checkpoints,

    scan_files,
    batch_process_files,

    get_file_size,
    copy_file_safe,
    create_backup
)

from .enhanced_metrics import EnhancedMetrics

__all__ = [
    'setup_logger',
    'get_logger',
    'log_model_info',
    'log_training_metrics',
    'log_gpu_memory',
    'log_data_info',
    'log_config',
    'log_recognition_failure',

    'ensure_dir',
    'safe_path_join',
    'get_relative_path',
    'validate_path_exists',

    'load_json',
    'save_json',
    'update_json',

    'load_csv',
    'load_csv_chunks',
    'save_csv',

    'load_lora_weights',
    'save_lora_weights',
    'list_checkpoints',

    'scan_files',
    'batch_process_files',

    'get_file_size',
    'copy_file_safe',
    'create_backup',

    'EnhancedMetrics'
]

__version__ = '1.1.0'