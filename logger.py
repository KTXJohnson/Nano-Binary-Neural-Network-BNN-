"""
Standardized logging system for Nano Binary Models.

This module provides a centralized logging system for all Nano Binary Models,
ensuring consistent logging across different models and experiments.
"""

import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch

from config import LoggingConfig


class NBNLogger:
    """
    Standardized logger for Nano Binary Network models.

    This class provides consistent logging functionality across all NBN models,
    including benchmark tracking, tensor conversion for JSON serialization,
    and automatic log file management.
    """

    def __init__(self, config: Optional[LoggingConfig] = None):
        """
        Initialize the logger with the given configuration.

        Args:
            config: Logging configuration. If None, default configuration is used.
        """
        self.config = config or LoggingConfig()
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up the logger based on the configuration."""
        # Create logger
        logger = logging.getLogger("NBNLogger")
        logger.setLevel(getattr(logging, self.config.level))

        # Close and clear existing handlers
        for handler in logger.handlers:
            handler.close()
        logger.handlers = []

        # Create formatter
        formatter = logging.Formatter(
            fmt=self.config.format,
            datefmt=self.config.datefmt
        )

        # Add console handler if enabled
        if self.config.console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        # Add file handler if enabled
        if self.config.file_output:
            # Create log directory if it doesn't exist
            os.makedirs(self.config.log_dir, exist_ok=True)

            # Generate log filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{self.config.file_prefix}_{timestamp}.log"
            log_path = os.path.join(self.config.log_dir, log_filename)

            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_file_size_mb * 1024 * 1024,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)

    def log_benchmark(self, category: str, data: Dict[str, Any]) -> None:
        """
        Log benchmark data for the given category.

        Args:
            category: Category of the benchmark (e.g., 'training', 'inference').
            data: Dictionary containing benchmark data.
        """
        if category not in self.benchmarks:
            self.benchmarks[category] = []

        # Add timestamp
        data['timestamp'] = datetime.now().isoformat()

        # Convert tensors to serializable format
        serializable_data = self._convert_tensors_to_json(data)

        # Add to benchmarks
        self.benchmarks[category].append(serializable_data)

        # Log summary
        self.info(f"Benchmark [{category}]: {json.dumps(serializable_data, indent=None)}")

    def _convert_tensors_to_json(self, data: Dict[str, Any], decimal_places: int = 2) -> Dict[str, Any]:
        """
        Convert PyTorch tensors to JSON-serializable format.

        Args:
            data: Dictionary that may contain PyTorch tensors.
            decimal_places: Number of decimal places to round float values to.
                           This ensures consistent precision for tensor values.

        Returns:
            Dictionary with tensors converted to lists or scalar values.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                # Convert tensor to Python scalar or list
                if value.numel() == 1:
                    # Round to specified decimal places for consistent precision
                    scalar_value = value.item()
                    if isinstance(scalar_value, float):
                        result[key] = round(scalar_value, decimal_places)
                    else:
                        result[key] = scalar_value
                else:
                    # Round each element if it's a float
                    list_value = value.tolist()
                    if all(isinstance(x, float) for x in list_value):
                        result[key] = [round(x, decimal_places) for x in list_value]
                    else:
                        result[key] = list_value
            elif isinstance(value, dict):
                # Recursively convert nested dictionaries
                result[key] = self._convert_tensors_to_json(value, decimal_places)
            elif isinstance(value, (list, tuple)):
                # Convert lists/tuples that might contain tensors
                result[key] = []
                for v in value:
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        scalar_value = v.item()
                        if isinstance(scalar_value, float):
                            result[key].append(round(scalar_value, decimal_places))
                        else:
                            result[key].append(scalar_value)
                    elif isinstance(v, torch.Tensor):
                        list_value = v.tolist()
                        if all(isinstance(x, float) for x in list_value):
                            result[key].append([round(x, decimal_places) for x in list_value])
                        else:
                            result[key].append(list_value)
                    else:
                        result[key].append(v)
            else:
                # Keep other types as is
                result[key] = value
        return result

    def save_benchmarks(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Save benchmarks to a JSON file.

        Args:
            filepath: Path to save the benchmarks. If None, a default path is used.

        Returns:
            Path to the saved file.
        """
        if filepath is None:
            # Create benchmarks directory if it doesn't exist
            benchmarks_dir = os.path.join(self.config.log_dir, "benchmarks")
            os.makedirs(benchmarks_dir, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(benchmarks_dir, f"benchmarks_{timestamp}.json")

        # Add summary statistics
        self.benchmarks['summary'] = self._generate_summary_stats()

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)

        self.info(f"Benchmarks saved to {filepath}")
        return filepath

    def close(self) -> None:
        """
        Close all file handlers and release resources.
        This should be called when the logger is no longer needed.
        """
        if hasattr(self, 'logger') and self.logger:
            for handler in self.logger.handlers:
                handler.close()
            self.logger.handlers = []

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """
        Generate summary statistics for benchmarks.

        Returns:
            Dictionary containing summary statistics.
        """
        summary = {
            'total_runtime': time.time() - self.start_time,
            'timestamp': datetime.now().isoformat(),
            'categories': list(self.benchmarks.keys()),
            'category_counts': {k: len(v) for k, v in self.benchmarks.items()},
        }

        # Add system info
        try:
            import psutil
            summary['system_info'] = {
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'memory_available': psutil.virtual_memory().available,
            }
        except ImportError:
            pass

        # Add PyTorch info
        summary['pytorch_info'] = {
            'version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            summary['pytorch_info']['cuda_device_count'] = torch.cuda.device_count()
            summary['pytorch_info']['cuda_device_name'] = torch.cuda.get_device_name(0)

        return summary


# Singleton instance for easy import
_logger_instance = None


def get_logger(config: Optional[LoggingConfig] = None) -> NBNLogger:
    """
    Get the singleton logger instance.

    Args:
        config: Logging configuration. If None, the existing configuration is used,
               or a default configuration is created if no logger exists yet.

    Returns:
        The singleton logger instance.
    """
    global _logger_instance
    if _logger_instance is None or config is not None:
        # Close the previous logger if it exists
        if _logger_instance is not None:
            _logger_instance.close()
        _logger_instance = NBNLogger(config)
    return _logger_instance
