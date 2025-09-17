"""
Base wave model for Nano Binary Networks.

This module provides a base class for wave-based models in Nano Binary Networks,
defining a common interface and shared functionality.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch import Tensor

from config import WaveConfig
from logger import get_logger


class BaseWaveModel(ABC):
    """
    Base class for wave-based models in Nano Binary Networks.

    This abstract class defines the common interface and shared functionality
    for all wave-based models, ensuring consistency across different implementations.
    """

    def __init__(self, config: Optional[WaveConfig] = None):
        """
        Initialize the base wave model.

        Args:
            config: Configuration for the wave model. If None, default configuration is used.
        """
        self.config = config or WaveConfig()
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__} with config: {self.config}")

        # Initialize time steps
        self.t = torch.linspace(
            self.config.time_range[0],
            self.config.time_range[1],
            self.config.time_steps,
            dtype=torch.float32
        )

        # Track performance metrics
        self.metrics: Dict[str, Any] = {}
        self.start_time = time.time()

    @abstractmethod
    def compute_wave(self, frequencies: Tensor, amplitudes: Optional[Tensor] = None) -> Tensor:
        """
        Compute the wave function for the given frequencies and amplitudes.

        Args:
            frequencies: Tensor of frequencies.
            amplitudes: Optional tensor of amplitudes. If None, default amplitudes are used.

        Returns:
            Tensor representing the wave function.
        """
        pass

    def log_interference(self, u1: Tensor, u2: Tensor, name: str, freqs1: Optional[Tensor] = None, freqs2: Optional[Tensor] = None) -> Dict[str, Any]:
        """
        Log interference between two wave functions.

        Args:
            u1: First wave function.
            u2: Second wave function.
            name: Name for the interference pattern.
            freqs1: Optional frequencies for the first wave function.
            freqs2: Optional frequencies for the second wave function.

        Returns:
            Dictionary of interference metrics.
        """
        interference = u1 - u2
        metrics = {
            'max_abs': torch.max(torch.abs(interference)).item(),
            'max_time': self.t[torch.argmax(torch.abs(interference))].item(),
            'norm': torch.sqrt(torch.mean(interference ** 2)).item(),
        }

        self.logger.info(f"Interference for {name}: {metrics}")

        # Log values at specific time steps
        for step in torch.arange(0, 1.01, 0.1):
            idx = torch.argmin(torch.abs(self.t - step))
            self.logger.info(f"t={step.item():.1f}s: interference={interference[idx].item():.8f}")

        # Calculate periods if frequencies are provided
        if freqs1 is not None and freqs2 is not None:
            periods = [1 / f.item() for f in torch.cat((freqs1, freqs2)) if f > 0]
            self.logger.info(f"Periods for {name}: {periods}")

        # Find alignment points
        high_thresh = 0.5 * torch.max(torch.cat((u1, u2)))
        align_t = self.t[(torch.abs(u1) > high_thresh) & (torch.abs(u2) > high_thresh)]
        if align_t.numel() > 0:
            self.logger.info(f"Alignment points at t: {align_t.tolist()}")

        # Store metrics for later analysis
        self.metrics[f"interference_{name}"] = metrics

        return metrics

    def log_performance(self, operation: str, data: Dict[str, Any]) -> None:
        """
        Log performance metrics for an operation.

        Args:
            operation: Name of the operation.
            data: Dictionary of performance data.
        """
        # Add timing information
        data['elapsed_time'] = time.time() - self.start_time

        # Log the benchmark
        self.logger.log_benchmark(operation, data)

    def save_metrics(self, filepath: Optional[str] = None) -> str:
        """
        Save metrics to a file.

        Args:
            filepath: Path to save the metrics. If None, a default path is used.

        Returns:
            Path to the saved file.
        """
        # Add final metrics
        self.metrics['total_runtime'] = time.time() - self.start_time

        # Log the metrics
        self.logger.log_benchmark("final_metrics", self.metrics)

        # Save all benchmarks
        return self.logger.save_benchmarks(filepath)
