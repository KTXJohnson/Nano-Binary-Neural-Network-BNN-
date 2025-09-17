"""
Configuration system for Nano Binary Models.

This module provides a centralized configuration system for all Nano Binary Models,
allowing for consistent parameter management across different models and experiments.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Union


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = "logs"
    level: str = "INFO"
    format: str = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"
    file_prefix: str = "nbn"
    console_output: bool = True
    file_output: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class WaveConfig:
    """Configuration for wave parameters."""
    frequency_range: List[float] = field(default_factory=lambda: [0.1, 200.0])
    amplitude_range: List[float] = field(default_factory=lambda: [1.0, 600000.0])
    phase_offset_range: List[float] = field(default_factory=lambda: [0.0, 6.28])
    viscosity: float = 0.5
    base_resistance: float = 1.0
    time_steps: int = 1000
    time_range: List[float] = field(default_factory=lambda: [0.0, 1.0])


@dataclass
class DictionaryConfig:
    """Configuration for dictionary loading."""
    limit: int = 50000
    cache_dir: str = "cache"
    use_cache: bool = True
    url: str = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"


@dataclass
class ProcessorConfig:
    """Configuration for NBN processors."""
    num_layers: int = 6
    layer_sizes: List[int] = field(default_factory=lambda: [100, 200, 300, 200, 100, 50])
    activation: str = "relu"
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    use_cuda: bool = True
    seed: int = 42


@dataclass
class Config:
    """Main configuration class that contains all sub-configurations."""
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wave: WaveConfig = field(default_factory=WaveConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    processor: ProcessorConfig = field(default_factory=ProcessorConfig)
    experiment_name: str = "default"
    output_dir: str = "output"

    def save(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Save configuration to a JSON file."""
        if filepath is None:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, f"{self.experiment_name}_config.json")

        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

        return filepath

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Config':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """Load configuration from a JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a Config object from a dictionary."""
        # Convert dictionaries to dataclasses
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        wave_config = WaveConfig(**config_dict.get('wave', {}))
        dictionary_config = DictionaryConfig(**config_dict.get('dictionary', {}))
        processor_config = ProcessorConfig(**config_dict.get('processor', {}))

        # Create and return the main config
        return cls(
            logging=logging_config,
            wave=wave_config,
            dictionary=dictionary_config,
            processor=processor_config,
            experiment_name=config_dict.get('experiment_name', 'default'),
            output_dir=config_dict.get('output_dir', 'output')
        )


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def merge_configs(base_config: Config, override_config: Dict[str, Any]) -> Config:
    """Merge a base configuration with override values."""
    base_dict = asdict(base_config)

    # Recursively update the base dictionary with override values
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d

    updated_dict = update_dict(base_dict, override_config)

    # Convert back to Config object
    return Config.from_dict(updated_dict)
