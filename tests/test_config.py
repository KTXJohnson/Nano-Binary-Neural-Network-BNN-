"""
Tests for the configuration system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config, LoggingConfig, WaveConfig, DictionaryConfig, ProcessorConfig, merge_configs, get_default_config


class TestConfig(unittest.TestCase):
    """Tests for the configuration system."""
    
    def test_default_config(self):
        """Test that the default configuration is created correctly."""
        config = get_default_config()
        
        # Check that all sub-configs are created
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertIsInstance(config.wave, WaveConfig)
        self.assertIsInstance(config.dictionary, DictionaryConfig)
        self.assertIsInstance(config.processor, ProcessorConfig)
        
        # Check default values
        self.assertEqual(config.experiment_name, "default")
        self.assertEqual(config.output_dir, "output")
        self.assertEqual(config.logging.log_dir, "logs")
        self.assertEqual(config.wave.viscosity, 0.5)
        self.assertEqual(config.dictionary.limit, 50000)
        self.assertEqual(config.processor.num_layers, 6)
    
    def test_save_load_config(self):
        """Test saving and loading a configuration."""
        config = get_default_config()
        
        # Modify some values
        config.experiment_name = "test_experiment"
        config.logging.log_dir = "test_logs"
        config.wave.viscosity = 0.7
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            filepath = temp_file.name
        
        try:
            saved_path = config.save(filepath)
            self.assertEqual(saved_path, filepath)
            
            # Load the configuration
            loaded_config = Config.load(filepath)
            
            # Check that values are preserved
            self.assertEqual(loaded_config.experiment_name, "test_experiment")
            self.assertEqual(loaded_config.logging.log_dir, "test_logs")
            self.assertEqual(loaded_config.wave.viscosity, 0.7)
            
            # Check that default values are preserved
            self.assertEqual(loaded_config.output_dir, "output")
            self.assertEqual(loaded_config.dictionary.limit, 50000)
            self.assertEqual(loaded_config.processor.num_layers, 6)
        finally:
            # Clean up
            os.unlink(filepath)
    
    def test_merge_configs(self):
        """Test merging configurations."""
        base_config = get_default_config()
        
        # Create override values
        override_config = {
            "experiment_name": "merged_experiment",
            "logging": {
                "log_dir": "merged_logs",
                "level": "DEBUG"
            },
            "wave": {
                "viscosity": 0.8
            }
        }
        
        # Merge configurations
        merged_config = merge_configs(base_config, override_config)
        
        # Check that override values are applied
        self.assertEqual(merged_config.experiment_name, "merged_experiment")
        self.assertEqual(merged_config.logging.log_dir, "merged_logs")
        self.assertEqual(merged_config.logging.level, "DEBUG")
        self.assertEqual(merged_config.wave.viscosity, 0.8)
        
        # Check that non-overridden values are preserved
        self.assertEqual(merged_config.output_dir, "output")
        self.assertEqual(merged_config.dictionary.limit, 50000)
        self.assertEqual(merged_config.processor.num_layers, 6)
        self.assertEqual(merged_config.logging.format, "%(asctime)s - %(levelname)s - %(message)s")


if __name__ == "__main__":
    unittest.main()