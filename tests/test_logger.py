"""
Tests for the logging system.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from config import LoggingConfig
from logger import NBNLogger, get_logger


class TestLogger(unittest.TestCase):
    """Tests for the logging system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for logs
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create a logging configuration that uses the temporary directory
        self.config = LoggingConfig(
            log_dir=self.temp_dir.name,
            console_output=False,  # Disable console output for tests
            file_output=True
        )

        # Create a logger with the test configuration
        self.logger = NBNLogger(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        # Close the logger to release file handlers
        if hasattr(self, 'logger') and self.logger:
            self.logger.close()

        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_logger_initialization(self):
        """Test that the logger is initialized correctly."""
        # Check that the logger has the correct configuration
        self.assertEqual(self.logger.config.log_dir, self.temp_dir.name)
        self.assertFalse(self.logger.config.console_output)
        self.assertTrue(self.logger.config.file_output)

        # Check that the benchmarks dictionary is initialized
        self.assertEqual(self.logger.benchmarks, {})

        # Check that the logger is created
        self.assertIsNotNone(self.logger.logger)

    def test_logging_methods(self):
        """Test the logging methods."""
        # Save the original logger and its handlers
        original_logger = self.logger.logger
        original_handlers = list(original_logger.handlers)

        try:
            # Mock the underlying logger to check that methods are called
            self.logger.logger = MagicMock()

            # Test each logging method
            self.logger.info("Info message")
            self.logger.logger.info.assert_called_once_with("Info message")

            self.logger.warning("Warning message")
            self.logger.logger.warning.assert_called_once_with("Warning message")

            self.logger.error("Error message")
            self.logger.logger.error.assert_called_once_with("Error message")

            self.logger.debug("Debug message")
            self.logger.logger.debug.assert_called_once_with("Debug message")
        finally:
            # Close the original handlers before restoring
            for handler in original_handlers:
                handler.close()

            # Restore the original logger
            self.logger.logger = original_logger

    def test_log_benchmark(self):
        """Test logging benchmarks."""
        # Create a benchmark with various data types
        benchmark_data = {
            "scalar": 1.23,
            "string": "test",
            "list": [1, 2, 3],
            "tensor_scalar": torch.tensor(4.56),
            "tensor_vector": torch.tensor([7.89, 0.12]),
            "nested": {
                "scalar": 3.45,
                "tensor": torch.tensor(6.78)
            }
        }

        # Log the benchmark
        self.logger.log_benchmark("test_category", benchmark_data)

        # Check that the benchmark is added to the benchmarks dictionary
        self.assertIn("test_category", self.logger.benchmarks)
        self.assertEqual(len(self.logger.benchmarks["test_category"]), 1)

        # Check that tensors are converted to Python types
        logged_data = self.logger.benchmarks["test_category"][0]
        self.assertIsInstance(logged_data["scalar"], float)
        self.assertIsInstance(logged_data["string"], str)
        self.assertIsInstance(logged_data["list"], list)
        self.assertIsInstance(logged_data["tensor_scalar"], float)
        self.assertIsInstance(logged_data["tensor_vector"], list)
        self.assertIsInstance(logged_data["nested"], dict)
        self.assertIsInstance(logged_data["nested"]["scalar"], float)
        self.assertIsInstance(logged_data["nested"]["tensor"], float)

        # Check that values are preserved
        self.assertEqual(logged_data["scalar"], 1.23)
        self.assertEqual(logged_data["string"], "test")
        self.assertEqual(logged_data["list"], [1, 2, 3])
        self.assertEqual(logged_data["tensor_scalar"], 4.56)
        self.assertEqual(logged_data["tensor_vector"], [7.89, 0.12])
        self.assertEqual(logged_data["nested"]["scalar"], 3.45)
        self.assertEqual(logged_data["nested"]["tensor"], 6.78)

        # Check that timestamp is added
        self.assertIn("timestamp", logged_data)

    def test_save_benchmarks(self):
        """Test saving benchmarks to a file."""
        # Add some benchmarks
        self.logger.log_benchmark("category1", {"value": 1.23})
        self.logger.log_benchmark("category2", {"value": 4.56})

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            filepath = temp_file.name

        try:
            saved_path = self.logger.save_benchmarks(filepath)
            self.assertEqual(saved_path, filepath)

            # Check that the file exists
            self.assertTrue(os.path.exists(filepath))

            # Load the file and check its contents
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check that categories are preserved
            self.assertIn("category1", data)
            self.assertIn("category2", data)

            # Check that values are preserved
            self.assertEqual(len(data["category1"]), 1)
            self.assertEqual(len(data["category2"]), 1)
            self.assertEqual(data["category1"][0]["value"], 1.23)
            self.assertEqual(data["category2"][0]["value"], 4.56)

            # Check that summary is added
            self.assertIn("summary", data)
            self.assertIn("total_runtime", data["summary"])
            self.assertIn("timestamp", data["summary"])
            self.assertIn("categories", data["summary"])
            self.assertIn("category_counts", data["summary"])
        finally:
            # Clean up
            os.unlink(filepath)

    def test_get_logger_singleton(self):
        """Test that get_logger returns a singleton instance."""
        # Get a logger with default configuration
        logger1 = get_logger()

        # Get another logger with default configuration
        logger2 = get_logger()

        # Check that they are the same instance
        self.assertIs(logger1, logger2)

        # Get a logger with a specific configuration
        config = LoggingConfig(log_dir="custom_logs")
        logger3 = get_logger(config)

        # Check that it's a different instance
        self.assertIsNot(logger1, logger3)

        # Check that it has the correct configuration
        self.assertEqual(logger3.config.log_dir, "custom_logs")


if __name__ == "__main__":
    unittest.main()
