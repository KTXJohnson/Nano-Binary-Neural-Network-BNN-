"""
Main script for running Nano Binary Network experiments.

This script serves as the entry point for running experiments with
Nano Binary Networks in a CI/CD pipeline.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
from config import Config, merge_configs, get_default_config
from logger import get_logger
from models.nano_wave import NanoWaveModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Nano Binary Network experiments")
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        default="nano_wave", 
        choices=["nano_wave", "fractal_wave", "tensor_wave"],
        help="Type of experiment to run"
    )
    
    parser.add_argument(
        "--text", 
        type=str, 
        help="Path to text file for training"
    )
    
    parser.add_argument(
        "--word-pairs", 
        type=str, 
        help="Path to JSON file containing word pairs to analyze"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    return parser.parse_args()


def load_config(args) -> Config:
    """
    Load configuration from file or create default.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Configuration object.
    """
    # Start with default config
    config = get_default_config()
    
    # Update with config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = merge_configs(config, config_dict)
    
    # Update with command line arguments
    override_config = {}
    
    if args.output:
        override_config["output_dir"] = args.output
    
    if args.log_level:
        override_config["logging"] = {"level": args.log_level}
    
    if args.experiment:
        override_config["experiment_name"] = args.experiment
    
    if override_config:
        config = merge_configs(config, override_config)
    
    return config


def load_text(args) -> str:
    """
    Load text from file or use sample text.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Text for training.
    """
    if args.text and os.path.exists(args.text):
        with open(args.text, 'r', encoding='utf-8') as f:
            return f.read()
    
    # Sample text if no file provided
    return """
    This is a sample text for training Nano Binary Networks.
    It contains various words and phrases that can be used to train the model.
    The model will learn patterns from this text and use them to analyze word pairs.
    """


def load_word_pairs(args):
    """
    Load word pairs from file or use sample pairs.
    
    Args:
        args: Command line arguments.
        
    Returns:
        List of (base, context) word pairs.
    """
    if args.word_pairs and os.path.exists(args.word_pairs):
        with open(args.word_pairs, 'r') as f:
            return json.load(f)
    
    # Sample word pairs if no file provided
    return [
        ("model", "training model"),
        ("word", "sample word"),
        ("text", "training text"),
        ("binary", "nano binary"),
        ("network", "neural network")
    ]


def run_nano_wave_experiment(config: Config, text: str, word_pairs: list) -> Dict[str, Any]:
    """
    Run a Nano Wave experiment.
    
    Args:
        config: Configuration object.
        text: Text for training.
        word_pairs: List of (base, context) word pairs to analyze.
        
    Returns:
        Dictionary of experiment results.
    """
    # Initialize the model
    model = NanoWaveModel(config.wave)
    
    # Run the experiment
    results = model.run_experiment(text, word_pairs)
    
    return results


def run_experiment(args):
    """
    Run the specified experiment.
    
    Args:
        args: Command line arguments.
        
    Returns:
        Dictionary of experiment results.
    """
    # Load configuration
    config = load_config(args)
    
    # Set up logger
    logger = get_logger(config.logging)
    logger.info(f"Starting {args.experiment} experiment")
    
    # Load data
    text = load_text(args)
    word_pairs = load_word_pairs(args)
    
    # Run the appropriate experiment
    if args.experiment == "nano_wave":
        results = run_nano_wave_experiment(config, text, word_pairs)
    else:
        logger.error(f"Experiment type {args.experiment} not implemented")
        results = {"error": f"Experiment type {args.experiment} not implemented"}
    
    # Save results
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results_path = os.path.join(output_dir, f"{config.experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    """Main entry point."""
    try:
        args = parse_args()
        results = run_experiment(args)
        print(f"Experiment completed successfully. Results saved to {os.path.join(args.output if args.output else 'output', f'{args.experiment}_results.json')}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())