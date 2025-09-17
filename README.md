# Nano Binary Models

This repository contains a collection of models and utilities for working with nano-scale binary neural networks, optimized for CI/CD pipeline training.

## Overview

The Nano Binary Models package provides a modular and standardized approach to working with wave-based neural networks, binary domain networks, and tensor-based models. The codebase has been refactored to improve maintainability, testability, and performance, with a focus on CI/CD pipeline integration.

## Key Features

- **Modular Architecture**: Clear separation of concerns with models, utilities, and configuration
- **Standardized Interfaces**: Consistent interfaces across different model types
- **Comprehensive Logging**: Detailed logging and benchmarking for performance analysis
- **Configuration System**: Flexible configuration system for model parameters
- **CI/CD Integration**: Command-line interface for easy integration with CI/CD pipelines
- **Testing Framework**: Unit tests for key components.

## Directory Structure

```
Nano_Binary_Models/
├── models/                  # Model implementations
│   ├── __init__.py
│   ├── base_wave.py         # Base class for wave models
│   └── nano_wave.py         # Nano Wave model implementation
├── utils/                   # Utility functions and classes
│   ├── __init__.py
│   └── dictionary_loader.py # Dictionary loading utilities
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_config.py       # Tests for configuration system
│   └── test_logger.py       # Tests for logging system
├── __init__.py              # Package initialization
├── config.py                # Configuration system
├── logger.py                # Logging system
├── main.py                  # Main entry point for experiments
└── requirements.txt         # Package dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone <https://github.com/KTXJohnson/Nano-Binary-Neural-Network-BNN-.git>
   cd Nano_Binary_Models
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Experiments

You can run experiments using the `main.py` script:

```
python main.py --experiment nano_wave --text path/to/text.txt --word-pairs path/to/word_pairs.json --output path/to/output
```

### Command-line Arguments

- `--config`: Path to configuration file (JSON)
- `--experiment`: Type of experiment to run (`nano_wave`, `fractal_wave`, `tensor_wave`)
- `--text`: Path to text file for training
- `--word-pairs`: Path to JSON file containing word pairs to analyze
- `--output`: Path to output directory
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Configuration

You can configure the models using a JSON configuration file:

```json
{
  "experiment_name": "custom_experiment",
  "output_dir": "custom_output",
  "logging": {
    "log_dir": "custom_logs",
    "level": "DEBUG"
  },
  "wave": {
    "frequency_range": [0.1, 300.0],
    "amplitude_range": [1.0, 800000.0],
    "viscosity": 0.7
  }
}
```

Pass the configuration file using the `--config` argument:

```
python main.py --config path/to/config.json
```

## CI/CD Pipeline Integration

### Example GitHub Actions Workflow

```yaml
name: Train and Evaluate Models

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r Nano_Binary_Models/requirements.txt
    - name: Run tests
      run: |
        cd Nano_Binary_Models
        python -m unittest discover tests
    - name: Train model
      run: |
        cd Nano_Binary_Models
        python main.py --experiment nano_wave --output results
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: model-results
        path: Nano_Binary_Models/results
```

### Example Jenkins Pipeline

```groovy
pipeline {
    agent {
        docker {
            image 'python:3.9'
        }
    }
    stages {
        stage('Setup') {
            steps {
                sh 'pip install -r Nano_Binary_Models/requirements.txt'
            }
        }
        stage('Test') {
            steps {
                sh 'cd Nano_Binary_Models && python -m unittest discover tests'
            }
        }
        stage('Train') {
            steps {
                sh 'cd Nano_Binary_Models && python main.py --experiment nano_wave --output results'
            }
        }
        stage('Archive') {
            steps {
                archiveArtifacts artifacts: 'Nano_Binary_Models/results/**/*', fingerprint: true
            }
        }
    }
}
```

## Extending the Framework

### Adding a New Model

1. Create a new model file in the `models` directory
2. Extend the appropriate base class (e.g., `BaseWaveModel`)
3. Implement the required methods
4. Add the model to the experiment runner in `main.py`

### Adding a New Utility

1. Create a new utility file in the `utils` directory
2. Implement the utility functions or classes
3. Import and use the utility in your models

### Adding Tests

1. Create a new test file in the `tests` directory
2. Implement tests using the `unittest` framework
3. Run the tests using `python -m unittest discover tests`

## Performance Optimization

The codebase has been optimized for performance in several ways:

1. **Vectorized Operations**: Using PyTorch tensors for efficient computation
2. **Caching**: Dictionary loading with caching for faster access
3. **Benchmarking**: Performance tracking for identifying bottlenecks
4. **Parallel Processing**: Support for GPU acceleration where available

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.