# ML Training Pipeline Architecture

## Overview

This document outlines the architecture for a modular machine learning training pipeline designed to classify penguin species based on the Palmer Penguins dataset. The pipeline follows functional programming principles, uses immutable data structures, and integrates with MLflow for experiment tracking.

## Problem Statement

The pipeline addresses the following problem:

- **Input**: Penguin measurement data from `data/penguins_size.csv`, including features such as culmen dimensions, flipper length, body mass, and demographic information.
- **Output**: Predictions of penguin species (Adelie, Chinstrap, Gentoo) along with the original dataset columns.
- **Evaluation**: Performance metrics including accuracy, recall, precision, and confusion matrix.

## Pipeline Components

The pipeline consists of the following modular components:

1. **Data Loading**
   - Responsible for loading data from CSV files using Polars
   - Returns immutable Polars DataFrames
   - Includes schema validation with Pydantic

2. **Data Preprocessing**
   - Feature engineering (e.g., normalization, encoding)
   - Missing value imputation
   - Dataset splitting (training, validation, test)
   - Returns processed DataFrames without modifying the original data

3. **Model Training**
   - Trains different ML models on the preprocessed data
   - Implements a common interface for various model algorithms
   - Supports hyperparameter tuning
   - Captures model artifacts and metrics using MLflow

4. **Model Evaluation**
   - Calculates evaluation metrics (accuracy, recall, precision)
   - Generates confusion matrices
   - Logs results to MLflow and structured logs

5. **Prediction Generation**
   - Applies trained model to test data
   - Outputs predictions with original data columns

## MLflow Integration with DuckDB

The pipeline leverages MLflow for experiment tracking with DuckDB as the backend:

1. **Experiment Management**
   - MLflow experiments are organized by model type and configuration
   - Experiment metadata stored in DuckDB tables
   - Enables easy comparison of different models and parameters

2. **Run Tracking**
   - Each run records parameters, metrics, and artifacts
   - DuckDB provides efficient querying of experiment results
   - Supports SQL-based analytics on experiment results

3. **Model Registry**
   - Registers best-performing models for deployment
   - Tracks model versions and associated metadata
   - Provides model lineage and provenance

## Configuration System

The pipeline uses a hierarchical configuration system:

1. **Core Configuration**
   - `PipelineConfig` as the base Pydantic model
   - Component-specific configurations inherit from base models
   - Default values provided for all parameters

2. **Configuration Sources**
   - YAML/TOML files for persistent configurations
   - Environment variables for runtime overrides
   - Command-line arguments for one-time customizations

3. **Configuration Validation**
   - Pydantic models validate configuration values
   - Type checking and constraint enforcement
   - Helpful error messages for misconfiguration

## CLI Implementation

The command-line interface provides the following functionality:

1. **Main Commands**
   - `train`: Run the full training pipeline
   - `preprocess`: Run only the preprocessing step
   - `evaluate`: Evaluate an existing model
   - `predict`: Generate predictions from a trained model

2. **Options and Arguments**
   - Configuration file path
   - Model selection
   - Output directory
   - Experiment tracking options

3. **Implementation**
   - Built using Typer for CLI argument parsing
   - Integrates with the configuration system
   - Provides helpful error messages and usage information

## Logging System

The pipeline implements a structured logging approach:

1. **Structured Logging**
   - Uses `structlog` for consistent log formats
   - Each log entry includes component, timestamp, and context
   - Supports various log levels (DEBUG, INFO, WARNING, ERROR)

2. **Log Destinations**
   - Console output for interactive use
   - File output for persistent logs
   - Integration with MLflow for experiment-specific logs

3. **Log Context**
   - Enriches logs with experiment and run IDs
   - Includes component-specific context
   - Facilitates log filtering and analysis

## Data Flow

The pipeline follows a functional data flow pattern:

```
Load Data → Preprocess → Train Model → Evaluate → Generate Predictions
     ↓           ↓           ↓            ↓              ↓
    Logs       Logs         Logs         Logs           Logs
     ↓           ↓           ↓            ↓              ↓
    MLflow ← Configuration → MLflow      MLflow        MLflow
```

## Directory Structure

```
vibe-data-science/
├── src/
│   └── vibe_data_science_2nd_try/
│       ├── config/              # Configuration models and loaders
│       ├── data/                # Data loading and preprocessing
│       ├── models/              # Model training and evaluation
│       ├── utils/               # Utility functions and logging
│       ├── cli.py               # CLI implementation
│       └── pipeline.py          # Pipeline orchestration
├── data/
│   └── penguins_size.csv        # Input dataset
├── configs/
│   └── default.yaml             # Default configuration
├── tests/                       # Test suite
├── mlruns/                      # MLflow artifacts
└── docs/
    ├── problem.md               # Problem definition
    └── pipeline_architecture.md # This document
```

## Implementation Approach

The pipeline follows these implementation principles:

1. **Functional Programming**
   - Pure functions with minimal side effects
   - Immutable data structures using Polars
   - Function composition for pipeline stages

2. **Type Safety**
   - Extensive type hints for function signatures
   - Pydantic models for configuration and data validation
   - Regular type checking with `ty`

3. **Testing Strategy**
   - Unit tests for individual components
   - Integration tests for the full pipeline
   - Fixtures for common test data and configurations

4. **Extensibility**
   - New models can be added by implementing standard interfaces
   - Preprocessing steps can be composed or replaced
   - Configuration-driven pipeline customization

5. **t-wada's TDD Style**
   - Test-driven development with a focus on small, incremental changes
   - Each feature implemented with corresponding tests
   - Refactoring guided by test coverage and design principles

## Candidate Models

The pipeline supports various machine learning models, including:

- LightGBM
- XGBoost
- CatBoost

## Next Steps

1. Implement the core pipeline components
2. Create configuration models using Pydantic
3. Set up MLflow with DuckDB backend
4. Develop CLI interface with Typer
5. Implement structured logging with contextual information
