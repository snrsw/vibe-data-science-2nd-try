# Vibe Data Science - Penguin Classification

A machine learning pipeline for classifying penguin species based on their measurements.

## Features

- Modular ML training pipeline with replaceable components
- MLflow integration with DuckDB backend for experiment tracking
- Configuration system using Pydantic for type safety
- Polars for efficient data handling
- Command-line interface with Typer
- Structured logging with structlog

## Installation

```bash
# Clone the repository
git clone https://github.com/snrsw/vibe-data-science-2nd-try.git
cd vibe-data-science-2nd-try

# Install dependencies
uv sync
```

## Data

Download the Palmer Penguins dataset:

```bash
# Download penguin dataset
uv run kaggle datasets download amulyas/penguin-size-dataset

# Unzip dataset
uv run unzip penguin-size-dataset.zip -d data/
```

## Usage

### Training a Model

```bash
# Train with default configuration
uv run python -m vibe_data_science_2nd_try.cli train

# Train with custom configuration using environment variables
VIBE_TRAIN_CONFIG_PATH=configs/custom.yaml uv run python -m vibe_data_science_2nd_try.cli train

# Train with specific model type
VIBE_TRAIN_MODEL_TYPE=lightgbm uv run python -m vibe_data_science_2nd_try.cli train
```

### Making Predictions

```bash
# Generate predictions using environment variables
VIBE_PREDICT_MODEL_PATH=output/lightgbm_model/model.pkl \
VIBE_PREDICT_DATA_PATH=data/penguins_size.csv \
VIBE_PREDICT_OUTPUT_PATH=predictions.csv \
uv run python -m vibe_data_science_2nd_try.cli predict
```

## Development

```bash
# Format code
uvx ruff format .

# Run linting with auto-fixes
uvx ruff check . --fix

# Type checking
uvx ty check .

# Run tests
uv run pytest tests
```

## Pipeline Architecture

See `docs/pipeline_architecture.md` for details on the ML pipeline architecture.