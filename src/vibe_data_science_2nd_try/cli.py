from pathlib import Path
from typing import Optional

import structlog
from pydantic_settings import BaseSettings, SettingsConfigDict

from vibe_data_science_2nd_try.config import PipelineConfig, load_config
from vibe_data_science_2nd_try.pipeline import run_pipeline, run_prediction
from vibe_data_science_2nd_try.utils import configure_logging, setup_mlflow

logger = structlog.get_logger()


class TrainSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIBE_TRAIN_",
        env_nested_delimiter="__",
    )
    
    config_path: Optional[Path] = None
    data_path: Optional[Path] = None
    model_type: Optional[str] = None
    output_dir: Optional[Path] = None


class PredictSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIBE_PREDICT_",
        env_nested_delimiter="__",
    )
    
    model_path: Path
    data_path: Path
    output_path: Path = Path("predictions.csv")
    config_path: Optional[Path] = None


def initialize_environment(
    config: PipelineConfig,
) -> None:
    # Set up logging
    configure_logging(
        level=config.log.level,
        output_path=config.log.output_path,
        include_timestamp=config.log.include_timestamp,
    )
    
    # Set up MLflow
    setup_mlflow(config.mlflow)
    
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)


def train(settings: Optional[TrainSettings] = None) -> None:
    """
    Train a penguin species classification model.
    
    Args:
        settings: Training settings
    """
    if settings is None:
        settings = TrainSettings()
    
    # Load configuration
    try:
        config = load_config(settings.config_path)
        
        # Override config values if specified
        if settings.data_path:
            config.data.input_path = settings.data_path
        if settings.model_type:
            if settings.model_type in ["lightgbm", "xgboost", "catboost"]:
                config.model.model_type = settings.model_type
            else:
                logger.warning(
                    "Invalid model_type, using default",
                    invalid_type=settings.model_type,
                    default=config.model.model_type
                )
        if settings.output_dir:
            config.output_dir = settings.output_dir
        
        # Initialize environment
        initialize_environment(config)
        
        # Run pipeline
        run_pipeline(config)
    
    except Exception as e:
        logger.exception("Error in training command", error=str(e))
        raise SystemExit(1)


def predict(settings: Optional[PredictSettings] = None) -> None:
    """
    Generate predictions using a trained model.
    
    Args:
        settings: Prediction settings
    """
    if settings is None:
        try:
            settings = PredictSettings()
        except Exception as e:
            logger.exception("Error loading prediction settings", error=str(e))
            raise SystemExit(1)
    
    # Load configuration
    try:
        config = load_config(settings.config_path)
        
        # Initialize environment
        initialize_environment(config)
        
        # Override data path
        config.data.input_path = settings.data_path
        
        # Run prediction
        run_prediction(settings.model_path, settings.data_path, settings.output_path, config)
    
    except Exception as e:
        logger.exception("Error in prediction command", error=str(e))
        raise SystemExit(1)


def main() -> None:
    """
    Main entry point for the CLI.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m vibe_data_science_2nd_try.cli [train|predict] [options]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        train()
    elif command == "predict":
        predict()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, predict")
        sys.exit(1)


if __name__ == "__main__":
    main()