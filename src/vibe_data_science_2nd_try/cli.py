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
    configure_logging(
        level=config.log.level,
        output_path=config.log.output_path,
        include_timestamp=config.log.include_timestamp,
    )

    setup_mlflow(config.mlflow)

    config.output_dir.mkdir(parents=True, exist_ok=True)


def train(settings: Optional[TrainSettings] = None) -> None:
    if settings is None:
        settings = TrainSettings()

    try:
        config = load_config(settings.config_path)

        if settings.data_path:
            config.data.input_path = settings.data_path
        if settings.model_type:
            if settings.model_type in ["lightgbm", "xgboost", "catboost"]:
                # Use the validated value directly
                if settings.model_type == "lightgbm":
                    config.model.model_type = "lightgbm"
                elif settings.model_type == "xgboost":
                    config.model.model_type = "xgboost"
                elif settings.model_type == "catboost":
                    config.model.model_type = "catboost"
            else:
                logger.warning(
                    "Invalid model_type, using default",
                    invalid_type=settings.model_type,
                    default=config.model.model_type,
                )
        if settings.output_dir:
            config.output_dir = settings.output_dir

        initialize_environment(config)

        run_pipeline(config)

    except Exception as e:
        logger.exception("Error in training command", error=str(e))
        raise SystemExit(1)


def predict(settings: Optional[PredictSettings] = None) -> None:
    if settings is None:
        try:
            settings = PredictSettings(
                model_path=Path("./output/lightgbm_model/model.pkl"),
                data_path=Path("./data/penguins_size.csv"),
            )
        except Exception as e:
            logger.exception("Error loading prediction settings", error=str(e))
            raise SystemExit(1)

    try:
        config = load_config(settings.config_path)

        initialize_environment(config)

        config.data.input_path = settings.data_path

        run_prediction(
            settings.model_path, settings.data_path, settings.output_path, config
        )

    except Exception as e:
        logger.exception("Error in prediction command", error=str(e))
        raise SystemExit(1)


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m vibe_data_science_2nd_try.cli [train|predict] [options]"
        )
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
