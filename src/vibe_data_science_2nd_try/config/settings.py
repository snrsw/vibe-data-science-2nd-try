from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    output_path: Path | None = None
    include_timestamp: bool = True


class DataConfig(BaseModel):
    input_path: Path = Path("data/penguins_size.csv")
    test_size: Annotated[float, Field(ge=0.0, le=0.5)] = 0.2
    validation_size: Annotated[float, Field(ge=0.0, le=0.5)] = 0.2
    random_state: int = 42


class ModelConfig(BaseModel):
    model_type: Literal["lightgbm", "xgboost", "catboost"] = "lightgbm"
    hyperparameters: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        model_type = self.model_type

        # Apply default hyperparameters if none provided
        if not self.hyperparameters:
            if model_type == "lightgbm":
                self.hyperparameters = {
                    "objective": "multiclass",
                    "num_class": 3,
                    "metric": "multi_logloss",
                    "boosting_type": "gbdt",
                    "num_leaves": 31,
                    "learning_rate": 0.05,
                    "feature_fraction": 0.9,
                }
            elif model_type == "xgboost":
                self.hyperparameters = {
                    "objective": "multi:softmax",
                    "num_class": 3,
                    "eval_metric": "mlogloss",
                    "eta": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }
            elif model_type == "catboost":
                self.hyperparameters = {
                    "loss_function": "MultiClass",
                    "eval_metric": "MultiClass",
                    "iterations": 1000,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "random_seed": 42,
                }


class MLFlowConfig(BaseModel):
    tracking_uri: str = "duckdb+artifact:///mlflow-artifacts/mlruns.duckdb"
    experiment_name: str = "penguin-classification"
    register_model: bool = True
    log_artifacts: bool = True


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIBE_",
        env_nested_delimiter="__",
        validate_default=True,
    )

    log: LogConfig = LogConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    output_dir: Path = Path("output")
