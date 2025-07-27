from pathlib import Path

import pytest

from vibe_data_science_2nd_try.config import PipelineConfig, ModelConfig, DataConfig


def test_pipeline_config_default_values():
    config = PipelineConfig()

    assert config.log.level == "INFO"
    assert config.data.input_path == Path("data/penguins_size.csv")
    assert config.model.model_type == "lightgbm"
    assert config.mlflow.experiment_name == "penguin-classification"
    assert config.output_dir == Path("output")


def test_model_config_hyperparameters_validation():
    lightgbm_config = ModelConfig(model_type="lightgbm")
    assert "objective" in lightgbm_config.hyperparameters
    assert lightgbm_config.hyperparameters["num_class"] == 3

    xgboost_config = ModelConfig(model_type="xgboost")
    assert "objective" in xgboost_config.hyperparameters
    assert xgboost_config.hyperparameters["num_class"] == 3

    catboost_config = ModelConfig(model_type="catboost")
    assert "loss_function" in catboost_config.hyperparameters
    assert catboost_config.hyperparameters["loss_function"] == "MultiClass"


def test_model_config_custom_hyperparameters():
    custom_params = {
        "objective": "multiclass",
        "num_class": 5,
        "custom_param": "value",
    }

    config = ModelConfig(model_type="lightgbm", hyperparameters=custom_params)
    assert config.hyperparameters["num_class"] == 5
    assert config.hyperparameters["custom_param"] == "value"


def test_data_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(test_size=-0.1))

    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(test_size=0.6))

    with pytest.raises(ValueError):
        PipelineConfig(data=DataConfig(validation_size=0.6))
