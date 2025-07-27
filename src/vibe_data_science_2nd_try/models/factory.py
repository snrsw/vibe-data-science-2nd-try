from typing import Any, Dict

import structlog

from .base import BaseModel
from .lightgbm_model import LightGBMModel

logger = structlog.get_logger()


def create_model(model_type: str, model_params: Dict[str, Any]) -> BaseModel:
    """
    Create a model instance of the specified type.

    Args:
        model_type: Type of model to create
        model_params: Parameters for the model

    Returns:
        A model instance of the specified type

    Raises:
        ValueError: If the specified model type is not supported
    """
    if model_type.lower() == "lightgbm":
        logger.info("Creating LightGBM model")
        return LightGBMModel(model_params)
    else:
        supported_models = ["lightgbm"]
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Supported models: {', '.join(supported_models)}"
        )
