from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl
import structlog
from pydantic import BaseModel


logger = structlog.get_logger()


class ModelResult(BaseModel):
    model_type: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    model_path: Path
    feature_importance: Dict[str, float] = {}


class BaseModel(ABC):
    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.model: Any = None
        self.feature_names: list[str] = []
        self.classes: list[str] = []
    
    @abstractmethod
    def train(
        self, 
        X_train: pl.DataFrame, 
        y_train: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of metrics from training
        """
        pass
    
    @abstractmethod
    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        Generate predictions for the provided features.
        
        Args:
            X: Features to predict on
            
        Returns:
            Series of predictions
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> Path:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
            
        Returns:
            Path where the model was saved
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Load a model from the specified path.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass