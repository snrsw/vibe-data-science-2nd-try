from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import structlog

from .base import BaseModel

logger = structlog.get_logger()


class LightGBMModel(BaseModel):
    def __init__(self, model_params: Dict[str, Any]):
        super().__init__(model_params)
        
        # Ensure required params for classification
        if "objective" not in self.model_params:
            self.model_params["objective"] = "multiclass"
        if "num_class" not in self.model_params and self.model_params["objective"] == "multiclass":
            self.model_params["num_class"] = 3
    
    def train(
        self, 
        X_train: pl.DataFrame, 
        y_train: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        logger.info("Training LightGBM model", params=self.model_params)
        
        # Store feature names
        self.feature_names = X_train.columns
        
        # Get unique classes from target
        self.classes = sorted(y_train.unique().to_list())
        
        # Create training dataset
        dtrain = lgb.Dataset(
            X_train.to_numpy(),
            label=y_train.to_numpy(),
            feature_name=self.feature_names,
        )
        
        # Create validation dataset if provided
        eval_results: Dict[str, Dict[str, list[float]]] = {}
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(
                X_val.to_numpy(),
                y_val.to_numpy(),
            )]
        
        # Train the model
        self.model = lgb.train(
            params=self.model_params,
            train_set=dtrain,
            valid_sets=[dtrain] + (eval_set or []),
            valid_names=["train"] + (["valid"] if eval_set else []),
            evals_result=eval_results,
            verbose_eval=False,
        )
        
        # Get final metrics
        metrics = {}
        if eval_results and "valid" in eval_results:
            for metric_name, values in eval_results["valid"].items():
                metrics[f"val_{metric_name}"] = values[-1]
        elif eval_results and "train" in eval_results:
            for metric_name, values in eval_results["train"].items():
                metrics[f"train_{metric_name}"] = values[-1]
        
        logger.info("LightGBM training completed", metrics=metrics)
        
        return metrics
    
    def predict(self, X: pl.DataFrame) -> pl.Series:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Generating predictions with LightGBM model", rows=X.shape[0])
        
        # Make predictions
        y_pred_proba = self.model.predict(X.to_numpy())
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert numeric predictions to class labels
        predictions = pl.Series(
            name="predicted_species",
            values=[self.classes[i] for i in y_pred]
        )
        
        return predictions
    
    def save(self, path: Union[str, Path]) -> Path:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model file
        model_path = save_path / "model.pkl"
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "classes": self.classes,
            },
            model_path,
        )
        
        logger.info("LightGBM model saved", path=str(model_path))
        
        return model_path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LightGBMModel':
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        # Load the model
        saved_data = joblib.load(load_path)
        
        # Create a new model instance
        model_instance = cls({})
        model_instance.model = saved_data["model"]
        model_instance.feature_names = saved_data["feature_names"]
        model_instance.classes = saved_data["classes"]
        
        logger.info("LightGBM model loaded", path=str(load_path))
        
        return model_instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get feature importance from model
        importance = self.model.feature_importance(importance_type="gain")
        
        # Map feature names to importance
        result = {}
        for idx, name in enumerate(self.feature_names):
            if idx < len(importance):
                result[name] = float(importance[idx])
        
        return result