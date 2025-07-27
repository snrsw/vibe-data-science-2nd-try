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

        if "objective" not in self.model_params:
            self.model_params["objective"] = "multiclass"
        if (
            "num_class" not in self.model_params
            and self.model_params["objective"] == "multiclass"
        ):
            self.model_params["num_class"] = 3

    def train(
        self,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: Optional[pl.DataFrame] = None,
        y_val: Optional[pl.Series] = None,
    ) -> Dict[str, float]:
        logger.info("Training LightGBM model", params=self.model_params)

        self.feature_names: list[str] = X_train.columns

        # Convert categorical targets to numeric indices
        unique_classes: list = sorted(y_train.unique().to_list())
        self.classes: list = unique_classes
        
        # Create a mapping of class names to numeric indices
        class_to_idx = {class_name: idx for idx, class_name in enumerate(unique_classes)}
        
        # Convert target values to numeric indices with explicit return_dtype
        y_train_numeric = y_train.map_elements(lambda x: class_to_idx.get(x, 0), return_dtype=pl.Int32)
        y_val_numeric = None if y_val is None else y_val.map_elements(lambda x: class_to_idx.get(x, 0), return_dtype=pl.Int32)

        dtrain: lgb.Dataset = lgb.Dataset(
            X_train.to_numpy(),
            label=y_train_numeric.to_numpy(),
            feature_name=self.feature_names,
        )

        eval_results: Dict[str, Dict[str, list[float]]] = {}
        eval_datasets = [dtrain]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None and y_val_numeric is not None:
            dval = lgb.Dataset(
                X_val.to_numpy(),
                label=y_val_numeric.to_numpy(),
                feature_name=self.feature_names,
            )
            eval_datasets.append(dval)
            valid_names.append("valid")

        self.model = lgb.train(
            params=self.model_params,
            train_set=dtrain,
            valid_sets=eval_datasets,
            valid_names=valid_names,
            callbacks=[lgb.record_evaluation(eval_results)],
        )

        metrics: Dict[str, float] = {}
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

        y_pred_proba: np.ndarray = self.model.predict(X.to_numpy())
        y_pred: np.ndarray = np.argmax(y_pred_proba, axis=1)

        predictions: pl.Series = pl.Series(
            name="predicted_species", values=[self.classes[i] for i in y_pred]
        )

        return predictions

    def save(self, path: Union[str, Path]) -> Path:
        if self.model is None:
            raise ValueError("Model not trained yet")

        save_path: Path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "classes": self.classes,
            },
            save_path,
        )

        logger.info("LightGBM model saved", path=str(save_path))

        return save_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LightGBMModel":
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        saved_data: Dict[str, Any] = joblib.load(load_path)

        model_instance: LightGBMModel = cls({})
        model_instance.model = saved_data["model"]
        model_instance.feature_names = saved_data["feature_names"]
        model_instance.classes = saved_data["classes"]

        logger.info("LightGBM model loaded", path=str(load_path))

        return model_instance

    def get_feature_importance(self) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance: np.ndarray = self.model.feature_importance(importance_type="gain")

        result: Dict[str, float] = {}
        for idx, name in enumerate(self.feature_names):
            if idx < len(importance):
                result[name] = float(importance[idx])

        return result
