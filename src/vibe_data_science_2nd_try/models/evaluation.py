from typing import Dict

import mlflow
import numpy as np
import polars as pl
import structlog
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from vibe_data_science_2nd_try.models.base import BaseModel

logger = structlog.get_logger()


def evaluate_model(
    model: BaseModel,
    X_test: pl.DataFrame,
    y_test: pl.Series,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test target values

    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = calculate_classification_metrics(y_test, y_pred)

    # Log metrics
    logger.info("Model evaluation complete", **metrics)

    return metrics


def calculate_classification_metrics(
    y_true: pl.Series,
    y_pred: pl.Series,
) -> Dict[str, float]:
    """
    Calculate classification metrics for the predictions.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy arrays
    y_true_np = y_true.to_numpy()
    y_pred_np = y_pred.to_numpy()

    # Get unique classes
    classes = sorted(set(y_true_np) | set(y_pred_np))

    # Calculate metrics
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true_np, y_pred_np))

    # Precision, Recall, F1 (macro)
    metrics["precision_macro"] = float(
        precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    )
    metrics["recall_macro"] = float(
        recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    )
    metrics["f1_macro"] = float(
        f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    )

    # Per-class metrics
    for i, cls in enumerate(classes):
        y_true_binary = np.array([1 if y == cls else 0 for y in y_true_np])
        y_pred_binary = np.array([1 if y == cls else 0 for y in y_pred_np])

        metrics[f"precision_{cls}"] = float(
            precision_score(y_true_binary, y_pred_binary, zero_division=0)
        )
        metrics[f"recall_{cls}"] = float(
            recall_score(y_true_binary, y_pred_binary, zero_division=0)
        )
        metrics[f"f1_{cls}"] = float(
            f1_score(y_true_binary, y_pred_binary, zero_division=0)
        )

    return metrics


def log_confusion_matrix(
    y_true: pl.Series,
    y_pred: pl.Series,
) -> None:
    """
    Calculate and log confusion matrix to MLflow.

    Args:
        y_true: True target values
        y_pred: Predicted target values
    """
    # Convert to numpy arrays
    y_true_np = y_true.to_numpy()
    y_pred_np = y_pred.to_numpy()

    # Get unique classes
    classes = sorted(set(y_true_np) | set(y_pred_np))

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_np, y_pred_np, labels=classes)

    # Log as a figure using Plotly
    import plotly.graph_objects as go
    import plotly.io as pio

    # Create heatmap figure
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=classes,
            y=classes,
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
        )
    )

    # Update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted"),
        yaxis=dict(title="True"),
        width=800,
        height=600,
    )

    # Save figure to file
    pio.write_image(fig, "confusion_matrix.png")

    # Log figure to MLflow
    mlflow.log_artifact("confusion_matrix.png")

    logger.info("Confusion matrix logged to MLflow")
