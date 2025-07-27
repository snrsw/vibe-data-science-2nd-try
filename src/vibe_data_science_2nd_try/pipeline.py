from pathlib import Path

import mlflow
import structlog
from mlflow.models import infer_signature

from vibe_data_science_2nd_try.config import PipelineConfig
from vibe_data_science_2nd_try.data import (
    PenguinDataSchema,
    load_penguin_data,
    preprocess_data,
    split_data,
)
from vibe_data_science_2nd_try.models import (
    create_model,
    evaluate_model,
    log_confusion_matrix,
)
from vibe_data_science_2nd_try.utils import mlflow_run

logger = structlog.get_logger()


def run_pipeline(config: PipelineConfig) -> None:
    """
    Run the full training pipeline.

    Args:
        config: Pipeline configuration
    """
    logger.info(
        "Starting pipeline",
        model_type=config.model.model_type,
        data_path=str(config.data.input_path),
    )

    with mlflow_run(
        run_name=f"penguin-classifier-{config.model.model_type}",
        params={
            "model_type": config.model.model_type,
            "input_path": str(config.data.input_path),
            "test_size": config.data.test_size,
            "validation_size": config.data.validation_size,
            "random_state": config.data.random_state,
            **config.model.hyperparameters,
        },
        log_artifacts=config.mlflow.log_artifacts,
    ):
        # Step 1: Load Data
        df = load_penguin_data(config.data.input_path)

        # Step 2: Split Data
        train_df, val_df, test_df = split_data(
            df,
            test_size=config.data.test_size,
            validation_size=config.data.validation_size,
            random_state=config.data.random_state,
        )

        # Step 3: Preprocess Data
        preprocessed_data = preprocess_data(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            imputation_strategy="median",
        )

        # Log data shapes
        mlflow.log_param("train_shape", preprocessed_data.train_features.shape)
        mlflow.log_param("val_shape", preprocessed_data.val_features.shape)
        mlflow.log_param("test_shape", preprocessed_data.test_features.shape)

        # Step 4: Create and Train Model
        model = create_model(config.model.model_type, config.model.hyperparameters)

        # Train the model
        train_metrics = model.train(
            preprocessed_data.train_features,
            preprocessed_data.train_target,
            preprocessed_data.val_features,
            preprocessed_data.val_target,
        )

        # Log training metrics
        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})

        # Step 5: Evaluate Model
        test_metrics = evaluate_model(
            model,
            preprocessed_data.test_features,
            preprocessed_data.test_target,
        )

        # Log test metrics
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Log confusion matrix
        predictions = model.predict(preprocessed_data.test_features)
        log_confusion_matrix(preprocessed_data.test_target, predictions)

        # Step 6: Log Feature Importance
        feature_importance = model.get_feature_importance()
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"importance_{feature}", importance)

        # Step 7: Save Model
        model_path = config.output_dir / f"{config.model.model_type}_model"
        model_path.mkdir(parents=True, exist_ok=True)
        saved_path = model.save(model_path / "model.pkl")

        # Log model to MLflow
        signature = infer_signature(
            preprocessed_data.train_features.to_pandas(),
            preprocessed_data.train_target.to_pandas(),
        )

        if config.model.model_type == "lightgbm":
            mlflow.lightgbm.log_model(
                model.model,
                "lightgbm_model",
                signature=signature,
                registered_model_name="penguin-classifier"
                if config.mlflow.register_model
                else None,
            )

        logger.info(
            "Pipeline completed successfully",
            model_path=str(saved_path),
            test_accuracy=test_metrics.get("accuracy", 0),
        )


def run_prediction(
    model_path: Path,
    data_path: Path,
    output_path: Path,
    config: PipelineConfig,
) -> None:
    """
    Run prediction using a trained model.

    Args:
        model_path: Path to trained model
        data_path: Path to input data
        output_path: Path to output predictions
        config: Pipeline configuration
    """
    logger.info(
        "Starting prediction", model_path=str(model_path), data_path=str(data_path)
    )

    # Load data
    df = load_penguin_data(data_path)

    # Determine model type from path
    if "lightgbm" in str(model_path).lower():
        from vibe_data_science_2nd_try.models.lightgbm_model import LightGBMModel

        model = LightGBMModel.load(model_path)
    else:
        raise ValueError(f"Unknown model type from path: {model_path}")

    # Make predictions
    predictions = model.predict(
        df.select(
            PenguinDataSchema.feature_columns + PenguinDataSchema.categorical_columns
        )
    )

    # Combine original data with predictions
    result = df.with_columns(predictions)

    # Save predictions to CSV
    result.write_csv(output_path)

    logger.info(
        "Prediction completed successfully",
        output_path=str(output_path),
        rows=result.shape[0],
    )
