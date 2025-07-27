from dataclasses import dataclass

import polars as pl
import structlog

from .schema import PenguinDataSchema

logger = structlog.get_logger()


@dataclass
class PreprocessingResult:
    train_features: pl.DataFrame
    train_target: pl.Series
    val_features: pl.DataFrame
    val_target: pl.Series
    test_features: pl.DataFrame
    test_target: pl.Series
    preprocessor_state: dict


def preprocess_data(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    imputation_strategy: str = "median",
) -> PreprocessingResult:
    """
    Preprocess the data for machine learning.

    This includes:
    - Handling missing values
    - Encoding categorical features
    - Feature scaling/normalization
    """
    logger.info("Starting data preprocessing", imputation_strategy=imputation_strategy)

    preprocessor_state = {}

    # Process target column
    train_target = train_df.select(PenguinDataSchema.target_column).to_series()
    val_target = val_df.select(PenguinDataSchema.target_column).to_series()
    test_target = test_df.select(PenguinDataSchema.target_column).to_series()

    # Extract feature columns
    train_features = train_df.select(
        PenguinDataSchema.feature_columns + PenguinDataSchema.categorical_columns
    )
    val_features = val_df.select(
        PenguinDataSchema.feature_columns + PenguinDataSchema.categorical_columns
    )
    test_features = test_df.select(
        PenguinDataSchema.feature_columns + PenguinDataSchema.categorical_columns
    )

    # Handle missing values in numerical features
    for col in PenguinDataSchema.feature_columns:
        if imputation_strategy == "median":
            # Calculate median on training data
            fill_value = train_features.select(pl.col(col)).drop_nulls().median()[0, 0]
            preprocessor_state[f"{col}_imputation_value"] = fill_value

            # Apply imputation
            train_features = train_features.with_columns(
                pl.col(col).fill_null(fill_value)
            )
            val_features = val_features.with_columns(pl.col(col).fill_null(fill_value))
            test_features = test_features.with_columns(
                pl.col(col).fill_null(fill_value)
            )
        elif imputation_strategy == "mean":
            # Calculate mean on training data
            fill_value = train_features.select(pl.col(col)).drop_nulls().mean()[0, 0]
            preprocessor_state[f"{col}_imputation_value"] = fill_value

            # Apply imputation
            train_features = train_features.with_columns(
                pl.col(col).fill_null(fill_value)
            )
            val_features = val_features.with_columns(pl.col(col).fill_null(fill_value))
            test_features = test_features.with_columns(
                pl.col(col).fill_null(fill_value)
            )

    # Handle categorical features
    for col in PenguinDataSchema.categorical_columns:
        # For categorical columns with missing values, fill with most frequent value
        if train_features.select(pl.col(col).null_count()).item() > 0:
            # Find most common value
            value_counts = train_features.select(pl.col(col)).drop_nulls().to_series().value_counts()
            most_common = value_counts.filter(pl.col(value_counts.columns[0]) == value_counts.row(0)[0])[0, 0]
            preprocessor_state[f"{col}_imputation_value"] = most_common

            # Apply imputation
            train_features = train_features.with_columns(
                pl.col(col).fill_null(most_common)
            )
            val_features = val_features.with_columns(pl.col(col).fill_null(most_common))
            test_features = test_features.with_columns(
                pl.col(col).fill_null(most_common)
            )

        # One-hot encode categorical columns
        unique_values = (
            train_features.select(pl.col(col))
            .unique()
            .sort(col)
            .to_series()
            .drop_nulls()
            .to_list()
        )
        preprocessor_state[f"{col}_categories"] = unique_values

        for value in unique_values:
            # Create one-hot encoded column
            column_name = f"{col}_{value}"
            train_features = train_features.with_columns(
                (pl.col(col) == value).cast(pl.Int8).alias(column_name)
            )
            val_features = val_features.with_columns(
                (pl.col(col) == value).cast(pl.Int8).alias(column_name)
            )
            test_features = test_features.with_columns(
                (pl.col(col) == value).cast(pl.Int8).alias(column_name)
            )

    # Drop original categorical columns after one-hot encoding
    train_features = train_features.drop(PenguinDataSchema.categorical_columns)
    val_features = val_features.drop(PenguinDataSchema.categorical_columns)
    test_features = test_features.drop(PenguinDataSchema.categorical_columns)

    # Normalize numerical features
    for col in PenguinDataSchema.feature_columns:
        # Calculate mean and std on training data
        col_mean = train_features.select(pl.col(col)).mean()[0, 0]
        col_std = train_features.select(pl.col(col)).std()[0, 0]

        # Store normalization parameters
        preprocessor_state[f"{col}_mean"] = col_mean
        preprocessor_state[f"{col}_std"] = col_std

        # Apply normalization
        train_features = train_features.with_columns(
            ((pl.col(col) - col_mean) / col_std).alias(col)
        )
        val_features = val_features.with_columns(
            ((pl.col(col) - col_mean) / col_std).alias(col)
        )
        test_features = test_features.with_columns(
            ((pl.col(col) - col_mean) / col_std).alias(col)
        )

    logger.info(
        "Data preprocessing complete",
        train_shape=train_features.shape,
        val_shape=val_features.shape,
        test_shape=test_features.shape,
    )

    return PreprocessingResult(
        train_features=train_features,
        train_target=train_target,
        val_features=val_features,
        val_target=val_target,
        test_features=test_features,
        test_target=test_target,
        preprocessor_state=preprocessor_state,
    )
