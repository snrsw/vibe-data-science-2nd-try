from pathlib import Path
import polars as pl
import structlog

from vibe_data_science_2nd_try.data.schema import PenguinDataSchema

logger = structlog.get_logger()


def load_penguin_data(file_path: str | Path) -> pl.DataFrame:
    path = Path(file_path)
    if not path.exists():
        logger.error("Data file not found", path=str(path))
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading penguin dataset", path=str(path))

    df: pl.DataFrame = pl.read_csv(source=path, null_values=["NA"])

    missing_columns = set(PenguinDataSchema.column_names) - set(df.columns)
    if missing_columns:
        logger.error("Missing required columns", missing_columns=list(missing_columns))
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column, dtype in PenguinDataSchema.dtypes.items():
        if column in df.columns:
            df = df.with_columns(pl.col(column).cast(dtype))

    logger.info(
        "Dataset loaded successfully",
        rows=df.shape[0],
        columns=df.shape[1],
        missing_values=df.null_count().sum(),
    )

    return df


def split_data(
    df: pl.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Split the dataframe into train, validation, and test sets.
    """
    logger.info(
        "Splitting dataset",
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )

    total_rows: int = df.shape[0]
    test_rows = int(total_rows * test_size)
    val_rows = int(total_rows * validation_size)
    train_rows: int = total_rows - test_rows - val_rows

    shuffled_df = df.with_row_index("__index").sample(
        fraction=1.0, shuffle=True, seed=random_state
    )

    train_df: pl.DataFrame = shuffled_df.filter(pl.col("__index") < train_rows).drop(
        "__index"
    )

    val_df: pl.DataFrame = shuffled_df.filter(
        (pl.col("__index") >= train_rows) & (pl.col("__index") < train_rows + val_rows)
    ).drop("__index")

    test_df: pl.DataFrame = shuffled_df.filter(
        pl.col("__index") >= train_rows + val_rows
    ).drop("__index")

    logger.info(
        "Dataset split complete",
        train_rows=train_df.shape[0],
        val_rows=val_df.shape[0],
        test_rows=test_df.shape[0],
    )

    return train_df, val_df, test_df
