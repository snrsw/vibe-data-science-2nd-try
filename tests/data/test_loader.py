from unittest.mock import patch, mock_open

import polars as pl
import pytest

from vibe_data_science_2nd_try.data.loader import load_penguin_data, split_data


@pytest.fixture
def sample_penguin_data():
    return pl.DataFrame(
        {
            "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Gentoo"],
            "island": ["Torgersen", "Dream", "Biscoe", "Torgersen", "Biscoe"],
            "culmen_length_mm": [39.1, 49.3, 46.5, 38.6, 45.7],
            "culmen_depth_mm": [18.7, 19.9, 17.9, 17.8, 17.3],
            "flipper_length_mm": [181.0, 203.0, 217.0, 193.0, 222.0],
            "body_mass_g": [3750.0, 3650.0, 4800.0, 3700.0, 5200.0],
            "sex": ["MALE", "FEMALE", "MALE", "FEMALE", "MALE"],
        }
    )


@pytest.fixture
def mock_csv_data():
    csv_data = (
        "species,island,culmen_length_mm,culmen_depth_mm,flipper_length_mm,body_mass_g,sex\n"
        "Adelie,Torgersen,39.1,18.7,181,3750,MALE\n"
        "Chinstrap,Dream,49.3,19.9,203,3650,FEMALE\n"
        "Gentoo,Biscoe,46.5,17.9,217,4800,MALE\n"
        "Adelie,Torgersen,38.6,17.8,193,3700,FEMALE\n"
        "Gentoo,Biscoe,45.7,17.3,222,5200,MALE\n"
    )
    return csv_data


def test_load_penguin_data_success(mock_csv_data):
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=mock_csv_data)),
        patch(
            "polars.read_csv",
            return_value=pl.DataFrame(
                {
                    "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Gentoo"],
                    "island": ["Torgersen", "Dream", "Biscoe", "Torgersen", "Biscoe"],
                    "culmen_length_mm": [39.1, 49.3, 46.5, 38.6, 45.7],
                    "culmen_depth_mm": [18.7, 19.9, 17.9, 17.8, 17.3],
                    "flipper_length_mm": [181.0, 203.0, 217.0, 193.0, 222.0],
                    "body_mass_g": [3750.0, 3650.0, 4800.0, 3700.0, 5200.0],
                    "sex": ["MALE", "FEMALE", "MALE", "FEMALE", "MALE"],
                }
            ),
        ),
    ):
        df = load_penguin_data("data/penguins_size.csv")
        assert df.shape == (5, 7)
        assert "species" in df.columns
        assert "culmen_length_mm" in df.columns


def test_load_penguin_data_file_not_found():
    with (
        patch("pathlib.Path.exists", return_value=False),
        pytest.raises(FileNotFoundError),
    ):
        load_penguin_data("nonexistent_file.csv")


def test_load_penguin_data_missing_columns():
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "polars.read_csv",
            return_value=pl.DataFrame(
                {
                    "island": ["Torgersen", "Dream"],
                    "culmen_length_mm": [39.1, 49.3],
                }
            ),
        ),
        pytest.raises(ValueError),
    ):
        load_penguin_data("data/penguins_size.csv")


def test_split_data(sample_penguin_data):
    train_df, val_df, test_df = split_data(
        sample_penguin_data, test_size=0.2, validation_size=0.2, random_state=42
    )

    # With 5 samples and test_size=0.2, val_size=0.2, we expect:
    # train=3, val=1, test=1
    assert train_df.shape[0] == 3
    assert val_df.shape[0] == 1
    assert test_df.shape[0] == 1

    # Make sure we didn't lose any rows
    assert (
        train_df.shape[0] + val_df.shape[0] + test_df.shape[0]
        == sample_penguin_data.shape[0]
    )

    # Verify all columns are preserved
    assert train_df.columns == sample_penguin_data.columns
    assert val_df.columns == sample_penguin_data.columns
    assert test_df.columns == sample_penguin_data.columns
