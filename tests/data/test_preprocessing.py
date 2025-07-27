import polars as pl
import pytest

from vibe_data_science_2nd_try.data.preprocessing import preprocess_data


@pytest.fixture
def sample_data():
    train_df = pl.DataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Gentoo", "Chinstrap"],
        "island": ["Torgersen", "Dream", "Biscoe", "Torgersen", "Biscoe", "Dream"],
        "culmen_length_mm": [39.1, 49.3, 46.5, 38.6, 45.7, 48.8],
        "culmen_depth_mm": [18.7, 19.9, 17.9, 17.8, 17.3, 19.5],
        "flipper_length_mm": [181.0, 203.0, 217.0, 193.0, 222.0, 197.0],
        "body_mass_g": [3750.0, 3650.0, 4800.0, 3700.0, 5200.0, 3500.0],
        "sex": ["MALE", "FEMALE", "MALE", "FEMALE", "MALE", None],
    })
    
    val_df = pl.DataFrame({
        "species": ["Adelie", "Gentoo"],
        "island": ["Dream", "Biscoe"],
        "culmen_length_mm": [37.8, 47.2],
        "culmen_depth_mm": [18.3, 17.1],
        "flipper_length_mm": [174.0, 214.0],
        "body_mass_g": [3400.0, 4650.0],
        "sex": ["MALE", "FEMALE"],
    })
    
    test_df = pl.DataFrame({
        "species": ["Chinstrap", "Adelie"],
        "island": [None, "Torgersen"],
        "culmen_length_mm": [46.9, None],
        "culmen_depth_mm": [16.6, 18.2],
        "flipper_length_mm": [193.0, 190.0],
        "body_mass_g": [3800.0, 3650.0],
        "sex": ["MALE", "FEMALE"],
    })
    
    return train_df, val_df, test_df


def test_preprocess_data_basic(sample_data):
    train_df, val_df, test_df = sample_data
    result = preprocess_data(train_df, val_df, test_df)
    
    # Check that all DataFrames and Series are returned
    assert isinstance(result.train_features, pl.DataFrame)
    assert isinstance(result.train_target, pl.Series)
    assert isinstance(result.val_features, pl.DataFrame)
    assert isinstance(result.val_target, pl.Series)
    assert isinstance(result.test_features, pl.DataFrame)
    assert isinstance(result.test_target, pl.Series)
    
    # Check that preprocessor state is returned
    assert isinstance(result.preprocessor_state, dict)
    
    # Check that the target column is correctly extracted
    assert result.train_target.name == "species"
    assert result.val_target.name == "species"
    assert result.test_target.name == "species"
    
    # Verify one-hot encoding
    assert "island_Torgersen" in result.train_features.columns
    assert "island_Dream" in result.train_features.columns
    assert "island_Biscoe" in result.train_features.columns
    assert "sex_MALE" in result.train_features.columns
    assert "sex_FEMALE" in result.train_features.columns
    
    # Check the original categorical columns are dropped
    assert "island" not in result.train_features.columns
    assert "sex" not in result.train_features.columns
    
    # Check numeric features are still present
    assert "culmen_length_mm" in result.train_features.columns
    assert "culmen_depth_mm" in result.train_features.columns
    assert "flipper_length_mm" in result.train_features.columns
    assert "body_mass_g" in result.train_features.columns


def test_preprocess_data_missing_values(sample_data):
    train_df, val_df, test_df = sample_data
    result = preprocess_data(train_df, val_df, test_df)
    
    # Check that missing values in test data were handled
    assert result.test_features.null_count().sum() == 0
    
    # Verify that island was imputed in test data
    assert "island_Torgersen" in result.test_features.columns
    assert "island_Dream" in result.test_features.columns
    assert "island_Biscoe" in result.test_features.columns
    
    # Verify that culmen_length_mm was imputed in test data
    assert not result.test_features["culmen_length_mm"].is_null().any()
    
    # Verify that sex was imputed in train data
    assert "sex_MALE" in result.train_features.columns
    assert "sex_FEMALE" in result.train_features.columns
    

def test_preprocess_data_normalization(sample_data):
    train_df, val_df, test_df = sample_data
    result = preprocess_data(train_df, val_df, test_df)
    
    # Check normalization params are stored
    assert "culmen_length_mm_mean" in result.preprocessor_state
    assert "culmen_length_mm_std" in result.preprocessor_state
    assert "culmen_depth_mm_mean" in result.preprocessor_state
    assert "culmen_depth_mm_std" in result.preprocessor_state
    assert "flipper_length_mm_mean" in result.preprocessor_state
    assert "flipper_length_mm_std" in result.preprocessor_state
    assert "body_mass_g_mean" in result.preprocessor_state
    assert "body_mass_g_std" in result.preprocessor_state
    
    # Check that features are normalized (mean ~0, std ~1)
    for col in ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]:
        mean = result.train_features[col].mean()
        std = result.train_features[col].std()
        assert abs(mean) < 1e-10  # Close to 0
        assert abs(std - 1.0) < 1e-10  # Close to 1