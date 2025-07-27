
import polars as pl
import pytest
from unittest.mock import MagicMock, patch

from vibe_data_science_2nd_try.models.lightgbm_model import LightGBMModel


@pytest.fixture
def sample_model_params():
    return {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
    }


@pytest.fixture
def sample_features():
    return pl.DataFrame({
        "culmen_length_mm": [39.1, 49.3, 46.5, 38.6, 45.7, 48.8],
        "culmen_depth_mm": [18.7, 19.9, 17.9, 17.8, 17.3, 19.5],
        "flipper_length_mm": [181.0, 203.0, 217.0, 193.0, 222.0, 197.0],
        "body_mass_g": [3750.0, 3650.0, 4800.0, 3700.0, 5200.0, 3500.0],
    })


@pytest.fixture
def sample_target():
    return pl.Series(name="species", values=[0, 1, 2, 0, 2, 1])


def test_lightgbm_model_init(sample_model_params):
    model = LightGBMModel(sample_model_params)
    assert model.model_params == sample_model_params
    assert model.model is None

    # Test with default params
    default_model = LightGBMModel({})
    assert "objective" in default_model.model_params
    assert default_model.model_params["objective"] == "multiclass"
    assert "num_class" in default_model.model_params
    assert default_model.model_params["num_class"] == 3


@patch("lightgbm.train")
def test_lightgbm_model_train(mock_train, sample_features, sample_target, sample_model_params):
    # Mock the lightgbm.train function
    mock_model = MagicMock()
    mock_train.return_value = mock_model

    # Create the model
    model = LightGBMModel(sample_model_params)

    # Train the model
    model.train(sample_features, sample_target)

    # Check that lightgbm.train was called
    assert mock_train.called

    # Check that model was set
    assert model.model is not None
    assert model.model == mock_model

    # Check that feature names were stored
    assert model.feature_names == sample_features.columns


@patch("lightgbm.train")
def test_lightgbm_model_train_with_validation(mock_train, sample_features, sample_target, sample_model_params):
    # Mock the lightgbm.train function
    mock_model = MagicMock()
    mock_train.return_value = mock_model

    # Create the model
    model = LightGBMModel(sample_model_params)

    # Train the model with validation data
    model.train(
        sample_features,
        sample_target,
        sample_features.slice(0, 2),
        sample_target.slice(0, 2)
    )

    # Check that lightgbm.train was called with validation set
    args, kwargs = mock_train.call_args
    assert len(kwargs["valid_names"]) == 2
    assert "valid" in kwargs["valid_names"]


@patch("lightgbm.train")
def test_lightgbm_model_predict(mock_train, sample_features, sample_target, sample_model_params):
    # Mock the lightgbm.train function
    mock_model = MagicMock()
    mock_model.predict.return_value = [
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.7, 0.1],
    ]
    mock_train.return_value = mock_model

    # Create and train the model
    model = LightGBMModel(sample_model_params)
    model.train(sample_features, sample_target)
    model.classes = ["Adelie", "Chinstrap", "Gentoo"]

    # Make predictions
    predictions = model.predict(sample_features)

    # Check that predict was called
    assert mock_model.predict.called

    # Check predictions
    assert len(predictions) == len(sample_features)
    assert predictions.name == "predicted_species"
    assert predictions[0] == "Adelie"
    assert predictions[1] == "Chinstrap"
    assert predictions[2] == "Gentoo"


@patch("lightgbm.train")
@patch("joblib.dump")
def test_lightgbm_model_save(mock_dump, mock_train, sample_features, sample_target, sample_model_params, tmp_path):
    # Mock the lightgbm.train function
    mock_model = MagicMock()
    mock_train.return_value = mock_model

    # Create and train the model
    model = LightGBMModel(sample_model_params)
    model.train(sample_features, sample_target)

    # Save the model
    save_path = tmp_path / "model.pkl"
    result = model.save(save_path)

    # Check that joblib.dump was called
    assert mock_dump.called

    # Check that the path was returned
    assert result == save_path


@patch("joblib.load")
def test_lightgbm_model_load(mock_load, tmp_path):
    # Mock the joblib.load function
    mock_model = MagicMock()
    mock_load.return_value = {
        "model": mock_model,
        "feature_names": ["feature1", "feature2"],
        "classes": ["class1", "class2"],
    }

    # Load the model
    model_path = tmp_path / "model.pkl"
    model_path.touch()  # Create empty file
    loaded_model = LightGBMModel.load(model_path)

    # Check that joblib.load was called
    assert mock_load.called

    # Check that the model was loaded correctly
    assert loaded_model.model == mock_model
    assert loaded_model.feature_names == ["feature1", "feature2"]
    assert loaded_model.classes == ["class1", "class2"]
