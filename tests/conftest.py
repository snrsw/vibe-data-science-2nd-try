
import polars as pl
import pytest

from vibe_data_science_2nd_try.config import PipelineConfig


@pytest.fixture
def sample_config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def sample_penguin_data():
    return pl.DataFrame({
        "species": ["Adelie", "Chinstrap", "Gentoo", "Adelie", "Gentoo", "Chinstrap"],
        "island": ["Torgersen", "Dream", "Biscoe", "Torgersen", "Biscoe", "Dream"],
        "culmen_length_mm": [39.1, 49.3, 46.5, 38.6, 45.7, 48.8],
        "culmen_depth_mm": [18.7, 19.9, 17.9, 17.8, 17.3, 19.5],
        "flipper_length_mm": [181.0, 203.0, 217.0, 193.0, 222.0, 197.0],
        "body_mass_g": [3750.0, 3650.0, 4800.0, 3700.0, 5200.0, 3500.0],
        "sex": ["MALE", "FEMALE", "MALE", "FEMALE", "MALE", "FEMALE"],
    })


@pytest.fixture
def temp_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir