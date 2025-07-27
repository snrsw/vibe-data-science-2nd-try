from enum import Enum
from typing import Any, ClassVar, Optional

import polars as pl
from pydantic import BaseModel


class PenguinSpecies(str, Enum):
    ADELIE = "Adelie"
    CHINSTRAP = "Chinstrap"
    GENTOO = "Gentoo"


class PenguinSex(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class PenguinIsland(str, Enum):
    TORGERSEN = "Torgersen"
    BISCOE = "Biscoe"
    DREAM = "Dream"


class PenguinRecord(BaseModel):
    species: PenguinSpecies
    island: PenguinIsland
    culmen_length_mm: Optional[float] = None
    culmen_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[PenguinSex] = None


class PenguinDataSchema(BaseModel):
    column_names: ClassVar[list[str]] = [
        "species",
        "island",
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
    ]

    feature_columns: ClassVar[list[str]] = [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    categorical_columns: ClassVar[list[str]] = [
        "island",
        "sex",
    ]

    target_column: ClassVar[str] = "species"

    dtypes: ClassVar[dict[str, Any]] = {
        "species": pl.Categorical,
        "island": pl.Categorical,
        "culmen_length_mm": pl.Float32,
        "culmen_depth_mm": pl.Float32,
        "flipper_length_mm": pl.Float32,
        "body_mass_g": pl.Float32,
        "sex": pl.Categorical,
    }


class PenguinPredictionRecord(BaseModel):
    species: PenguinSpecies
    island: PenguinIsland
    culmen_length_mm: Optional[float] = None
    culmen_depth_mm: Optional[float] = None
    flipper_length_mm: Optional[float] = None
    body_mass_g: Optional[float] = None
    sex: Optional[PenguinSex] = None
    predicted_species: PenguinSpecies
