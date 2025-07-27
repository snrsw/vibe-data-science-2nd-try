from pathlib import Path

import tomli
import yaml

from .settings import PipelineConfig


def load_config_from_yaml(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    return PipelineConfig.model_validate(config_dict)


def load_config_from_toml(path: str | Path) -> PipelineConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(config_path, "rb") as f:
        config_dict = tomli.load(f)
    
    return PipelineConfig.model_validate(config_dict)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    if path is None:
        return PipelineConfig()
    
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if config_path.suffix.lower() in (".yaml", ".yml"):
        return load_config_from_yaml(config_path)
    elif config_path.suffix.lower() == ".toml":
        return load_config_from_toml(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")