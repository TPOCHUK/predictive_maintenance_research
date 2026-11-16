"""Project paths and model settings."""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


def load_yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PipelineConfig:
    """Validated feature-pipeline configuration."""

    base_features: tuple[str, ...]
    engineered_features: tuple[str, ...]
    flag_columns: dict[str, str]
    baseline_cycles: int
    detector_window: int
    sensor_directions: dict[str, int]

    def __post_init__(self) -> None:
        if self.baseline_cycles <= 0:
            raise ValueError("baseline_cycles must be positive.")

        if self.detector_window <= 0:
            raise ValueError("detector_window must be positive.")

        if not self.base_features:
            raise ValueError("base_features must not be empty.")

        if not self.engineered_features:
            raise ValueError("engineered_features must not be empty.")

        required_models = {"lstm", "cnn"}
        missing_models = required_models - self.flag_columns.keys()

        if missing_models:
            raise ValueError(f"Missing flag columns for models: {sorted(missing_models)}")

        if not self.sensor_directions:
            raise ValueError("sensor_directions must not be empty.")

        invalid_directions = {
            sensor: direction
            for sensor, direction in self.sensor_directions.items()
            if direction not in (-1, 1)
        }

        if invalid_directions:
            raise ValueError(
                f"Sensor directions must be either -1 or 1. Invalid values: {invalid_directions}"
            )


def load_pipeline_config() -> PipelineConfig:
    raw = load_yaml("pipeline_config.yaml")

    features = raw["features"]
    feature_engineering = raw["feature_engineering"]

    return PipelineConfig(
        base_features=tuple(features["base_features"]),
        engineered_features=tuple(features["engineered_features"]),
        flag_columns=dict(features["flag_columns"]),
        baseline_cycles=feature_engineering["baseline_cycles"],
        detector_window=feature_engineering["detector_window"],
        sensor_directions=dict(feature_engineering["sensor_directions"]),
    )


@dataclass(frozen=True)
class ModelSpec:
    """Settings needed to load and run a model."""

    name: str
    architecture: str
    weights: Path
    window_size: int
    input_size: int
    flag_mode: str
    hidden_size: int | None = None
    dropout: float = 0.3
    checkpoint_key: str | None = None


def load_model_specs() -> dict[str, ModelSpec]:
    raw = load_yaml("model_config.yaml")["models"]
    specs: dict[str, ModelSpec] = {}
    for name, cfg in raw.items():
        if "architecture" not in cfg:
            continue
        specs[name] = ModelSpec(
            name=name,
            architecture=cfg["architecture"],
            weights=PROJECT_ROOT / cfg["weights"],
            window_size=cfg["window_size"],
            input_size=cfg["input_size"],
            flag_mode=cfg["flag_mode"],
            hidden_size=cfg.get("hidden_size"),
            dropout=cfg.get("dropout", 0.3),
            checkpoint_key=cfg.get("checkpoint_key"),
        )
    return specs
