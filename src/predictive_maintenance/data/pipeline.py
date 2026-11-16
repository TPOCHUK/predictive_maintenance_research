"""Turn raw data into model input windows."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .artifacts import (
    load_detector,
    load_feature_metadata,
    load_model_scaler,
)
from .feature_engineering import (
    CNN_FLAG_COLUMN,
    CNN_INPUT_COLUMNS,
    LSTM_FLAG_COLUMN,
    LSTM_INPUT_COLUMNS,
    DegradationScore,
    build_features,
)
from .preprocessing import (
    make_last_windows,
    prepare_data,
    transform_data,
)

MODEL_CONFIGS = {
    "lstm": {
        "columns": LSTM_INPUT_COLUMNS,
        "flag_mode": "warmup",
        "flag_column": LSTM_FLAG_COLUMN,
    },
    "cnn": {
        "columns": CNN_INPUT_COLUMNS,
        "flag_mode": "available",
        "flag_column": CNN_FLAG_COLUMN,
    },
}


@dataclass
class FeaturePipeline:
    detector: DegradationScore
    scaler: StandardScaler
    flag_mode: str = "warmup"
    window_size: int = 80
    model_type: str = "lstm"
    columns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.model_type = self.model_type.lower()

        if self.model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_type {self.model_type!r}. Expected one of {tuple(MODEL_CONFIGS)}."
            )

        config = MODEL_CONFIGS[self.model_type]
        expected_columns = config["columns"]

        if self.flag_mode != config["flag_mode"]:
            raise ValueError(
                f"{self.model_type} requires flag_mode="
                f"{config['flag_mode']!r}, got {self.flag_mode!r}."
            )

        if not self.columns:
            self.columns = list(expected_columns)
        elif self.columns != expected_columns:
            raise ValueError(f"Feature order mismatch for {self.model_type}.")

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")

        if self.scaler.n_features_in_ != len(self.columns):
            raise ValueError(
                f"Scaler expects {self.scaler.n_features_in_} features, "
                f"pipeline has {len(self.columns)}."
            )

    @classmethod
    def from_artifacts(
        cls,
        model_type: str = "lstm",
        window_size: int = 80,
    ) -> "FeaturePipeline":
        model_type = model_type.lower()

        if model_type not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_type {model_type!r}. Expected one of {tuple(MODEL_CONFIGS)}."
            )

        config = MODEL_CONFIGS[model_type]
        expected_columns = config["columns"]

        metadata = load_feature_metadata()

        if model_type not in metadata:
            raise ValueError(f"Model {model_type!r} is missing from feature_metadata.json.")

        saved_config = metadata[model_type]

        if saved_config["features"] != expected_columns:
            raise ValueError(
                f"Feature metadata mismatch for {model_type}. "
                "Check feature order and training configuration."
            )

        if saved_config["n_features"] != len(expected_columns):
            raise ValueError(f"Invalid n_features in metadata for {model_type}.")

        if saved_config["flag_mode"] != config["flag_mode"]:
            raise ValueError(f"Invalid flag_mode in metadata for {model_type}.")

        if saved_config["window_size"] != window_size:
            raise ValueError(
                f"{model_type} was trained with window_size="
                f"{saved_config['window_size']}, not {window_size}."
            )

        return cls(
            detector=load_detector(),
            scaler=load_model_scaler(model_type),
            flag_mode=config["flag_mode"],
            window_size=window_size,
            model_type=model_type,
            columns=list(expected_columns),
        )

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Додаємо ознаки, масштабуємо їх і збираємо останні 80 циклів."""

        df = prepare_data(df)

        df = build_features(df, self.detector)

        config = MODEL_CONFIGS[self.model_type]

        df = transform_data(
            df,
            scaler=self.scaler,
            scaler_columns=self.columns,
            flag_mode=self.flag_mode,
            flag_column=config["flag_column"],
        )

        X = make_last_windows(
            df,
            feature_columns=self.columns,
            window_size=self.window_size,
        )

        unit_ids = np.sort(df["unit_id"].unique())

        return unit_ids, X
