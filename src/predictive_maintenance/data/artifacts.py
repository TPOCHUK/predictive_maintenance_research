"""Load and save preprocessing files."""

import json
from pathlib import Path

from sklearn.preprocessing import StandardScaler

from ..config import ARTIFACTS_DIR
from .feature_engineering import DegradationScore
from .preprocessing import load_scaler

DETECTOR_JSON = ARTIFACTS_DIR / "degradation_detector.json"
DETECTOR_PKL = ARTIFACTS_DIR / "degradation_detector.pkl"

SCALER_PATHS = {
    "lstm": ARTIFACTS_DIR / "scaler_lstm_fd001.pkl",
    "cnn": ARTIFACTS_DIR / "scaler_cnn_fd001.pkl",
}

FEATURE_METADATA_PATH = ARTIFACTS_DIR / "feature_metadata.json"


def save_detector(detector: DegradationScore, path: Path = DETECTOR_JSON) -> Path:
    path.write_text(
        json.dumps(detector.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_detector(path: Path = DETECTOR_JSON) -> DegradationScore:
    return DegradationScore.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_model_scaler(model_type: str) -> StandardScaler:
    model_type = model_type.lower()

    if model_type not in SCALER_PATHS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Expected one of {tuple(SCALER_PATHS)}."
        )

    return load_scaler(SCALER_PATHS[model_type])


def load_feature_metadata() -> dict:
    if not FEATURE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Feature metadata not found: {FEATURE_METADATA_PATH}")

    return json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))
