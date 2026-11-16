import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .feature_engineering import (
    CNN_FLAG_COLUMN,
    FLAG_COLUMN,
    LSTM_FLAG_COLUMN,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
SCALER_PATH = PROJECT_ROOT / "artifacts" / "scaler_fd001.pkl"

COLUMNS = ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]

REQUIRED_COLUMNS = COLUMNS

FLAG_MODES = ("warmup", "available", "constant_1")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def validate_data(df: pd.DataFrame, data_name: str = "Data") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{data_name} must be a pandas DataFrame.")

    if df.empty:
        raise ValueError(f"{data_name} is empty.")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{data_name} is missing columns: {sorted(missing)}")

    numeric_data = df[REQUIRED_COLUMNS].select_dtypes(include=np.number)

    if len(numeric_data.columns) != len(REQUIRED_COLUMNS):
        raise TypeError(f"{data_name} contains non-numeric columns.")

    values = numeric_data.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError(f"{data_name} contains missing or infinite values.")

    if (df["unit_id"] <= 0).any():
        raise ValueError(f"{data_name}: unit_id must be positive.")

    if (df["cycle"] <= 0).any():
        raise ValueError(f"{data_name}: cycle must be positive.")

    if (df[["unit_id", "cycle"]] % 1 != 0).any().any():
        raise ValueError(f"{data_name}: unit_id and cycle must be whole numbers.")

    if df.duplicated(subset=["unit_id", "cycle"]).any():
        raise ValueError(f"{data_name} contains duplicate unit_id/cycle pairs.")


def prepare_data(df: pd.DataFrame, add_rul: bool = False) -> pd.DataFrame:
    validate_data(df)

    out = df.copy()
    out = out.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    if add_rul:
        max_cycles = out.groupby("unit_id")["cycle"].transform("max")
        out["RUL"] = max_cycles - out["cycle"]

    return out


def load_fd001(data_dir: Path = DATA_DIR):
    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMNS)

    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMNS)

    rul = pd.read_csv(rul_path, sep=r"\s+", header=None)[0].to_numpy(dtype=np.float32)

    train = prepare_data(train, add_rul=True)
    test = prepare_data(test)

    if len(rul) != test["unit_id"].nunique():
        raise ValueError("Number of test RUL values does not match number of test engines.")

    if not np.isfinite(rul).all() or (rul < 0).any():
        raise ValueError("Invalid values in test RUL.")

    return train, test, rul


def add_degradation_flag(
    df: pd.DataFrame,
    mode: str = "warmup",
    flag_column: str = FLAG_COLUMN,
) -> pd.DataFrame:
    """Add the degradation flag used by a model."""

    if "degradation_score" not in df.columns:
        raise ValueError("Missing degradation_score column.")

    if mode not in FLAG_MODES:
        raise ValueError(f"Unknown flag mode {mode!r}; expected one of {FLAG_MODES}")

    if flag_column not in (LSTM_FLAG_COLUMN, CNN_FLAG_COLUMN):
        raise ValueError(f"Unknown flag column: {flag_column}")

    out = df.copy()
    missing = out["degradation_score"].isna()

    if mode == "warmup":
        flag = missing
    elif mode == "available":
        flag = ~missing
    else:
        flag = pd.Series(True, index=out.index)

    out[flag_column] = flag.astype("float32")

    return out


def load_scaler(path: Path = SCALER_PATH) -> StandardScaler:
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")

    scaler = joblib.load(path)

    if not isinstance(scaler, StandardScaler):
        raise TypeError("Loaded artifact is not a StandardScaler.")

    return scaler


def fit_scaler(
    df: pd.DataFrame,
    feature_columns: list[str],
    flag_mode: str = "warmup",
    flag_column: str = FLAG_COLUMN,
) -> tuple[StandardScaler, list[str]]:
    if "degradation_score" not in df.columns:
        raise ValueError("Missing degradation_score column.")

    out = add_degradation_flag(
        df,
        mode=flag_mode,
        flag_column=flag_column,
    )

    missing = set(feature_columns) - set(out.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    scaler_columns = feature_columns + [flag_column]

    scaler = StandardScaler()
    scaler.fit(out[scaler_columns].fillna(0).to_numpy(dtype=np.float64))

    return scaler, scaler_columns


def transform_data(
    df: pd.DataFrame,
    scaler: StandardScaler,
    scaler_columns: list[str],
    flag_mode: str = "warmup",
    flag_column: str = FLAG_COLUMN,
) -> pd.DataFrame:
    out = add_degradation_flag(
        df,
        mode=flag_mode,
        flag_column=flag_column,
    )

    missing = set(scaler_columns) - set(out.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    if scaler.n_features_in_ != len(scaler_columns):
        raise ValueError("Scaler feature count does not match scaler_columns.")

    out[scaler_columns] = scaler.transform(out[scaler_columns].fillna(0).to_numpy(dtype=np.float64))

    return out


def make_train_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    if "RUL" not in df.columns:
        raise ValueError("Training data must contain RUL.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    X, y = [], []

    for _, engine in df.groupby("unit_id", sort=False):
        engine = engine.sort_values("cycle")

        features = engine[feature_columns].to_numpy(dtype=np.float32)
        rul = engine["RUL"].to_numpy(dtype=np.float32)

        for end in range(window_size, len(engine) + 1):
            X.append(features[end - window_size : end])
            y.append(rul[end - 1])

    if not X:
        raise ValueError(f"No complete windows of size {window_size} found.")

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_last_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int = 80,
) -> np.ndarray:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    X = []

    for _, engine in df.groupby("unit_id", sort=True):
        engine = engine.sort_values("cycle")
        features = engine[feature_columns].to_numpy(dtype=np.float32)

        if len(features) < window_size:
            # Repeat the first row; do not invent future readings.
            padding = np.repeat(
                features[:1],
                window_size - len(features),
                axis=0,
            )
            features = np.vstack([padding, features])

        X.append(features[-window_size:])

    if not X:
        raise ValueError("No engine data available for windows.")

    return np.asarray(X, dtype=np.float32)
