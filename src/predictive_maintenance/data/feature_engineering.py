import numpy as np
import pandas as pd

from ..config import load_pipeline_config

PIPELINE_CONFIG = load_pipeline_config()

BASE_FEATURES = list(PIPELINE_CONFIG.base_features)
ENGINEERED_FEATURES = list(PIPELINE_CONFIG.engineered_features)

LSTM_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
LSTM_FLAG_COLUMN = PIPELINE_CONFIG.flag_columns["lstm"]
LSTM_INPUT_COLUMNS = LSTM_FEATURES + [LSTM_FLAG_COLUMN]

CNN_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES
CNN_FLAG_COLUMN = PIPELINE_CONFIG.flag_columns["cnn"]
CNN_INPUT_COLUMNS = CNN_FEATURES + [CNN_FLAG_COLUMN]

# Old names kept for other modules.
FLAG_COLUMN = LSTM_FLAG_COLUMN
MODEL_INPUT_COLUMNS = LSTM_INPUT_COLUMNS


class DegradationScore:
    """Directional sensor-deviation score."""

    def __init__(
        self,
        baseline_cycles: int | None = None,
        window: int | None = None,
        directions: dict[str, int] | None = None,
    ):

        if baseline_cycles is None:
            baseline_cycles = PIPELINE_CONFIG.baseline_cycles

        if window is None:
            window = PIPELINE_CONFIG.detector_window

        if directions is None:
            directions = PIPELINE_CONFIG.sensor_directions

        self.baseline_cycles = baseline_cycles
        self.window = window
        self.sensors = list(directions)
        self.direction = pd.Series(directions, dtype=int)
        self.scale_: pd.Series | None = None

    def _engine_score(self, engine: pd.DataFrame) -> pd.Series:
        if self.scale_ is None:
            raise RuntimeError("DegradationScore must be fitted first.")

        values = engine[self.sensors]
        # Compare each engine with its own first cycles.
        baseline = values.iloc[: self.baseline_cycles].median()
        deviation = values.sub(baseline).div(self.scale_)
        directional = deviation.mul(self.direction).clip(lower=0)
        raw = directional.mean(axis=1)

        return raw.rolling(self.window, min_periods=1).mean()

    def fit(self, train_df):
        residuals = []

        for _, eng in train_df.sort_values("cycle").groupby("unit_id", sort=False):
            calib = eng[self.sensors].iloc[: self.baseline_cycles]
            residuals.append(calib.sub(calib.median()))

        resid = pd.concat(residuals, ignore_index=True)
        # Median-based scale is less sensitive to noisy readings.
        self.scale_ = resid.abs().median().mul(1.4826).clip(lower=1e-6)

        return self

    def to_dict(self) -> dict:
        if self.scale_ is None:
            raise RuntimeError("DegradationScore must be fitted first.")

        return {
            "version": 1,
            "baseline_cycles": self.baseline_cycles,
            "window": self.window,
            "directions": {sensor: int(direction) for sensor, direction in self.direction.items()},
            "scale": {k: float(v) for k, v in self.scale_.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DegradationScore":
        expected_directions = PIPELINE_CONFIG.sensor_directions

        if d["directions"] != expected_directions:
            raise ValueError(
                "Detector directions in artifact differ from "
                "pipeline_config.yaml; retrain or fix the configuration."
            )

        if d["baseline_cycles"] != PIPELINE_CONFIG.baseline_cycles:
            raise ValueError(
                "Detector baseline_cycles in artifact differ from "
                "pipeline_config.yaml; retrain or fix the configuration."
            )

        if d["window"] != PIPELINE_CONFIG.detector_window:
            raise ValueError(
                "Detector window in artifact differs from "
                "pipeline_config.yaml; retrain or fix the configuration."
            )

        obj = cls(
            baseline_cycles=d["baseline_cycles"],
            window=d["window"],
            directions=d["directions"],
        )
        obj.scale_ = pd.Series(d["scale"], dtype=float)[obj.sensors]

        return obj

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.scale_ is None:
            raise RuntimeError("DegradationScore must be fitted first.")

        missing = set(self.sensors + ["unit_id", "cycle"]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        result = pd.Series(
            np.nan,
            index=df.index,
            name="degradation_score",
            dtype=float,
        )

        for _, engine in df.groupby("unit_id", sort=False):
            engine = engine.sort_values("cycle")
            score = self._engine_score(engine).copy()
            score.iloc[: self.baseline_cycles] = np.nan
            result.loc[engine.index] = score.to_numpy()

        return result


def build_features(df: pd.DataFrame, detector: DegradationScore) -> pd.DataFrame:
    """Add model features and a degradation score."""

    out = df.copy()
    out["degradation_score"] = detector.transform(out)

    return out
