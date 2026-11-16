from threading import Lock

import numpy as np
import pandas as pd
import torch

from ..config import load_model_specs
from ..data.pipeline import FeaturePipeline
from .model_loader import load_model


def predict_windows(model: torch.nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        out = (
            model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy().reshape(-1)
        )
    if not np.isfinite(out).all():
        raise ValueError("Model produced NaN or infinite predictions.")
    return out


class RULPredictor:
    """Load a model and its feature pipeline."""

    def __init__(self, model_name: str = "lstm", device: str = "cpu"):
        spec = load_model_specs()[model_name]
        self.device = device
        self._prediction_lock = Lock()

        self.pipeline = FeaturePipeline.from_artifacts(
            model_type=model_name, window_size=spec.window_size
        )
        self.model = load_model(spec, device)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Готуємо історію кожного двигуна і рахуємо один прогноз на нього."""
        # API requests share one model.
        with self._prediction_lock:
            unit_ids, X = self.pipeline.transform(df)
            history_cycles = df.groupby("unit_id").size().reindex(unit_ids).to_numpy()
            return pd.DataFrame(
                {
                    "unit_id": unit_ids,
                    "predicted_rul": predict_windows(self.model, X, self.device),
                    "history_cycles": history_cycles,
                    "is_warmup": history_cycles < self.pipeline.window_size,
                }
            )
