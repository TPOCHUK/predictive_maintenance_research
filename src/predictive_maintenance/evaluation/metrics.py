"""Model evaluation metrics."""

import numpy as np


def nasa_score(y_true, y_pred) -> float:
    """Return the C-MAPSS score with a higher cost for late predictions."""
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sum(np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)))


def regression_metrics(y_true, y_pred, rul_cap: float | None = None) -> dict[str, float]:
    """Return regression metrics with an optional RUL limit."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("Predictions contain NaN or inf.")
    if rul_cap is not None:
        y, p = np.minimum(y, rul_cap), np.minimum(p, rul_cap)
    err = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "R2": float(1 - np.sum(err**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "NASA Score": nasa_score(y, p),
    }
