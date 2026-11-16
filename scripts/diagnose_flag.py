"""Check preprocessing for active models."""

from predictive_maintenance.config import load_model_specs
from predictive_maintenance.data.pipeline import FeaturePipeline
from predictive_maintenance.data.preprocessing import load_fd001
from predictive_maintenance.evaluation.metrics import regression_metrics
from predictive_maintenance.inference.model_loader import load_model
from predictive_maintenance.inference.predictor import predict_windows

_, test, y = load_fd001()
for name, spec in load_model_specs().items():
    model = load_model(spec)
    _, X = FeaturePipeline.from_artifacts(
        model_type=name,
        window_size=spec.window_size,
    ).transform(test)
    metrics = regression_metrics(y, predict_windows(model, X))
    print(
        f"{name.upper():5} flag={spec.flag_mode:10} "
        f"MAE={metrics['MAE']:7.2f} "
        f"RMSE={metrics['RMSE']:7.2f} "
        f"R2={metrics['R2']:7.3f}"
    )
