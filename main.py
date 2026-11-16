import numpy as np
import pandas as pd

from predictive_maintenance.config import load_model_specs, load_yaml
from predictive_maintenance.data.pipeline import FeaturePipeline
from predictive_maintenance.data.preprocessing import load_fd001
from predictive_maintenance.evaluation.metrics import regression_metrics
from predictive_maintenance.inference.model_loader import load_model
from predictive_maintenance.inference.predictor import predict_windows

DEVICE = "cpu"


def _print_table(title: str, rows: dict[str, dict]) -> None:
    print(f"\n========== {title} ==========")
    print(pd.DataFrame(rows).T.rename_axis("Model").to_string(float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    print("Loading FD001 test dataset...")
    _, test_df, y_true = load_fd001()

    unit_ids = np.sort(test_df["unit_id"].unique())
    if not np.array_equal(unit_ids, np.arange(1, len(y_true) + 1)):
        raise ValueError("Test engine IDs do not match official RUL order.")
    print(f"Test engines: {len(unit_ids)}")

    cap = load_yaml("pipeline_config.yaml")["evaluation"]["rul_cap"]
    raw, capped = {}, {}

    for name, spec in load_model_specs().items():
        print(f"\nEvaluating {name.upper()} (features={spec.input_size})...")

        pipe = FeaturePipeline.from_artifacts(model_type=name, window_size=spec.window_size)
        ids, X = pipe.transform(test_df)

        expected = (len(y_true), spec.window_size, spec.input_size)
        if X.shape != expected or not np.array_equal(ids, unit_ids):
            raise ValueError(f"Unexpected test windows {X.shape}, expected {expected}")

        pred = predict_windows(load_model(spec, DEVICE), X, DEVICE)
        raw[name.upper()] = regression_metrics(y_true, pred)
        capped[name.upper()] = regression_metrics(y_true, pred, rul_cap=cap)

    _print_table("FD001 TEST (raw RUL)", raw)
    _print_table(f"FD001 TEST (RUL capped at {cap}, compare with Colab)", capped)


if __name__ == "__main__":
    main()
