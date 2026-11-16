import numpy as np

from predictive_maintenance.config import load_model_specs
from predictive_maintenance.data.artifacts import load_feature_metadata
from predictive_maintenance.data.preprocessing import load_fd001
from predictive_maintenance.inference.predictor import RULPredictor


def test_partial_history_models_match_runtime_artifacts():
    metadata = load_feature_metadata()
    specs = load_model_specs()
    _, test, _ = load_fd001()
    one_cycle = test.groupby("unit_id", sort=True).head(1)

    for model_name in ("lstm", "cnn"):
        assert specs[model_name].input_size == 27
        assert metadata[model_name]["n_features"] == 27
        assert metadata[model_name]["trained_on_partial_histories"] is True
        assert metadata[model_name]["minimum_history_cycles"] == 1

        result = RULPredictor(model_name).predict(one_cycle)

        assert len(result) == test["unit_id"].nunique()
        assert result["history_cycles"].eq(1).all()
        assert result["is_warmup"].all()
        assert np.isfinite(result["predicted_rul"]).all()
