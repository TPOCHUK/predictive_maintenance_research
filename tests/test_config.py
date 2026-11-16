from predictive_maintenance.config import load_pipeline_config


def test_load_pipeline_config():
    config = load_pipeline_config()

    assert config.baseline_cycles == 20
    assert config.detector_window == 10

    assert config.sensor_directions["sensor_4"] == 1
    assert config.sensor_directions["sensor_7"] == -1

    assert config.engineered_features == ("degradation_score",)
    assert config.flag_columns["lstm"] == "degradation_warmup"
    assert config.flag_columns["cnn"] == "degradation_available"

    assert all(direction in (-1, 1) for direction in config.sensor_directions.values())
