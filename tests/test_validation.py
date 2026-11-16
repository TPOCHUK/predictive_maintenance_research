import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from predictive_maintenance.api import dependencies
from predictive_maintenance.api.main import app
from predictive_maintenance.data.preprocessing import validate_data
from predictive_maintenance.inference.predictor import RULPredictor
from predictive_maintenance.schemas.telemetry import TelemetryInput


def measurement(**changes):
    row = {"unit_id": 1, "cycle": 1}
    row.update({f"setting_{i}": 0.0 for i in range(1, 4)})
    row.update({f"sensor_{i}": 1.0 for i in range(1, 22)})
    return {**row, **changes}


@pytest.mark.parametrize("value", [float("inf"), float("nan"), -float("inf")])
def test_sensor_values_must_be_finite(value):
    with pytest.raises(ValidationError):
        TelemetryInput.model_validate(measurement(sensor_1=value))


def test_duplicate_cycles_return_422(monkeypatch):
    monkeypatch.setattr(dependencies, "predictor", RULPredictor())
    row = measurement()
    response = TestClient(app).post("/predict", json={"measurements": [row, row]})
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]


@pytest.mark.parametrize("field", ["unit_id", "cycle"])
def test_dataframe_ids_must_be_whole_numbers(field):
    with pytest.raises(ValueError, match="whole numbers"):
        validate_data(pd.DataFrame([measurement(**{field: 1.5})]))
